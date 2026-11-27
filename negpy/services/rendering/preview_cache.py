from __future__ import annotations

import threading
from collections.abc import Collection
from dataclasses import dataclass
from typing import Hashable, Optional

import numpy as np

from negpy.domain.types import AppConfig, Dimensions, ImageBuffer
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _entry_bytes(buffer: ImageBuffer, metadata: dict) -> int:
    """Count retained array allocations, including shared views, once per entry."""
    allocations: dict[int, int] = {}
    for value in (buffer, *metadata.values()):
        if isinstance(value, np.ndarray):
            while isinstance(value.base, np.ndarray):
                value = value.base
            allocations[id(value)] = int(value.nbytes)
    return sum(allocations.values())


@dataclass(frozen=True)
class PreviewCacheKey:
    file_hash: str
    use_camera_wb: bool
    workspace_color_space: str
    full_resolution: bool
    demosaic: str = "Auto"
    half: int = 0
    split_x: float = 0.5
    crop_rect: tuple[float, float, float, float] | None = None
    gutter_thickness: float = 0.0
    split_axis: str = "x"
    positive_decode: str = ""
    highlight_mode: int = 0
    bake_camera_wb: bool = False
    lens_token: str = ""

    def as_tuple(self) -> Hashable:
        return (
            self.file_hash,
            self.use_camera_wb,
            self.workspace_color_space,
            self.full_resolution,
            self.demosaic,
            self.half,
            round(self.split_x, 6),
            self.crop_rect,
            round(self.gutter_thickness, 6),
            self.split_axis,
            self.positive_decode,
            self.highlight_mode,
            self.bake_camera_wb,
            self.lens_token,
        )


@dataclass
class _Entry:
    buffer: ImageBuffer
    dims: Dimensions
    metadata: dict
    byte_size: int


@dataclass(frozen=True)
class PreviewCacheUsage:
    entries: int
    bytes_used: int
    entries_remaining: int
    bytes_remaining: int
    reclaimable_entries: int
    reclaimable_bytes: int


class PreviewBufferCache:
    """
    In-memory LRU for decoded linear preview buffers. Evicts by entry count and approximate RSS.

    One manager is shared by batch workers that decode several frames at once, so every
    public method holds the lock: the LRU list and the entry map must move together, and
    an unguarded ``_order.remove`` raises when two threads recycle the same key.
    """

    def __init__(self, app_config: Optional[AppConfig] = None) -> None:
        self._app = app_config or APP_CONFIG
        self._order: list[Hashable] = []
        self._data: dict[Hashable, _Entry] = {}
        self._lock = threading.RLock()

    def get(self, key: PreviewCacheKey) -> Optional[tuple[ImageBuffer, Dimensions, dict]]:
        t = key.as_tuple()
        with self._lock:
            ent = self._data.get(t)
            if ent is None:
                return None
            self._order.remove(t)
            self._order.append(t)
            return ent.buffer, ent.dims, ent.metadata

    def peek(self, key: PreviewCacheKey) -> Optional[tuple[ImageBuffer, Dimensions, dict]]:
        """Like ``get``, but leaves the LRU order as it is."""
        with self._lock:
            ent = self._data.get(key.as_tuple())
            return None if ent is None else (ent.buffer, ent.dims, ent.metadata)

    def put(
        self,
        key: PreviewCacheKey,
        buffer: ImageBuffer,
        dims: Dimensions,
        metadata: dict,
        *,
        protected_file_hashes: Collection[str] = (),
        preserve_full_resolution: bool = False,
    ) -> None:
        t = key.as_tuple()
        b = _entry_bytes(buffer, metadata)
        if b > self._app.preview_cache_max_bytes:
            # The byte cap would evict it immediately, so do not churn the cache, or evict everything
            # else on the way out.
            logger.debug("preview cache skip: %d B entry exceeds %d B cap", b, self._app.preview_cache_max_bytes)
            return
        with self._lock:
            if key.full_resolution:
                # Full-res entries get a small slot budget (default 2: the active frame plus the one
                # navigated from, so going back is instant). Unbudgeted, HQ buffers of hundreds of MB
                # each would evict every small preview through the byte cap.
                limit = max(1, self._app.preview_cache_max_full_res_entries)
                full = [k for k in self._order if isinstance(k, tuple) and len(k) > 3 and k[3] and k != t]
                need_drop = max(0, len(full) - (limit - 1))
                if need_drop:
                    prot = frozenset(protected_file_hashes)
                    unprotected = [k for k in full if not self._is_protected(k, prot, False)]
                    protected_list = [k for k in full if self._is_protected(k, prot, False)]
                    for other in (unprotected + protected_list)[:need_drop]:
                        self._remove_key(other)
            if t in self._data:
                self._order.remove(t)
            self._data[t] = _Entry(buffer=buffer, dims=dims, metadata=dict(metadata), byte_size=b)
            self._order.append(t)
            self._evict_if_needed(frozenset(protected_file_hashes), preserve_full_resolution)

    def invalidate_path_hash(self, file_hash: str) -> None:
        with self._lock:
            to_drop = [k for k in self._order if isinstance(k, tuple) and k and k[0] == file_hash]
            for t in to_drop:
                self._remove_key(t)

    def clear(self) -> None:
        with self._lock:
            self._order.clear()
            self._data.clear()

    def contains(self, key: PreviewCacheKey) -> bool:
        with self._lock:
            return key.as_tuple() in self._data

    def usage(
        self,
        *,
        protected_file_hashes: Collection[str] = (),
        preserve_full_resolution: bool = False,
        for_full_resolution: bool = False,
    ) -> PreviewCacheUsage:
        with self._lock:
            entries = len(self._data)
            bytes_used = sum(entry.byte_size for entry in self._data.values())
            protected = frozenset(protected_file_hashes)
            reclaimable = [key for key in self._order if not self._is_protected(key, protected, preserve_full_resolution)]
            entries_remaining = max(0, self._app.preview_cache_max_entries - entries)
            reclaimable_entries = len(reclaimable)
            if for_full_resolution:
                max_full = max(1, self._app.preview_cache_max_full_res_entries)
                full_keys = [key for key in self._order if isinstance(key, tuple) and len(key) > 3 and bool(key[3])]
                full_reclaimable = [key for key in full_keys if not self._is_protected(key, protected, False)]
                full_free = max(0, max_full - len(full_keys))
                overall_slots = entries_remaining + reclaimable_entries
                full_slots = full_free + len(full_reclaimable)
                effective_slots = min(overall_slots, full_slots)
                entries_remaining = min(entries_remaining, full_free)
                reclaimable_entries = max(0, effective_slots - entries_remaining)
            return PreviewCacheUsage(
                entries=entries,
                bytes_used=bytes_used,
                entries_remaining=entries_remaining,
                bytes_remaining=max(0, self._app.preview_cache_max_bytes - bytes_used),
                reclaimable_entries=reclaimable_entries,
                reclaimable_bytes=sum(self._data[key].byte_size for key in reclaimable),
            )

    def _remove_key(self, t: Hashable) -> None:
        self._data.pop(t, None)
        if t in self._order:
            self._order.remove(t)

    @staticmethod
    def _is_protected(
        key: Hashable,
        protected_file_hashes: frozenset[str],
        preserve_full_resolution: bool,
    ) -> bool:
        if not isinstance(key, tuple) or not key or not isinstance(key[0], str):
            return False
        if preserve_full_resolution and len(key) > 3 and bool(key[3]):
            return True
        if not protected_file_hashes:
            return False
        file_hash = key[0]
        if file_hash in protected_file_hashes:
            return True
        parts = file_hash.split("|", 2)
        return len(parts) == 3 and parts[0] in {"rgb", "hdr", "stitch"} and parts[1] in protected_file_hashes

    def _evict_if_needed(
        self,
        protected_file_hashes: frozenset[str] = frozenset(),
        preserve_full_resolution: bool = False,
    ) -> None:
        max_n = self._app.preview_cache_max_entries
        max_b = self._app.preview_cache_max_bytes

        def total_bytes() -> int:
            return sum(self._data[k].byte_size for k in self._order)

        def evict(reason: str) -> bool:
            key = next(
                (item for item in self._order if not self._is_protected(item, protected_file_hashes, preserve_full_resolution)),
                None,
            )
            if key is None:
                return False
            self._remove_key(key)
            logger.debug("preview cache evict (%s): dropped entry", reason)
            return True

        while len(self._order) > max_n and self._order:
            if not evict("count"):
                break

        while total_bytes() > max_b and self._order:
            if not evict("bytes"):
                break
