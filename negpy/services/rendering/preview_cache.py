from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional


from negpy.domain.types import AppConfig, Dimensions, ImageBuffer
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreviewCacheKey:
    file_hash: str
    use_camera_wb: bool
    workspace_color_space: str
    full_resolution: bool

    def as_tuple(self) -> Hashable:
        return (
            self.file_hash,
            self.use_camera_wb,
            self.workspace_color_space,
            self.full_resolution,
        )


@dataclass
class _Entry:
    buffer: ImageBuffer
    dims: Dimensions
    metadata: dict
    byte_size: int


class PreviewBufferCache:
    """
    In-memory LRU for decoded linear preview buffers. Evicts by entry count and approximate RSS.
    """

    def __init__(self, app_config: Optional[AppConfig] = None) -> None:
        self._app = app_config or APP_CONFIG
        self._order: list[Hashable] = []
        self._data: dict[Hashable, _Entry] = {}

    def get(self, key: PreviewCacheKey) -> Optional[tuple[ImageBuffer, Dimensions, dict]]:
        t = key.as_tuple()
        ent = self._data.get(t)
        if ent is None:
            return None
        self._order.remove(t)
        self._order.append(t)
        return ent.buffer, ent.dims, ent.metadata

    def put(self, key: PreviewCacheKey, buffer: ImageBuffer, dims: Dimensions, metadata: dict) -> None:
        t = key.as_tuple()
        b = int(buffer.nbytes)
        if b > self._app.preview_cache_max_bytes:
            # The byte cap would evict it immediately — don't churn the cache
            # (and don't evict everything else on the way out).
            logger.debug("preview cache skip: %d B entry exceeds %d B cap", b, self._app.preview_cache_max_bytes)
            return
        if key.full_resolution:
            # Full-res entries get a small slot budget (default 2: the active
            # frame plus the one navigated from, so going back is instant).
            # Unbudgeted, HQ buffers (hundreds of MB each) would evict every
            # small preview through the byte cap.
            limit = max(1, self._app.preview_cache_max_full_res_entries)
            full = [k for k in self._order if isinstance(k, tuple) and k[3] and k != t]
            for other in full[: max(0, len(full) - (limit - 1))]:  # trim oldest beyond budget
                self._remove_key(other)
        if t in self._data:
            self._order.remove(t)
        self._data[t] = _Entry(buffer=buffer, dims=dims, metadata=dict(metadata), byte_size=b)
        self._order.append(t)
        self._evict_if_needed()

    def invalidate_path_hash(self, file_hash: str) -> None:
        to_drop = [k for k in self._order if isinstance(k, tuple) and k and k[0] == file_hash]
        for t in to_drop:
            self._remove_key(t)

    def clear(self) -> None:
        self._order.clear()
        self._data.clear()

    def _remove_key(self, t: Hashable) -> None:
        self._data.pop(t, None)
        if t in self._order:
            self._order.remove(t)

    def _evict_if_needed(self) -> None:
        max_n = self._app.preview_cache_max_entries
        max_b = self._app.preview_cache_max_bytes

        def total_bytes() -> int:
            return sum(self._data[k].byte_size for k in self._order)

        while len(self._order) > max_n and self._order:
            t = self._order.pop(0)
            self._data.pop(t, None)
            logger.debug("preview cache evict (count): dropped entry")

        while total_bytes() > max_b and self._order:
            t = self._order.pop(0)
            self._data.pop(t, None)
            logger.debug("preview cache evict (bytes): dropped entry")
