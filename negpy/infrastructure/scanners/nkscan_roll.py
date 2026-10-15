# SPDX-License-Identifier: GPL-3.0-or-later
"""RollSession over nkscan: measure the strip once, then cut the previews out of that pass.

The first native RollSession. Where PerFrameRollSession guesses a frame's position from an
index and an offset the operator tunes by eye, nkscan measures every boundary on the loaded
film in one pass and hands that pass back, so a preview costs no scanning at all — a nudged
boundary re-previews without touching the scanner.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from typing import Any

import numpy as np

from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.nkscan_backend import _offset_units, _progress_bridge, _stack_rgb
from negpy.infrastructure.scanners.roll import RollPreview, effective_pitch_mm
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_PREVIEW_DEPTH_DPI = 0  # the strip pass has its own resolution; nothing chooses it


def slice_frame(strip: np.ndarray, rect: tuple[int, int, int, int], columns: np.ndarray | None) -> np.ndarray | None:
    """The frame's own pixels out of the strip pass, or None when it falls outside.

    `columns` is the feed address of each column. The top and bottom of a rect find their columns.
    """
    if columns is None or len(columns) != strip.shape[1]:
        return None
    top, _left, bottom, _right = rect
    # A frame before the first column starts at the first column. A frame after the last column
    # stops at the last column. Nikon Scan shows the frame in the same way.
    first = int(np.searchsorted(columns, top))
    last = int(np.searchsorted(columns, bottom))
    if last - first < 2:
        return None
    return strip[:, first:last]


class NkscanRollSession:
    """One measured strip: preview its frames, nudge their boundaries, release the unit."""

    def __init__(
        self,
        backend: Any,
        device: ScannerDevice,
        session: Any | None,
        model: str,
        *,
        dpi: int = _PREVIEW_DEPTH_DPI,
        film_format: str | None = None,
        film_type: str = "negative",
    ) -> None:
        self._backend = backend
        self._device = device
        self._session = session
        self._model = model
        self._dpi = int(dpi)
        self._film_format = film_format
        self._film_type = film_type
        self._offsets: dict[int, float] = {}
        self._approved: set[int] = set()
        # Only the fallback meters, and one strip is one exposure decision.
        self._exposures: dict[str, int] | None = None
        self._closed = False
        self.slot_count = len(backend.measured_frames(device.id, film_format, film_type))
        # An absolute rect can be re-addressed backwards, unlike a within-frame offset.
        self.offset_range = (-1.0, 1.0)
        self.supports_single_slot_preview = True
        self._pitch_mm = effective_pitch_mm(device.capabilities)

    # ── preview ───────────────────────────────────────────────────────

    def preview(
        self, slots: Iterable[int], *, cancel: threading.Event, progress: Callable[..., None] | None = None
    ) -> Iterator[RollPreview]:
        self._require_open()
        if cancel.is_set():
            return
        frames = self._ensure_frames(cancel, progress)
        if not frames:
            return
        for slot in slots:
            if cancel.is_set():
                return
            if not 1 <= slot <= len(frames):
                # The dialog may ask for more slots than this strip turned out to hold.
                continue
            offset = self._reached(slot)
            try:
                rgb = self._preview_one(slot, cancel, progress)
            except Exception as error:
                if cancel.is_set():
                    return
                logger.warning("Preview of slot %s failed: %s", slot, error)
                yield RollPreview(slot=slot, error=str(error), offset=offset, needs_approval=self._needs_approval(slot))
                continue
            cut = partial(self.cut, slot) if self.thumbnail is not None else None
            yield RollPreview(slot=slot, rgb=rgb, offset=offset, needs_approval=self._needs_approval(slot), cut=cut)

    def cut(self, slot: int, offset: float) -> RollPreview | None:
        """The slot at `offset`, cut from the strip pass. This does not use the unit. None if the frame is not on the pass."""
        strip = self.thumbnail
        if strip is None or not 1 <= slot <= len(self._backend.frames(self._device.id)):
            return None
        self.set_offset(slot, offset)
        tile = slice_frame(strip, self._rect(slot), self._backend.column_addresses(self._device.id))
        if tile is None:
            return None
        return RollPreview(
            slot=slot,
            rgb=tile,
            offset=self._reached(slot),
            needs_approval=self._needs_approval(slot),
            cut=partial(self.cut, slot),
        )

    @property
    def thumbnail(self) -> np.ndarray | None:
        """The strip pass every preview is cut from, where the mechanism took one."""
        return self._backend.strip_pass(self._device.id)

    def _preview_one(self, slot: int, cancel: threading.Event, progress: Callable[..., None] | None) -> np.ndarray:
        rect = self._rect(slot)
        strip = self.thumbnail
        if strip is not None:
            tile = slice_frame(strip, rect, self._backend.column_addresses(self._device.id))
            if tile is not None:
                return tile
            logger.info("Slot %s falls outside the strip pass; scanning it instead", slot)
        return self._scan_preview(rect, cancel, progress)

    def _scan_preview(self, rect: tuple[int, int, int, int], cancel: threading.Event, progress: Callable[..., None] | None) -> np.ndarray:
        """A pass of one frame, for a mechanism that measured the film without a strip pass."""
        with self._backend._dropping(self._device.id), self._backend._mapped_errors():
            result = self._backend.scan_frame(
                self._unit(),
                rect,
                dpi=self._dpi or None,
                samples=1,
                infrared=False,
                clean=False,
                exposures=self._exposures,
                progress=_progress_bridge(progress, cancel),
                **self._backend.metering(self._device.id, self._film_type, self._film_format),
            )
        if self._exposures is None:
            self._exposures = dict(result.exposures)
        return _stack_rgb(result.colors)

    def _ensure_frames(self, cancel: threading.Event, progress: Callable[..., None] | None) -> list[tuple[int, int, int, int]]:
        frames = self._backend.measured_frames(self._device.id, self._film_format, self._film_type)
        if not frames:
            with self._backend._dropping(self._device.id):
                self._backend.discover_frames(
                    self._unit(),
                    self._device.id,
                    film_format=self._film_format,
                    film_type=self._film_type,
                    progress=_progress_bridge(progress, cancel),
                )
            frames = self._backend.frames(self._device.id)
        self.slot_count = len(frames)
        return frames

    # ── boundaries ────────────────────────────────────────────────────

    def set_offset(self, slot: int, offset: float) -> None:
        lo, hi = self.offset_range
        self._offsets[slot] = max(lo, min(hi, float(offset)))

    def approve(self, slot: int) -> None:
        self._approved.add(slot)

    def _needs_approval(self, slot: int) -> bool:
        """Every boundary here was measured rather than addressed, so it needs a look."""
        return slot not in self._approved

    def _rect(self, slot: int) -> tuple[int, int, int, int]:
        """The rect of the slot, moved by its offset and kept on the strip pass.

        The offset is a fraction of one frame pitch. The scan gets the same offset in millimetres.
        The backend applies the same limit to the two, thus the tile and the scan show the same film.
        """
        rect = self._backend.frames(self._device.id)[slot - 1]
        offset_mm = self._offsets.get(slot, 0.0) * self._pitch_mm
        return self._backend.nudged(self._device.id, rect, offset_mm, self._optical_dpi())

    def _reached(self, slot: int) -> float:
        """The offset that the slot uses after the limit, as a fraction of the pitch."""
        asked = self._offsets.get(slot, 0.0)
        frames = self._backend.frames(self._device.id)
        if not 1 <= slot <= len(frames) or not self._pitch_mm:
            return asked
        optical = self._optical_dpi()
        moved = self._rect(slot)[0] - frames[slot - 1][0]
        if moved == _offset_units(asked * self._pitch_mm, optical):
            return asked
        return moved * 25.4 / optical / self._pitch_mm

    def _optical_dpi(self) -> int:
        return self._backend.optical_dpi(self._device.id) or int(self._unit().capabilities.optical_dpi)

    def _unit(self) -> Any:
        """The staged unit. This stages the unit again if the backend opened it again. A tile from the strip pass does not need it."""
        if self._session is None or self._session is not self._backend._units.get(self._device.id):
            self._session, self._model = self._backend._open(self._device.id)
        return self._session

    # ── lifetime ──────────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        # The unit stays open in the backend.
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Strip session for {self._device.id} is closed")
