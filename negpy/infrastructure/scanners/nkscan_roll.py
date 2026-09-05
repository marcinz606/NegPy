# SPDX-License-Identifier: GPL-3.0-or-later
"""RollSession over nkscan: measure the strip once, then cut the previews out of that pass.

The first native RollSession. Where PerFrameRollSession guesses a frame's position from an
index and an offset the operator tunes by eye, nkscan measures every boundary on the loaded
film in one pass and hands that pass back, so a preview costs no scanning at all — a nudged
boundary re-previews without touching the scanner.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import suppress
from typing import Any

import numpy as np

from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.nkscan_backend import _offset_units, _progress_bridge, _shift_frame, _stack_rgb
from negpy.infrastructure.scanners.roll import RollPreview, effective_pitch_mm
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_PREVIEW_DEPTH_DPI = 0  # the strip pass has its own resolution; nothing chooses it


def thumbnail_scale(optical_dpi: int, thumbnail_dpi: int) -> float:
    """Stage addresses per thumbnail column.

    A column is one line pitch of film and the pass starts at the axis origin, so a column is
    a feed address. The pitch is a whole number of addresses; the resolution the unit reports
    for the pass is that pitch rounded down, so the trip back through it has to round.
    """
    if optical_dpi <= 0 or thumbnail_dpi <= 0:
        return 0.0
    return float(round(optical_dpi / thumbnail_dpi))


def slice_frame(strip: np.ndarray, rect: tuple[int, int, int, int], scale: float) -> np.ndarray | None:
    """The frame's own pixels out of the strip pass, or None when it falls outside.

    Columns are feed addresses from the axis start, so a rect maps straight onto them.
    """
    if scale <= 0:
        return None
    top, _left, bottom, _right = rect
    first = int(round(top / scale))
    last = int(round(bottom / scale))
    if first < 0 or last > strip.shape[1] or last - first < 2:
        return None
    return strip[:, first:last]


class NkscanRollSession:
    """One measured strip: preview its frames, nudge their boundaries, release the unit."""

    def __init__(
        self,
        backend: Any,
        device: ScannerDevice,
        session: Any,
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
        self.slot_count = len(backend.frames(device.id))
        # An absolute rect can be re-addressed backwards, unlike a within-frame offset.
        self.offset_range = (-1.0, 1.0)
        self.supports_single_slot_preview = True
        self._pitch_mm = effective_pitch_mm(device.capabilities)

    # ── preview ───────────────────────────────────────────────────────

    def preview(self, slots: Iterable[int], *, cancel: threading.Event) -> Iterator[RollPreview]:
        self._require_open()
        if cancel.is_set():
            return
        frames = self._ensure_frames(cancel)
        if not frames:
            return
        for slot in slots:
            if cancel.is_set():
                return
            if not 1 <= slot <= len(frames):
                # The dialog may ask for more slots than this strip turned out to hold.
                continue
            offset = self._offsets.get(slot, 0.0)
            try:
                rgb = self._preview_one(slot, cancel)
            except Exception as error:
                if cancel.is_set():
                    return
                logger.warning("Preview of slot %s failed: %s", slot, error)
                yield RollPreview(slot=slot, error=str(error), offset=offset, needs_approval=self._needs_approval(slot))
                continue
            yield RollPreview(slot=slot, rgb=rgb, offset=offset, needs_approval=self._needs_approval(slot))

    @property
    def thumbnail(self) -> np.ndarray | None:
        """The strip pass every preview is cut from, where the mechanism took one."""
        return self._backend.strip_pass(self._device.id)

    def _preview_one(self, slot: int, cancel: threading.Event) -> np.ndarray:
        rect = self._rect(slot)
        strip = self.thumbnail
        if strip is not None:
            tile = slice_frame(strip, rect, self._scale())
            if tile is not None:
                return tile
            logger.info("Slot %s falls outside the strip pass; scanning it instead", slot)
        return self._scan_preview(rect, cancel)

    def _scale(self) -> float:
        caps = self._session.capabilities
        dpi = tuple(caps.thumbnail_dpi)
        return thumbnail_scale(int(caps.optical_dpi), int(dpi[0]) if dpi else 0)

    def _scan_preview(self, rect: tuple[int, int, int, int], cancel: threading.Event) -> np.ndarray:
        """A pass of one frame, for a mechanism that measured the film without a strip pass."""
        with self._backend._mapped_errors():
            result = self._backend.scan_frame(
                self._session,
                rect,
                dpi=self._dpi or None,
                samples=1,
                infrared=False,
                clean=False,
                lock_white_balance=self._backend.locks_white_balance(self._film_type),
                exposures=self._exposures,
                progress=_progress_bridge(None, cancel),
            )
        if self._exposures is None:
            self._exposures = dict(result.exposures)
        return _stack_rgb(result.colors)

    def _ensure_frames(self, cancel: threading.Event) -> list[tuple[int, int, int, int]]:
        frames = self._backend.frames(self._device.id)
        if not frames:
            self._backend.discover_frames(
                self._session,
                self._device.id,
                film_format=self._film_format,
                film_type=self._film_type,
                progress=_progress_bridge(None, cancel),
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
        """The slot's rect, slid by its offset.

        The offset is a fraction of one frame pitch, the same units the fine scan receives as
        millimetres, so preview and scan land on the same film.
        """
        rect = self._backend.frames(self._device.id)[slot - 1]
        offset_mm = self._offsets.get(slot, 0.0) * self._pitch_mm
        optical = int(self._session.capabilities.optical_dpi)
        return _shift_frame(rect, _offset_units(offset_mm, optical))

    # ── lifetime ──────────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            self._session.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Strip session for {self._device.id} is closed")
