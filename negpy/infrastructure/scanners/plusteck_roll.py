# SPDX-License-Identifier: GPL-3.0-or-later
"""RollSession over plusteck: one real low-res tray pass, sliced into per-slot previews.

Takes nkscan_roll.py's shape, measuring the whole strip once and cutting previews out of
that pass, for the Plustek OpticFilm 135i. Measuring here means plusteck's own Preview
(Color, Low 600 dpi, its fastest real capture) with plusteck's per-slot half-frame split
finding the boundaries, instead of a second copy of that heuristic.

set_offset and approve are no-ops: the boundaries are plusteck's auto-split. Nudging them
would go through its POST .../split endpoint with an explicit left_end/right_start.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.plusteck_backend import (
    _ensure_all_split,
    _http_json,
    _read_half_frame,
    _wait_for_scan,
)
from negpy.infrastructure.scanners.roll import RollPreview
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


class PlusteckRollSession:
    """One measured tray: preview its half-frame slots, release the unit.

    `slot_count` starts at 0 and only becomes accurate after the first
    `preview()` call actually runs the one real pass this needs -- see
    ScannerCapabilities.roll_discovery's own docstring for why that's
    correct here rather than a bug (the tray's real slot count, 0-6 loaded
    positions each possibly split into 2 half-frames, isn't knowable any
    other way on this transport).
    """

    def __init__(self, base_url: str, device: ScannerDevice, *, dpi: int) -> None:
        del dpi  # plusteck's own Preview is fixed to its fastest real tier, not caller-chosen
        self._base_url = base_url
        self._device = device
        self._closed = False
        self._frames: list[Path] | None = None
        self.slot_count = 0
        self.offset_range = (0.0, 0.0)
        # Previewing here means one full low-res tray pass, same cost
        # whether the caller asks for one slot or all of them -- there is
        # no cheaper "just this slot" path to offer.
        self.supports_single_slot_preview = False

    def preview(self, slots: Iterable[int], *, cancel: threading.Event) -> Iterator[RollPreview]:
        self._require_open()
        if cancel.is_set():
            return
        self._ensure_previewed(cancel)
        assert self._frames is not None
        for slot in slots:
            if cancel.is_set():
                return
            index = slot - 1
            if not 0 <= index < len(self._frames):
                continue  # the dialog may ask for more slots than this tray load turned out to hold
            try:
                rgb = _read_half_frame(self._frames[index])
            except Exception as error:
                logger.warning("Preview of slot %s failed: %s", slot, error)
                yield RollPreview(slot=slot, error=str(error), needs_approval=True)
                continue
            # Every preview needs approval: plusteck's split boundary is content-based
            # auto-detection, documented in its split_output.py as unreliable on real photo
            # content, rather than the measured boundary nkscan's strip pass gives.
            yield RollPreview(slot=slot, rgb=rgb, needs_approval=True)

    def discover(self, cancel: threading.Event) -> int:
        """Run the tray-measuring Preview pass if it hasn't already, and report the slot count.

        The `detect_frames` half of the roll_discovery contract (see
        `ScannerCapabilities.roll_discovery`'s own docstring): the same one real pass
        `preview()` needs, without asking the caller to iterate slots it doesn't want
        yet -- a "whole strip" scan request has none to name until this runs.
        """
        self._require_open()
        self._ensure_previewed(cancel)
        return self.slot_count

    def set_offset(self, slot: int, offset: float) -> None:
        del slot, offset  # not supported yet -- see module docstring

    def approve(self, slot: int) -> None:
        del slot  # no-op to match set_offset -- nothing here actually gates on approval yet

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Roll session for {self._device.id} is closed")

    def _ensure_previewed(self, cancel: threading.Event) -> None:
        if self._frames is not None:
            return
        session_id = f"negpy_preview_{int(time.time() * 1000)}"
        response = _http_json(
            "POST",
            self._base_url,
            "/api/scanner/scan/preview",
            {"session_id": session_id},
        )
        if response.get("scan_state") == "error":
            raise RuntimeError(response.get("scan_error") or "preview request rejected")
        _wait_for_scan(self._base_url, session_id, None, cancel)
        self._frames = _ensure_all_split(self._base_url, session_id)
        self.slot_count = len(self._frames)
