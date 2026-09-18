# SPDX-License-Identifier: GPL-3.0-or-later
"""RollSession over plusteck: one real low-res tray pass, sliced into per-slot previews.

Mirrors nkscan_roll.py's shape (measure the whole strip once, cut previews out of
that one pass) but for plusteck's Plustek OpticFilm 135i, where "measuring" means
running plusteck's own Preview (Color, Low 600 dpi -- the fastest tier with a real
capture) and letting plusteck's own per-slot half-frame split do the boundary
detection, rather than reimplementing that heuristic a second time here.

No boundary nudging yet (set_offset/approve are no-ops): plusteck's own auto-split
is what this reads, and re-splitting with an explicit left_end/right_start via
plusteck's own POST .../split endpoint would be the natural way to add it later,
not something to build blind against a UI this adapter can't drive to verify.
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
            # needs_approval=True unconditionally: plusteck's own split boundary is
            # content-based auto-detection, explicitly documented (plusteck's
            # split_output.py) as unreliable on real photo content -- never
            # measured/exact the way nkscan's strip pass is.
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
