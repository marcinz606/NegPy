"""Roll-scanning service backed by the optional `coolscanpy` package.

Mirrors `negpy.services.scanning.service.ScannerService`: one class that
orchestrates device discovery, the hardware workflow, and writing results to
disk in NegPy's conventional layout. Where `ScannerService` wraps a single
ad-hoc `Device.scan()`, this wraps coolscanpy's whole-roll workflow
(preview -> approve -> batch fine-scan) exposed by
`negpy.infrastructure.scanners.coolscanpy_roll`.

Entirely inert if `coolscanpy` is not installed: `available()` is a cheap
presence check re-exported from that module, and every other method only
reaches coolscanpy (transitively, through `coolscanpy_roll`) once actually
called.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from datetime import date as _date
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np
import tifffile

from negpy.infrastructure.scanners import coolscanpy_roll
from negpy.kernel.system.logging import get_logger
from negpy.services.scanning.templating import render_scan_filename

if TYPE_CHECKING:
    import coolscanpy

logger = get_logger(__name__)

# Re-exported so callers only need this module: `from negpy.services.scanning
# .roll_service import available`, matching how the plain Scan sidebar checks
# `_sane_available()` before showing its device combo.
available = coolscanpy_roll.available


@dataclasses.dataclass(frozen=True)
class RollFrameOutput:
    """Where one scanned frame's files landed on disk."""

    slot: int
    rgb_path: str
    ir_path: str | None
    receipt_path: str


class RollScanningService:
    """Orchestrates coolscanpy device/roll lifecycle and output writing.

    One roll open at a time, matching `coolscanpy.Device.roll()`'s own
    single-reservation lock -- call `close()` (or use this as a context
    manager) before opening another.
    """

    def __init__(self) -> None:
        self._roll: coolscanpy_roll.RollHandle | None = None

    # -- device / roll lifecycle -----------------------------------------

    def list_devices(self) -> "list[coolscanpy.DeviceInfo]":
        return coolscanpy_roll.list_devices()

    def open_roll(self, device_id: str | None = None, *, material: "coolscanpy.Material | None" = None) -> None:
        """Open a device and its roll extension. Call `close()` when done."""
        if self._roll is not None:
            raise RuntimeError("a roll is already open on this service; call close() first")
        self._roll = coolscanpy_roll.open_roll(device_id, material=material)

    def close(self) -> None:
        """Idempotent. Ends the roll reservation and releases the device."""
        if self._roll is not None:
            self._roll.close()
            self._roll = None

    def __enter__(self) -> "RollScanningService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- preview / approval ------------------------------------------------

    def preview(self, slots: Iterable[int] | None = None, *, on_progress=None) -> list:
        return self._require_roll().preview(slots, on_progress=on_progress)

    def set_spacing_offset(self, slot: int, offset_rows: int) -> None:
        self._require_roll().set_spacing_offset(slot, offset_rows)

    def approve(self, slot: int) -> None:
        self._require_roll().approve(slot)

    def needs_approval(self, slot: int) -> bool:
        return self._require_roll().needs_approval(slot)

    # -- scanning ----------------------------------------------------------

    def scan_many(self, slots: Iterable[int], *, on_progress=None) -> Iterator["coolscanpy.Frame"]:
        yield from self._require_roll().scan_many(slots, on_progress=on_progress)

    def safe_stop(self) -> None:
        if self._roll is not None:
            self._roll.safe_stop()

    # -- writing -------------------------------------------------------------

    def write_frame(self, frame: "coolscanpy.Frame", output_folder: str, filename_pattern: str) -> RollFrameOutput:
        """Write one scanned `Frame` to disk: a 16-bit TIFF master, an `_IR`
        TIFF sidecar when the frame carries an infrared plane (matching
        `writer.write_tiff_16bit`'s own `<basename>_IR.tif` convention), and
        a `_receipt.json` sidecar holding the full `Receipt`.

        `filename_pattern` is the same Jinja2 template the plain scan path
        uses (variables: `date`, `seq`), but `seq` is seeded from the frame's
        physical slot number rather than probed for the next free name: a
        roll slot is already a stable identity, so re-scanning slot 5
        replaces slot 5's old files instead of piling up a `..._002` beside
        them.

        `frame.ir_validity` (the per-pixel IR confidence mask) is not
        persisted here -- it isn't part of NegPy's existing scan-output
        convention and nothing downstream reads it yet.
        """
        os.makedirs(output_folder, exist_ok=True)
        date_str = _date.today().strftime("%Y%m%d")
        basename = render_scan_filename(filename_pattern, date_str, frame.slot)
        base_path = os.path.join(output_folder, basename)

        rgb_path = _atomic_write_tiff(base_path + ".tif", frame.rgb, photometric="rgb")

        ir_path = None
        if frame.ir is not None:
            ir_path = _atomic_write_tiff(base_path + "_IR.tif", frame.ir, photometric="minisblack")

        receipt_path = base_path + "_receipt.json"
        _atomic_write_json(receipt_path, dataclasses.asdict(frame.receipt))

        return RollFrameOutput(slot=frame.slot, rgb_path=rgb_path, ir_path=ir_path, receipt_path=receipt_path)

    # -- internals -----------------------------------------------------------

    def _require_roll(self) -> coolscanpy_roll.RollHandle:
        if self._roll is None:
            raise RuntimeError("no roll is open; call open_roll() first")
        return self._roll


def _ensure_uint16(array: np.ndarray) -> np.ndarray:
    return array if array.dtype == np.uint16 else array.astype(np.uint16)


def _atomic_write_tiff(path: str, array: np.ndarray, *, photometric: str) -> str:
    """Write `array` to `path` via a temp file + rename, matching the
    atomic-write convention `negpy.services.scanning.writer` already uses
    for the plain scan path."""
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        tifffile.imwrite(tmp_path, _ensure_uint16(array), photometric=photometric, compression="lzw")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def _atomic_write_json(path: str, payload: dict) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path) or ".", delete=False, suffix=".part", encoding="utf-8") as tmp:
            tmp_path = tmp.name
            json.dump(payload, tmp, default=str, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path
