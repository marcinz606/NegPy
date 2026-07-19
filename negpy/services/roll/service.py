"""Roll-scanning service backed by the optional `coolscanpy` package.

Sibling of `negpy.services.capture.service.CaptureService`: one class that
orchestrates the hardware workflow and writes results to disk in NegPy's
conventional layout, on top of an infrastructure-layer adapter for an
optional dependency (`negpy.infrastructure.roll.coolscanpy_roll`, mirroring
`negpy.infrastructure.capture.gphoto`). It also mirrors the older, simpler
`negpy.services.scanning.service.ScannerService`: where that wraps a single
ad-hoc `Device.scan()`, this wraps coolscanpy's whole-roll workflow
(preview -> approve -> batch fine-scan).

Entirely inert if `coolscanpy` is not installed: `available()` is a cheap
presence check re-exported from that module, and every other method only
reaches coolscanpy (transitively, through `coolscanpy_roll`) once actually
called.

`write_frame` writes a captured frame across up to three output tiers --
unrepaired, repaired, positive -- each independently selectable and each
derived from the one before it. See its docstring for the tier definitions,
the write order, and how a tier that cannot be produced degrades instead of
losing the tiers that still can be. Tier 2 (repaired) is backed by
`negpy.infrastructure.roll.repair`, an optional-engine seam with nothing
registered in this codebase; Tier 3 (positive) is backed by
`negpy.services.roll.positive`, which reuses NegPy's own rendering engine.
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

from negpy.infrastructure.roll import coolscanpy_roll
from negpy.infrastructure.roll import repair as roll_repair
from negpy.infrastructure.roll.repair import RepairMode
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.roll import positive as roll_positive
from negpy.services.scanning.templating import render_scan_filename

if TYPE_CHECKING:
    import coolscanpy

logger = get_logger(__name__)

# Re-exported so callers only need this module: `from negpy.services.roll
# .service import available`, matching how the plain Scan sidebar checks
# `_sane_available()` before showing its device combo.
available = coolscanpy_roll.available


class RollScanningError(RuntimeError):
    """Raised for a roll-scanning failure that originates in this service's
    own orchestration (lifecycle misuse), as opposed to one translated from
    coolscanpy itself. Mirrors `negpy.services.capture.service.CaptureError`."""


@dataclasses.dataclass(frozen=True)
class RollFrameOutput:
    """Where one scanned frame's files landed on disk, one pair of fields per
    tier. A tier that was not selected, or could not be produced, leaves its
    field(s) `None` -- the receipt at `receipt_path` records which of those
    two it was and, for a tier that did write, its provenance."""

    slot: int
    rgb_path: str | None  # Tier 1 (unrepaired): RGB plane
    ir_path: str | None  # Tier 1 (unrepaired): infrared plane
    repaired_rgb_path: str | None  # Tier 2 (repaired): RGB plane
    repaired_ir_path: str | None  # Tier 2 (repaired): infrared plane (Tier 1's own, retained unchanged)
    positive_path: str | None  # Tier 3 (positive)
    receipt_path: str


class RollScanningService:
    """Orchestrates coolscanpy device/roll lifecycle and output writing.

    One roll open at a time, matching `coolscanpy.Device.roll()`'s own
    single-reservation lock -- call `close()` (or use this as a context
    manager) before opening another.
    """

    def __init__(self) -> None:
        self._roll: coolscanpy_roll.RollHandle | None = None
        # Built lazily, on the first frame that actually needs Tier 3 -- see
        # `_get_image_processor()`. Most batches never touch it.
        self._image_processor: ImageProcessor | None = None

    # -- device / roll lifecycle -----------------------------------------

    def list_devices(self) -> "list[coolscanpy.DeviceInfo]":
        return coolscanpy_roll.list_devices()

    def open_roll(self, device_id: str | None = None, *, material: "coolscanpy.Material | None" = None) -> None:
        """Open a device and its roll extension. Call `close()` when done."""
        if self._roll is not None:
            raise RollScanningError("a roll is already open on this service; call close() first")
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

    def preview(
        self, slots: Iterable[int] | None = None, *, on_progress: "coolscanpy.ProgressCallback | None" = None
    ) -> "list[coolscanpy.Thumbnail]":
        return self._require_roll().preview(slots, on_progress=on_progress)

    def set_spacing_offset(self, slot: int, offset_rows: int) -> None:
        self._require_roll().set_spacing_offset(slot, offset_rows)

    def approve(self, slot: int) -> None:
        self._require_roll().approve(slot)

    def needs_approval(self, slot: int) -> bool:
        return self._require_roll().needs_approval(slot)

    # -- scanning ----------------------------------------------------------

    def scan_many(
        self, slots: Iterable[int], *, on_progress: "coolscanpy.ProgressCallback | None" = None
    ) -> Iterator["coolscanpy.Frame"]:
        yield from self._require_roll().scan_many(slots, on_progress=on_progress)

    def safe_stop(self) -> None:
        if self._roll is not None:
            self._roll.safe_stop()

    # -- writing -------------------------------------------------------------

    def write_frame(
        self,
        frame: "coolscanpy.Frame",
        output_folder: str,
        filename_pattern: str,
        *,
        write_unrepaired: bool = True,
        write_repaired: bool = False,
        write_positive: bool = False,
        repair_mode: str = RepairMode.EXACT.value,
    ) -> RollFrameOutput:
        """Write one scanned `Frame` to disk across up to three tiers, plus a
        `_receipt.json` sidecar. Each tier is independently selectable; any
        combination of the three flags is valid.

        Tier 1, unrepaired (`write_unrepaired`): the frame exactly as
        captured -- a 16-bit RGB TIFF plus an `_IR` sidecar when the frame
        carries an infrared plane, matching `writer.write_tiff_16bit`'s own
        `<basename>_IR.tif` convention. This is the archival master: the
        only tier the scanner itself can reproduce. A failure writing it is
        allowed to raise, unlike Tier 2/3 below.

        Tier 2, repaired (`write_repaired`): Tier 1 with infrared-guided
        dust/scratch repair applied through `negpy.infrastructure.roll
        .repair`, written as `<basename>_repaired.tif` plus a
        `<basename>_repaired_IR.tif` sidecar. That sidecar is Tier 1's own
        infrared plane, unchanged -- repair consumes infrared to find
        defects, it does not produce a repaired version of it, and keeping
        the original lets a later re-repair under a different mode start
        from the same evidence Tier 1 captured.

        Tier 3, positive (`write_positive`): Tier 2 inverted through
        `negpy.services.roll.positive`, written as `<basename>_positive.tif`.
        Requested independently of `write_repaired`, but always derived from
        Tier 2's in-memory result -- repair runs whenever either flag needs
        it, so a positive is never silently missing the repair pass a caller
        asked for, and never from a Tier 1 or Tier 2 file already on disk
        (an upper tier is never derived from a lower tier's file, only from
        the in-memory data the lower tier is also optionally written from).

        Tier 2 and Tier 3 degrade instead of raising when they cannot be
        produced -- no repair engine registered, no infrared plane to guide
        one, the engine itself fails, or the render fails -- recording why in
        the receipt and leaving the corresponding `RollFrameOutput` path(s)
        `None`. Tier 1 always writes regardless (or Tier 2, for a Tier-3
        failure): losing the archival capture to a problem in a derived,
        regenerable tier is exactly what this must not do.

        `filename_pattern` is the same Jinja2 template the plain scan path
        uses (variables: `date`, `seq`), but `seq` is seeded from the frame's
        physical slot number rather than probed for the next free name: a
        roll slot is already a stable identity, so re-scanning slot 5
        replaces slot 5's old files instead of piling up a `..._002` beside
        them, for every tier.

        `frame.ir_validity` (the per-pixel IR confidence mask) is not
        persisted here -- it isn't part of NegPy's existing scan-output
        convention and nothing downstream reads it yet.
        """
        try:
            resolved_repair_mode = RepairMode(repair_mode)
        except ValueError:
            resolved_repair_mode = RepairMode.EXACT

        os.makedirs(output_folder, exist_ok=True)
        date_str = _date.today().strftime("%Y%m%d")
        basename = render_scan_filename(filename_pattern, date_str, frame.slot)
        base_path = os.path.join(output_folder, basename)

        outputs: dict = {}

        # -- Tier 1: unrepaired -----------------------------------------------
        rgb_path = ir_path = None
        if write_unrepaired:
            rgb_path = _atomic_write_tiff(base_path + ".tif", frame.rgb, photometric="rgb")
            if frame.ir is not None:
                ir_path = _atomic_write_tiff(base_path + "_IR.tif", frame.ir, photometric="minisblack")
            outputs["unrepaired"] = {"written": True, "rgb_path": rgb_path, "ir_path": ir_path}
        else:
            outputs["unrepaired"] = {"written": False, "status": "not selected"}

        # -- Tier 2: repaired (also feeds Tier 3, even when not itself written) --
        outputs["repaired"] = {"written": False, "status": "not selected"}
        repair_result: roll_repair.RepairResult | None = None
        repaired_rgb_path = repaired_ir_path = None
        if write_repaired or write_positive:
            if frame.ir is None:
                outputs["repaired"] = {"written": False, "status": "unavailable: frame has no infrared plane to guide repair"}
            elif not roll_repair.available():
                outputs["repaired"] = {"written": False, "status": "unavailable: no dust-repair engine registered"}
            else:
                try:
                    repair_result = roll_repair.repair(frame.rgb, frame.ir, resolved_repair_mode)
                except Exception as error:
                    outputs["repaired"] = {"written": False, "status": f"repair failed: {error}"}
                else:
                    entry = {
                        "engine": repair_result.engine,
                        "engine_version": repair_result.engine_version,
                        "mode": str(repair_result.mode),
                    }
                    if write_repaired:
                        repaired_rgb_path = _atomic_write_tiff(base_path + "_repaired.tif", repair_result.rgb, photometric="rgb")
                        repaired_ir_path = _atomic_write_tiff(base_path + "_repaired_IR.tif", frame.ir, photometric="minisblack")
                        entry.update(written=True, rgb_path=repaired_rgb_path, ir_path=repaired_ir_path)
                    else:
                        entry.update(written=False, status="not selected (computed in memory for the positive)")
                    outputs["repaired"] = entry

        # -- Tier 3: positive -----------------------------------------------
        outputs["positive"] = {"written": False, "status": "not selected"}
        positive_path = None
        if write_positive:
            if repair_result is None:
                tier2_status = outputs["repaired"].get("status", "unavailable")
                outputs["positive"] = {"written": False, "status": f"unavailable: Tier 2 (repaired) could not be produced ({tier2_status})"}
            elif not roll_positive.available():
                outputs["positive"] = {"written": False, "status": "unavailable: inversion path not available"}
            else:
                try:
                    result = roll_positive.render_positive(repair_result.rgb, processor=self._get_image_processor())
                except Exception as error:
                    outputs["positive"] = {"written": False, "status": f"inversion failed: {error}"}
                else:
                    positive_path = _atomic_write_tiff(base_path + "_positive.tif", result.rgb, photometric="rgb")
                    outputs["positive"] = {
                        "written": True,
                        "rgb_path": positive_path,
                        "inversion_path": "negpy.services.rendering.image_processor.ImageProcessor.run_pipeline",
                        "render_intent": result.render_intent,
                        "process_mode": result.process_mode,
                        "auto_exposure": result.auto_exposure,
                        "negpy_version": result.negpy_version,
                        "repair_engine": repair_result.engine,
                        "repair_engine_version": repair_result.engine_version,
                        "repair_mode": str(repair_result.mode),
                    }

        receipt_path = base_path + "_receipt.json"
        receipt_payload = dataclasses.asdict(frame.receipt)
        receipt_payload["outputs"] = outputs
        _atomic_write_json(receipt_path, receipt_payload)

        return RollFrameOutput(
            slot=frame.slot,
            rgb_path=rgb_path,
            ir_path=ir_path,
            repaired_rgb_path=repaired_rgb_path,
            repaired_ir_path=repaired_ir_path,
            positive_path=positive_path,
            receipt_path=receipt_path,
        )

    def _get_image_processor(self) -> ImageProcessor:
        """Built on first use and reused across a batch: its constructor
        probes for GPU acceleration, worth paying once per service instance
        rather than once per frame, even though Tier-3 rendering itself
        always runs on the CPU engine (see `positive.render_positive`)."""
        if self._image_processor is None:
            self._image_processor = ImageProcessor()
        return self._image_processor

    # -- internals -----------------------------------------------------------

    def _require_roll(self) -> coolscanpy_roll.RollHandle:
        if self._roll is None:
            raise RollScanningError("no roll is open; call open_roll() first")
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
