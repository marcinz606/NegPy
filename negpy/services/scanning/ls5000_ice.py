"""Roll-workflow service layer for portable Digital ICE captures.

One selected roll slot becomes one dual-RGBI acquisition, one on-disk bundle,
one engine invocation, and one published cleaned TIFF with a provenance
receipt.  The three steps are deliberately separate functions with the bundle
as the only hand-off between acquisition and processing: the scanner handle is
closed before the engine runs, and a processing failure leaves the bundle on
disk as the recovery artifact instead of forcing a rescan.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from negpy.infrastructure.scanners.dice_dual_source_runner import (
    DiceDualSourcePlan,
    Libsane,
    acquire_dual_sources,
    load_capture_bundle,
    write_capture_bundle,
)
from negpy.infrastructure.scanners.portable_digital_ice_adapter import (
    PortableDigitalIceResult,
    apply_portable_digital_ice,
)
from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.ls5000_roll_outputs import _write_receipt_exclusive
from negpy.services.scanning.ls5000_sane_rgb import (
    LS5000_FINE_WIDTH,
    LS5000_FULL_WINDOW_ROWS,
    LS5000_NATIVE_DPI,
    orient_sane_rgb_for_storage,
)

ICE_FRAME_RECEIPT_KIND = "negpy.ls5000-ice-frame"
ICE_FRAME_RECEIPT_VERSION = 1


class IceRollError(RuntimeError):
    """A roll ICE step failed in a way the worker must surface verbatim."""


def acquire_ice_bundle(
    *,
    device_id: str,
    plan: DiceDualSourcePlan,
    bundle_root: Path,
    run_id: str,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """Acquire one dual-RGBI pair and persist it; the handle never outlives this call.

    The SANE session is opened and closed inside this function so the caller
    can prove the scanner is released before any processing starts.  The
    bundle is written after both closes: the arrays are already in memory and
    disk I/O has no business extending a hardware reservation.
    """

    sane = Libsane()
    try:
        identity = sane.require_ls5000(device_id)
        device = sane.open(identity.device_id, identity=identity)
        try:
            capture = acquire_dual_sources(device, plan, progress=progress)
        finally:
            device.close()
    finally:
        sane.close()
    return write_capture_bundle(
        bundle_root,
        device_id=capture.scanner_identity.device_id,
        plan=plan,
        capture=capture,
        run_id=run_id,
    )


@dataclass(frozen=True)
class ProcessedIceFrame:
    """One engine invocation's output plus the identity evidence around it."""

    ice: PortableDigitalIceResult
    plan: DiceDualSourcePlan
    bundle_manifest_sha256: str
    device_model: str


def process_ice_bundle(
    bundle_dir: Path,
    *,
    backend: str,
    progress: Callable[[object], None] | None = None,
) -> ProcessedIceFrame:
    """Reload a bundle through its full integrity gate and run the engine."""

    capture, plan = load_capture_bundle(bundle_dir)
    receipt = json.loads((Path(bundle_dir) / "receipt.json").read_text(encoding="utf-8"))
    manifest_sha256 = str(receipt["manifest_sha256"])
    ice = apply_portable_digital_ice(
        capture,
        plan=plan,
        backend=backend,
        progress=progress,
    )
    identity = capture.scanner_identity
    return ProcessedIceFrame(
        ice=ice,
        plan=plan,
        bundle_manifest_sha256=manifest_sha256,
        device_model=f"{identity.vendor} {identity.model}",
    )


def build_ice_receipt(
    processed: ProcessedIceFrame,
    *,
    roll_slot: int,
    boundary_offset_rows: int,
) -> dict[str, Any]:
    """Assemble the provenance receipt published beside the cleaned TIFF."""

    return {
        "kind": ICE_FRAME_RECEIPT_KIND,
        "version": ICE_FRAME_RECEIPT_VERSION,
        "roll_slot": roll_slot,
        "boundary_offset_rows": boundary_offset_rows,
        "plan": processed.plan.semantic_dict(),
        "bundle_manifest_sha256": processed.bundle_manifest_sha256,
        "backend": {
            "requested": processed.ice.requested_backend.value,
            "used": processed.ice.used_backend.value,
            "selection_reason": processed.ice.selection_reason,
        },
        "engine_receipt": asdict(processed.ice.receipt),
    }


def publish_ice_frame(
    processed: ProcessedIceFrame,
    *,
    service: Any,
    output_folder: str,
    filename_pattern: str,
    roll_slot: int,
    boundary_offset_rows: int,
) -> str:
    """Write the cleaned RGB master and its receipt; fail without leftovers.

    The cleaned frame is scanner-native portrait like every other LS-5000
    master, so it takes the same storage rotation.  ``write_result`` reserves
    the ``_SCAN.json`` basename alongside the TIFF, which is exactly where the
    receipt lands; a receipt failure removes the TIFF so a master can never
    exist without its evidence.
    """

    cleaned = processed.ice.cleaned_rgb16
    expected = (LS5000_FULL_WINDOW_ROWS, LS5000_FINE_WIDTH, 3)
    if (
        not isinstance(cleaned, np.ndarray)
        or cleaned.dtype != np.uint16
        or cleaned.shape != expected
    ):
        shape = getattr(cleaned, "shape", None)
        dtype = getattr(cleaned, "dtype", None)
        raise IceRollError(
            f"ICE output must be a {expected[0]}x{expected[1]} uint16 RGB "
            f"raster; got shape={shape}, dtype={dtype}"
        )
    result = ScanResult(
        rgb=cleaned,
        ir=None,
        dpi=LS5000_NATIVE_DPI,
        device_model=processed.device_model,
    )
    storage_result = orient_sane_rgb_for_storage(result)
    rgb_path = service.write_result(
        result=storage_result,
        output_folder=output_folder,
        filename_pattern=filename_pattern,
        output_format="TIFF",
        slot=roll_slot,
    )
    try:
        receipt = build_ice_receipt(
            processed,
            roll_slot=roll_slot,
            boundary_offset_rows=boundary_offset_rows,
        )
        rgb_file = Path(rgb_path)
        receipt["output"] = {
            "rgb": {
                "path": rgb_file.name,
                "sha256": _sha256_file(rgb_file),
                "bytes": rgb_file.stat().st_size,
            }
        }
        receipt_path = rgb_file.with_name(rgb_file.name.removesuffix(".tif") + "_SCAN.json")
        _write_receipt_exclusive(receipt_path, receipt)
    except BaseException:
        Path(rgb_path).unlink(missing_ok=True)
        raise
    return rgb_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ICE_FRAME_RECEIPT_KIND",
    "ICE_FRAME_RECEIPT_VERSION",
    "IceRollError",
    "ProcessedIceFrame",
    "acquire_ice_bundle",
    "build_ice_receipt",
    "process_ice_bundle",
    "publish_ice_frame",
]
