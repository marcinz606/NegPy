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
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from negpy.infrastructure.scanners.dice_dual_source_runner import (
    DiceDualSourcePlan,
    Libsane,
    acquire_dual_sources,
    load_capture_bundle,
    verify_capture_bundle,
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


ICE_HYBRID_RECEIPT_KIND = "negpy.ls5000-ice-hybrid-frame"

#: Environment variables the launcher sets to make hybrid repair available.
#: Every value is a path or hash the hybrid CLI needs; all seven are required
#: because a partially configured hybrid runtime must not look available.
_HYBRID_ENV = {
    "cli": "NEGPY_HYBRID_CLI",
    "iopaint_python": "NEGPY_HYBRID_IOPAINT_PYTHON",
    "iopaint_executable": "NEGPY_HYBRID_IOPAINT_EXECUTABLE",
    "iopaint_source_sha256": "NEGPY_HYBRID_IOPAINT_SOURCE_SHA256",
    "model_dir": "NEGPY_HYBRID_MODEL_DIR",
    "model_weights": "NEGPY_HYBRID_MODEL_WEIGHTS",
    "model_weights_sha256": "NEGPY_HYBRID_MODEL_WEIGHTS_SHA256",
}
_HYBRID_DEVICE_ENV = "NEGPY_HYBRID_INPAINT_DEVICE"
_HYBRID_PATH_KEYS = ("cli", "iopaint_python", "iopaint_executable", "model_dir", "model_weights")


@dataclass(frozen=True)
class HybridRepairConfig:
    """Where the hybrid CLI, its pinned IOPaint runtime, and the weights live."""

    cli: Path
    iopaint_python: Path
    iopaint_executable: Path
    iopaint_source_sha256: str
    model_dir: Path
    model_weights: Path
    model_weights_sha256: str
    inpaint_device: str = "cpu"

    @classmethod
    def from_env(cls) -> "HybridRepairConfig":
        """Build the config from the launcher's environment, or refuse.

        Raises IceRollError naming exactly what is missing, so the UI can show
        the reason and the worker can fail before any hardware is touched.
        """

        values: dict[str, str] = {}
        missing: list[str] = []
        for field, variable in _HYBRID_ENV.items():
            value = os.environ.get(variable, "").strip()
            if not value:
                missing.append(variable)
            values[field] = value
        if missing:
            raise IceRollError(
                "hybrid repair is not configured; launch through the NegPy ICE "
                "launcher script, which sets: " + ", ".join(missing)
            )
        for field in _HYBRID_PATH_KEYS:
            if not Path(values[field]).exists():
                raise IceRollError(
                    f"hybrid repair path does not exist: {_HYBRID_ENV[field]}="
                    f"{values[field]}"
                )
        device = os.environ.get(_HYBRID_DEVICE_ENV, "cpu").strip() or "cpu"
        if device not in {"cpu", "mps", "cuda"}:
            raise IceRollError(
                f"{_HYBRID_DEVICE_ENV} must be cpu, mps, or cuda; got {device!r}"
            )
        return cls(
            cli=Path(values["cli"]),
            iopaint_python=Path(values["iopaint_python"]),
            iopaint_executable=Path(values["iopaint_executable"]),
            iopaint_source_sha256=values["iopaint_source_sha256"],
            model_dir=Path(values["model_dir"]),
            model_weights=Path(values["model_weights"]),
            model_weights_sha256=values["model_weights_sha256"],
            inpaint_device=device,
        )


def hybrid_availability() -> tuple[bool, str]:
    """UI-facing probe: is hybrid repair configured, and if not, why not."""

    try:
        config = HybridRepairConfig.from_env()
    except IceRollError as error:
        return False, str(error)
    return True, f"hybrid repair ready (inpaint device: {config.inpaint_device})"


@dataclass(frozen=True)
class HybridRunOutputs:
    """Artifacts one hybrid CLI invocation produced for one frame.

    Carries the bundle identity facts the publisher needs, read once from the
    verified manifest, so the worker never re-opens the bundle.
    """

    out_dir: Path
    hybrid_rgb16: np.ndarray
    pure_rgb16: np.ndarray
    synth_mask_path: Path
    hybrid_receipt: dict[str, Any]
    routing: dict[str, Any]
    bundle_manifest_sha256: str
    plan_semantic: dict[str, Any]
    device_model: str


def run_hybrid_repair(
    bundle_dir: Path,
    out_dir: Path,
    *,
    config: HybridRepairConfig,
    backend: str,
    timeout_seconds: float = 1800.0,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> HybridRunOutputs:
    """Run the hybrid CLI on a verified bundle's own arrays.

    The bundle already contains exactly the two inputs the hybrid CLI takes,
    and its acquisition evidence is what justifies the CLI's required
    focus-exposure-locked assertion. The CLI runs the engine itself with
    diagnostics enabled, so this replaces the plain engine invocation rather
    than following it.
    """

    root = Path(bundle_dir)
    manifest = verify_capture_bundle(root)
    same_frame_id = str(manifest["same_frame_id"])
    bundle_receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    manifest_sha256 = str(bundle_receipt["manifest_sha256"])
    plan_semantic = manifest["plan"]
    if not isinstance(plan_semantic, dict):
        raise IceRollError("capture bundle plan is not an object")
    identity = manifest["scanner_identity"]
    if not isinstance(identity, dict):
        raise IceRollError("capture bundle has no scanner identity")
    device_model = f"{identity.get('vendor', '')} {identity.get('model', '')}".strip()
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise IceRollError(f"hybrid output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(config.cli),
        "--prepass",
        str(root / "prepass_rgbi.npy"),
        "--main",
        str(root / "main_rgbi.npy"),
        "--out",
        str(out_dir),
        "--same-frame-id",
        same_frame_id,
        "--assert-focus-exposure-locked",
        "--backend",
        backend,
        "--iopaint-python",
        str(config.iopaint_python),
        "--iopaint-executable",
        str(config.iopaint_executable),
        "--iopaint-source-manifest-sha256",
        config.iopaint_source_sha256,
        "--model-dir",
        str(config.model_dir),
        "--model-weights",
        str(config.model_weights),
        "--model-weights-sha256",
        config.model_weights_sha256,
        "--inpaint-device",
        config.inpaint_device,
        "--inpaint-threads",
        "4",
        "--inpaint-seed",
        "0",
    ]
    completed = runner(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise IceRollError(
            f"hybrid repair CLI failed with exit {completed.returncode}: "
            f"{detail[-2000:]}"
        )
    hybrid_path = out_dir / "output-hybrid.rgb16.npy"
    pure_path = out_dir / "output.rgb16.npy"
    mask_path = out_dir / "synth-mask.png"
    receipt_path = out_dir / "hybrid-receipt.json"
    routing_path = out_dir / "routing.json"
    for required in (hybrid_path, pure_path, mask_path, receipt_path, routing_path):
        if not required.is_file():
            raise IceRollError(
                f"hybrid repair CLI succeeded but {required.name} is missing "
                f"from {out_dir}"
            )
    return HybridRunOutputs(
        out_dir=out_dir,
        hybrid_rgb16=np.load(hybrid_path, allow_pickle=False, mmap_mode="r"),
        pure_rgb16=np.load(pure_path, allow_pickle=False, mmap_mode="r"),
        synth_mask_path=mask_path,
        hybrid_receipt=json.loads(receipt_path.read_text(encoding="utf-8")),
        routing=json.loads(routing_path.read_text(encoding="utf-8")),
        bundle_manifest_sha256=manifest_sha256,
        plan_semantic=plan_semantic,
        device_model=device_model,
    )


def publish_hybrid_frame(
    outputs: HybridRunOutputs,
    *,
    service: Any,
    output_folder: str,
    filename_pattern: str,
    roll_slot: int,
    boundary_offset_rows: int,
) -> str:
    """Publish the hybrid master with its pure ICE sibling and evidence.

    The hybrid master is the file the user edits. The pure ICE master is kept
    beside it as ``<base>_ICE.tif`` because inside the synthesis mask the pure
    reconstruction is otherwise lost once the bundle is deleted; outside the
    mask the two are byte-identical by the hybrid's own contract. The
    synthesis mask lands as ``<base>_SYNTH.png`` and one receipt binds all
    three by hash. A failure removes everything this call wrote.
    """

    expected = (LS5000_FULL_WINDOW_ROWS, LS5000_FINE_WIDTH, 3)
    for label, array in (("hybrid", outputs.hybrid_rgb16), ("pure", outputs.pure_rgb16)):
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.uint16
            or array.shape != expected
        ):
            raise IceRollError(
                f"{label} ICE output must be a {expected[0]}x{expected[1]} "
                f"uint16 RGB raster; got shape={getattr(array, 'shape', None)}, "
                f"dtype={getattr(array, 'dtype', None)}"
            )
    hybrid_storage = orient_sane_rgb_for_storage(
        ScanResult(
            rgb=outputs.hybrid_rgb16,
            ir=None,
            dpi=LS5000_NATIVE_DPI,
            device_model=outputs.device_model,
        )
    )
    rgb_path = service.write_result(
        result=hybrid_storage,
        output_folder=output_folder,
        filename_pattern=filename_pattern,
        output_format="TIFF",
        slot=roll_slot,
    )
    written: list[Path] = [Path(rgb_path)]
    try:
        base = rgb_path.removesuffix(".tif")
        ice_path = Path(base + "_ICE.tif")
        synth_path = Path(base + "_SYNTH.png")
        receipt_path = Path(base + "_SCAN.json")
        for sibling in (ice_path, synth_path):
            if sibling.exists():
                raise IceRollError(
                    f"stray file blocks hybrid publication: {sibling}"
                )
        from negpy.services.scanning.writer import write_tiff_16bit

        pure_storage = orient_sane_rgb_for_storage(
            ScanResult(
                rgb=outputs.pure_rgb16,
                ir=None,
                dpi=LS5000_NATIVE_DPI,
                device_model=outputs.device_model,
            )
        )
        write_tiff_16bit(pure_storage, base + "_ICE")
        written.append(ice_path)
        shutil.copyfile(outputs.synth_mask_path, synth_path)
        written.append(synth_path)
        synthesis = outputs.hybrid_receipt.get("synthesis", {})
        receipt = {
            "kind": ICE_HYBRID_RECEIPT_KIND,
            "version": 1,
            "roll_slot": roll_slot,
            "boundary_offset_rows": boundary_offset_rows,
            "plan": outputs.plan_semantic,
            "bundle_manifest_sha256": outputs.bundle_manifest_sha256,
            "synthesis": synthesis,
            "outputs": {
                "hybrid_rgb": {
                    "path": Path(rgb_path).name,
                    "sha256": _sha256_file(Path(rgb_path)),
                    "bytes": Path(rgb_path).stat().st_size,
                },
                "ice_rgb": {
                    "path": ice_path.name,
                    "sha256": _sha256_file(ice_path),
                    "bytes": ice_path.stat().st_size,
                },
                "synth_mask": {
                    "path": synth_path.name,
                    "sha256": _sha256_file(synth_path),
                    "bytes": synth_path.stat().st_size,
                },
            },
            "hybrid_receipt": outputs.hybrid_receipt,
        }
        _write_receipt_exclusive(receipt_path, receipt)
        written.append(receipt_path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return rgb_path


__all__ = [
    "ICE_FRAME_RECEIPT_KIND",
    "ICE_FRAME_RECEIPT_VERSION",
    "ICE_HYBRID_RECEIPT_KIND",
    "HybridRepairConfig",
    "HybridRunOutputs",
    "IceRollError",
    "ProcessedIceFrame",
    "acquire_ice_bundle",
    "build_ice_receipt",
    "hybrid_availability",
    "process_ice_bundle",
    "publish_hybrid_frame",
    "publish_ice_frame",
    "run_hybrid_repair",
]
