"""Optional post-import infrared dust repair via the digital-fauxice engine.

This wraps ``portable-digital-ice`` (import name ``portable_digital_ice``), a
byte-exact, independently reverse-engineered reimplementation of the Nikon
Coolscan's Digital ICE infrared dust repair. It is an optional dependency:
every function here degrades to a reported status instead of raising when the
engine, or its optional ``fauxce-hybrid`` companion, is not installed.

Read this before wiring a caller to this module. The input contract is
narrower than "an RGB master plus its IR sidecar":

The engine repairs one main scan using a *second*, earlier capture of the
same physical frame: a 285 dpi RGBI prepass that supplies per-frame
calibration the main pass depends on. Its own docs are explicit that this
is not a resampling convenience and cannot be reconstructed after the fact:
"Reconstructing or guessing it from the main scan would move outside the
byte-exact claim." NegPy's generic import/scanning writer still lacks that
paired capture, but the Coolscan roll path now supplies its scanner-bound
285 dpi prepass, validity mask, and main RGBI acquisition directly. Calls
without that real prepass report ``SKIPPED`` rather than fabricate one. See
``docs/FAUXICE_IR_REPAIR.md`` for the exact/hybrid split
(``fauxice_hybrid_runner.py`` covers the external hybrid path).

This is a one-shot repair, not a per-render pipeline stage. It accepts an
optional progress callback and cancellation event, mirroring the shape
``ScannerService.run_scan`` already uses for its own long operation
(``negpy/services/scanning/service.py``), so a future background worker can
wrap a repair call the same way ``ScanWorker`` wraps a scan.

A repaired frame is published as a new companion file
(``<basename>_FAUXICE.tif``) next to the untouched original, never as an
in-place rewrite: NegPy keys stored edits by
the source file's content hash (see ``StorageRepository`` and
``negpy/services/assets/sidecar.py``), so silently rewriting the master
would orphan any edits already saved against it. The original ``_IR.tif``
companion is only ever opened for reading here.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import tifffile

from negpy.kernel.system.logging import get_logger
from negpy.services.repair.fauxice_hybrid_runner import (
    HybridRunCancelled,
    HybridRunError,
    HybridRuntimeConfig,
    run_hybrid_repair,
)

logger = get_logger(__name__)

ENGINE_DISTRIBUTION = "portable-digital-ice"
ENGINE_IMPORT_NAME = "portable_digital_ice"
HYBRID_IMPORT_NAME = "fauxce_hybrid"
ENGINE_HOME = "https://github.com/rohanpandula/digital-fauxice"

IR_SIDECAR_SUFFIX = "_IR"
REPAIR_OUTPUT_SUFFIX = "_FAUXICE"
REPAIR_SIDECAR_SUFFIX = "_FAUXICE"
SYNTH_MASK_SUFFIX = "_FAUXICE_SYNTH"

# The engine currently validates exactly one acquisition profile (see its
# own profile.py: LS5000Selector8NormalProfile). These are not arbitrary
# defaults; they are the only dpi pair it accepts today.
PREPASS_DPI = 285
MAIN_DPI = 4000


class RepairStatus(str, Enum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


class RepairMode(str, Enum):
    EXACT = "exact"
    HYBRID = "hybrid"


class FauxiceRepairCancelled(RuntimeError):
    """A cooperative stop requested by the caller."""


def engine_available() -> bool:
    """True if the core digital-fauxice engine is importable.

    Uses ``find_spec`` rather than a real import so an availability check
    never pays the engine's import cost (or runs any stub's side effects)
    just to answer "is this installed."
    """
    return importlib.util.find_spec(ENGINE_IMPORT_NAME) is not None


def hybrid_available(runtime: HybridRuntimeConfig | None = None) -> bool:
    """True only when an explicit external hybrid runtime was supplied.

    The companion deliberately runs in a separate Python environment, so an
    in-process import probe is both irrelevant and wrong on NegPy's supported
    Python 3.13 runtime.  ``HybridRuntimeConfig`` is validated when built and
    carries every executable/path/hash needed at the process boundary.
    """

    if runtime is None:
        return False
    try:
        runtime.validate_files()
    except ValueError:
        return False
    return True


def _engine_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(ENGINE_DISTRIBUTION)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _format_fraction(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.4%}"


@dataclass(frozen=True)
class FauxiceRepairConfig:
    """Off-by-default settings for the post-import Digital ICE repair step.

    This mirrors NegPy's existing "bool = False" feature-flag idiom (see
    ``RetouchConfig.ir_dust_remove`` in ``negpy/features/retouch/models.py``)
    without living inside ``WorkspaceConfig``. ``WorkspaceConfig`` holds
    cheap per-render recipe parameters ``DarkroomEngine`` replays on every
    slider drag; this step is an explicit, one-shot external process a
    caller (a worker, a CLI, a future sidebar action) triggers deliberately,
    so it gets its own config object instead.
    """

    enabled: bool = False
    mode: RepairMode = RepairMode.EXACT
    backend: str = "auto"


@dataclass(frozen=True)
class FauxiceRepairResult:
    """The outcome of one ``repair_ir_dust`` call."""

    status: RepairStatus
    reason: str
    mode_requested: RepairMode
    mode_resolved: RepairMode | None = None
    repaired_rgb16: npt.NDArray[np.uint16] | None = None
    engine_version: str | None = None
    backend_requested: str | None = None
    backend_used: str | None = None
    backend_selection_reason: str | None = None
    hybrid_mask_png: bytes | None = None
    hybrid_mask_sha256: str | None = None
    hybrid_mask: npt.NDArray[np.bool_] | None = None
    hybrid_synthesis_fraction: float | None = None
    hybrid_routing_counts: dict[str, int] | None = None
    hybrid_receipt: bytes | None = None
    hybrid_receipt_sha256: str | None = None
    hybrid_provenance_class: str | None = None
    hybrid_receipt_output_rgb_sha256: str | None = None
    native_output_rgb_sha256: str | None = None

    def provenance(self) -> dict:
        """JSON-serializable provenance record; the sidecar's payload."""

        payload: dict = {
            "kind": "negpy.fauxice-ir-repair",
            "version": 1,
            "status": self.status.value,
            "reason": self.reason,
            "mode_requested": self.mode_requested.value,
            "mode_resolved": self.mode_resolved.value if self.mode_resolved is not None else None,
            "engine": {
                "package": ENGINE_IMPORT_NAME,
                "version": self.engine_version,
                "backend_requested": self.backend_requested,
                "backend_used": self.backend_used,
                "backend_selection_reason": self.backend_selection_reason,
            },
        }
        if self.mode_resolved is RepairMode.HYBRID:
            payload["hybrid"] = {
                "package": HYBRID_IMPORT_NAME,
                "disclosure_mask_sha256": self.hybrid_mask_sha256,
                "synthesis_fraction": self.hybrid_synthesis_fraction,
                # Region/pixel counts straight from the tool's own receipt
                # (routing.counts); None when the receipt did not carry the
                # expected keys rather than guessing at partial data.
                "routing_counts": self.hybrid_routing_counts,
            }
        return payload


def _disabled_result(mode: RepairMode) -> FauxiceRepairResult:
    return FauxiceRepairResult(
        status=RepairStatus.SKIPPED,
        reason="disabled by configuration (FauxiceRepairConfig.enabled is False)",
        mode_requested=mode,
    )


def _unavailable_result(mode: RepairMode) -> FauxiceRepairResult:
    return FauxiceRepairResult(
        status=RepairStatus.UNAVAILABLE,
        reason=(
            f"{ENGINE_IMPORT_NAME} is not installed; install the optional "
            f"'fauxice' dependency group from {ENGINE_HOME} "
            "(see docs/FAUXICE_IR_REPAIR.md)"
        ),
        mode_requested=mode,
    )


def repair_ir_dust(
    rgb: npt.NDArray[np.uint16],
    ir: npt.NDArray[np.uint16],
    *,
    same_frame_id: str,
    config: FauxiceRepairConfig,
    prepass_rgbi: npt.NDArray[np.uint16] | None = None,
    validity_mask: npt.NDArray[np.bool_] | None = None,
    hybrid_runtime: HybridRuntimeConfig | None = None,
    hybrid_subprocess_runner: Callable[..., "subprocess.CompletedProcess[str]"] | None = None,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> FauxiceRepairResult:
    """Repair infrared-flagged dust in ``rgb`` using the digital-fauxice engine.

    ``rgb`` (HxWx3 uint16) and ``ir`` (HxW uint16) are the two halves of
    NegPy's own single-pass capture (``negpy/services/scanning/writer.py``
    writes them from one ``ScanResult``), so reassembling them into one
    HxWx4 RGBI array is a lossless relabeling, not a reconstruction.

    ``prepass_rgbi`` is different in kind: seeing the module docstring's
    prepass note before wiring a caller to this function. Pass ``None``
    (the default) unless a real 285 dpi acquisition of the same frame is
    available; passing anything reconstructed or guessed would silently
    break the engine's exactness claim.

    Malformed array shapes/dtypes raise ``ValueError`` immediately (a caller
    bug, not an expected runtime state). Missing preconditions (feature
    disabled, engine not installed, no prepass, hybrid unavailable) never
    raise; they come back as a status on the result, so a caller can show
    it to a user without a try/except.

    ``progress`` and ``cancel`` follow ``ScannerService.run_scan``'s own
    shape (a ``0.0``-``1.0`` callback plus a ``threading.Event``) so a
    background worker can wire them the same way ``ScanWorker`` wires a
    scan. The exact engine polls its native cooperative callback; the hybrid
    runner polls the event while its isolated process group runs, terminates
    that group on cancellation, and reports coarse phase progress.
    """

    mode_requested = config.mode

    if not config.enabled:
        return _disabled_result(mode_requested)
    if not engine_available():
        return _unavailable_result(mode_requested)

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint16:
        raise ValueError("rgb must be an HxWx3 uint16 array")
    if ir.ndim != 2 or ir.dtype != np.uint16:
        raise ValueError("ir must be an HxW uint16 array")
    if ir.shape != rgb.shape[:2]:
        raise ValueError(f"ir shape {ir.shape} does not match rgb shape {rgb.shape[:2]}")
    if validity_mask is not None and validity_mask.shape != rgb.shape[:2]:
        raise ValueError("validity_mask must match rgb's height and width")
    if prepass_rgbi is not None and (prepass_rgbi.ndim != 3 or prepass_rgbi.shape[2] != 4 or prepass_rgbi.dtype != np.uint16):
        raise ValueError("prepass_rgbi must be an HxWx4 uint16 array")
    if not same_frame_id.strip():
        raise ValueError("same_frame_id must be non-empty")

    if prepass_rgbi is None:
        return FauxiceRepairResult(
            status=RepairStatus.SKIPPED,
            reason=(
                "no paired 285 dpi prepass acquisition available for this frame; "
                "Digital ICE requires one captured alongside the main scan and "
                "cannot be reconstructed from it (see docs/FAUXICE_IR_REPAIR.md)"
            ),
            mode_requested=mode_requested,
        )

    if cancel is not None and cancel.is_set():
        raise FauxiceRepairCancelled(
            "cancelled by caller before repair started"
        )

    main_rgbi = np.dstack([rgb, ir])

    mode_resolved = RepairMode.EXACT
    hybrid_note = ""
    hybrid_outcome = None

    if mode_requested is RepairMode.HYBRID:
        if hybrid_runtime is None:
            hybrid_note = "hybrid mode requested but no hybrid runtime is configured; degraded to exact repair. "
        elif not hybrid_available(hybrid_runtime):
            try:
                hybrid_runtime.validate_files()
            except ValueError as error:
                detail = str(error)
            else:  # pragma: no cover - defensive against a racing runtime
                detail = "runtime artifacts changed during validation"
            hybrid_note = f"hybrid mode requested but the configured hybrid runtime is unavailable ({detail}); degraded to exact repair. "
        elif cancel is not None and cancel.is_set():
            raise FauxiceRepairCancelled(
                "cancelled before the hybrid run started"
            )
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="negpy-fauxice-hybrid-") as scratch:
                    hybrid_outcome = run_hybrid_repair(
                        main_rgbi,
                        prepass_rgbi,
                        same_frame_id=same_frame_id,
                        backend=config.backend,
                        runtime=hybrid_runtime,
                        scratch_dir=Path(scratch),
                        runner=hybrid_subprocess_runner,
                        progress=progress,
                        cancel=cancel,
                    )
                mode_resolved = RepairMode.HYBRID
            except HybridRunCancelled as error:
                raise FauxiceRepairCancelled(
                    f"hybrid repair cancelled: {error}"
                ) from error
            except HybridRunError as error:
                hybrid_note = f"hybrid mode requested but the fauxce-hybrid run failed ({error}); degraded to exact repair. "

    result_kwargs: dict
    if mode_resolved is RepairMode.HYBRID and hybrid_outcome is not None:
        repaired = hybrid_outcome.hybrid_rgb16
        result_kwargs = {
            "engine_version": hybrid_outcome.engine_version,
            "backend_requested": hybrid_outcome.backend_requested,
            "backend_used": hybrid_outcome.backend_used,
            "backend_selection_reason": hybrid_outcome.backend_selection_reason,
            "hybrid_mask_png": hybrid_outcome.synth_mask_png,
            "hybrid_mask_sha256": hybrid_outcome.synth_mask_sha256,
            "hybrid_mask": hybrid_outcome.synth_mask,
            "hybrid_synthesis_fraction": hybrid_outcome.synthesis_fraction,
            "hybrid_routing_counts": hybrid_outcome.routing_counts,
            "hybrid_receipt": hybrid_outcome.receipt,
            "hybrid_receipt_sha256": hybrid_outcome.receipt_sha256,
            "hybrid_provenance_class": hybrid_outcome.provenance_class,
            "hybrid_receipt_output_rgb_sha256": (hybrid_outcome.output_rgb16_sha256),
            "native_output_rgb_sha256": hybrid_outcome.output_rgb16_sha256,
        }
        region_note = ""
        if result_kwargs["hybrid_routing_counts"] is not None:
            region_note = f"; {result_kwargs['hybrid_routing_counts']['final_regions']} region(s) routed"
        reason = (
            f"applied via {HYBRID_IMPORT_NAME} (engine {result_kwargs['engine_version']}, "
            f"backend {result_kwargs['backend_used']}); synthesis fraction "
            f"{_format_fraction(result_kwargs['hybrid_synthesis_fraction'])}{region_note}"
        )
    else:
        from portable_digital_ice import ProcessingCancelled

        try:
            repaired, backend_info = _run_exact(
                main_rgbi,
                prepass_rgbi,
                same_frame_id=same_frame_id,
                backend=config.backend,
                progress=progress,
                cancel=cancel,
            )
        except ProcessingCancelled as error:
            raise FauxiceRepairCancelled(
                f"{hybrid_note}cancelled before completion: {error}"
            ) from error
        except (ValueError, TypeError, RuntimeError) as error:
            return FauxiceRepairResult(
                status=RepairStatus.SKIPPED,
                reason=f"{hybrid_note}engine rejected the acquisition: {error}",
                mode_requested=mode_requested,
                mode_resolved=None,
            )
        result_kwargs = {
            "engine_version": _engine_version(),
            "native_output_rgb_sha256": hashlib.sha256(
                np.array(repaired, dtype="<u2", order="C", copy=True).tobytes(order="C")
            ).hexdigest(),
            **backend_info,
        }
        reason = (
            f"{hybrid_note}applied via {ENGINE_IMPORT_NAME} {result_kwargs['engine_version']} (backend {result_kwargs['backend_used']})"
        )

    if validity_mask is not None:
        repaired = np.where(validity_mask[:, :, None], repaired, rgb)
    repaired = np.array(repaired, dtype="<u2", order="C", copy=True)
    result_kwargs["native_output_rgb_sha256"] = hashlib.sha256(memoryview(repaired).cast("B")).hexdigest()

    return FauxiceRepairResult(
        status=RepairStatus.APPLIED,
        reason=reason,
        mode_requested=mode_requested,
        mode_resolved=mode_resolved,
        repaired_rgb16=repaired,
        **result_kwargs,
    )


def _run_exact(
    main_rgbi: npt.NDArray[np.uint16],
    prepass_rgbi: npt.NDArray[np.uint16],
    *,
    same_frame_id: str,
    backend: str,
    progress: Callable[[float], None] | None,
    cancel: threading.Event | None,
) -> tuple[npt.NDArray[np.uint16], dict]:
    """Call the installed engine directly.

    Only reached after ``engine_available()`` is true. The import is local
    to this function, never at module scope, so importing this module never
    requires the optional engine to be installed.

    ``progress``/``cancel`` are NegPy's own ``Callable[[float], None]`` /
    ``threading.Event`` shapes; the engine's native callbacks take a
    ``ProcessingProgress`` object and a zero-argument ``() -> bool`` poll
    function respectively, so this is where the two conventions meet.
    """

    from portable_digital_ice import (
        AcquisitionEpoch,
        ComputeBackend,
        DualRGBIAcquisition,
        ProcessingJob,
        ProcessingMode,
        RGBI16Frame,
        ScannerModel,
        process,
    )

    prepass_frame = RGBI16Frame(prepass_rgbi, AcquisitionEpoch.PREPASS, PREPASS_DPI, f"{same_frame_id}-prepass")
    main_frame = RGBI16Frame(main_rgbi, AcquisitionEpoch.MAIN, MAIN_DPI, f"{same_frame_id}-main")
    acquisition = DualRGBIAcquisition(prepass_frame, main_frame, same_frame_id)
    job = ProcessingJob(
        acquisition=acquisition,
        scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=MAIN_DPI,
        bit_depth=16,
        focus_exposure_locked=True,
    )

    engine_progress = None
    if progress is not None:

        def engine_progress(step: object) -> None:
            total = getattr(step, "total", 0) or 0
            completed = getattr(step, "completed", 0)
            progress(float(completed) / float(total) if total else 0.0)

    engine_cancelled = cancel.is_set if cancel is not None else None

    routed = process(
        job,
        backend=ComputeBackend(backend),
        progress=engine_progress,
        cancelled=engine_cancelled,
    )
    selection = routed.selection
    return routed.result.output_rgb16, {
        "backend_requested": selection.requested.value,
        "backend_used": selection.used.value,
        "backend_selection_reason": selection.reason,
    }


def default_ir_path(rgb_path: Path) -> Path:
    """The sidecar path NegPy's own scanning writer produces: ``<stem>_IR.tif``."""
    return rgb_path.with_name(rgb_path.stem + IR_SIDECAR_SUFFIX + rgb_path.suffix)


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    return np.asarray(tifffile.imread(path))


def _load_validity_mask(path: Path) -> np.ndarray:
    mask = _load_array(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask if mask.dtype == np.bool_ else mask.astype(bool)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, default=str) + "\n").encode("utf-8"))


def _atomic_write_tiff(path: Path, rgb16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".part.tif")
    os.close(fd)
    try:
        tifffile.imwrite(tmp_name, rgb16, photometric="rgb", compression="lzw")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _write_sidecar(path: Path, result: FauxiceRepairResult, *, rgb_path: Path, ir_path: Path) -> None:
    payload = result.provenance()
    payload["source"] = {"rgb": rgb_path.name, "ir": ir_path.name}
    if result.status is RepairStatus.APPLIED:
        output = {"repaired_rgb": rgb_path.stem + REPAIR_OUTPUT_SUFFIX + rgb_path.suffix}
        if result.mode_resolved is RepairMode.HYBRID and result.hybrid_mask_png is not None:
            output["disclosure_mask"] = rgb_path.stem + SYNTH_MASK_SUFFIX + ".png"
        payload["output"] = output
    _atomic_write_json(path, payload)


def repair_frame_files(
    rgb_path: Path,
    *,
    config: FauxiceRepairConfig,
    ir_path: Path | None = None,
    prepass_path: Path | None = None,
    validity_mask_path: Path | None = None,
    same_frame_id: str | None = None,
    hybrid_runtime: HybridRuntimeConfig | None = None,
    hybrid_subprocess_runner: Callable[..., "subprocess.CompletedProcess[str]"] | None = None,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> FauxiceRepairResult:
    """Load a NegPy-imported RGB master plus its ``_IR`` companion and repair it.

    Always writes a provenance sidecar (``<stem>_FAUXICE.json``), even when
    the result is not ``APPLIED``, so a caller can show why. Only writes the
    repaired master (``<stem>_FAUXICE.tif``, and the disclosure mask PNG for
    a hybrid run) when the result is ``APPLIED``; a skipped or unavailable
    attempt otherwise leaves every existing file byte-for-byte untouched.
    ``ir_path`` is never opened for writing.

    ``prepass_path``, if given, may be a ``.npy`` array or a TIFF holding
    the paired 285 dpi acquisition (HxWx4 uint16). NegPy's own import
    pipeline does not produce one today (see the module docstring), so this
    parameter exists for whatever future acquisition path can supply it, and
    for tests.
    """

    rgb_path = Path(rgb_path)
    ir_path = Path(ir_path) if ir_path is not None else default_ir_path(rgb_path)
    frame_id = same_frame_id or rgb_path.stem
    sidecar_path = rgb_path.with_name(rgb_path.stem + REPAIR_SIDECAR_SUFFIX + ".json")

    if not config.enabled:
        result = _disabled_result(config.mode)
    elif not engine_available():
        result = _unavailable_result(config.mode)
    elif not ir_path.is_file():
        result = FauxiceRepairResult(
            status=RepairStatus.SKIPPED,
            reason=f"no IR companion found at {ir_path.name}",
            mode_requested=config.mode,
        )
    else:
        rgb = np.asarray(tifffile.imread(rgb_path))
        ir = np.asarray(tifffile.imread(ir_path))
        prepass = _load_array(Path(prepass_path)) if prepass_path is not None else None
        validity_mask = _load_validity_mask(Path(validity_mask_path)) if validity_mask_path is not None else None
        result = repair_ir_dust(
            rgb,
            ir,
            same_frame_id=frame_id,
            config=config,
            prepass_rgbi=prepass,
            validity_mask=validity_mask,
            hybrid_runtime=hybrid_runtime,
            hybrid_subprocess_runner=hybrid_subprocess_runner,
            progress=progress,
            cancel=cancel,
        )

    _write_sidecar(sidecar_path, result, rgb_path=rgb_path, ir_path=ir_path)

    if result.status is RepairStatus.APPLIED and result.repaired_rgb16 is not None:
        output_path = rgb_path.with_name(rgb_path.stem + REPAIR_OUTPUT_SUFFIX + rgb_path.suffix)
        _atomic_write_tiff(output_path, result.repaired_rgb16)
        if result.hybrid_mask_png is not None:
            mask_path = rgb_path.with_name(rgb_path.stem + SYNTH_MASK_SUFFIX + ".png")
            _atomic_write_bytes(mask_path, result.hybrid_mask_png)

    return result


__all__ = [
    "ENGINE_HOME",
    "ENGINE_IMPORT_NAME",
    "HYBRID_IMPORT_NAME",
    "FauxiceRepairConfig",
    "FauxiceRepairResult",
    "HybridRuntimeConfig",
    "RepairMode",
    "RepairStatus",
    "default_ir_path",
    "engine_available",
    "hybrid_available",
    "repair_frame_files",
    "repair_ir_dust",
]
