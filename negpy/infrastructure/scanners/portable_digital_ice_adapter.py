"""Optional bridge from validated RGBI captures to portable Digital ICE.

This module deliberately sits between verified acquisition and NegPy's
orientation and color pipeline. It accepts a dual-source capture whose scanner
identity, ordering, shapes, and locked state have been re-derived, builds the
engine's typed contracts, and returns a new RGB image. The source arrays remain
independent archival artifacts.

``portable_digital_ice`` is an optional dependency.  Selecting ``off`` never
imports it.  Explicit CPU and CUDA requests fail instead of substituting a
different backend.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from negpy.infrastructure.scanners.dice_dual_source_runner import (
    DiceDualSourcePlan,
    DualSourceCapture,
    validate_dual_source_capture,
)


UInt16Array = npt.NDArray[np.uint16]
ProgressCallback = Callable[[object], None]
EXPECTED_PROFILE_ID = "nikon-ls5000-selector8-normal-metric4000"
RNG_MODULUS = 1 << 24
STARTUP_STAGE_COUNT = 6


class PortableDigitalIceBackend(StrEnum):
    """The only Digital ICE choices NegPy exposes.

    ``CPU`` is the engine's reference implementation and takes roughly an hour
    for a 4000 dpi frame.  ``CPU_FAST`` is its compiled equivalent and takes
    about nine seconds; the engine proves the two byte-identical on a synthetic
    job before it will run the compiled path.  ``AUTO`` is deliberately absent:
    a backend that silently changes under a scan is not evidence.
    """

    OFF = "off"
    CPU = "cpu"
    CPU_FAST = "cpu-fast"
    CUDA = "cuda"


#: Engine exception class names that mean "this backend cannot run here", by
#: the backend that was explicitly asked for.  Each is reported as an
#: unavailable backend rather than a failed scan, and never falls back.
_UNAVAILABLE_ENGINE_ERRORS: dict[PortableDigitalIceBackend, tuple[str, str]] = {
    PortableDigitalIceBackend.CPU_FAST: ("CpuFastUnavailable", "compiled CPU"),
    PortableDigitalIceBackend.CUDA: ("CudaBackendUnavailable", "CUDA"),
}


class PortableDigitalIceUnavailable(RuntimeError):
    """The requested optional engine or compute backend cannot run."""


class PortableDigitalIceIntegrityError(RuntimeError):
    """The optional engine returned a result that breaks its public contract."""


@dataclass(frozen=True)
class PortableDigitalIceAvailability:
    """Cheap install facts for UI display; no self-test, no kernel launch."""

    engine_installed: bool
    engine_detail: str
    cpu_fast_installed: bool
    cpu_fast_detail: str
    cuda_installed: bool
    cuda_detail: str


def availability_summary() -> PortableDigitalIceAvailability:
    """Report which backends could plausibly run, cheaply enough for a UI.

    This answers "is it installed", not "will it pass its self-test" — the
    proof of byte parity stays in :func:`probe_backend`, which a scan worker
    runs before touching hardware.  Keeping this summary import-only means a
    sidebar can populate a backend selector without paying seconds of compile
    or synthetic-job time on the GUI thread.
    """

    try:
        engine = _load_engine()
    except PortableDigitalIceUnavailable as error:
        detail = str(error)
        return PortableDigitalIceAvailability(
            engine_installed=False,
            engine_detail=detail,
            cpu_fast_installed=False,
            cpu_fast_detail=detail,
            cuda_installed=False,
            cuda_detail=detail,
        )
    engine_detail = getattr(engine, "__file__", None) or "installed"
    try:
        import numba  # noqa: F401 — availability probe only
    except Exception as error:  # pragma: no cover - depends on environment
        cpu_fast_installed = False
        cpu_fast_detail = f"numba is not importable: {error}"
    else:
        cpu_fast_installed = True
        cpu_fast_detail = f"numba {getattr(numba, '__version__', 'unknown')}"
    try:
        cuda_module = import_module("portable_digital_ice.cuda_backend")
        summary = cuda_module.cuda_device_summary()
    except Exception as error:
        cuda_installed = False
        cuda_detail = f"CUDA backend is not usable: {error}"
    else:
        cuda_installed = True
        cuda_detail = str(summary)
    return PortableDigitalIceAvailability(
        engine_installed=True,
        engine_detail=str(engine_detail),
        cpu_fast_installed=cpu_fast_installed,
        cpu_fast_detail=cpu_fast_detail,
        cuda_installed=cuda_installed,
        cuda_detail=cuda_detail,
    )


def probe_backend(backend: PortableDigitalIceBackend | str) -> None:
    """Prove the requested backend can run before any scanner is touched.

    ``OFF`` needs nothing and ``CPU`` needs only the engine import.  The
    compiled backends run the engine's own byte-parity self-test, which is
    cached per process and doubles as compile warmup, so a scanner session is
    never spent discovering a backend that would have refused to start.
    """

    requested = _normalise_backend(backend)
    if requested is PortableDigitalIceBackend.OFF:
        return
    engine = _load_engine()
    if requested is PortableDigitalIceBackend.CPU:
        return
    self_test_name = {
        PortableDigitalIceBackend.CPU_FAST: "cpu_fast_self_test",
        PortableDigitalIceBackend.CUDA: "cuda_self_test",
    }[requested]
    self_test = getattr(getattr(engine, "backend", None), self_test_name, None)
    if not callable(self_test):
        raise PortableDigitalIceUnavailable(
            f"portable Digital ICE engine does not provide {self_test_name}; "
            "install portable-digital-ice 0.2.0 or newer"
        )
    try:
        self_test()
    except Exception as error:
        label = _UNAVAILABLE_ENGINE_ERRORS[requested][1]
        raise PortableDigitalIceUnavailable(
            f"portable Digital ICE {label} backend is unavailable: {error}"
        ) from error


@dataclass(frozen=True)
class PortableDigitalIceReceipt:
    """Stable, serializable evidence for one adapter invocation."""

    status: str
    same_frame_id: str
    prepass_evidence_id: str
    main_evidence_id: str
    prepass_sha256: str
    main_sha256: str
    output_sha256: str
    output_shape: tuple[int, int, int]
    profile_id: str | None
    attempted_pixels: int
    written_pixels: int
    changed_pixels: int
    public_rng_advances: int
    final_rng_state: int | None
    startup_attempted_per_stage: tuple[int, ...] | None
    startup_rng_advances_per_stage: tuple[int, ...] | None
    startup_final_rng_state: int | None


@dataclass(frozen=True)
class PortableDigitalIceResult:
    """A non-destructive RGB output and its backend/engine evidence."""

    cleaned_rgb16: UInt16Array
    requested_backend: PortableDigitalIceBackend
    used_backend: PortableDigitalIceBackend
    selection_reason: str
    receipt: PortableDigitalIceReceipt


def _sha256_uint16(array: UInt16Array) -> str:
    little_endian = array.astype("<u2", copy=False)
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _owned_rgbi16(pixels: npt.ArrayLike, *, label: str) -> UInt16Array:
    array = np.asarray(pixels)
    if array.dtype != np.dtype(np.uint16):
        raise TypeError(f"{label} must have dtype uint16")
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"{label} must have shape HxWx4")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{label} cannot have an empty dimension")
    owned = np.array(array, dtype=np.uint16, order="C", copy=True)
    owned.flags.writeable = False
    return owned


def _load_engine() -> ModuleType:
    try:
        return import_module("portable_digital_ice")
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing == "portable_digital_ice" or missing.startswith(
            "portable_digital_ice."
        ):
            raise PortableDigitalIceUnavailable(
                "portable Digital ICE is not installed; install "
                "https://github.com/rohanpandula/digital-fauxice"
            ) from error
        raise


def _normalise_backend(
    backend: PortableDigitalIceBackend | str,
) -> PortableDigitalIceBackend:
    try:
        return PortableDigitalIceBackend(backend)
    except ValueError as error:
        choices = ", ".join(member.value for member in PortableDigitalIceBackend)
        raise ValueError(f"Digital ICE backend must be one of: {choices}") from error


def _engine_backend(value: object) -> PortableDigitalIceBackend:
    raw = getattr(value, "value", value)
    try:
        backend = PortableDigitalIceBackend(raw)
    except (TypeError, ValueError) as error:
        raise PortableDigitalIceIntegrityError(
            f"portable engine reported an unknown backend {raw!r}"
        ) from error
    if backend is PortableDigitalIceBackend.OFF:
        raise PortableDigitalIceIntegrityError(
            "portable engine reported the adapter-only off backend"
        )
    return backend


def _validated_output(pixels: object, *, expected_shape: tuple[int, int, int]) -> UInt16Array:
    array = np.asarray(pixels)
    if array.dtype != np.dtype(np.uint16) or array.shape != expected_shape:
        raise PortableDigitalIceIntegrityError(
            "portable engine returned "
            f"dtype={array.dtype}, shape={array.shape}; expected uint16 {expected_shape}"
        )
    return np.array(array, dtype=np.uint16, order="C", copy=True)


def _off_result(
    *,
    prepass: UInt16Array,
    main: UInt16Array,
    same_frame_id: str,
    prepass_evidence_id: str,
    main_evidence_id: str,
) -> PortableDigitalIceResult:
    output = np.array(main[:, :, :3], dtype=np.uint16, order="C", copy=True)
    receipt = PortableDigitalIceReceipt(
        status="bypassed",
        same_frame_id=same_frame_id,
        prepass_evidence_id=prepass_evidence_id,
        main_evidence_id=main_evidence_id,
        prepass_sha256=_sha256_uint16(prepass),
        main_sha256=_sha256_uint16(main),
        output_sha256=_sha256_uint16(output),
        output_shape=output.shape,
        profile_id=None,
        attempted_pixels=0,
        written_pixels=0,
        changed_pixels=0,
        public_rng_advances=0,
        final_rng_state=None,
        startup_attempted_per_stage=None,
        startup_rng_advances_per_stage=None,
        startup_final_rng_state=None,
    )
    return PortableDigitalIceResult(
        cleaned_rgb16=output,
        requested_backend=PortableDigitalIceBackend.OFF,
        used_backend=PortableDigitalIceBackend.OFF,
        selection_reason="Digital ICE disabled",
        receipt=receipt,
    )


def _engine_receipt(
    *,
    routed: Any,
    output: UInt16Array,
    prepass: UInt16Array,
    main: UInt16Array,
    same_frame_id: str,
    prepass_evidence_id: str,
    main_evidence_id: str,
) -> PortableDigitalIceReceipt:
    result = routed.result
    replay = result.replay
    if result.profile_id != EXPECTED_PROFILE_ID:
        raise PortableDigitalIceIntegrityError(
            "portable engine reported an unexpected profile "
            f"{result.profile_id!r}; expected {EXPECTED_PROFILE_ID!r}"
        )
    output_sha256 = _sha256_uint16(output)
    if replay.output_sha256 != output_sha256:
        raise PortableDigitalIceIntegrityError(
            "portable engine output does not match its SHA-256 receipt"
        )
    replay_shape = tuple(replay.shape)
    if replay_shape != output.shape:
        raise PortableDigitalIceIntegrityError(
            f"portable engine receipt shape {replay_shape} does not match output {output.shape}"
        )
    startup = replay.startup
    counters = {
        "attempted_pixels": replay.attempted_pixels,
        "written_pixels": replay.written_pixels,
        "changed_pixels": replay.changed_pixels,
        "public_rng_advances": replay.public_rng_advances,
    }
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise PortableDigitalIceIntegrityError(
            f"portable engine reported invalid replay counters: {counters}"
        )
    if replay.written_pixels > replay.attempted_pixels:
        raise PortableDigitalIceIntegrityError(
            "portable engine wrote more pixels than it attempted"
        )
    if replay.changed_pixels > replay.written_pixels:
        raise PortableDigitalIceIntegrityError(
            "portable engine changed more pixels than it wrote"
        )
    if (
        type(replay.final_rng_state) is not int
        or not 0 <= replay.final_rng_state < RNG_MODULUS
    ):
        raise PortableDigitalIceIntegrityError(
            "portable engine reported an invalid final RNG state"
        )
    if startup is not None:
        attempted = tuple(startup.attempted_per_stage)
        advances = tuple(startup.rng_advances_per_stage)
        if len(attempted) != STARTUP_STAGE_COUNT or len(advances) != STARTUP_STAGE_COUNT:
            raise PortableDigitalIceIntegrityError(
                "portable engine reported the wrong number of startup stages"
            )
        if any(type(value) is not int or value < 0 for value in (*attempted, *advances)):
            raise PortableDigitalIceIntegrityError(
                "portable engine reported invalid startup counters"
            )
        if (
            type(startup.final_rng_state) is not int
            or not 0 <= startup.final_rng_state < RNG_MODULUS
        ):
            raise PortableDigitalIceIntegrityError(
                "portable engine reported an invalid startup RNG state"
            )
    return PortableDigitalIceReceipt(
        status="processed",
        same_frame_id=same_frame_id,
        prepass_evidence_id=prepass_evidence_id,
        main_evidence_id=main_evidence_id,
        prepass_sha256=_sha256_uint16(prepass),
        main_sha256=_sha256_uint16(main),
        output_sha256=output_sha256,
        output_shape=output.shape,
        profile_id=EXPECTED_PROFILE_ID,
        attempted_pixels=int(replay.attempted_pixels),
        written_pixels=int(replay.written_pixels),
        changed_pixels=int(replay.changed_pixels),
        public_rng_advances=int(replay.public_rng_advances),
        final_rng_state=int(replay.final_rng_state),
        startup_attempted_per_stage=(
            None
            if startup is None
            else tuple(int(value) for value in startup.attempted_per_stage)
        ),
        startup_rng_advances_per_stage=(
            None
            if startup is None
            else tuple(int(value) for value in startup.rng_advances_per_stage)
        ),
        startup_final_rng_state=(
            None if startup is None else int(startup.final_rng_state)
        ),
    )


def _apply_arrays_unverified(
    prepass_rgbi16: npt.ArrayLike,
    main_rgbi16: npt.ArrayLike,
    *,
    same_frame_id: str,
    backend: PortableDigitalIceBackend | str,
    prepass_evidence_id: str = "negpy-prepass-rgbi16",
    main_evidence_id: str = "negpy-main-rgbi16",
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> PortableDigitalIceResult:
    """Low-level array bridge used only after acquisition evidence is checked.

    The prepass must be the native 285 dpi RGBI source and the main input must
    be the locked 4000 dpi RGBI source from the same physical frame.  This
    adapter does not rotate, flip, invert, crop, color-manage, or overwrite
    either source array.
    """

    requested = _normalise_backend(backend)
    if not same_frame_id.strip():
        raise ValueError("same_frame_id must be non-empty")
    if not prepass_evidence_id.strip() or not main_evidence_id.strip():
        raise ValueError("evidence IDs must be non-empty")

    # Own immutable copies before crossing the optional-package boundary.  In
    # particular, the caller's archival arrays can never become an output
    # buffer or be changed by a backend implementation.
    prepass = _owned_rgbi16(prepass_rgbi16, label="prepass RGBI")
    main = _owned_rgbi16(main_rgbi16, label="main RGBI")

    if requested is PortableDigitalIceBackend.OFF:
        return _off_result(
            prepass=prepass,
            main=main,
            same_frame_id=same_frame_id,
            prepass_evidence_id=prepass_evidence_id,
            main_evidence_id=main_evidence_id,
        )

    engine = _load_engine()
    acquisition = engine.DualRGBIAcquisition(
        prepass=engine.RGBI16Frame(
            prepass,
            engine.AcquisitionEpoch.PREPASS,
            285,
            prepass_evidence_id,
        ),
        main=engine.RGBI16Frame(
            main,
            engine.AcquisitionEpoch.MAIN,
            4000,
            main_evidence_id,
        ),
        same_frame_id=same_frame_id,
    )
    job = engine.ProcessingJob(
        acquisition=acquisition,
        scanner_model=engine.ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=engine.ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=4000,
        bit_depth=16,
        focus_exposure_locked=True,
    )

    try:
        routed = engine.process(
            job,
            backend=requested.value,
            progress=progress,
            cancelled=None if cancel is None else cancel.is_set,
        )
    except Exception as error:
        unavailable = _UNAVAILABLE_ENGINE_ERRORS.get(requested)
        if unavailable is not None and (
            error.__class__.__name__ == unavailable[0]
            or isinstance(error, ModuleNotFoundError)
        ):
            raise PortableDigitalIceUnavailable(
                f"portable Digital ICE {unavailable[1]} backend is unavailable: "
                f"{error}"
            ) from error
        raise

    used = _engine_backend(routed.selection.used)
    reported_requested = _engine_backend(routed.selection.requested)
    if reported_requested is not requested or used is not requested:
        raise PortableDigitalIceIntegrityError(
            "portable engine changed an explicit backend request: "
            f"requested={requested.value}, reported={reported_requested.value}, "
            f"used={used.value}"
        )
    expected_shape = (main.shape[0], main.shape[1], 3)
    output = _validated_output(
        routed.result.output_rgb16,
        expected_shape=expected_shape,
    )
    receipt = _engine_receipt(
        routed=routed,
        output=output,
        prepass=prepass,
        main=main,
        same_frame_id=same_frame_id,
        prepass_evidence_id=prepass_evidence_id,
        main_evidence_id=main_evidence_id,
    )
    return PortableDigitalIceResult(
        cleaned_rgb16=output,
        requested_backend=requested,
        used_backend=used,
        selection_reason=str(routed.selection.reason),
        receipt=receipt,
    )


def apply_portable_digital_ice(
    capture: DualSourceCapture,
    *,
    plan: DiceDualSourcePlan,
    backend: PortableDigitalIceBackend | str,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> PortableDigitalIceResult:
    """Process one verified same-frame capture before later image transforms."""

    try:
        validate_dual_source_capture(capture, plan)
    except (TypeError, ValueError, RuntimeError) as error:
        raise PortableDigitalIceIntegrityError(
            f"Digital ICE acquisition evidence is invalid: {error}"
        ) from error
    evidence_root = capture.same_frame_id
    return _apply_arrays_unverified(
        capture.prepass_rgbi,
        capture.main_rgbi,
        same_frame_id=capture.same_frame_id,
        backend=backend,
        prepass_evidence_id=f"{evidence_root}:prepass-rgbi16",
        main_evidence_id=f"{evidence_root}:main-rgbi16",
        progress=progress,
        cancel=cancel,
    )


__all__ = [
    "PortableDigitalIceAvailability",
    "PortableDigitalIceBackend",
    "PortableDigitalIceIntegrityError",
    "PortableDigitalIceReceipt",
    "PortableDigitalIceResult",
    "PortableDigitalIceUnavailable",
    "apply_portable_digital_ice",
    "availability_summary",
    "probe_backend",
]
