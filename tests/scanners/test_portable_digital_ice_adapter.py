"""Contracts for NegPy's optional portable Digital ICE boundary."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from negpy.infrastructure.scanners import portable_digital_ice_adapter as adapter
from negpy.infrastructure.scanners import dice_dual_source_runner as dual
from negpy.infrastructure.scanners.params import ScannerCaptureState


@dataclass(frozen=True)
class _FakeFrame:
    pixels: np.ndarray
    epoch: str
    resolution_dpi: int
    evidence_id: str


@dataclass(frozen=True)
class _FakeAcquisition:
    prepass: _FakeFrame
    main: _FakeFrame
    same_frame_id: str


@dataclass(frozen=True)
class _FakeJob:
    acquisition: _FakeAcquisition
    scanner_model: str
    mode: str
    selector: int
    resolution_metric: int
    bit_depth: int
    focus_exposure_locked: bool


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        array.astype("<u2", copy=False).tobytes(order="C")
    ).hexdigest()


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    prepass = np.arange(8 * 10 * 4, dtype=np.uint16).reshape(8, 10, 4)
    main = np.arange(12 * 16 * 4, dtype=np.uint16).reshape(12, 16, 4)
    return prepass, main


def _verified_capture() -> tuple[dual.DualSourceCapture, dual.DiceDualSourcePlan]:
    plan = dual.DiceDualSourcePlan(
        window=dual.PixelWindow(0, 0, 111, 111),
        transport="mounted",
    )
    prepass = np.arange(8 * 8 * 4, dtype=np.uint16).reshape(8, 8, 4)
    main = np.arange(112 * 112 * 4, dtype=np.uint16).reshape(112, 112, 4)
    events: list[dict[str, object]] = []

    def event(kind: str, **fields: object) -> None:
        events.append({"event": kind, **fields})

    for name, value in (
        ("preview", False),
        ("infrared", False),
        ("samples_per_scan", 1),
        ("depth", 16),
        ("negative", False),
        ("tl_x", 0),
        ("tl_y", 0),
        ("br_x", 111),
        ("br_y", 111),
        ("infrared", True),
        ("resolution", 285),
        ("autofocus", True),
        ("ae", True),
    ):
        event("set", option=name, value=value, readback_verified=True)
    event(
        "read_begin",
        epoch="prepass",
        source="dedicated_single_sample_rgbi",
    )
    event("read_end", epoch="prepass", shape=[8, 8, 4])
    event("capture_state_read")
    for name, value in (
        ("autofocus", False),
        ("ae", False),
        ("focus", 216),
        ("exposure", 1.0),
        ("red_exposure", 1370.0),
        ("green_exposure", 1290.0),
        ("blue_exposure", 1120.0),
        ("resolution", 4000),
    ):
        event("set", option=name, value=value, readback_verified=True)
    event(
        "read_begin",
        epoch="main",
        source="dedicated_single_sample_rgbi",
    )
    event("read_end", epoch="main", shape=[112, 112, 4])
    event("capture_state_verified")
    assertions = dual._derive_assertions(events, plan)
    capture = dual.DualSourceCapture(
        prepass_rgbi=prepass,
        main_rgbi=main,
        capture_state=ScannerCaptureState(216, 1.0, 1370.0, 1290.0, 1120.0),
        scanner_identity=dual.ScannerIdentity(
            device_id="coolscan3:usb:test",
            vendor="Nikon",
            model="LS-5000 ED",
            kind="film scanner",
        ),
        same_frame_id="verified-frame-20",
        events=tuple(events),
        assertions=assertions,
    )
    return capture, plan


def _successful_engine(calls: list[dict[str, Any]]) -> SimpleNamespace:
    def process(job, *, backend, progress, cancelled):
        event = SimpleNamespace(phase="reconstruction", completed=4, total=12)
        if progress is not None:
            progress(event)
        output = np.array(job.acquisition.main.pixels[:, :, :3], copy=True)
        output[:, :, 0] += 1
        startup = SimpleNamespace(
            attempted_per_stage=(0, 1, 2, 3, 4, 5),
            rng_advances_per_stage=(5, 4, 3, 2, 1, 0),
            final_rng_state=12_357,
        )
        replay = SimpleNamespace(
            shape=output.shape,
            startup=startup,
            attempted_pixels=91,
            written_pixels=37,
            public_rng_advances=123,
            final_rng_state=45_678,
            output_sha256=_sha256(output),
            changed_pixels=29,
        )
        calls.append(
            {
                "job": job,
                "backend": backend,
                "cancelled": cancelled,
                "output": output,
                "event": event,
            }
        )
        return SimpleNamespace(
            result=SimpleNamespace(
                output_rgb16=output,
                replay=replay,
                profile_id="nikon-ls5000-selector8-normal-metric4000",
            ),
            selection=SimpleNamespace(
                requested=SimpleNamespace(value=backend),
                used=SimpleNamespace(value=backend),
                reason=f"explicit {backend.upper()} request",
            ),
        )

    return SimpleNamespace(
        AcquisitionEpoch=SimpleNamespace(PREPASS="prepass", MAIN="main"),
        RGBI16Frame=_FakeFrame,
        DualRGBIAcquisition=_FakeAcquisition,
        ProcessingJob=_FakeJob,
        ScannerModel=SimpleNamespace(
            NIKON_SUPER_COOLSCAN_5000_ED="nikon-super-coolscan-5000-ed"
        ),
        ProcessingMode=SimpleNamespace(NORMAL="normal"),
        process=process,
    )


def test_off_is_a_dependency_free_non_destructive_passthrough(monkeypatch) -> None:
    prepass, main = _arrays()
    prepass_before = prepass.copy()
    main_before = main.copy()

    def must_not_import():
        raise AssertionError("off must not import the optional package")

    monkeypatch.setattr(adapter, "_load_engine", must_not_import)

    result = adapter._apply_arrays_unverified(
        prepass,
        main,
        same_frame_id="frame-20",
        backend=adapter.PortableDigitalIceBackend.OFF,
    )

    assert result.requested_backend is adapter.PortableDigitalIceBackend.OFF
    assert result.used_backend is adapter.PortableDigitalIceBackend.OFF
    assert result.receipt.status == "bypassed"
    assert result.receipt.profile_id is None
    assert np.array_equal(result.cleaned_rgb16, main[:, :, :3])
    assert not np.shares_memory(result.cleaned_rgb16, main)
    assert np.array_equal(prepass, prepass_before)
    assert np.array_equal(main, main_before)
    result.cleaned_rgb16[0, 0, 0] = 65_535
    assert main[0, 0, 0] == main_before[0, 0, 0]


def test_cpu_builds_exact_engine_contracts_and_returns_bound_receipt(monkeypatch) -> None:
    prepass, main = _arrays()
    prepass_before = prepass.copy()
    main_before = main.copy()
    calls: list[dict[str, Any]] = []
    engine = _successful_engine(calls)
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)
    progress: list[object] = []
    cancel = threading.Event()

    result = adapter._apply_arrays_unverified(
        prepass,
        main,
        same_frame_id="gold-200-frame-20",
        backend="cpu",
        prepass_evidence_id="capture/prepass",
        main_evidence_id="capture/main",
        progress=progress.append,
        cancel=cancel,
    )

    [call] = calls
    job = call["job"]
    assert call["backend"] == "cpu"
    assert job.acquisition.prepass.resolution_dpi == 285
    assert job.acquisition.prepass.epoch == "prepass"
    assert job.acquisition.main.resolution_dpi == 4000
    assert job.acquisition.main.epoch == "main"
    assert job.acquisition.same_frame_id == "gold-200-frame-20"
    assert job.scanner_model == "nikon-super-coolscan-5000-ed"
    assert job.mode == "normal"
    assert job.selector == 8
    assert job.resolution_metric == 4000
    assert job.bit_depth == 16
    assert job.focus_exposure_locked is True
    assert job.acquisition.prepass.pixels.flags.writeable is False
    assert job.acquisition.main.pixels.flags.writeable is False
    assert not np.shares_memory(job.acquisition.prepass.pixels, prepass)
    assert not np.shares_memory(job.acquisition.main.pixels, main)
    assert progress == [call["event"]]
    assert call["cancelled"]() is False
    cancel.set()
    assert call["cancelled"]() is True

    assert result.requested_backend is adapter.PortableDigitalIceBackend.CPU
    assert result.used_backend is adapter.PortableDigitalIceBackend.CPU
    assert result.selection_reason == "explicit CPU request"
    assert result.receipt.status == "processed"
    assert result.receipt.same_frame_id == "gold-200-frame-20"
    assert result.receipt.prepass_evidence_id == "capture/prepass"
    assert result.receipt.main_evidence_id == "capture/main"
    assert result.receipt.prepass_sha256 == _sha256(prepass)
    assert result.receipt.main_sha256 == _sha256(main)
    assert result.receipt.output_sha256 == _sha256(result.cleaned_rgb16)
    assert result.receipt.output_shape == (12, 16, 3)
    assert result.receipt.attempted_pixels == 91
    assert result.receipt.written_pixels == 37
    assert result.receipt.changed_pixels == 29
    assert result.receipt.public_rng_advances == 123
    assert result.receipt.final_rng_state == 45_678
    assert result.receipt.startup_attempted_per_stage == (0, 1, 2, 3, 4, 5)
    assert result.receipt.startup_rng_advances_per_stage == (5, 4, 3, 2, 1, 0)
    assert result.receipt.startup_final_rng_state == 12_357
    assert not np.shares_memory(result.cleaned_rgb16, call["output"])
    assert np.array_equal(prepass, prepass_before)
    assert np.array_equal(main, main_before)


def test_auto_is_not_an_exposed_backend_and_fails_before_import(monkeypatch) -> None:
    prepass, main = _arrays()

    def must_not_import():
        raise AssertionError("invalid backend reached optional import")

    monkeypatch.setattr(adapter, "_load_engine", must_not_import)

    with pytest.raises(ValueError, match="off, cpu, cpu-fast, cuda"):
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="auto",
        )


def test_missing_optional_package_fails_with_install_instruction(monkeypatch) -> None:
    prepass, main = _arrays()

    def missing(_name: str):
        error = ModuleNotFoundError("No module named 'portable_digital_ice'")
        error.name = "portable_digital_ice"
        raise error

    monkeypatch.setattr(adapter, "import_module", missing)

    with pytest.raises(
        adapter.PortableDigitalIceUnavailable,
        match="github.com/rohanpandula/digital-fauxice",
    ):
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="cpu",
        )


def test_cuda_unavailable_fails_without_cpu_fallback(monkeypatch) -> None:
    prepass, main = _arrays()
    attempted: list[str] = []
    unavailable = type("CudaBackendUnavailable", (RuntimeError,), {})
    engine = _successful_engine([])

    def process(_job, *, backend, progress, cancelled):
        attempted.append(backend)
        raise unavailable("no usable CUDA device")

    engine.process = process
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(
        adapter.PortableDigitalIceUnavailable,
        match="CUDA backend is unavailable.*no usable CUDA device",
    ):
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="cuda",
        )

    assert attempted == ["cuda"]


def test_cpu_fast_routes_to_the_compiled_backend(monkeypatch) -> None:
    prepass, main = _arrays()
    calls: list[dict[str, Any]] = []
    engine = _successful_engine(calls)
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    result = adapter._apply_arrays_unverified(
        prepass,
        main,
        same_frame_id="frame-20",
        backend="cpu-fast",
    )

    [call] = calls
    assert call["backend"] == "cpu-fast"
    assert result.requested_backend is adapter.PortableDigitalIceBackend.CPU_FAST
    assert result.used_backend is adapter.PortableDigitalIceBackend.CPU_FAST
    assert result.receipt.status == "processed"


def test_cpu_fast_unavailable_fails_without_reference_fallback(monkeypatch) -> None:
    prepass, main = _arrays()
    attempted: list[str] = []
    unavailable = type("CpuFastUnavailable", (RuntimeError,), {})
    engine = _successful_engine([])

    def process(_job, *, backend, progress, cancelled):
        attempted.append(backend)
        raise unavailable("numba is not installed")

    engine.process = process
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(
        adapter.PortableDigitalIceUnavailable,
        match="compiled CPU backend is unavailable.*numba is not installed",
    ):
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="cpu-fast",
        )

    # The reference backend is a different scan, not a silent substitute for a
    # compiled one that could not start.
    assert attempted == ["cpu-fast"]


def test_cancellation_callback_and_engine_exception_pass_through(monkeypatch) -> None:
    prepass, main = _arrays()
    cancel = threading.Event()
    cancel.set()
    cancelled = type("ProcessingCancelled", (RuntimeError,), {})
    engine = _successful_engine([])

    def process(_job, *, backend, progress, cancelled: Any):
        assert backend == "cpu"
        assert cancelled() is True
        raise cancelled_error

    cancelled_error = cancelled("stopped")
    engine.process = process
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(cancelled) as raised:
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="cpu",
            cancel=cancel,
        )

    assert raised.value is cancelled_error


def test_engine_cannot_silently_change_the_explicit_backend(monkeypatch) -> None:
    prepass, main = _arrays()
    calls: list[dict[str, Any]] = []
    engine = _successful_engine(calls)
    original_process = engine.process

    def process(*args, **kwargs):
        routed = original_process(*args, **kwargs)
        routed.selection.used.value = "cpu"
        return routed

    engine.process = process
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(
        adapter.PortableDigitalIceIntegrityError,
        match="changed an explicit backend request",
    ):
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="cuda",
        )


def test_engine_output_must_match_its_receipt(monkeypatch) -> None:
    prepass, main = _arrays()
    engine = _successful_engine([])
    original_process = engine.process

    def process(*args, **kwargs):
        routed = original_process(*args, **kwargs)
        routed.result.replay.output_sha256 = "0" * 64
        return routed

    engine.process = process
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(
        adapter.PortableDigitalIceIntegrityError,
        match="does not match its SHA-256 receipt",
    ):
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="cpu",
        )


def test_invalid_capture_contract_fails_before_optional_import(monkeypatch) -> None:
    prepass, main = _arrays()

    def must_not_import():
        raise AssertionError("bad arrays reached optional import")

    monkeypatch.setattr(adapter, "_load_engine", must_not_import)

    with pytest.raises(TypeError, match="prepass RGBI must have dtype uint16"):
        adapter._apply_arrays_unverified(
            prepass.astype(np.float32),
            main,
            same_frame_id="frame-20",
            backend="cpu",
        )
    with pytest.raises(ValueError, match="main RGBI must have shape HxWx4"):
        adapter._apply_arrays_unverified(
            prepass,
            main[:, :, :3],
            same_frame_id="frame-20",
            backend="cuda",
        )


def test_public_adapter_requires_and_processes_verified_capture(monkeypatch) -> None:
    capture, plan = _verified_capture()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter, "_load_engine", lambda: _successful_engine(calls))

    result = adapter.apply_portable_digital_ice(
        capture,
        plan=plan,
        backend="cpu",
    )

    assert result.receipt.same_frame_id == capture.same_frame_id
    assert result.receipt.prepass_evidence_id.endswith(":prepass-rgbi16")
    assert result.receipt.main_evidence_id.endswith(":main-rgbi16")
    assert result.cleaned_rgb16.shape == (112, 112, 3)
    assert len(calls) == 1


def test_public_adapter_rederives_capture_evidence_before_import(monkeypatch) -> None:
    capture, plan = _verified_capture()
    forged = dual.DualSourceCapture(
        prepass_rgbi=capture.prepass_rgbi,
        main_rgbi=capture.main_rgbi,
        capture_state=capture.capture_state,
        scanner_identity=capture.scanner_identity,
        same_frame_id=capture.same_frame_id,
        events=(),
        assertions={"all_passed": True},
    )

    def must_not_import():
        raise AssertionError("invalid acquisition evidence reached optional import")

    monkeypatch.setattr(adapter, "_load_engine", must_not_import)

    with pytest.raises(
        adapter.PortableDigitalIceIntegrityError,
        match="acquisition evidence is invalid",
    ):
        adapter.apply_portable_digital_ice(forged, plan=plan, backend="cpu")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda routed: setattr(routed.result, "profile_id", "different-profile"),
            "unexpected profile",
        ),
        (
            lambda routed: setattr(routed.result.replay, "written_pixels", 92),
            "more pixels than it attempted",
        ),
        (
            lambda routed: setattr(routed.result.replay.startup, "attempted_per_stage", (0,)),
            "wrong number of startup stages",
        ),
        (
            lambda routed: setattr(routed.result.replay, "final_rng_state", 1 << 24),
            "invalid final RNG state",
        ),
    ),
)
def test_engine_receipt_invariants_fail_closed(monkeypatch, mutation, message) -> None:
    prepass, main = _arrays()
    engine = _successful_engine([])
    original_process = engine.process

    def process(*args, **kwargs):
        routed = original_process(*args, **kwargs)
        mutation(routed)
        return routed

    engine.process = process
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(adapter.PortableDigitalIceIntegrityError, match=message):
        adapter._apply_arrays_unverified(
            prepass,
            main,
            same_frame_id="frame-20",
            backend="cpu",
        )


def test_real_optional_package_contract_when_installed() -> None:
    pytest.importorskip("portable_digital_ice")
    capture, plan = _verified_capture()

    result = adapter.apply_portable_digital_ice(
        capture,
        plan=plan,
        backend="cpu",
    )

    assert result.cleaned_rgb16.shape == (112, 112, 3)
    assert result.receipt.output_sha256 == _sha256(result.cleaned_rgb16)


def test_real_optional_package_compiled_cpu_matches_reference_when_installed() -> None:
    """The compiled backend is a speed choice, not a different result."""

    pytest.importorskip("portable_digital_ice")
    capture, plan = _verified_capture()

    reference = adapter.apply_portable_digital_ice(capture, plan=plan, backend="cpu")
    try:
        compiled = adapter.apply_portable_digital_ice(
            capture,
            plan=plan,
            backend="cpu-fast",
        )
    except adapter.PortableDigitalIceUnavailable as error:
        pytest.skip(f"compiled CPU backend is not installed: {error}")

    assert compiled.used_backend is adapter.PortableDigitalIceBackend.CPU_FAST
    assert compiled.receipt.output_sha256 == reference.receipt.output_sha256
    assert np.array_equal(compiled.cleaned_rgb16, reference.cleaned_rgb16)


def test_probe_off_never_imports_and_cpu_needs_only_the_engine(monkeypatch) -> None:
    def must_not_import():
        raise AssertionError("off must not import the optional package")

    monkeypatch.setattr(adapter, "_load_engine", must_not_import)
    adapter.probe_backend("off")

    engine = SimpleNamespace()  # no backend module at all
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)
    adapter.probe_backend("cpu")


def test_probe_runs_the_engine_self_test_for_compiled_backends(monkeypatch) -> None:
    ran: list[str] = []
    engine = SimpleNamespace(
        backend=SimpleNamespace(
            cpu_fast_self_test=lambda: ran.append("cpu-fast"),
            cuda_self_test=lambda: ran.append("cuda"),
        )
    )
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    adapter.probe_backend("cpu-fast")
    adapter.probe_backend(adapter.PortableDigitalIceBackend.CUDA)

    assert ran == ["cpu-fast", "cuda"]


def test_probe_reports_a_failed_self_test_as_unavailable(monkeypatch) -> None:
    unavailable = type("CpuFastUnavailable", (RuntimeError,), {})

    def failing_self_test() -> None:
        raise unavailable("numba is not importable")

    engine = SimpleNamespace(backend=SimpleNamespace(cpu_fast_self_test=failing_self_test))
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(
        adapter.PortableDigitalIceUnavailable,
        match="compiled CPU backend is unavailable.*numba is not importable",
    ):
        adapter.probe_backend("cpu-fast")


def test_probe_refuses_an_engine_without_the_self_test(monkeypatch) -> None:
    engine = SimpleNamespace(backend=SimpleNamespace())
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    with pytest.raises(
        adapter.PortableDigitalIceUnavailable,
        match="0.2.0 or newer",
    ):
        adapter.probe_backend("cpu-fast")


def test_availability_summary_without_the_engine_reports_everything_missing(monkeypatch) -> None:
    def missing() -> None:
        raise adapter.PortableDigitalIceUnavailable("portable Digital ICE is not installed")

    monkeypatch.setattr(adapter, "_load_engine", missing)

    summary = adapter.availability_summary()

    assert summary.engine_installed is False
    assert summary.cpu_fast_installed is False
    assert summary.cuda_installed is False
    assert "not installed" in summary.engine_detail


def test_availability_summary_is_import_only_and_reports_cuda_reason(monkeypatch) -> None:
    engine = SimpleNamespace(__file__="/fake/portable_digital_ice/__init__.py")
    monkeypatch.setattr(adapter, "_load_engine", lambda: engine)

    def fake_import(name: str):
        assert name == "portable_digital_ice.cuda_backend"
        raise ModuleNotFoundError("No module named 'cupy'")

    monkeypatch.setattr(adapter, "import_module", fake_import)

    summary = adapter.availability_summary()

    assert summary.engine_installed is True
    # NegPy pins numba, so the compiled-CPU probe reflects the real venv.
    assert summary.cpu_fast_installed is True
    assert summary.cuda_installed is False
    assert "cupy" in summary.cuda_detail
