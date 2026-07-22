"""Tests for the fauxice IR repair adapter, with stub engine/hybrid modules.

Neither ``portable_digital_ice`` nor ``fauxce_hybrid`` is installed in this
environment (they are optional dependencies distributed from GitHub
releases, not PyPI), so every test that needs "the engine is installed"
installs a small fake module into ``sys.modules`` first via
``_install_stub_engine`` below, rather than requiring a real install.
``importlib.util.find_spec`` resolves an already-imported module through
``sys.modules``, so this is enough to make ``engine_available()`` (and a
real ``from portable_digital_ice import ...``) see the stub.

The stub engine's ``process()`` inverts the 16-bit main RGB plane
(``65535 - value``) as its "repair": deterministic, trivially distinguished
from the untouched input, and safely within uint16 range for the small
fixture values used here.
"""

from __future__ import annotations

import importlib.machinery
import hashlib
import json
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import tifffile

from negpy.services.repair.fauxice_hybrid_runner import (
    HybridRunResult,
    HybridRuntimeConfig,
)
from negpy.services.repair.fauxice_ir_repair import (
    ENGINE_IMPORT_NAME,
    HYBRID_IMPORT_NAME,
    FauxiceRepairCancelled,
    FauxiceRepairConfig,
    RepairMode,
    RepairStatus,
    engine_available,
    hybrid_available,
    repair_frame_files,
    repair_ir_dust,
)

INVERT = np.uint16(65535)


def _install_stub_engine(monkeypatch: pytest.MonkeyPatch, **process_kwargs: object) -> types.ModuleType:
    """Register a minimal fake ``portable_digital_ice`` in ``sys.modules``."""

    module = types.ModuleType(ENGINE_IMPORT_NAME)
    module.__spec__ = importlib.machinery.ModuleSpec(ENGINE_IMPORT_NAME, loader=None)

    class AcquisitionEpoch:
        PREPASS = "prepass"
        MAIN = "main"

    class ComputeBackend(str):
        def __new__(cls, value: str) -> "ComputeBackend":
            return str.__new__(cls, value)

        @property
        def value(self) -> str:
            return str(self)

    class ProcessingMode:
        NORMAL = "normal"

    class ScannerModel:
        NIKON_SUPER_COOLSCAN_5000_ED = "nikon-ls5000"

    class ProcessingCancelled(RuntimeError):
        pass

    @dataclass
    class RGBI16Frame:
        pixels: np.ndarray
        epoch: str
        resolution_dpi: int
        evidence_id: str

    @dataclass
    class DualRGBIAcquisition:
        prepass: RGBI16Frame
        main: RGBI16Frame
        same_frame_id: str

    @dataclass
    class ProcessingJob:
        acquisition: DualRGBIAcquisition
        scanner_model: str
        mode: str
        selector: int
        resolution_metric: int
        bit_depth: int
        focus_exposure_locked: bool

    @dataclass
    class _Progress:
        completed: int
        total: int

    @dataclass
    class _Selection:
        requested: object
        used: object
        reason: str

    @dataclass
    class _Result:
        output_rgb16: np.ndarray

    @dataclass
    class _Routed:
        result: _Result
        selection: _Selection

    calls_made = process_kwargs.get("calls", None)

    def process(
        job: ProcessingJob,
        *,
        backend,
        output_rgb16=None,
        progress=None,
        cancelled=None,
        export_diagnostics: bool = False,
    ) -> _Routed:
        if calls_made is not None:
            calls_made.append(job)
        if progress is not None:
            progress(_Progress(completed=0, total=2))
        if cancelled is not None and cancelled():
            raise ProcessingCancelled("stub engine cancelled before completion")
        if progress is not None:
            progress(_Progress(completed=2, total=2))
        main_rgb = job.acquisition.main.pixels[:, :, :3]
        output = (INVERT - main_rgb.astype(np.int64)).astype(np.uint16)
        backend_value = ComputeBackend(backend)
        return _Routed(
            result=_Result(output_rgb16=output),
            selection=_Selection(requested=backend_value, used=backend_value, reason="stub selection"),
        )

    module.AcquisitionEpoch = AcquisitionEpoch
    module.ComputeBackend = ComputeBackend
    module.DualRGBIAcquisition = DualRGBIAcquisition
    module.ProcessingJob = ProcessingJob
    module.ProcessingMode = ProcessingMode
    module.RGBI16Frame = RGBI16Frame
    module.ScannerModel = ScannerModel
    module.ProcessingCancelled = ProcessingCancelled
    module.process = process

    monkeypatch.setitem(sys.modules, ENGINE_IMPORT_NAME, module)
    return module


def _install_stub_hybrid(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Register a bare-presence fake ``fauxce_hybrid`` (only ``find_spec`` needs it)."""

    module = types.ModuleType(HYBRID_IMPORT_NAME)
    module.__spec__ = importlib.machinery.ModuleSpec(HYBRID_IMPORT_NAME, loader=None)
    monkeypatch.setitem(sys.modules, HYBRID_IMPORT_NAME, module)
    return module


class _FlippingCancelEvent:
    """``threading.Event`` look-alike: unset for the first N calls, set after."""

    def __init__(self, set_after_calls: int) -> None:
        self._calls = 0
        self._set_after = set_after_calls

    def is_set(self) -> bool:
        self._calls += 1
        return self._calls > self._set_after


def _rgb(height: int, width: int, base: int) -> np.ndarray:
    return np.full((height, width, 3), base, dtype=np.uint16)


def _ir(height: int, width: int, base: int) -> np.ndarray:
    return np.full((height, width), base, dtype=np.uint16)


def _prepass(height: int, width: int) -> np.ndarray:
    return np.full((height, width, 4), 2000, dtype=np.uint16)


def _hybrid_runtime(tmp_path: Path) -> HybridRuntimeConfig:
    hybrid_python = tmp_path / "hybrid" / "bin" / "python"
    executable = tmp_path / "hybrid" / "bin" / "fauxce-hybrid"
    iopaint_python = tmp_path / "iopaint" / "bin" / "python"
    iopaint_executable = tmp_path / "iopaint" / "bin" / "iopaint"
    model_dir = tmp_path / "models"
    model_weights = model_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt"
    for path in (hybrid_python, executable, iopaint_python, iopaint_executable):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
    model_weights.parent.mkdir(parents=True, exist_ok=True)
    model_weights.write_bytes(b"stub-model")
    return HybridRuntimeConfig(
        hybrid_python=hybrid_python,
        executable=executable,
        core_source_manifest_sha256="c" * 64,
        hybrid_source_manifest_sha256="d" * 64,
        iopaint_python=iopaint_python,
        iopaint_executable=iopaint_executable,
        iopaint_source_manifest_sha256="a" * 64,
        model_dir=model_dir,
        model_weights=model_weights,
        model_weights_sha256=hashlib.sha256(b"stub-model").hexdigest(),
    )


def _hybrid_receipt(*, fraction: float = 0.02) -> dict:
    return {
        "core": {
            "version": "0.3.0",
            "backend": {"requested": "cpu", "used": "cpu", "reason": "stub"},
        },
        "synthesis": {"fraction": fraction},
        "routing": {
            "counts": {
                "final_regions": 2,
                "synthesis_pixels": 40,
                "frame_pixels": 400,
                "at_floor_pixels": 60,
            }
        },
    }


def _stub_hybrid_runner(hybrid_rgb16: np.ndarray, *, receipt: dict | None = None) -> Callable:
    def runner(argv, **kwargs) -> subprocess.CompletedProcess:
        out_dir = Path(argv[argv.index("--out") + 1])
        out_dir.mkdir(parents=True)
        np.save(out_dir / "output-hybrid.rgb16.npy", hybrid_rgb16, allow_pickle=False)
        (out_dir / "synth-mask.png").write_bytes(b"mask-bytes")
        (out_dir / "hybrid-receipt.json").write_text(json.dumps(receipt or _hybrid_receipt()), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return runner


def _hybrid_outcome(hybrid_rgb16: np.ndarray) -> HybridRunResult:
    mask = np.zeros(hybrid_rgb16.shape[:2], dtype=np.bool_)
    receipt = b'{"schema":"fauxce-hybrid-receipt-v2"}'
    mask_bytes = b"mask-bytes"
    return HybridRunResult(
        hybrid_rgb16=np.ascontiguousarray(hybrid_rgb16),
        synth_mask_png=mask_bytes,
        synth_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        synth_mask=mask,
        receipt=receipt,
        receipt_sha256=hashlib.sha256(receipt).hexdigest(),
        acquisition_manifest_sha256="1" * 64,
        main_rgbi_sha256="2" * 64,
        prepass_rgbi_sha256="3" * 64,
        output_rgb16_sha256=hashlib.sha256(hybrid_rgb16.astype("<u2").tobytes()).hexdigest(),
        provenance_class="caller_asserted_bare_npy",
        synthesis_fraction=0.02,
        engine_version="0.3.0",
        backend_requested="cpu",
        backend_used="cpu",
        backend_selection_reason="stub",
        routing_counts={
            "final_regions": 2,
            "synthesis_pixels": 40,
            "frame_pixels": 400,
            "at_floor_pixels": 60,
        },
    )


# ---------------------------------------------------------------------------
# Availability gating
# ---------------------------------------------------------------------------


class TestAvailabilityGating:
    def test_engine_unavailable_when_distribution_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "negpy.services.repair.fauxice_ir_repair.importlib.util.find_spec",
            lambda _name: None,
        )
        assert engine_available() is False

    def test_hybrid_unavailable_by_default(self) -> None:
        assert hybrid_available() is False

    def test_engine_available_with_stub_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        assert engine_available() is True

    def test_hybrid_available_with_explicit_external_runtime(self, tmp_path: Path) -> None:
        assert hybrid_available(_hybrid_runtime(tmp_path)) is True

    def test_engine_and_hybrid_availability_are_independent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        assert engine_available() is True
        assert hybrid_available() is False


# ---------------------------------------------------------------------------
# Disabled / unavailable / no-prepass short circuits
# ---------------------------------------------------------------------------


class TestShortCircuits:
    def test_disabled_config_skips_without_checking_engine(self) -> None:
        config = FauxiceRepairConfig(enabled=False)
        result = repair_ir_dust(_rgb(4, 4, 1000), _ir(4, 4, 500), same_frame_id="f1", config=config, prepass_rgbi=_prepass(4, 4))
        assert result.status is RepairStatus.SKIPPED
        assert "disabled" in result.reason

    def test_engine_unavailable_reports_unavailable_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "negpy.services.repair.fauxice_ir_repair.importlib.util.find_spec",
            lambda _name: None,
        )
        config = FauxiceRepairConfig(enabled=True)
        result = repair_ir_dust(_rgb(4, 4, 1000), _ir(4, 4, 500), same_frame_id="f1", config=config, prepass_rgbi=_prepass(4, 4))
        assert result.status is RepairStatus.UNAVAILABLE
        assert ENGINE_IMPORT_NAME in result.reason

    def test_missing_prepass_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        result = repair_ir_dust(_rgb(4, 4, 1000), _ir(4, 4, 500), same_frame_id="f1", config=config)
        assert result.status is RepairStatus.SKIPPED
        assert "prepass" in result.reason


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_malformed_rgb_shape_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        with pytest.raises(ValueError, match="rgb"):
            repair_ir_dust(
                np.zeros((4, 4), dtype=np.uint16),
                _ir(4, 4, 500),
                same_frame_id="f1",
                config=config,
                prepass_rgbi=_prepass(4, 4),
            )

    def test_mismatched_ir_shape_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        with pytest.raises(ValueError, match="shape"):
            repair_ir_dust(
                _rgb(4, 4, 1000),
                _ir(8, 8, 500),
                same_frame_id="f1",
                config=config,
                prepass_rgbi=_prepass(4, 4),
            )

    def test_empty_same_frame_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        with pytest.raises(ValueError, match="same_frame_id"):
            repair_ir_dust(
                _rgb(4, 4, 1000),
                _ir(4, 4, 500),
                same_frame_id="  ",
                config=config,
                prepass_rgbi=_prepass(4, 4),
            )

    def test_malformed_prepass_shape_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        with pytest.raises(ValueError, match="prepass_rgbi"):
            repair_ir_dust(
                _rgb(4, 4, 1000),
                _ir(4, 4, 500),
                same_frame_id="f1",
                config=config,
                prepass_rgbi=np.zeros((4, 4, 3), dtype=np.uint16),
            )


# ---------------------------------------------------------------------------
# Mode selection: exact
# ---------------------------------------------------------------------------


class TestExactMode:
    def test_default_mode_is_exact(self) -> None:
        assert FauxiceRepairConfig().mode is RepairMode.EXACT

    def test_exact_mode_applies_via_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.EXACT)
        rgb = _rgb(4, 4, 1000)
        result = repair_ir_dust(rgb, _ir(4, 4, 500), same_frame_id="f1", config=config, prepass_rgbi=_prepass(4, 4))
        assert result.status is RepairStatus.APPLIED
        assert result.mode_requested is RepairMode.EXACT
        assert result.mode_resolved is RepairMode.EXACT
        assert result.repaired_rgb16 is not None
        np.testing.assert_array_equal(result.repaired_rgb16, INVERT - rgb.astype(np.int64))
        assert result.backend_used == "auto"

    def test_exact_mode_honours_explicit_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.EXACT, backend="cpu")
        result = repair_ir_dust(_rgb(4, 4, 1000), _ir(4, 4, 500), same_frame_id="f1", config=config, prepass_rgbi=_prepass(4, 4))
        assert result.backend_requested == "cpu"
        assert result.backend_used == "cpu"

    def test_validity_mask_restricts_repair_to_valid_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        rgb = _rgb(4, 4, 1000)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True

        result = repair_ir_dust(
            rgb,
            _ir(4, 4, 500),
            same_frame_id="f1",
            config=config,
            prepass_rgbi=_prepass(4, 4),
            validity_mask=mask,
        )

        repaired = result.repaired_rgb16
        assert repaired is not None
        # Inside the mask: repaired (inverted) value.
        np.testing.assert_array_equal(repaired[1:3, 1:3], INVERT - rgb[1:3, 1:3].astype(np.int64))
        # Outside the mask: untouched original.
        np.testing.assert_array_equal(repaired[0, :], rgb[0, :])
        np.testing.assert_array_equal(repaired[3, :], rgb[3, :])

    def test_engine_rejection_reports_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _install_stub_engine(monkeypatch)

        def failing_process(*args, **kwargs):
            raise ValueError("synthetic profile rejection")

        module.process = failing_process
        config = FauxiceRepairConfig(enabled=True)
        result = repair_ir_dust(_rgb(4, 4, 1000), _ir(4, 4, 500), same_frame_id="f1", config=config, prepass_rgbi=_prepass(4, 4))
        assert result.status is RepairStatus.SKIPPED
        assert "rejected the acquisition" in result.reason
        assert "synthetic profile rejection" in result.reason


# ---------------------------------------------------------------------------
# Progress and cancellation (exact path)
# ---------------------------------------------------------------------------


class TestProgressAndCancellation:
    def test_progress_callback_receives_fractional_updates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        observed: list[float] = []

        result = repair_ir_dust(
            _rgb(4, 4, 1000),
            _ir(4, 4, 500),
            same_frame_id="f1",
            config=config,
            prepass_rgbi=_prepass(4, 4),
            progress=observed.append,
        )

        assert result.status is RepairStatus.APPLIED
        assert observed == [0.0, 1.0]

    def test_cancel_before_start_skips_without_calling_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []
        _install_stub_engine(monkeypatch, calls=calls)
        config = FauxiceRepairConfig(enabled=True)
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(FauxiceRepairCancelled, match="cancelled"):
            repair_ir_dust(
                _rgb(4, 4, 1000),
                _ir(4, 4, 500),
                same_frame_id="f1",
                config=config,
                prepass_rgbi=_prepass(4, 4),
                cancel=cancel,
            )

        assert calls == []  # the engine was never invoked

    def test_cancel_during_exact_run_skips_with_clear_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        # Unset for the adapter's pre-flight check, set by the time the stub
        # engine polls it mid-run.
        cancel = _FlippingCancelEvent(set_after_calls=1)

        with pytest.raises(
            FauxiceRepairCancelled,
            match="cancelled before completion",
        ):
            repair_ir_dust(
                _rgb(4, 4, 1000),
                _ir(4, 4, 500),
                same_frame_id="f1",
                config=config,
                prepass_rgbi=_prepass(4, 4),
                cancel=cancel,
            )


# ---------------------------------------------------------------------------
# Mode selection: hybrid, and hybrid-selected-but-absent degradation
# ---------------------------------------------------------------------------


class TestHybridMode:
    def test_hybrid_mode_applies_via_hybrid_runner(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        _install_stub_hybrid(monkeypatch)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.HYBRID)
        hybrid_output = np.full((4, 4, 3), 9999, dtype=np.uint16)
        monkeypatch.setattr(
            "negpy.services.repair.fauxice_ir_repair.run_hybrid_repair",
            lambda *args, **kwargs: _hybrid_outcome(hybrid_output),
        )

        result = repair_ir_dust(
            _rgb(4, 4, 1000),
            _ir(4, 4, 500),
            same_frame_id="f1",
            config=config,
            prepass_rgbi=_prepass(4, 4),
            hybrid_runtime=_hybrid_runtime(tmp_path),
            hybrid_subprocess_runner=_stub_hybrid_runner(hybrid_output),
        )

        assert result.status is RepairStatus.APPLIED
        assert result.mode_requested is RepairMode.HYBRID
        assert result.mode_resolved is RepairMode.HYBRID
        np.testing.assert_array_equal(result.repaired_rgb16, hybrid_output)
        assert result.hybrid_synthesis_fraction == 0.02
        assert result.hybrid_routing_counts == {
            "final_regions": 2,
            "synthesis_pixels": 40,
            "frame_pixels": 400,
            "at_floor_pixels": 60,
        }
        assert "2 region(s) routed" in result.reason

    def test_hybrid_validity_composite_has_its_own_final_output_hash(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        hybrid_output = np.full((4, 4, 3), 9999, dtype=np.uint16)
        outcome = _hybrid_outcome(hybrid_output)
        monkeypatch.setattr(
            "negpy.services.repair.fauxice_ir_repair.run_hybrid_repair",
            lambda *args, **kwargs: outcome,
        )
        source = _rgb(4, 4, 1000)
        validity = np.ones((4, 4), dtype=np.bool_)
        validity[1, 2] = False

        result = repair_ir_dust(
            source,
            _ir(4, 4, 500),
            same_frame_id="validity-bound",
            config=FauxiceRepairConfig(enabled=True, mode=RepairMode.HYBRID),
            prepass_rgbi=_prepass(4, 4),
            validity_mask=validity,
            hybrid_runtime=_hybrid_runtime(tmp_path),
        )

        expected = hybrid_output.copy()
        expected[1, 2] = source[1, 2]
        np.testing.assert_array_equal(result.repaired_rgb16, expected)
        assert result.native_output_rgb_sha256 == hashlib.sha256(expected.astype("<u2").tobytes()).hexdigest()
        assert result.hybrid_receipt_output_rgb_sha256 == outcome.output_rgb16_sha256

    def test_hybrid_mode_without_runtime_degrades_to_exact_truthfully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.HYBRID)

        result = repair_ir_dust(_rgb(4, 4, 1000), _ir(4, 4, 500), same_frame_id="f1", config=config, prepass_rgbi=_prepass(4, 4))

        assert result.status is RepairStatus.APPLIED
        assert result.mode_requested is RepairMode.HYBRID
        assert result.mode_resolved is RepairMode.EXACT
        assert "no hybrid runtime is configured" in result.reason
        assert "degraded to exact" in result.reason

    def test_hybrid_run_failure_degrades_to_exact_never_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        _install_stub_hybrid(monkeypatch)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.HYBRID)

        def broken_runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="model runtime missing")

        result = repair_ir_dust(
            _rgb(4, 4, 1000),
            _ir(4, 4, 500),
            same_frame_id="f1",
            config=config,
            prepass_rgbi=_prepass(4, 4),
            hybrid_runtime=_hybrid_runtime(tmp_path),
            hybrid_subprocess_runner=broken_runner,
        )

        assert result.status is RepairStatus.APPLIED
        assert result.mode_resolved is RepairMode.EXACT
        assert "fauxce-hybrid run failed" in result.reason
        assert result.repaired_rgb16 is not None

    def test_hybrid_mode_cancelled_before_start_skips_entirely(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        _install_stub_hybrid(monkeypatch)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.HYBRID)
        cancel = threading.Event()
        cancel.set()

        def unreachable_runner(argv, **kwargs):
            raise AssertionError("hybrid subprocess must not run once cancelled")

        # The top-level pre-flight check (shared with exact mode) fires
        # before the mode is even inspected, so a cancellation requested up
        # front skips the whole call, hybrid included, without touching the
        # subprocess runner.
        with pytest.raises(
            FauxiceRepairCancelled,
            match="cancelled by caller before repair started",
        ):
            repair_ir_dust(
                _rgb(4, 4, 1000),
                _ir(4, 4, 500),
                same_frame_id="f1",
                config=config,
                prepass_rgbi=_prepass(4, 4),
                hybrid_runtime=_hybrid_runtime(tmp_path),
                hybrid_subprocess_runner=unreachable_runner,
                cancel=cancel,
            )

    def test_hybrid_mode_cancelled_between_preflight_and_hybrid_start(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Covers the hybrid-specific cancel check, distinct from the top-level one.

        The top-level pre-flight check and the engine's own cooperative
        cancellation both read ``cancel.is_set()``; a stateful fake flips
        from unset to set between calls to simulate another thread calling
        ``cancel.set()`` in that narrow window, the scenario the extra
        hybrid-branch check exists for.
        """
        _install_stub_engine(monkeypatch)
        _install_stub_hybrid(monkeypatch)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.HYBRID)
        # Call 1: the top-level pre-flight check (unset). Call 2: the
        # hybrid-branch check (set). Call 3: the exact fallback's
        # cooperative check inside the stub engine (stays set).
        cancel = _FlippingCancelEvent(set_after_calls=1)

        def unreachable_runner(argv, **kwargs):
            raise AssertionError("hybrid subprocess must not run once cancelled")

        with pytest.raises(
            FauxiceRepairCancelled,
            match="cancelled before the hybrid run started",
        ):
            repair_ir_dust(
                _rgb(4, 4, 1000),
                _ir(4, 4, 500),
                same_frame_id="f1",
                config=config,
                prepass_rgbi=_prepass(4, 4),
                hybrid_runtime=_hybrid_runtime(tmp_path),
                hybrid_subprocess_runner=unreachable_runner,
                cancel=cancel,
            )


# ---------------------------------------------------------------------------
# repair_frame_files: tmp-file invocation, sidecars, IR immutability
# ---------------------------------------------------------------------------


class TestRepairFrameFiles:
    def _write_frame(self, tmp_path: Path, *, rgb_base: int = 1000, ir_base: int = 500) -> tuple[Path, Path]:
        rgb_path = tmp_path / "frame001.tif"
        ir_path = tmp_path / "frame001_IR.tif"
        tifffile.imwrite(rgb_path, _rgb(4, 4, rgb_base), photometric="rgb")
        tifffile.imwrite(ir_path, _ir(4, 4, ir_base), photometric="minisblack")
        return rgb_path, ir_path

    def test_writes_repaired_output_and_sidecar(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        rgb_path, ir_path = self._write_frame(tmp_path)
        prepass_path = tmp_path / "frame001_prepass.npy"
        np.save(prepass_path, _prepass(4, 4), allow_pickle=False)
        config = FauxiceRepairConfig(enabled=True)

        result = repair_frame_files(rgb_path, config=config, prepass_path=prepass_path)

        assert result.status is RepairStatus.APPLIED
        output_path = tmp_path / "frame001_FAUXICE.tif"
        sidecar_path = tmp_path / "frame001_FAUXICE.json"
        assert output_path.is_file()
        assert sidecar_path.is_file()

        written = np.asarray(tifffile.imread(output_path))
        np.testing.assert_array_equal(written, result.repaired_rgb16)

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["status"] == "applied"
        assert payload["engine"]["package"] == ENGINE_IMPORT_NAME
        assert payload["source"] == {"rgb": rgb_path.name, "ir": ir_path.name}
        assert payload["output"]["repaired_rgb"] == output_path.name

    def test_missing_ir_companion_skips_without_writing_repaired_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        rgb_path = tmp_path / "frame002.tif"
        tifffile.imwrite(rgb_path, _rgb(4, 4, 1000), photometric="rgb")
        config = FauxiceRepairConfig(enabled=True)

        result = repair_frame_files(rgb_path, config=config)

        assert result.status is RepairStatus.SKIPPED
        assert "no IR companion" in result.reason
        assert not (tmp_path / "frame002_FAUXICE.tif").exists()
        assert (tmp_path / "frame002_FAUXICE.json").is_file()

    def test_disabled_config_writes_sidecar_only(self, tmp_path: Path) -> None:
        rgb_path, _ = self._write_frame(tmp_path)
        config = FauxiceRepairConfig(enabled=False)

        result = repair_frame_files(rgb_path, config=config)

        assert result.status is RepairStatus.SKIPPED
        sidecar_path = tmp_path / "frame001_FAUXICE.json"
        assert sidecar_path.is_file()
        assert not (tmp_path / "frame001_FAUXICE.tif").exists()

    def test_ir_file_untouched_on_disk_after_repair(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        rgb_path, ir_path = self._write_frame(tmp_path)
        prepass_path = tmp_path / "frame001_prepass.npy"
        np.save(prepass_path, _prepass(4, 4), allow_pickle=False)
        before = ir_path.read_bytes()
        config = FauxiceRepairConfig(enabled=True)

        result = repair_frame_files(rgb_path, config=config, prepass_path=prepass_path)

        assert result.status is RepairStatus.APPLIED
        after = ir_path.read_bytes()
        assert before == after

    def test_ir_array_not_mutated_in_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        config = FauxiceRepairConfig(enabled=True)
        ir = _ir(4, 4, 500)
        ir_before = ir.copy()

        result = repair_ir_dust(_rgb(4, 4, 1000), ir, same_frame_id="f1", config=config, prepass_rgbi=_prepass(4, 4))

        assert result.status is RepairStatus.APPLIED
        np.testing.assert_array_equal(ir, ir_before)

    def test_provenance_sidecar_written_even_when_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "negpy.services.repair.fauxice_ir_repair.importlib.util.find_spec",
            lambda _name: None,
        )
        rgb_path, _ = self._write_frame(tmp_path)
        config = FauxiceRepairConfig(enabled=True)

        result = repair_frame_files(rgb_path, config=config)

        assert result.status is RepairStatus.UNAVAILABLE
        sidecar_path = tmp_path / "frame001_FAUXICE.json"
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["status"] == "unavailable"
        assert "output" not in payload

    def test_hybrid_provenance_sidecar_records_disclosure_mask_and_routing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_stub_engine(monkeypatch)
        _install_stub_hybrid(monkeypatch)
        rgb_path, _ = self._write_frame(tmp_path)
        prepass_path = tmp_path / "frame001_prepass.npy"
        np.save(prepass_path, _prepass(4, 4), allow_pickle=False)
        config = FauxiceRepairConfig(enabled=True, mode=RepairMode.HYBRID)
        hybrid_output = np.full((4, 4, 3), 9999, dtype=np.uint16)
        monkeypatch.setattr(
            "negpy.services.repair.fauxice_ir_repair.run_hybrid_repair",
            lambda *args, **kwargs: _hybrid_outcome(hybrid_output),
        )

        result = repair_frame_files(
            rgb_path,
            config=config,
            prepass_path=prepass_path,
            hybrid_runtime=_hybrid_runtime(tmp_path),
            hybrid_subprocess_runner=_stub_hybrid_runner(hybrid_output),
        )

        assert result.status is RepairStatus.APPLIED
        assert result.mode_resolved is RepairMode.HYBRID
        mask_path = tmp_path / "frame001_FAUXICE_SYNTH.png"
        assert mask_path.is_file()
        assert mask_path.read_bytes() == b"mask-bytes"

        sidecar_path = tmp_path / "frame001_FAUXICE.json"
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["hybrid"]["routing_counts"]["final_regions"] == 2
        assert payload["output"]["disclosure_mask"] == mask_path.name

    def test_no_overwrite_of_existing_ir_when_ir_path_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller-supplied ir_path is also never opened for writing."""
        _install_stub_engine(monkeypatch)
        rgb_path = tmp_path / "custom.tif"
        ir_path = tmp_path / "custom_infrared.tif"
        tifffile.imwrite(rgb_path, _rgb(4, 4, 1000), photometric="rgb")
        tifffile.imwrite(ir_path, _ir(4, 4, 500), photometric="minisblack")
        prepass_path = tmp_path / "prepass.npy"
        np.save(prepass_path, _prepass(4, 4), allow_pickle=False)
        before = ir_path.read_bytes()
        config = FauxiceRepairConfig(enabled=True)

        result = repair_frame_files(rgb_path, config=config, ir_path=ir_path, prepass_path=prepass_path)

        assert result.status is RepairStatus.APPLIED
        assert ir_path.read_bytes() == before
