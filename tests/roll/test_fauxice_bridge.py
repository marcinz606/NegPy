from __future__ import annotations

import builtins
import hashlib
import io
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from negpy.infrastructure.roll import fauxice_bridge
from negpy.infrastructure.roll import repair as roll_repair
from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig
from negpy.services.repair.fauxice_ir_repair import (
    FauxiceRepairResult,
    RepairMode as FauxiceMode,
    RepairStatus,
)


def test_internal_repair_import_failure_is_not_silently_downgraded(
    monkeypatch,
) -> None:
    """Only the external engine may be optional; broken NegPy code is fatal."""

    target = "negpy.services.repair.fauxice_ir_repair"
    real_import = builtins.__import__

    def fail_internal_import(name, *args, **kwargs):
        if name == target:
            raise ImportError("sentinel internal packaging failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_internal_import)
    path = Path(fauxice_bridge.__file__)
    spec = importlib.util.spec_from_file_location(
        "negpy.infrastructure.roll._fauxice_bridge_import_probe",
        path,
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError, match="sentinel internal packaging failure"):
        spec.loader.exec_module(probe)


def _png(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(stream, format="PNG")
    return stream.getvalue()


def _acquisition() -> roll_repair.RepairAcquisition:
    main = np.arange(4 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    prepass = np.arange(2 * 2 * 4, dtype=np.uint16).reshape(2, 2, 4)
    validity = np.ones(main.shape[:2], dtype=np.bool_)
    validity[0, -1] = False
    return roll_repair.RepairAcquisition.from_arrays(
        acquisition_id="dice-" + "1" * 64,
        slot=9,
        reservation_id="reservation-009",
        capture_attempt_id="fine-slot-9-attempt-001",
        storage_transform=roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
        evidence_sha256="2" * 64,
        main_rgbi=main,
        prepass_rgbi=prepass,
        ir_validity=validity,
    )


def _runtime(tmp_path: Path) -> HybridRuntimeConfig:
    return HybridRuntimeConfig(
        hybrid_python=tmp_path / "hybrid" / "bin" / "python",
        executable=tmp_path / "hybrid" / "bin" / "fauxce-hybrid",
        core_source_manifest_sha256="1" * 64,
        hybrid_source_manifest_sha256="2" * 64,
        iopaint_python=tmp_path / "python",
        iopaint_executable=tmp_path / "iopaint",
        iopaint_source_manifest_sha256="3" * 64,
        model_dir=tmp_path / "models",
        model_weights=tmp_path / "models" / "big-lama.pt",
        model_weights_sha256="4" * 64,
    )


def test_bridge_runs_hybrid_in_scanner_native_orientation_and_rotates_once(monkeypatch, tmp_path: Path) -> None:
    acquisition = _acquisition()
    runtime = _runtime(tmp_path)
    native_repaired = np.ascontiguousarray(
        np.where(
            acquisition.ir_validity[..., None],
            acquisition.main_rgbi[..., :3] + 19,
            acquisition.main_rgbi[..., :3],
        )
    )
    native_mask = np.zeros(acquisition.main_rgbi.shape[:2], dtype=np.bool_)
    native_mask[1, 2] = True
    native_mask[0, -1] = True  # routed, but invalid IR means not applied
    applied_mask = np.ascontiguousarray(native_mask & acquisition.ir_validity)
    native_mask_png = _png(native_mask)
    hybrid_receipt = b'{"schema":"fauxce-hybrid-receipt-v2"}'
    calls = []

    def fake_repair(rgb, ir, **kwargs):
        calls.append((rgb, ir, kwargs))
        return FauxiceRepairResult(
            status=RepairStatus.APPLIED,
            reason="hybrid applied",
            mode_requested=FauxiceMode.HYBRID,
            mode_resolved=FauxiceMode.HYBRID,
            repaired_rgb16=native_repaired,
            engine_version="0.3.0",
            backend_requested="auto",
            backend_used="cpu-fast",
            backend_selection_reason="parity self-test passed",
            hybrid_mask_png=native_mask_png,
            hybrid_mask_sha256=hashlib.sha256(native_mask_png).hexdigest(),
            hybrid_mask=native_mask,
            hybrid_synthesis_fraction=2 / native_mask.size,
            hybrid_routing_counts={
                "final_regions": 1,
                "synthesis_pixels": 2,
                "frame_pixels": native_mask.size,
                "at_floor_pixels": 2,
            },
            hybrid_receipt=hybrid_receipt,
            hybrid_receipt_sha256=hashlib.sha256(hybrid_receipt).hexdigest(),
            hybrid_provenance_class="caller_asserted_bare_npy",
            native_output_rgb_sha256=hashlib.sha256(native_repaired.astype("<u2").tobytes()).hexdigest(),
        )

    monkeypatch.setattr(fauxice_bridge, "repair_ir_dust", fake_repair)

    result = fauxice_bridge._FauxiceEngine().repair(
        acquisition,
        roll_repair.RepairMode.HYBRID,
        hybrid_runtime=runtime,
    )

    assert len(calls) == 1
    rgb, ir, kwargs = calls[0]
    np.testing.assert_array_equal(rgb, acquisition.main_rgbi[..., :3])
    np.testing.assert_array_equal(ir, acquisition.main_rgbi[..., 3])
    np.testing.assert_array_equal(kwargs["prepass_rgbi"], acquisition.prepass_rgbi)
    np.testing.assert_array_equal(kwargs["validity_mask"], acquisition.ir_validity)
    assert kwargs["same_frame_id"] == acquisition.acquisition_id
    assert kwargs["hybrid_runtime"] is runtime
    np.testing.assert_array_equal(result.rgb, acquisition.storage_rgb(native_repaired))
    assert result.mode_requested is roll_repair.RepairMode.HYBRID
    assert result.mode_resolved is roll_repair.RepairMode.HYBRID
    assert result.degraded is False
    assert result.routed_native_synthesis_mask_png == native_mask_png
    assert result.routed_native_synthesis_mask_sha256 == hashlib.sha256(
        native_mask_png
    ).hexdigest()
    with Image.open(io.BytesIO(result.native_synthesis_mask_png)) as image:
        final_native_mask = np.asarray(image.convert("L")) != 0
    np.testing.assert_array_equal(final_native_mask, applied_mask)
    assert result.synthesis_fraction == 1 / native_mask.size
    assert result.storage_synthesis_mask_shape == (3, 4)
    assert result.synthesis_mask_transform == roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM
    with Image.open(io.BytesIO(result.storage_synthesis_mask_png)) as image:
        storage_mask = np.asarray(image.convert("L")) != 0
    np.testing.assert_array_equal(
        storage_mask,
        acquisition.storage_mask(applied_mask),
    )
    assert result.hybrid_receipt == hybrid_receipt


def test_bridge_reports_requested_hybrid_as_actual_exact_without_runtime(
    monkeypatch,
) -> None:
    acquisition = _acquisition()
    native_repaired = np.ascontiguousarray(acquisition.main_rgbi[..., :3] + 7)

    def fake_repair(rgb, ir, **kwargs):
        assert kwargs["hybrid_runtime"] is None
        return FauxiceRepairResult(
            status=RepairStatus.APPLIED,
            reason=("hybrid mode requested but no hybrid runtime is configured; degraded to exact repair"),
            mode_requested=FauxiceMode.HYBRID,
            mode_resolved=FauxiceMode.EXACT,
            repaired_rgb16=native_repaired,
            engine_version="0.3.0",
            backend_requested="auto",
            backend_used="cpu-fast",
            backend_selection_reason="parity self-test passed",
            native_output_rgb_sha256=hashlib.sha256(native_repaired.astype("<u2").tobytes()).hexdigest(),
        )

    monkeypatch.setattr(fauxice_bridge, "repair_ir_dust", fake_repair)

    result = fauxice_bridge._FauxiceEngine().repair(
        acquisition,
        roll_repair.RepairMode.HYBRID,
        hybrid_runtime=None,
    )

    assert result.mode_requested is roll_repair.RepairMode.HYBRID
    assert result.mode_resolved is roll_repair.RepairMode.EXACT
    assert result.degraded is True
    assert "no hybrid runtime" in result.reason
    assert result.storage_synthesis_mask_png is None
    assert result.hybrid_receipt is None
