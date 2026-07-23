from __future__ import annotations

import hashlib
import json
import os
import fcntl
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coolscanpy.protocol.ls5000_single_pass.density import (
    NikonDensityFrameOwnershipReceipt,
)
from coolscanpy.types import (
    ArtifactEvidence,
    ClippingTelemetry,
    ExposureVector,
    FocusDetailTelemetry,
    Receipt,
    TransportSmearAssessment,
)

from negpy.infrastructure.roll import repair as roll_repair
from negpy.services.roll import deep_acceptance
from negpy.services.roll import service as roll_service
from negpy.services.roll.service import RollFrameOutput, RollScanningService
from tests.roll.test_native_builder import (
    FRESH_FINGERPRINT_SHA256,
    RESERVATION_ID,
    REVIEWED_FINGERPRINT_SHA256,
    _PayloadReceipt,
    _density_document,
    _ownership_document,
    _synthetic_native_evidence,
)
from tests.roll.test_service import _valid_hybrid_result


class _Runtime:
    core_source_manifest_sha256 = "1" * 64
    hybrid_source_manifest_sha256 = "2" * 64
    iopaint_source_manifest_sha256 = "3" * 64
    model_weights_sha256 = "4" * 64
    inpaint_device = "cpu"
    inpaint_threads = 1
    inpaint_seed = 0

    def __init__(self) -> None:
        self.validated = False

    def validate_files(self) -> None:
        self.validated = True


def _runtime_bound_hybrid_result(
    acquisition: roll_repair.RepairAcquisition,
    runtime: _Runtime,
) -> roll_repair.RepairResult:
    base = _valid_hybrid_result(acquisition)
    document = json.loads(base.hybrid_receipt)
    assertion = {
        "assertions": {
            "focus_exposure_locked": True,
            "same_frame_id": acquisition.acquisition_id,
        },
        "inputs": {
            "main": {"raw_sha256": acquisition.main_rgbi_sha256},
            "prepass": {"raw_sha256": acquisition.prepass_rgbi_sha256},
        },
        "provenance_class": "caller_asserted_bare_npy",
        "schema": "negpy.fauxce-hybrid-acquisition-assertion-v1",
    }
    document.update(
        core={
            "backend": {
                "reason": base.backend_selection_reason,
                "requested": base.backend_requested,
                "used": base.backend_used,
            },
            "source_manifest_sha256": runtime.core_source_manifest_sha256,
            "version": base.engine_version,
        },
        generation={"hybrid_source_manifest_sha256": (runtime.hybrid_source_manifest_sha256)},
        inpainting={"invoked": False},
        inputs={
            "geometry": {
                "mask_shape": list(acquisition.main_rgbi.shape[:2]),
                "output_shape": [*acquisition.main_rgbi.shape[:2], 3],
            },
            "main": {
                "canonical_encoding": "uint16_little_endian_c_order",
                "raw_sha256": acquisition.main_rgbi_sha256,
                "shape": list(acquisition.main_rgbi.shape),
            },
            "prepass": {
                "canonical_encoding": "uint16_little_endian_c_order",
                "raw_sha256": acquisition.prepass_rgbi_sha256,
                "shape": list(acquisition.prepass_rgbi.shape),
            },
            "provenance": {
                "basis": "caller_asserted",
                "source_manifest_sha256": hashlib.sha256(deep_acceptance._canonical_json(assertion)).hexdigest(),
            },
        },
    )
    payload = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    return replace(
        base,
        hybrid_receipt=payload,
        hybrid_receipt_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _production_frame() -> tuple[SimpleNamespace, roll_repair.RepairAcquisition]:
    slot = 5
    storage_rgb = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    storage_rgb += np.uint16(10_000)
    storage_ir = np.arange(4 * 6, dtype=np.uint16).reshape(4, 6)
    storage_ir += np.uint16(2_000)
    storage_validity = np.ones((4, 6), dtype=np.bool_)
    meter = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    native_rgbi = np.ascontiguousarray(
        np.rot90(
            np.dstack((storage_rgb, storage_ir)),
            k=-1,
            axes=(0, 1),
        )
    )
    native_validity = np.ascontiguousarray(np.rot90(storage_validity, k=-1, axes=(0, 1)))
    acquisition_id, evidence_sha256 = roll_service._derive_digital_ice_producer_binding(
        slot=slot,
        reservation_id=RESERVATION_ID,
        capture_attempt_id="capture-d",
        main_rgbi=native_rgbi,
        prepass_rgbi=meter,
        ir_validity=native_validity,
    )
    acquisition = roll_repair.RepairAcquisition.from_arrays(
        acquisition_id=acquisition_id,
        slot=slot,
        reservation_id=RESERVATION_ID,
        capture_attempt_id="capture-d",
        storage_transform=roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
        evidence_sha256=evidence_sha256,
        main_rgbi=native_rgbi,
        prepass_rgbi=meter,
        ir_validity=native_validity,
    )

    def artifact(array: np.ndarray) -> ArtifactEvidence:
        contiguous = np.ascontiguousarray(array)
        return ArtifactEvidence(
            sha256=hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
            byte_length=contiguous.nbytes,
            shape=contiguous.shape,
            dtype=str(contiguous.dtype),
        )

    ownership_document = _ownership_document()
    ownership = NikonDensityFrameOwnershipReceipt.from_dict(ownership_document)
    receipt = Receipt(
        version=1,
        slot=slot,
        spacing_offset=0,
        dpi=4_000,
        depth=16,
        device_id="usb:2:7",
        device_model="Nikon LS-5000 ED 1.03",
        reviewed_fingerprint_sha256=REVIEWED_FINGERPRINT_SHA256,
        fresh_fingerprint_sha256=FRESH_FINGERPRINT_SHA256,
        manual_approval=None,
        exposure=ExposureVector(1, 1.0, 1.0, 1.0, 1.0),
        split_alignment=None,
        clipping=ClippingTelemetry((0.0, 0.0, 0.0), 65_535.0, 0.01, False),
        focus_detail=FocusDetailTelemetry("laplacian", "measured", 1.0, 1.0),
        transport_smear=TransportSmearAssessment(
            "clean",
            None,
            0,
            0,
            None,
            None,
            None,
            None,
            "clean",
        ),
        artifacts={"rgb": artifact(storage_rgb), "ir": artifact(storage_ir)},
        nikon_density_ownership=ownership,
    )
    frame = SimpleNamespace(
        slot=slot,
        rgb=storage_rgb,
        ir=storage_ir,
        ir_validity=storage_validity,
        meter_rgbi=meter,
        receipt=receipt,
        nikon_density_evidence=_PayloadReceipt(_density_document()),
        nikon_density_ownership=_PayloadReceipt(ownership_document),
        nikon_exact_builder_evidence=_synthetic_native_evidence(),
        prepare_digital_ice=lambda: acquisition,
    )
    return frame, acquisition


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _output(root: Path, slot: int) -> RollFrameOutput:
    base = root / f"acceptance_slot{slot:02d}"
    return RollFrameOutput(
        slot=slot,
        rgb_path=str(base.with_suffix(".tif")),
        ir_path=str(base.with_name(base.name + "_IR.tif")),
        repaired_rgb_path=str(base.with_name(base.name + "_repaired.tif")),
        repaired_ir_path=str(base.with_name(base.name + "_repaired_IR.tif")),
        positive_path=str(base.with_name(base.name + "_positive.tif")),
        receipt_path=str(base.with_name(base.name + "_receipt.json")),
        synthesis_mask_path=str(base.with_name(base.name + "_repaired_SYNTH.png")),
        native_synthesis_mask_path=str(root / ".negpy-dice-hybrid" / f"slot-{slot}" / "native.png"),
        hybrid_receipt_path=str(root / ".negpy-dice-hybrid" / f"slot-{slot}" / "hybrid-receipt.json"),
    )


def _completed_receipt_fixture(
    root: Path,
    *,
    slot: int = 1,
) -> tuple[RollFrameOutput, dict[str, object]]:
    output = _output(root, slot)

    def touch(path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"slot-{slot}:{path.name}".encode())
        return str(path)

    for value in (
        output.rgb_path,
        output.ir_path,
        output.repaired_rgb_path,
        output.repaired_ir_path,
        output.positive_path,
        output.synthesis_mask_path,
        output.native_synthesis_mask_path,
        output.hybrid_receipt_path,
    ):
        touch(Path(value))

    dice_dir = root / ".negpy-dice-acquisition" / f"slot-{slot}"
    dice_binding = dice_dir / "acquisition-binding.json"
    dice_prepass = dice_dir / "prepass.rgbi16.npy"
    dice_validity = dice_dir / "ir-validity.npy"
    for path in (dice_binding, dice_prepass, dice_validity):
        touch(path)

    hybrid_dir = Path(output.hybrid_receipt_path).parent
    routed_mask = hybrid_dir / "routed.png"
    hybrid_binding = hybrid_dir / "binding.json"
    for path in (routed_mask, hybrid_binding):
        touch(path)

    native_dir = root / ".negpy-native-builder" / f"slot-{slot}"
    native_paths = {
        "builder_receipt": native_dir / "native-builder-receipt.json",
        "analyzer_rgb": native_dir / "analyzer-rgb-u16le.bin",
        "evidence_receipt": native_dir / "native-builder-evidence.json",
        "frame_ownership_receipt": (native_dir / "nikon-density-frame-ownership.json"),
        "density_evidence_receipt": native_dir / "nikon-density-evidence.json",
    }
    lut_paths = [native_dir / f"builder-preF-{channel}.bin" for channel in "rgb"]
    for path in (*native_paths.values(), *lut_paths):
        touch(path)

    retained = {
        **{key: {"path": str(path)} for key, path in native_paths.items()},
        "native_per_acquisition_builder": True,
        "pre_f_luts": [{"path": str(path)} for path in lut_paths],
    }
    document: dict[str, object] = {
        "artifacts": {"ir": {}, "rgb": {}},
        "depth": 16,
        "device_id": "usb:2:7",
        "device_model": "Nikon LS-5000 ED 1.03",
        "dpi": 4_000,
        "outputs": {
            "native_color_evidence": {
                "native_per_acquisition_builder": True,
                "retained": True,
                "retained_builder_evidence": retained,
            },
            "positive": {
                "builder_validated": True,
                "cms_verified": True,
                "color_mode": "nikon-exact",
                "exact_nikon_color": True,
                "native_per_acquisition_builder": True,
                "retained_builder_evidence": retained,
                "rgb_path": output.positive_path,
                "written": True,
            },
            "repair_acquisition_evidence": {
                "artifacts": {
                    "ir_validity": {"path": str(dice_validity)},
                    "prepass_rgbi": {"path": str(dice_prepass)},
                },
                "binding": {"path": str(dice_binding)},
                "replayable": True,
                "retained": True,
                "sources": {
                    "storage_ir_tiff": {"path": output.ir_path},
                    "storage_rgb_tiff": {"path": output.rgb_path},
                },
            },
            "repaired": {
                "degraded": False,
                "disclosure_mask": {
                    "applied_final": {
                        "native": {"path": output.native_synthesis_mask_path},
                        "storage": {"path": output.synthesis_mask_path},
                    },
                    "routed_raw": {"native": {"path": str(routed_mask)}},
                },
                "hybrid_evidence_binding": {"path": str(hybrid_binding)},
                "hybrid_receipt": {"path": output.hybrid_receipt_path},
                "ir_path": output.repaired_ir_path,
                "mode_requested": "hybrid",
                "mode_resolved": "hybrid",
                "rgb_path": output.repaired_rgb_path,
                "written": True,
            },
            "unrepaired": {
                "ir_path": output.ir_path,
                "rgb_path": output.rgb_path,
                "written": True,
            },
        },
        "slot": slot,
        "version": 1,
    }
    receipt_path = Path(output.receipt_path)
    receipt_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    lock_dir = root / ".negpy-locks"
    lock_dir.mkdir()
    lock_path = lock_dir / (hashlib.sha256(str(receipt_path).encode()).hexdigest() + ".lock")
    lock_path.write_bytes(b"")
    return output, document


def _approval(slot: int, reviewed: str) -> dict[str, object]:
    return {
        "reviewed_fingerprint_sha256": reviewed,
        "slot": slot,
        "spacing_offset": 0,
        "thumbnail_sha256": _digest(f"thumbnail-{slot}"),
        "reviewed_lookup_row": (slot - 1) * 143,
        "reviewed_native_origin": (slot - 1) * 5_959,
        "review_reasons": ["strip boundary requires review"],
    }


def _fake_audits(root: Path) -> tuple[list[RollFrameOutput], list[deep_acceptance._FrameAudit]]:
    reservation = "reservation-six-frame"
    preview = _digest("preview")
    preview_identity = _digest("preview-identity")
    transport_table = _digest("transport-table")
    reviewed = _digest("reviewed")
    fresh = _digest("fresh")
    transport_material = {
        "reservation_id": reservation,
        "batch_session_id": reservation,
        "preview_sha256": preview,
        "preview_identity_sha256": preview_identity,
        "transport_table_sha256": transport_table,
        "reviewed_fingerprint_sha256": reviewed,
        "fresh_fingerprint_sha256": fresh,
        "selected_slots": list(deep_acceptance.SLOTS),
    }
    transport_identity = hashlib.sha256(
        json.dumps(
            transport_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    outputs: list[RollFrameOutput] = []
    audits: list[deep_acceptance._FrameAudit] = []
    for slot in deep_acceptance.SLOTS:
        output = _output(root, slot)
        outputs.append(output)
        visible = {
            Path(output.rgb_path),
            Path(output.ir_path),
            Path(output.repaired_rgb_path),
            Path(output.repaired_ir_path),
            Path(output.positive_path),
            Path(output.receipt_path),
            Path(output.synthesis_mask_path),
        }
        for path in visible:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"slot-{slot}:{path.name}".encode())
        ownership = SimpleNamespace(
            reservation_id=reservation,
            batch_session_id=reservation,
            preview_sha256=preview,
            preview_identity_sha256=preview_identity,
            transport_table_sha256=transport_table,
            transport_identity_sha256=transport_identity,
            reviewed_fingerprint_sha256=reviewed,
            fresh_fingerprint_sha256=fresh,
            frame_capture_attempt_id=f"attempt-{slot}",
            frame_index=slot,
            frame_total=6,
            selected_slots=deep_acceptance.SLOTS,
            selected_slot=slot,
        )
        receipt = {
            "device_id": "usb:2:7",
            "device_model": "Nikon LS-5000 ED 1.03",
            "spacing_offset": 0,
            "manual_approval": (_approval(slot, reviewed) if slot in deep_acceptance.APPROVED_SLOTS else None),
        }
        receipt_path = Path(output.receipt_path)
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        lock_dir = root / ".negpy-locks"
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_dir / (hashlib.sha256(str(receipt_path).encode()).hexdigest() + ".lock")
        lock_path.write_bytes(b"")
        visible.add(lock_path)
        audits.append(
            deep_acceptance._FrameAudit(
                summary={
                    "slot": slot,
                    "frame_receipt": {
                        "path": str(receipt_path),
                        "bytes": receipt_path.stat().st_size,
                        "sha256": receipt_sha,
                    },
                    "referenced_file_count": len(visible),
                    "referenced_files": sorted(str(path) for path in visible),
                },
                referenced_files=frozenset(visible),
                receipt_path=receipt_path,
                receipt=receipt,
                ownership=ownership,
                builder_receipt=SimpleNamespace(density_evidence_receipt_sha256=_digest("density")),
                output_artifacts={field: getattr(output, field) for field in deep_acceptance._OUTPUT_FIELDS},
            )
        )
    return outputs, audits


def test_six_frame_batch_closes_cross_frame_and_exact_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs, audits = _fake_audits(tmp_path)
    runtime = _Runtime()
    iterator = iter(audits)

    def validate_locked_frame(*args: object, **kwargs: object):
        del args, kwargs
        for audit in audits:
            lock_path = deep_acceptance._receipt_lock_path(audit.receipt_path)
            descriptor = os.open(lock_path, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
        return next(iterator)

    monkeypatch.setattr(
        deep_acceptance,
        "_validate_frame",
        validate_locked_frame,
    )
    audit_lock = tmp_path / "deep-audit.lock"
    audit_lock.write_bytes(b"")

    result = deep_acceptance.validate_six_frame_batch(
        outputs,
        output_dir=tmp_path,
        allowed_output_lock_name=audit_lock.name,
        builder=object(),
        evaluator=object(),
        hybrid_runtime=runtime,
    )

    assert result["status"] == "passed"
    assert result["slots"] == [1, 2, 3, 4, 5, 6]
    assert result["approved_slots"] == [1, 6]
    assert result["device_id"] == "usb:2:7"
    assert result["device_model"] == "Nikon LS-5000 ED 1.03"
    assert result["reviewed_fingerprint_sha256"] == audits[0].ownership.reviewed_fingerprint_sha256
    assert result["manual_approval_bindings"] == [
        {
            "slot": slot,
            "binding_sha256": deep_acceptance._validate_manual_approval(
                audits[slot - 1],
                expected=True,
            )["binding_sha256"],
        }
        for slot in sorted(deep_acceptance.APPROVED_SLOTS)
    ]
    assert result["inventory"]["visible_file_count"] == 42
    assert result["inventory"]["exact"] is True
    assert result["inventory"]["allowed_output_lock_path"] == str(audit_lock)
    assert len(result["referenced_files"]) == 48
    assert str(audit_lock) not in result["referenced_files"]
    assert all(Path(path).is_absolute() for path in result["referenced_files"])
    assert runtime.validated is True
    json.dumps(result, allow_nan=False)


def test_completed_frame_reports_deterministic_absolute_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs, audits = _fake_audits(tmp_path)
    audit = audits[0]
    monkeypatch.setattr(
        deep_acceptance,
        "_validate_frame",
        lambda *args, **kwargs: audit,
    )

    result = deep_acceptance.validate_completed_frame(
        outputs[0],
        output_dir=tmp_path,
        builder=object(),
        evaluator=object(),
        hybrid_runtime=_Runtime(),
    )

    assert result["referenced_files"] == sorted(result["referenced_files"])
    assert result["referenced_file_count"] == len(result["referenced_files"])
    assert all(Path(path).is_absolute() for path in result["referenced_files"])
    assert any("/.negpy-locks/" in path for path in result["referenced_files"])


def test_manual_approval_rejects_string_review_reasons(tmp_path: Path) -> None:
    _, audits = _fake_audits(tmp_path)
    approval = audits[0].receipt["manual_approval"]
    assert isinstance(approval, dict)
    approval["review_reasons"] = "not-a-json-list"

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="approval reasons are malformed",
    ):
        deep_acceptance._validate_manual_approval(audits[0], expected=True)


def test_collect_completed_frame_files_returns_exact_cheap_inventory(
    tmp_path: Path,
) -> None:
    output, _ = _completed_receipt_fixture(tmp_path)

    files = deep_acceptance.collect_completed_frame_files(
        output,
        output_dir=tmp_path,
        expected_slot=1,
    )

    assert files == sorted(files)
    assert len(files) == 23
    assert len(set(files)) == 23
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in files)
    assert output.receipt_path in files
    assert any("/.negpy-locks/" in path for path in files)


def test_collect_completed_frame_files_rejects_extra_receipt_path(
    tmp_path: Path,
) -> None:
    output, document = _completed_receipt_fixture(tmp_path)
    extra = tmp_path / "extra.bin"
    extra.write_bytes(b"extra")
    outputs = document["outputs"]
    assert isinstance(outputs, dict)
    positive = outputs["positive"]
    assert isinstance(positive, dict)
    positive["unexpected"] = {"path": str(extra)}
    Path(output.receipt_path).write_text(
        json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="receipt path inventory changed",
    ):
        deep_acceptance.collect_completed_frame_files(
            output,
            output_dir=tmp_path,
            expected_slot=1,
        )


def test_collect_completed_frame_files_rejects_noncanonical_or_mismatched_output(
    tmp_path: Path,
) -> None:
    output, document = _completed_receipt_fixture(tmp_path)
    Path(output.receipt_path).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="canonical frame-receipt encoding",
    ):
        deep_acceptance.collect_completed_frame_files(
            output,
            output_dir=tmp_path,
            expected_slot=1,
        )

    Path(output.receipt_path).write_text(
        json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    wrong_positive = tmp_path / "wrong-positive.tif"
    wrong_positive.write_bytes(b"wrong")
    mismatched = replace(output, positive_path=str(wrong_positive))
    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="positive_path differs from its receipt",
    ):
        deep_acceptance.collect_completed_frame_files(
            mismatched,
            output_dir=tmp_path,
            expected_slot=1,
        )


def test_collect_completed_frame_files_rejects_nonabsolute_required_path(
    tmp_path: Path,
) -> None:
    output, document = _completed_receipt_fixture(tmp_path)
    outputs = document["outputs"]
    assert isinstance(outputs, dict)
    unrepaired = outputs["unrepaired"]
    assert isinstance(unrepaired, dict)
    unrepaired["rgb_path"] = "relative-frame.tif"
    Path(output.receipt_path).write_text(
        json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="absolute artifact path",
    ):
        deep_acceptance.collect_completed_frame_files(
            output,
            output_dir=tmp_path,
            expected_slot=1,
        )


def test_collect_completed_frame_files_refuses_a_busy_production_lock(
    tmp_path: Path,
) -> None:
    output, _ = _completed_receipt_fixture(tmp_path)
    receipt_path = Path(output.receipt_path)
    lock_path = deep_acceptance._receipt_lock_path(receipt_path)
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            deep_acceptance.DeepAcceptanceError,
            match="busy in another process",
        ):
            deep_acceptance.collect_completed_frame_files(
                output,
                output_dir=tmp_path,
                expected_slot=1,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_completed_frame_runs_full_production_replay_end_to_end(
    fake_repair_engine: object,
    tmp_path: Path,
) -> None:
    del fake_repair_engine
    runtime = _Runtime()
    frame, expected_acquisition = _production_frame()

    class Engine:
        def repair(
            self,
            acquisition: roll_repair.RepairAcquisition,
            mode: roll_repair.RepairMode,
            **kwargs: object,
        ) -> roll_repair.RepairResult:
            del kwargs
            assert mode is roll_repair.RepairMode.HYBRID
            assert acquisition.acquisition_id == expected_acquisition.acquisition_id
            return _runtime_bound_hybrid_result(acquisition, runtime)

    roll_repair.register_engine(Engine())
    output = RollScanningService(hybrid_runtime=runtime).write_frame(
        frame,
        str(tmp_path),
        '20260723_{{ "%03d" % seq }}',
        write_unrepaired=True,
        write_repaired=True,
        write_positive=True,
        repair_mode="hybrid",
        positive_mode="nikon-exact",
    )

    cheap_files = deep_acceptance.collect_completed_frame_files(
        output,
        output_dir=tmp_path,
        expected_slot=5,
    )
    result = deep_acceptance.validate_completed_frame(
        output,
        output_dir=tmp_path,
        expected_slot=5,
        hybrid_runtime=runtime,
    )

    assert result["status"] == "passed"
    assert result["slot"] == 5
    assert result["referenced_file_count"] == 23
    assert result["referenced_files"] == cheap_files
    assert result["acquisition_id"] == expected_acquisition.acquisition_id
    assert runtime.validated is True


def test_six_frame_batch_rejects_one_unreferenced_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs, audits = _fake_audits(tmp_path)
    iterator = iter(audits)
    monkeypatch.setattr(
        deep_acceptance,
        "_validate_frame",
        lambda *args, **kwargs: next(iterator),
    )
    (tmp_path / "unexpected.txt").write_text("not part of the batch")

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="inventory file set changed",
    ):
        deep_acceptance.validate_six_frame_batch(
            outputs,
            output_dir=tmp_path,
            builder=object(),
            evaluator=object(),
            hybrid_runtime=_Runtime(),
        )


def test_six_frame_batch_ignores_regular_finder_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs, audits = _fake_audits(tmp_path)
    iterator = iter(audits)
    monkeypatch.setattr(
        deep_acceptance,
        "_validate_frame",
        lambda *args, **kwargs: next(iterator),
    )
    (tmp_path / ".DS_Store").write_bytes(b"finder metadata")

    result = deep_acceptance.validate_six_frame_batch(
        outputs,
        output_dir=tmp_path,
        builder=object(),
        evaluator=object(),
        hybrid_runtime=_Runtime(),
    )

    assert result["status"] == "passed"
    assert result["inventory"]["visible_file_count"] == 42


def test_six_frame_batch_rejects_finder_metadata_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs, audits = _fake_audits(tmp_path)
    iterator = iter(audits)
    monkeypatch.setattr(
        deep_acceptance,
        "_validate_frame",
        lambda *args, **kwargs: next(iterator),
    )
    outside = tmp_path.parent / "outside-metadata"
    outside.write_bytes(b"not metadata")
    (tmp_path / ".DS_Store").symlink_to(outside)

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="file entry is unsafe",
    ):
        deep_acceptance.validate_six_frame_batch(
            outputs,
            output_dir=tmp_path,
            builder=object(),
            evaluator=object(),
            hybrid_runtime=_Runtime(),
        )


def test_six_frame_batch_rejects_cross_frame_fingerprint_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs, audits = _fake_audits(tmp_path)
    audits[3].ownership.fresh_fingerprint_sha256 = _digest("different-fresh")
    iterator = iter(audits)
    monkeypatch.setattr(
        deep_acceptance,
        "_validate_frame",
        lambda *args, **kwargs: next(iterator),
    )

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="disagree on reservation, preview, transport, or fingerprint",
    ):
        deep_acceptance.validate_six_frame_batch(
            outputs,
            output_dir=tmp_path,
            builder=object(),
            evaluator=object(),
            hybrid_runtime=_Runtime(),
        )


def test_six_frame_batch_binds_canonical_live_run_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    outputs, audits = _fake_audits(output_dir)
    iterator = iter(audits)
    monkeypatch.setattr(
        deep_acceptance,
        "_validate_frame",
        lambda *args, **kwargs: next(iterator),
    )
    contact_path = tmp_path / "reviewed-contact-sheet.png"
    contact_path.write_bytes(b"pinned contact sheet")
    contact_sha256 = hashlib.sha256(contact_path.read_bytes()).hexdigest()
    approval_rows = [
        deep_acceptance._validate_manual_approval(
            audits[slot - 1],
            expected=True,
        )
        for slot in sorted(deep_acceptance.APPROVED_SLOTS)
    ]
    review_document = {
        "approvals": approval_rows,
        "contact_sheet": {
            "bytes": contact_path.stat().st_size,
            "path": str(contact_path),
            "sha256": contact_sha256,
        },
        "preview_session": {
            "bytes": 2,
            "path": str(tmp_path / "preview-session.json"),
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "review_basis": ("visual-inspection-of-six-frame-contact-sheet-and-canonical-restored-thumbnails"),
        "reviewed_fingerprint_sha256": (audits[0].ownership.reviewed_fingerprint_sha256),
        "schema": "negpy.ls5000-reviewed-approval.v1",
    }
    review_path = tmp_path / "reviewed-approval.json"
    review_path.write_bytes(
        deep_acceptance._canonical_json(
            review_document,
            newline=True,
            ensure_ascii=True,
        )
    )
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    run_receipt = {
        "approved_slots": [1, 6],
        "close": {
            "iterator": {"attempted": True, "succeeded": True},
            "roll": {"attempted": True, "succeeded": True},
        },
        "deep_acceptance": {"slots": [1, 2, 3, 4, 5, 6], "status": "passed"},
        "device_id": "usb:2:7",
        "eject_requested": False,
        "frames": [
            {
                "artifacts": frame.output_artifacts,
                "expected_slot": frame.summary["slot"],
                "frame_receipt_sha256": frame.summary["frame_receipt"]["sha256"],
                "slot": frame.summary["slot"],
            }
            for frame in audits
        ],
        "operation_state": {
            "batch_exhausted": True,
            "verified_slots": [1, 2, 3, 4, 5, 6],
        },
        "output_dir": str(output_dir),
        "output_lease": {
            "acquired": True,
            "release_attempted": True,
            "released": True,
        },
        "phase": "succeeded",
        "reviewed_approval": {
            "bytes": review_path.stat().st_size,
            "contact_sheet": {
                "path": str(contact_path),
                "sha256": contact_sha256,
            },
            "expected_sha256": review_sha256,
            "path": str(review_path),
            "reviewed_fingerprint_sha256": (audits[0].ownership.reviewed_fingerprint_sha256),
            "verified_sha256": review_sha256,
        },
        "retry_count": 0,
        "schema": "negpy.ls5000-live-acceptance.v2",
        "settings": {
            "filename_pattern": 'acceptance_slot{{ "%02d" % seq }}',
            "positive_mode": "nikon-exact",
            "repair_mode": "hybrid",
            "write_positive": True,
            "write_repaired": True,
            "write_unrepaired": True,
        },
        "slots": [1, 2, 3, 4, 5, 6],
        "status": "succeeded",
    }
    run_path = tmp_path / "run-receipt.json"
    run_path.write_bytes(
        deep_acceptance._canonical_json(
            run_receipt,
            newline=True,
            ensure_ascii=True,
        )
    )

    result = deep_acceptance.validate_six_frame_batch(
        outputs,
        output_dir=output_dir,
        run_receipt_path=run_path,
        builder=object(),
        evaluator=object(),
        hybrid_runtime=_Runtime(),
    )

    assert result["run_receipt"]["path"] == str(run_path)
    assert result["run_receipt"]["sha256"] == hashlib.sha256(run_path.read_bytes()).hexdigest()


def test_artifact_row_rejects_hash_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bound bytes")
    row = {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": "0" * 64,
    }

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="SHA-256 changed",
    ):
        deep_acceptance._artifact_row(
            row,
            root=tmp_path.resolve(),
            label="test artifact",
        )


def test_regular_file_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    artifact = real / "artifact.bin"
    artifact.write_bytes(b"payload")
    alias = tmp_path / "alias"
    os.symlink(real, alias)

    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="symlink or non-directory parent",
    ):
        deep_acceptance._regular_file(
            alias / artifact.name,
            root=tmp_path.resolve(),
            label="test artifact",
        )


def test_runtime_receipt_binding_rejects_retained_input_drift() -> None:
    main = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    prepass = np.arange(16, dtype=np.uint16).reshape(2, 2, 4)

    def raw_digest(array: np.ndarray) -> str:
        canonical = np.array(array, dtype="<u2", order="C", copy=True)
        return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()

    acquisition = SimpleNamespace(
        acquisition_id="dice-" + "a" * 64,
        main_rgbi=main,
        main_rgbi_sha256=raw_digest(main),
        prepass_rgbi=prepass,
        prepass_rgbi_sha256=raw_digest(prepass),
    )
    assertion = {
        "assertions": {
            "focus_exposure_locked": True,
            "same_frame_id": acquisition.acquisition_id,
        },
        "inputs": {
            "main": {"raw_sha256": acquisition.main_rgbi_sha256},
            "prepass": {"raw_sha256": acquisition.prepass_rgbi_sha256},
        },
        "provenance_class": "caller_asserted_bare_npy",
        "schema": "negpy.fauxce-hybrid-acquisition-assertion-v1",
    }
    receipt = {
        "core": {
            "backend": {
                "reason": "pinned",
                "requested": "auto",
                "used": "cpu",
            },
            "source_manifest_sha256": _Runtime.core_source_manifest_sha256,
        },
        "generation": {"hybrid_source_manifest_sha256": (_Runtime.hybrid_source_manifest_sha256)},
        "inpainting": {"invoked": False},
        "inputs": {
            "geometry": {
                "mask_shape": [2, 3],
                "output_shape": [2, 3, 3],
            },
            "main": {
                "canonical_encoding": "uint16_little_endian_c_order",
                "raw_sha256": acquisition.main_rgbi_sha256,
                "shape": [2, 3, 4],
            },
            "prepass": {
                "canonical_encoding": "uint16_little_endian_c_order",
                "raw_sha256": acquisition.prepass_rgbi_sha256,
                "shape": [2, 2, 4],
            },
            "provenance": {
                "basis": "caller_asserted",
                "source_manifest_sha256": hashlib.sha256(deep_acceptance._canonical_json(assertion)).hexdigest(),
            },
        },
    }
    repaired = {
        "backend_requested": "auto",
        "backend_used": "cpu",
        "backend_selection_reason": "pinned",
    }
    deep_acceptance._runtime_receipt_binding(
        receipt,
        acquisition=acquisition,
        repaired=repaired,
        runtime=_Runtime(),
        label="test",
    )

    receipt["inputs"]["main"]["raw_sha256"] = "0" * 64
    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="main input does not bind",
    ):
        deep_acceptance._runtime_receipt_binding(
            receipt,
            acquisition=acquisition,
            repaired=repaired,
            runtime=_Runtime(),
            label="test",
        )


def test_dice_sidecar_paths_must_match_replay_archive(tmp_path: Path) -> None:
    evidence = tmp_path / ".negpy-dice-acquisition" / "token"
    evidence.mkdir(parents=True)
    binding_path = evidence / "acquisition-binding.json"
    rgb_path = tmp_path / "frame.tif"
    ir_path = tmp_path / "frame_IR.tif"
    rgb_path.write_bytes(b"rgb")
    ir_path.write_bytes(b"ir")

    outer_artifacts: dict[str, dict[str, object]] = {}
    bound_artifacts: dict[str, dict[str, object]] = {}
    for key, filename, payload in (
        ("prepass_rgbi", "prepass.rgbi16.npy", b"prepass"),
        ("ir_validity", "ir-validity.npy", b"validity"),
    ):
        path = evidence / filename
        path.write_bytes(payload)
        bound = {
            "bytes": len(payload),
            "dtype": "<u2" if key == "prepass_rgbi" else "|b1",
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_sha256": _digest(key),
            "relative_path": filename,
            "shape": [1],
        }
        bound_artifacts[key] = bound
        outer_artifacts[key] = {**bound, "path": str(path)}

    outer_sources: dict[str, dict[str, object]] = {}
    bound_sources: dict[str, dict[str, object]] = {}
    for key, path, payload in (
        ("storage_rgb_tiff", rgb_path, b"rgb"),
        ("storage_ir_tiff", ir_path, b"ir"),
    ):
        bound = {
            "dtype": "<u2",
            "orientation": "upright-storage",
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "relative_path": os.path.relpath(path, evidence),
            "shape": [1],
        }
        bound_sources[key] = bound
        outer_sources[key] = {**bound, "path": str(path)}

    acquisition = {"acquisition_id": "dice-" + "a" * 64}
    replay = {"complete": True}
    binding = {
        "acquisition": acquisition,
        "artifacts": bound_artifacts,
        "replay": replay,
        "schema": "negpy.dice-acquisition-replay-v1",
        "sources": bound_sources,
    }
    binding_bytes = json.dumps(binding).encode()
    binding_path.write_bytes(binding_bytes)
    dice = {
        "acquisition_id": acquisition["acquisition_id"],
        "artifacts": outer_artifacts,
        "replay": replay,
        "schema": "negpy.dice-acquisition-replay-v1",
        "sources": outer_sources,
    }
    referenced: set[Path] = set()
    deep_acceptance._validate_dice_archive_paths(
        dice,
        binding_path=binding_path,
        binding_bytes=binding_bytes,
        root=tmp_path.resolve(),
        rgb_path=rgb_path,
        ir_path=ir_path,
        referenced=referenced,
        label="slot 1",
    )
    assert referenced == {
        evidence / "prepass.rgbi16.npy",
        evidence / "ir-validity.npy",
    }

    duplicate = evidence / "duplicate.npy"
    duplicate.write_bytes(b"prepass")
    outer_artifacts["prepass_rgbi"]["path"] = str(duplicate)
    with pytest.raises(
        deep_acceptance.DeepAcceptanceError,
        match="path differs from the replay binding",
    ):
        deep_acceptance._validate_dice_archive_paths(
            dice,
            binding_path=binding_path,
            binding_bytes=binding_bytes,
            root=tmp_path.resolve(),
            rgb_path=rgb_path,
            ir_path=ir_path,
            referenced=set(),
            label="slot 1",
        )
