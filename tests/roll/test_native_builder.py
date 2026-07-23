"""Native Nikon Stage-1 builder tests; no scanner or VM required."""

from __future__ import annotations

import json
import os
import struct
import types
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from negpy.infrastructure.roll import repair as roll_repair
from negpy.services.roll import exact_color
from negpy.services.roll import service as roll_service
from negpy.services.roll.native_builder import (
    NativeBuilderEvidence,
    adapt_native_builder_evidence,
    build_native_builder_receipt,
)
from negpy.services.roll.portable_builder import PortableStage1Builder
from negpy.services.roll.portable_cms import PortableCMSOnEvaluator
from negpy.services.roll.service import RollScanningService


CAPTURE_D = Path(
    os.environ.get(
        "CAPTURE_D",
        "/Volumes/isos/NikonRE/session20260719/capture-d",
    )
)
CAPTURE_D_PREF_SHA256 = (
    "46a0d68ae20c72088e64a1144a0d38bf692f15f506539bbe94eb563fe437c976",
    "23eda81294817e7a2a31f1488544a6f8d3e7ac817f22d43c8a39882565c34b95",
    "3cfc61c06bac49c4c28e69afe99af01366fae6bf5ea88954f688592a8e2756bb",
)
DENSITY_ALGORITHM_ID = "ls5000-md3-10088810-layout1-u16-proven-inputs-macos-binary64-exact-v6"
FRAME_OWNERSHIP_STATUS = "proven-exact-reservation-preview-registration-and-transport"
RESERVATION_ID = "reservation-test-001"
PREVIEW_SHA256 = sha256(b"test-preview").hexdigest()
PREVIEW_IDENTITY_SHA256 = sha256(b"test-preview-identity").hexdigest()
TRANSPORT_TABLE_SHA256 = sha256(b"test-transport-table").hexdigest()
REVIEWED_FINGERPRINT_SHA256 = sha256(b"test-reviewed-registration").hexdigest()
FRESH_FINGERPRINT_SHA256 = sha256(b"test-fresh-registration").hexdigest()
DENSITY_WIRE_SHA256 = PREVIEW_SHA256
DENSITY_CHILD_SHA256 = sha256(b"test-only-capture-d-density-child").hexdigest()


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _ownership_document() -> dict[str, object]:
    material = {
        "reservation_id": RESERVATION_ID,
        "batch_session_id": RESERVATION_ID,
        "preview_sha256": PREVIEW_SHA256,
        "preview_identity_sha256": PREVIEW_IDENTITY_SHA256,
        "transport_table_sha256": TRANSPORT_TABLE_SHA256,
        "reviewed_fingerprint_sha256": REVIEWED_FINGERPRINT_SHA256,
        "fresh_fingerprint_sha256": FRESH_FINGERPRINT_SHA256,
        "selected_slots": [5],
    }
    return {
        "schema_version": 1,
        "scope": "reservation-preview-frame",
        "binding_status": FRAME_OWNERSHIP_STATUS,
        "session_reservation_retained": True,
        **material,
        "transport_identity_sha256": sha256(_canonical_json(material)).hexdigest(),
        "frame_capture_attempt_id": "capture-d",
        "frame_index": 1,
        "frame_total": 1,
        "selected_slot": 5,
    }


def _density_document(
    *,
    numerators: tuple[int, int, int] = (57_114, 48_036, 32_683),
    density_f03: tuple[int, int, int] = (70_307, 136_614, 125_470),
    source_payload_bytes: int = 6_250_496,
    densities: tuple[float, float, float] = (
        struct.unpack(">d", bytes.fromhex("3fd8b159777b9d5f"))[0],
        struct.unpack(">d", bytes.fromhex("3fe9cc75f7f6705a"))[0],
        struct.unpack(">d", bytes.fromhex("3ff0b0dae0533338"))[0],
    ),
) -> dict[str, object]:
    binding_identity = {
        "session_id": RESERVATION_ID,
        "capture_attempt_id": "capture-d",
        "scan_identity": "capture-d-slot",
    }
    return {
        "schema_version": 1,
        "scope": "reservation-preview",
        "per_frame_binding_status": "requires-explicit-frame-ownership-receipt",
        "preview_identity_sha256": PREVIEW_IDENTITY_SHA256,
        "source_payload_bytes": source_payload_bytes,
        "calibration_binding": {
            "calibration": {
                "session_id": RESERVATION_ID,
                "numerators_rgb": list(numerators),
            },
            "capture_attempt_id": binding_identity["capture_attempt_id"],
            "scan_identity": binding_identity["scan_identity"],
        },
        "source_binding": {
            **binding_identity,
            "resolution_dpi": 97,
            "wire_sha256": DENSITY_WIRE_SHA256,
            "child_buffer_sha256": DENSITY_CHILD_SHA256,
        },
        "exposure_binding": {
            **binding_identity,
            "density_f03_exposures_raw_10ns_rgb": list(density_f03),
        },
        "result": {
            **binding_identity,
            "algorithm_id": DENSITY_ALGORITHM_ID,
            "promotable": True,
            "source_wire_sha256": DENSITY_WIRE_SHA256,
            "source_child_buffer_sha256": DENSITY_CHILD_SHA256,
            "numerators_rgb": list(numerators),
            "density_f03_denominators_raw_10ns_rgb": list(density_f03),
            "densities_rgb": list(densities),
            "density_binary64_be_hex_rgb": [struct.pack(">d", value).hex() for value in densities],
        },
    }


@dataclass(frozen=True)
class _PayloadReceipt:
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return self.payload


@dataclass(frozen=True)
class _RejectingOwnershipReceipt(_PayloadReceipt):
    def validate_evidence(self, evidence: object) -> None:
        del evidence
        raise ValueError("preview identity changed")


@dataclass(frozen=True)
class _FrameReceipt:
    version: int
    slot: int
    nikon_density_ownership: object | None


def _bound_repair_frame_fields(*, slot: int = 5) -> dict[str, object]:
    """Create one valid, immutable scanner-native repair acquisition."""

    storage_rgb = np.full((2, 3, 3), 12_000, dtype=np.uint16)
    storage_ir = np.full((2, 3), 3_000, dtype=np.uint16)
    storage_validity = np.ones(storage_rgb.shape[:2], dtype=np.bool_)
    meter_rgbi = np.zeros((1, 1, 4), dtype=np.uint16)
    native_rgbi = np.ascontiguousarray(
        np.rot90(
            np.dstack((storage_rgb, storage_ir)),
            k=-1,
            axes=(0, 1),
        )
    )
    native_validity = np.ascontiguousarray(np.rot90(storage_validity, k=-1, axes=(0, 1)))
    capture_attempt_id = f"capture-d-slot-{slot}"
    acquisition_id, evidence_sha256 = roll_service._derive_digital_ice_producer_binding(
        slot=slot,
        reservation_id=RESERVATION_ID,
        capture_attempt_id=capture_attempt_id,
        main_rgbi=native_rgbi,
        prepass_rgbi=meter_rgbi,
        ir_validity=native_validity,
    )
    acquisition = roll_repair.RepairAcquisition.from_arrays(
        acquisition_id=acquisition_id,
        slot=slot,
        reservation_id=RESERVATION_ID,
        capture_attempt_id=capture_attempt_id,
        storage_transform=roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
        evidence_sha256=evidence_sha256,
        main_rgbi=native_rgbi,
        prepass_rgbi=meter_rgbi,
        ir_validity=native_validity,
    )
    return {
        "rgb": storage_rgb,
        "ir": storage_ir,
        "ir_validity": storage_validity,
        "meter_rgbi": meter_rgbi,
        "prepare_digital_ice": lambda: acquisition,
    }


def _capture_d_evidence() -> NativeBuilderEvidence:
    if not CAPTURE_D.is_dir():
        pytest.skip("archived capture-d is unavailable")
    descriptor = (CAPTURE_D / "analyzer-desc.bin").read_bytes()
    width, height, r0, c0, r1, c1 = struct.unpack_from("<6I", descriptor)
    numerators = struct.unpack_from("<3I", descriptor, 0x1C)
    final_f02 = struct.unpack_from("<3I", descriptor, 0x28)
    densities = struct.unpack_from("<3d", descriptor, 0x40)
    analyzer = np.frombuffer((CAPTURE_D / "analyzer-pixels.bin").read_bytes(), dtype="<u2").reshape(height, width, 3)
    numerators = tuple(numerators)
    density_f03 = (70_307, 136_614, 125_470)
    densities = tuple(densities)
    ownership = _ownership_document()
    density = _density_document(numerators=numerators, density_f03=density_f03, densities=densities)
    ownership_payload = _canonical_json(ownership)
    density_payload = _canonical_json(density)
    return NativeBuilderEvidence(
        session_id=RESERVATION_ID,
        capture_attempt_id="capture-d",
        scan_identity="capture-d-slot",
        slot=5,
        # Stage-3 capture-d predates the 97-dpi ownership closure. These are
        # explicitly test-only identities; the fixture validates builder math
        # without claiming the later density source belongs to capture-d.
        density_source_wire_sha256=DENSITY_WIRE_SHA256,
        density_source_child_sha256=DENSITY_CHILD_SHA256,
        calibration_numerators=numerators,
        density_f03_denominators=density_f03,
        densities=densities,
        density_arithmetic=DENSITY_ALGORITHM_ID,
        frame_ownership_status=FRAME_OWNERSHIP_STATUS,
        frame_ownership_receipt=ownership_payload,
        frame_ownership_receipt_sha256=sha256(ownership_payload).hexdigest(),
        density_evidence_receipt=density_payload,
        density_evidence_receipt_sha256=sha256(density_payload).hexdigest(),
        reservation_id=RESERVATION_ID,
        batch_session_id=RESERVATION_ID,
        preview_sha256=PREVIEW_SHA256,
        preview_identity_sha256=PREVIEW_IDENTITY_SHA256,
        transport_table_sha256=TRANSPORT_TABLE_SHA256,
        transport_identity_sha256=ownership["transport_identity_sha256"],
        reviewed_fingerprint_sha256=REVIEWED_FINGERPRINT_SHA256,
        fresh_fingerprint_sha256=FRESH_FINGERPRINT_SHA256,
        frame_index=1,
        frame_total=1,
        selected_slots=(5,),
        analyzer_rgb=analyzer,
        analyzer_rgb_sha256=sha256(analyzer.astype("<u2", copy=False).tobytes()).hexdigest(),
        analyzer_resolution_dpi=285,
        analyzer_rectangle=(r0, c0, r1, c1),
        final_f02_denominators=tuple(final_f02),
    )


def _synthetic_native_evidence(
    *, source_payload_bytes: int = 6_250_496
) -> NativeBuilderEvidence:
    """Small deterministic provenance fixture whose analyzer is self-contained."""

    numerators = (57_114, 48_036, 32_683)
    density_f03 = (70_307, 136_614, 125_470)
    densities = (
        struct.unpack(">d", bytes.fromhex("3fd8b159777b9d5f"))[0],
        struct.unpack(">d", bytes.fromhex("3fe9cc75f7f6705a"))[0],
        struct.unpack(">d", bytes.fromhex("3ff0b0dae0533338"))[0],
    )
    analyzer = np.full((425, 281, 3), 20_000, dtype=np.uint16)
    ownership = _ownership_document()
    ownership_payload = _canonical_json(ownership)
    density_payload = _canonical_json(
        _density_document(
            numerators=numerators,
            density_f03=density_f03,
            source_payload_bytes=source_payload_bytes,
            densities=densities,
        )
    )
    return NativeBuilderEvidence(
        session_id=RESERVATION_ID,
        capture_attempt_id="capture-d",
        scan_identity="capture-d-slot",
        slot=5,
        density_source_wire_sha256=DENSITY_WIRE_SHA256,
        density_source_child_sha256=DENSITY_CHILD_SHA256,
        calibration_numerators=numerators,
        density_f03_denominators=density_f03,
        densities=densities,
        density_arithmetic=DENSITY_ALGORITHM_ID,
        frame_ownership_status=FRAME_OWNERSHIP_STATUS,
        frame_ownership_receipt=ownership_payload,
        frame_ownership_receipt_sha256=sha256(ownership_payload).hexdigest(),
        density_evidence_receipt=density_payload,
        density_evidence_receipt_sha256=sha256(density_payload).hexdigest(),
        reservation_id=RESERVATION_ID,
        batch_session_id=RESERVATION_ID,
        preview_sha256=PREVIEW_SHA256,
        preview_identity_sha256=PREVIEW_IDENTITY_SHA256,
        transport_table_sha256=TRANSPORT_TABLE_SHA256,
        transport_identity_sha256=ownership["transport_identity_sha256"],
        reviewed_fingerprint_sha256=REVIEWED_FINGERPRINT_SHA256,
        fresh_fingerprint_sha256=FRESH_FINGERPRINT_SHA256,
        frame_index=1,
        frame_total=1,
        selected_slots=(5,),
        analyzer_rgb=analyzer,
        analyzer_rgb_sha256=sha256(analyzer.astype("<u2", copy=False).tobytes()).hexdigest(),
        analyzer_resolution_dpi=285,
        analyzer_rectangle=(0, 0, 424, 280),
        final_f02_denominators=(50_000, 50_000, 50_000),
    )


def _synthetic_native_frame() -> object:
    ownership = _PayloadReceipt(_ownership_document())
    return types.SimpleNamespace(
        slot=5,
        **_bound_repair_frame_fields(),
        receipt=_FrameReceipt(
            version=1,
            slot=5,
            nikon_density_ownership=ownership,
        ),
        nikon_density_evidence=_PayloadReceipt(_density_document()),
        nikon_density_ownership=ownership,
        nikon_exact_builder_evidence=_synthetic_native_evidence(),
    )


def _coolscanpy_capture_d_evidence() -> object:
    try:
        from coolscanpy.protocol.ls5000_single_pass.density import (
            NikonExactBuilderEvidence,
        )
    except ImportError:
        pytest.skip("installed Coolscanpy lacks the exact builder producer")
    native = _capture_d_evidence()
    return NikonExactBuilderEvidence(**{name: getattr(native, name) for name in NativeBuilderEvidence.__dataclass_fields__})


def _coolscanpy_built_frame() -> object:
    try:
        from coolscanpy.protocol.ls5000_single_pass.density import (
            assemble_density_calibration,
            build_nikon_density_evidence,
            build_nikon_density_frame_ownership,
            build_nikon_exact_builder_evidence,
            decode_density_calibration_read,
        )
        from coolscanpy.types import (
            Frame,
            build_digital_ice_acquisition_evidence,
        )
    except ImportError:
        pytest.skip("installed Coolscanpy lacks the exact builder producer")
    native = _capture_d_evidence()
    calibration = assemble_density_calibration(
        [
            decode_density_calibration_read(
                bytes.fromhex(f"28008c000{color}0300000a80"),
                bytes.fromhex(payload),
            )
            for color, payload in enumerate(
                (
                    "8c20000000040000df1a",
                    "8c20000000040000bba4",
                    "8c200000000400007fab",
                ),
                start=1,
            )
        ],
        session_id=RESERVATION_ID,
    )
    samples = np.concatenate(
        (
            np.full(96, 45_000, dtype=np.uint16),
            np.full(96, 40_000, dtype=np.uint16),
            np.full(96, 32_000, dtype=np.uint16),
        )
    ).astype(">u2")
    density_source = bytes(100 * 1_024) + samples.tobytes() + bytes(448) + bytes((6_104 - 101) * 1_024)
    density = build_nikon_density_evidence(
        density_source,
        calibration=calibration,
        density_f03_exposures_raw_10ns=(70_307, 136_614, 125_470),
        session_id=RESERVATION_ID,
        capture_attempt_id="preview-attempt-1",
        scan_identity="reservation-test-001-density-97dpi-preview",
    )
    ownership = build_nikon_density_frame_ownership(
        density,
        reservation_id=RESERVATION_ID,
        batch_session_id=RESERVATION_ID,
        transport_table_sha256=TRANSPORT_TABLE_SHA256,
        reviewed_fingerprint_sha256=REVIEWED_FINGERPRINT_SHA256,
        fresh_fingerprint_sha256=FRESH_FINGERPRINT_SHA256,
        frame_capture_attempt_id="fine-slot-5-attempt-1",
        frame_index=1,
        frame_total=1,
        selected_slots=(5,),
        selected_slot=5,
    )
    exact_builder = build_nikon_exact_builder_evidence(
        density,
        ownership,
        analyzer_rgb=native.analyzer_rgb,
        final_f02_denominators=native.final_f02_denominators,
    )
    repair_fields = _bound_repair_frame_fields()
    digital_ice_evidence = build_digital_ice_acquisition_evidence(
        slot=5,
        reservation_id=RESERVATION_ID,
        capture_attempt_id="fine-slot-5-attempt-1",
        storage_rgb=repair_fields["rgb"],
        storage_ir=repair_fields["ir"],
        storage_ir_validity=repair_fields["ir_validity"],
        meter_rgbi=repair_fields["meter_rgbi"],
    )
    return Frame(
        slot=5,
        rgb=repair_fields["rgb"],
        ir=repair_fields["ir"],
        ir_validity=repair_fields["ir_validity"],
        receipt=_FrameReceipt(
            version=1,
            slot=5,
            nikon_density_ownership=ownership,
        ),
        meter_rgbi=repair_fields["meter_rgbi"],
        nikon_density_evidence=density,
        nikon_exact_builder_evidence=exact_builder,
        digital_ice_evidence=digital_ice_evidence,
    )


def test_exact_coolscanpy_producer_is_narrowly_adapted_and_revalidated() -> None:
    producer = _coolscanpy_capture_d_evidence()
    adapted = adapt_native_builder_evidence(producer)

    assert type(adapted) is NativeBuilderEvidence
    assert adapted.analyzer_rgb_sha256 == producer.analyzer_rgb_sha256
    receipt = build_native_builder_receipt(adapted)
    assert tuple(sha256(blob).hexdigest() for blob in receipt.pre_f_luts) == (CAPTURE_D_PREF_SHA256)


def test_coolscanpy_built_frame_runs_through_negpy_native_service(
    fake_repair_engine,
    tmp_path: Path,
) -> None:
    frame = _coolscanpy_built_frame()

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is not None
    positive = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]
    assert positive["native_per_acquisition_builder"] is True


def test_archived_capture_builder_math_reproduces_all_three_pref_luts_byte_exact() -> None:
    receipt = build_native_builder_receipt(_capture_d_evidence())

    assert tuple(sha256(blob).hexdigest() for blob in receipt.pre_f_luts) == CAPTURE_D_PREF_SHA256
    payload = exact_color.builder_receipt_payload(receipt)
    assert payload["native_per_acquisition_builder"] is True
    assert payload["density_source"]["resolution_dpi"] == 97
    assert payload["analyzer_source"]["resolution_dpi"] == 285
    assert payload["density_source"]["f03_denominators_raw_10ns_rgb"] != payload["analyzer_source"]["final_f02_denominators_raw_10ns_rgb"]


@pytest.mark.parametrize("source_payload_bytes", (6_250_496, 5_804_032))
def test_native_builder_accepts_each_proven_roll_preview_geometry(
    source_payload_bytes: int,
) -> None:
    receipt = build_native_builder_receipt(
        _synthetic_native_evidence(source_payload_bytes=source_payload_bytes)
    )

    assert exact_color.builder_receipt_payload(receipt)["native_per_acquisition_builder"] is True


@pytest.mark.parametrize("source_payload_bytes", (5_804_031, 5_804_033, 6_250_495, 6_250_497))
def test_native_builder_rejects_near_miss_roll_preview_geometry(
    source_payload_bytes: int,
) -> None:
    with pytest.raises(
        exact_color.ExactColorUnavailable,
        match="density evidence belongs to a different preview or reservation",
    ):
        build_native_builder_receipt(
            _synthetic_native_evidence(source_payload_bytes=source_payload_bytes)
        )


def test_native_receipt_runs_through_stage1_and_cms_with_native_application_binding() -> None:
    receipt = build_native_builder_receipt(_capture_d_evidence())
    source = np.array([[[0, 1, 2], [65_535, 42_000, 17]]], dtype=np.uint16)

    result = exact_color.evaluate_exact_color(
        source,
        builder_receipt=receipt,
        builder=PortableStage1Builder(chunk_pixels=1),
        evaluator=PortableCMSOnEvaluator(chunk_pixels=1),
    )

    application = exact_color.receipt_payload(result.builder_application_receipt)
    assert application["native_per_acquisition_builder"] is True
    assert application["scope"] == exact_color.NATIVE_BUILDER_SCOPE
    assert application["native_evidence_sha256"] == receipt.evidence_sha256


@pytest.mark.parametrize("evidence_source", ["negpy", "coolscanpy"])
def test_service_auto_builds_native_receipt_and_retains_analyzer_evidence(
    fake_repair_engine,
    tmp_path: Path,
    evidence_source: str,
) -> None:
    ownership = _PayloadReceipt(_ownership_document())
    density = _PayloadReceipt(_density_document())

    @dataclass(frozen=True)
    class Receipt:
        version: int = 1
        slot: int = 5
        nikon_density_ownership: object = ownership

    frame = types.SimpleNamespace(
        slot=5,
        **_bound_repair_frame_fields(),
        receipt=Receipt(),
        nikon_density_evidence=density,
        nikon_density_ownership=ownership,
        nikon_exact_builder_evidence=(_capture_d_evidence() if evidence_source == "negpy" else _coolscanpy_capture_d_evidence()),
    )

    service = RollScanningService()
    output = service.write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is not None
    entry = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]
    assert entry["native_per_acquisition_builder"] is True
    assert entry["inversion_path"] == "native-per-acquisition-builder-and-verified-portable-cms"
    retained = entry["retained_builder_evidence"]
    assert retained["scope"] == exact_color.NATIVE_BUILDER_SCOPE
    analyzer = Path(retained["analyzer_rgb"]["path"]).read_bytes()
    assert sha256(analyzer).hexdigest() == retained["analyzer_rgb"]["sha256"]
    ownership_payload = Path(retained["frame_ownership_receipt"]["path"]).read_bytes()
    assert sha256(ownership_payload).hexdigest() == retained["frame_ownership_receipt"]["sha256"]
    density_payload = Path(retained["density_evidence_receipt"]["path"]).read_bytes()
    assert sha256(density_payload).hexdigest() == retained["density_evidence_receipt"]["sha256"]


def test_service_retains_native_builder_evidence_when_positive_is_not_selected(
    tmp_path: Path,
) -> None:
    output = RollScanningService().write_frame(
        _synthetic_native_frame(),
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=True,
        write_repaired=False,
        write_positive=False,
    )

    receipt = json.loads(Path(output.receipt_path).read_text())
    assert receipt["outputs"]["positive"] == {
        "written": False,
        "status": "not selected",
    }
    entry = receipt["outputs"]["native_color_evidence"]
    assert entry["retained"] is True
    assert entry["builder_receipt_sha256"] == entry["retained_builder_evidence"]["builder_receipt"]["sha256"]
    for artifact in (
        entry["retained_builder_evidence"]["builder_receipt"],
        entry["retained_builder_evidence"]["evidence_receipt"],
        entry["retained_builder_evidence"]["frame_ownership_receipt"],
        entry["retained_builder_evidence"]["density_evidence_receipt"],
        entry["retained_builder_evidence"]["analyzer_rgb"],
        *entry["retained_builder_evidence"]["pre_f_luts"],
    ):
        payload = Path(artifact["path"]).read_bytes()
        assert len(payload) == artifact["bytes"]
        assert sha256(payload).hexdigest() == artifact["sha256"]

    reloaded = exact_color.load_native_builder_receipt(entry["retained_builder_evidence"]["builder_receipt"]["path"])
    assert reloaded.sha256 == entry["builder_receipt_sha256"]
    replay = exact_color.evaluate_exact_color(
        np.array([[[1, 2, 3], [65_535, 42_000, 17]]], dtype=np.uint16),
        builder_receipt=reloaded,
        builder=PortableStage1Builder(chunk_pixels=1),
        evaluator=PortableCMSOnEvaluator(chunk_pixels=1),
    )
    assert replay.builder_receipt.sha256 == entry["builder_receipt_sha256"]


def test_retained_native_builder_loader_rejects_a_symlinked_artifact(
    tmp_path: Path,
) -> None:
    output = RollScanningService().write_frame(
        _synthetic_native_frame(),
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_repaired=False,
        write_positive=False,
    )
    retained = json.loads(Path(output.receipt_path).read_text())["outputs"]["native_color_evidence"]["retained_builder_evidence"]
    target = Path(retained["pre_f_luts"][1]["path"])
    real = target.with_name("real-builder-preF-g.bin")
    target.replace(real)
    target.symlink_to(real)

    with pytest.raises(exact_color.ExactColorUnavailable, match="non-symlink"):
        exact_color.load_native_builder_receipt(retained["builder_receipt"]["path"])


def test_retained_native_builder_loader_does_not_block_on_a_fifo_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = RollScanningService().write_frame(
        _synthetic_native_frame(),
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_repaired=False,
        write_positive=False,
    )
    retained = json.loads(Path(output.receipt_path).read_text())["outputs"]["native_color_evidence"]["retained_builder_evidence"]
    target = Path(retained["pre_f_luts"][1]["path"])
    saved = target.with_name("saved-builder-preF-g.bin")
    real_open = os.open
    swapped = False

    def swap_to_fifo_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == target:
            swapped = True
            target.replace(saved)
            os.mkfifo(target)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(exact_color.os, "open", swap_to_fifo_then_open)

    with pytest.raises(
        exact_color.ExactColorIntegrityError,
        match="changed while being read",
    ):
        exact_color.load_native_builder_receipt(retained["builder_receipt"]["path"])


def test_retained_native_builder_loader_rederives_luts_instead_of_trusting_rehashed_files(
    tmp_path: Path,
) -> None:
    output = RollScanningService().write_frame(
        _synthetic_native_frame(),
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_repaired=False,
        write_positive=False,
    )
    retained = json.loads(Path(output.receipt_path).read_text())["outputs"]["native_color_evidence"]["retained_builder_evidence"]
    lut_path = Path(retained["pre_f_luts"][0]["path"])
    changed_lut = bytearray(lut_path.read_bytes())
    changed_lut[13_579] ^= 1
    lut_path.write_bytes(changed_lut)
    receipt_path = Path(retained["builder_receipt"]["path"])
    envelope = json.loads(receipt_path.read_bytes())
    envelope["pre_f_luts"][0]["sha256"] = sha256(changed_lut).hexdigest()
    receipt_path.write_bytes(_canonical_json(envelope))

    with pytest.raises(
        exact_color.ExactColorIntegrityError,
        match="fresh native derivation",
    ):
        exact_color.load_native_builder_receipt(receipt_path)


def test_retained_native_builder_loader_requires_its_content_addressed_directory(
    tmp_path: Path,
) -> None:
    output = RollScanningService().write_frame(
        _synthetic_native_frame(),
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_repaired=False,
        write_positive=False,
    )
    retained = json.loads(Path(output.receipt_path).read_text())["outputs"]["native_color_evidence"]["retained_builder_evidence"]
    original_receipt = Path(retained["builder_receipt"]["path"])
    renamed_directory = original_receipt.parent.with_name("wrong-content-address")
    original_receipt.parent.rename(renamed_directory)

    with pytest.raises(
        exact_color.ExactColorIntegrityError,
        match="content-addressed directory",
    ):
        exact_color.load_native_builder_receipt(renamed_directory / original_receipt.name)


def test_service_keeps_native_builder_evidence_when_exact_tiff_verification_fails(
    fake_repair_engine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def reject_tiff(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise exact_color.ExactColorIntegrityError("test TIFF rejection")

    monkeypatch.setattr(roll_service, "_verify_exact_positive_tiff", reject_tiff)

    output = RollScanningService().write_frame(
        _synthetic_native_frame(),
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_repaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    receipt = json.loads(Path(output.receipt_path).read_text())
    assert "test TIFF rejection" in receipt["outputs"]["positive"]["status"]
    retained = receipt["outputs"]["native_color_evidence"]
    assert retained["retained"] is True
    assert not (tmp_path / "005_positive.tif").exists()
    for artifact in (
        retained["retained_builder_evidence"]["builder_receipt"],
        retained["retained_builder_evidence"]["evidence_receipt"],
        retained["retained_builder_evidence"]["analyzer_rgb"],
        *retained["retained_builder_evidence"]["pre_f_luts"],
    ):
        assert Path(artifact["path"]).is_file()


def test_noncanonical_dice_binding_withholds_positive_but_keeps_native_color_evidence(
    fake_repair_engine,
    tmp_path: Path,
) -> None:
    frame = _synthetic_native_frame()
    forged_acquisition = replace(
        frame.prepare_digital_ice(),
        evidence_sha256="0" * 64,
    )
    frame.prepare_digital_ice = lambda: forged_acquisition

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_repaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert fake_repair_engine.calls == []
    assert output.positive_path is None
    receipt = json.loads(Path(output.receipt_path).read_text())["outputs"]
    assert "producer evidence SHA-256 changed" in receipt["repaired"]["status"]
    assert "Tier 2" in receipt["positive"]["status"]
    native = receipt["native_color_evidence"]
    assert native["retained"] is True
    assert Path(native["retained_builder_evidence"]["builder_receipt"]["path"]).is_file()


def test_service_refuses_explicit_builder_evidence_without_current_frame_ownership(fake_repair_engine, tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Receipt:
        version: int = 1
        slot: int = 5

    frame = types.SimpleNamespace(
        slot=5,
        **_bound_repair_frame_fields(),
        receipt=Receipt(),
        nikon_exact_builder_evidence=_capture_d_evidence(),
    )

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    status = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]["status"]
    assert "frame ownership receipt is missing" in status


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("preview_sha256", sha256(b"new-preview").hexdigest()),
        ("reservation_id", "new-reservation"),
        ("batch_session_id", "new-reservation"),
        ("transport_table_sha256", sha256(b"moved-transport-table").hexdigest()),
        ("reviewed_fingerprint_sha256", sha256(b"new-registration").hexdigest()),
        ("fresh_fingerprint_sha256", sha256(b"film-moved").hexdigest()),
        ("frame_capture_attempt_id", "another-frame-attempt"),
        ("selected_slot", 6),
    ],
    ids=[
        "new-preview",
        "new-reservation",
        "new-batch",
        "transport-change",
        "re-registration",
        "film-move",
        "new-frame-attempt",
        "different-slot",
    ],
)
def test_service_refuses_changed_frame_ownership_identity(
    fake_repair_engine,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    ownership_document = _ownership_document()
    ownership_document[field] = replacement
    ownership = _PayloadReceipt(ownership_document)
    density = _PayloadReceipt(_density_document())
    frame = types.SimpleNamespace(
        slot=5,
        **_bound_repair_frame_fields(),
        receipt=_FrameReceipt(version=1, slot=5, nikon_density_ownership=ownership),
        nikon_density_evidence=density,
        nikon_density_ownership=ownership,
        nikon_exact_builder_evidence=_capture_d_evidence(),
    )

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    status = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]["status"]
    assert "frame ownership does not match the native builder evidence" in status


def test_service_refuses_density_evidence_from_a_new_preview(fake_repair_engine, tmp_path: Path) -> None:
    ownership = _PayloadReceipt(_ownership_document())
    density_document = _density_document()
    density_document["preview_identity_sha256"] = sha256(b"new-preview-identity").hexdigest()
    density = _PayloadReceipt(density_document)
    frame = types.SimpleNamespace(
        slot=5,
        **_bound_repair_frame_fields(),
        receipt=_FrameReceipt(version=1, slot=5, nikon_density_ownership=ownership),
        nikon_density_evidence=density,
        nikon_density_ownership=ownership,
        nikon_exact_builder_evidence=_capture_d_evidence(),
    )

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    status = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]["status"]
    assert "density evidence does not match the native builder evidence" in status


def test_service_refuses_frame_and_public_receipt_ownership_disagreement(fake_repair_engine, tmp_path: Path) -> None:
    ownership = _PayloadReceipt(_ownership_document())
    changed = _ownership_document()
    changed["fresh_fingerprint_sha256"] = sha256(b"film-moved").hexdigest()
    public_ownership = _PayloadReceipt(changed)
    frame = types.SimpleNamespace(
        slot=5,
        **_bound_repair_frame_fields(),
        receipt=_FrameReceipt(version=1, slot=5, nikon_density_ownership=public_ownership),
        nikon_density_evidence=_PayloadReceipt(_density_document()),
        nikon_density_ownership=ownership,
        nikon_exact_builder_evidence=_capture_d_evidence(),
    )

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    status = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]["status"]
    assert "frame and public receipt disagree" in status


def test_service_honors_ownership_objects_runtime_validation(fake_repair_engine, tmp_path: Path) -> None:
    ownership = _RejectingOwnershipReceipt(_ownership_document())
    frame = types.SimpleNamespace(
        slot=5,
        **_bound_repair_frame_fields(),
        receipt=_FrameReceipt(version=1, slot=5, nikon_density_ownership=ownership),
        nikon_density_evidence=_PayloadReceipt(_density_document()),
        nikon_density_ownership=ownership,
        nikon_exact_builder_evidence=_capture_d_evidence(),
    )

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    status = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]["status"]
    assert "frame ownership is invalid: preview identity changed" in status


def test_service_refuses_generic_session_density_evidence_without_frame_ownership(fake_repair_engine, tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Receipt:
        version: int = 1
        slot: int = 5

    storage_rgb = np.full((2, 3, 3), 12_000, dtype=np.uint16)
    storage_ir = np.full((2, 3), 3_000, dtype=np.uint16)
    storage_rgbi = np.concatenate((storage_rgb, storage_ir[..., None]), axis=2)
    native_rgbi = np.ascontiguousarray(np.rot90(storage_rgbi, k=-1, axes=(0, 1)))
    prepass_rgbi = np.zeros((1, 1, 4), dtype=np.uint16)
    native_validity = np.ones(native_rgbi.shape[:2], dtype=np.bool_)
    acquisition_id, evidence_sha256 = roll_service._derive_digital_ice_producer_binding(
        slot=5,
        reservation_id="reservation-005",
        capture_attempt_id="fine-slot-05-attempt-001",
        main_rgbi=native_rgbi,
        prepass_rgbi=prepass_rgbi,
        ir_validity=native_validity,
    )
    acquisition = roll_repair.RepairAcquisition.from_arrays(
        acquisition_id=acquisition_id,
        slot=5,
        reservation_id="reservation-005",
        capture_attempt_id="fine-slot-05-attempt-001",
        storage_transform=roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
        evidence_sha256=evidence_sha256,
        main_rgbi=native_rgbi,
        prepass_rgbi=prepass_rgbi,
        ir_validity=native_validity,
    )
    frame = types.SimpleNamespace(
        slot=5,
        rgb=storage_rgb,
        ir=storage_ir,
        ir_validity=None,
        receipt=Receipt(),
        meter_rgbi=None,
        nikon_density_session_evidence=object(),
        prepare_digital_ice=lambda: acquisition,
    )

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    entry = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]
    assert "no native builder evidence" in entry["status"]


def test_native_builder_rejects_conflated_density_and_final_exposure_triplets() -> None:
    evidence = _capture_d_evidence()
    density_payload = _canonical_json(
        _density_document(
            numerators=evidence.calibration_numerators,
            density_f03=evidence.final_f02_denominators,
            densities=evidence.densities,
        )
    )
    candidate = replace(
        evidence,
        density_f03_denominators=evidence.final_f02_denominators,
        density_evidence_receipt=density_payload,
        density_evidence_receipt_sha256=sha256(density_payload).hexdigest(),
    )

    with pytest.raises(exact_color.ExactColorUnavailable, match="conflated"):
        build_native_builder_receipt(candidate)


def test_native_builder_rejects_analyzer_mutation_after_identity_was_bound() -> None:
    evidence = _capture_d_evidence()
    analyzer = evidence.analyzer_rgb.copy()
    candidate = replace(evidence, analyzer_rgb=analyzer)
    analyzer[0, 0, 0] ^= 1

    with pytest.raises(exact_color.ExactColorUnavailable, match="does not match its SHA-256"):
        build_native_builder_receipt(candidate)


def test_service_rejects_native_evidence_for_a_different_frame_slot(fake_repair_engine, tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Receipt:
        version: int = 1
        slot: int = 6

    frame = types.SimpleNamespace(
        slot=6,
        **_bound_repair_frame_fields(slot=6),
        receipt=Receipt(),
        nikon_exact_builder_evidence=_capture_d_evidence(),
    )

    output = RollScanningService().write_frame(
        frame,
        str(tmp_path),
        '{{ "%03d" % seq }}',
        write_unrepaired=False,
        write_positive=True,
        positive_mode="nikon-exact",
    )

    assert output.positive_path is None
    status = json.loads(Path(output.receipt_path).read_text())["outputs"]["positive"]["status"]
    assert "different slot" in status


def test_native_receipt_rejects_evidence_mutation() -> None:
    receipt = build_native_builder_receipt(_capture_d_evidence())
    tampered = replace(receipt, evidence_payload=receipt.evidence_payload + b" ")

    with pytest.raises(exact_color.ExactColorIntegrityError, match="evidence does not match"):
        exact_color.builder_receipt_payload(tampered)
