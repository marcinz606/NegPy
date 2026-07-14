"""Content-bound provenance contracts for full-negative source pairs."""

from __future__ import annotations

import pytest

from negpy.services.scanning.provenance import (
    ArtifactEvidence,
    CaptureStateEvidence,
    FullNegativeProvenanceManifest,
    PlanIdentity,
    ReconstructionBindingRecord,
    SourceCaptureRecord,
    SourcePairRecord,
    canonical_semantic_sha256,
)


ROLL_ID = "11111111-1111-4111-8111-111111111111"
INSERTION_ID = "22222222-2222-4222-8222-222222222222"
SESSION_ID = "33333333-3333-4333-8333-333333333333"
FILE_SHA = "a" * 64


def _identity() -> PlanIdentity:
    return PlanIdentity(ROLL_ID, INSERTION_ID, SESSION_ID)


def _artifact(role: str, digit: str) -> ArtifactEvidence:
    channels = 3 if role in {"rgb4x", "rgb1x_proxy"} else 2
    shape = (5959, 3946, channels) if channels == 3 else (5959, 3946)
    return ArtifactEvidence(
        path=f"captures/window003-{role}.tif",
        sha256=digit * 64,
        byte_length=123456,
        shape=shape,
        dtype="uint16" if role != "ir_valid_mask" else "uint8",
        page_count=1,
        payload_within_file=True,
        x_resolution=(4000, 1),
        y_resolution=(4000, 1),
        resolution_unit="INCH",
    )


def _measured_state() -> CaptureStateEvidence:
    return CaptureStateEvidence(
        mode="measured",
        replayed_from_capture_id=None,
        focus_position=216,
        exposure_multiplier=1.0,
        red_exposure_us=1370.0,
        green_exposure_us=1290.0,
        blue_exposure_us=1120.0,
    )


def _capture(
    physical_frame: int,
    *,
    state: CaptureStateEvidence,
) -> SourceCaptureRecord:
    return SourceCaptureRecord.create(
        identity=_identity(),
        physical_frame=physical_frame,
        geometry={
            "physical_frame": physical_frame,
            "subframe_mm": 0.0,
            "br_y_device_px": 5958,
        },
        recipe={
            "strategy": "coolscan-rgb4-rgbi1-v1",
            "dpi": 4000,
            "depth": 16,
            "rgb_samples": 4,
            "ir_samples": 1,
        },
        capture_state=state,
        passes=(
            {"role": "rgb4x", "lines": 5959, "depth": 16, "samples_per_scan": 4},
            {"role": "rgbi1x", "lines": 5959, "depth": 16, "samples_per_scan": 1},
        ),
        artifacts={
            "rgb4x": _artifact("rgb4x", "1"),
            "rgb1x_proxy": _artifact("rgb1x_proxy", "2"),
            "ir1x": _artifact("ir1x", "3"),
            "aligned_ir": _artifact("aligned_ir", "4"),
            "ir_valid_mask": _artifact("ir_valid_mask", "5"),
        },
        split_alignment={"mode": "identity", "dx_px": 0.0, "dy_px": 0.0},
    )


def test_plan_identity_round_trips_canonical_uuids() -> None:
    identity = PlanIdentity(
        roll_id=ROLL_ID,
        insertion_id=INSERTION_ID,
        session_id=SESSION_ID,
    )

    assert PlanIdentity.from_dict(identity.to_dict()) == identity


def test_plan_identity_rejects_reusing_one_uuid_for_two_identity_scopes() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        PlanIdentity(
            roll_id=ROLL_ID,
            insertion_id=ROLL_ID,
            session_id=SESSION_ID,
        )


def test_artifact_semantic_hash_excludes_path_but_includes_tiff_metadata() -> None:
    first = ArtifactEvidence(
        path="captures/window003-rgb4x.tif",
        sha256=FILE_SHA,
        byte_length=123456,
        shape=(5959, 3946, 3),
        dtype="uint16",
        page_count=1,
        payload_within_file=True,
        x_resolution=(4000, 1),
        y_resolution=(4000, 1),
        resolution_unit="INCH",
    )
    moved = ArtifactEvidence.from_dict({**first.to_dict(), "path": "archive/renamed-rgb4x.tif"})
    wrong_dpi = ArtifactEvidence.from_dict({**first.to_dict(), "x_resolution": [3999, 1]})

    assert canonical_semantic_sha256(first) == canonical_semantic_sha256(moved)
    assert canonical_semantic_sha256(first) != canonical_semantic_sha256(wrong_dpi)


def test_capture_state_distinguishes_measured_state_from_exact_replay() -> None:
    measured = _measured_state()
    source_id = "b" * 64
    replayed = CaptureStateEvidence.from_dict(
        {
            **measured.to_dict(),
            "mode": "replayed",
            "replayed_from_capture_id": source_id,
        }
    )

    assert replayed.replayed_from_capture_id == source_id
    assert replayed.values_dict() == measured.values_dict()


def test_source_capture_binding_covers_state_recipe_geometry_passes_and_artifacts() -> None:
    capture = SourceCaptureRecord.create(
        identity=_identity(),
        physical_frame=3,
        geometry={"physical_frame": 3, "subframe_mm": 0.0, "br_y_device_px": 5958},
        recipe={
            "strategy": "coolscan-rgb4-rgbi1-v1",
            "dpi": 4000,
            "depth": 16,
            "rgb_samples": 4,
            "ir_samples": 1,
        },
        capture_state=_measured_state(),
        passes=(
            {"role": "rgb4x", "lines": 5959, "depth": 16, "samples_per_scan": 4},
            {"role": "rgbi1x", "lines": 5959, "depth": 16, "samples_per_scan": 1},
        ),
        artifacts={
            "rgb4x": _artifact("rgb4x", "1"),
            "rgb1x_proxy": _artifact("rgb1x_proxy", "2"),
            "ir1x": _artifact("ir1x", "3"),
            "aligned_ir": _artifact("aligned_ir", "4"),
            "ir_valid_mask": _artifact("ir_valid_mask", "5"),
        },
        split_alignment={"mode": "identity", "dx_px": 0.0, "dy_px": 0.0},
    )

    assert SourceCaptureRecord.from_dict(capture.to_dict()) == capture
    assert capture.capture_id == capture.binding_sha256


def test_source_pair_binds_ordered_consecutive_captures_and_exact_state_replay() -> None:
    first = _capture(3, state=_measured_state())
    second = _capture(
        4,
        state=CaptureStateEvidence(
            mode="replayed",
            replayed_from_capture_id=first.capture_id,
            **_measured_state().values_dict(),
        ),
    )

    pair = SourcePairRecord.create(logical_frame=3, first=first, second=second)

    assert pair.first_capture_id == first.capture_id
    assert pair.second_capture_id == second.capture_id
    assert SourcePairRecord.from_dict(pair.to_dict()) == pair


def test_reconstruction_binding_covers_pair_registration_crop_algorithm_and_outputs() -> None:
    first = _capture(3, state=_measured_state())
    second = _capture(
        4,
        state=CaptureStateEvidence(
            mode="replayed",
            replayed_from_capture_id=first.capture_id,
            **_measured_state().values_dict(),
        ),
    )
    pair = SourcePairRecord.create(logical_frame=3, first=first, second=second)

    reconstruction = ReconstructionBindingRecord.create(
        identity=_identity(),
        logical_frame=3,
        source_pair=pair,
        registration_binding_sha256="c" * 64,
        algorithm_signature={"algorithm": "film-base-core-pair", "version": 1},
        pitch_rows=5959,
        crop_start_row=1020,
        artifacts={
            "rgb": _artifact("rgb4x", "6"),
            "ir": _artifact("aligned_ir", "7"),
            "ir_valid_mask": _artifact("ir_valid_mask", "8"),
        },
    )

    assert reconstruction.source_pair_binding_sha256 == pair.binding_sha256
    assert ReconstructionBindingRecord.from_dict(reconstruction.to_dict()) == reconstruction


def test_manifest_last_json_roundtrip_strictly_revalidates_the_complete_graph() -> None:
    first = _capture(3, state=_measured_state())
    second = _capture(
        4,
        state=CaptureStateEvidence(
            mode="replayed",
            replayed_from_capture_id=first.capture_id,
            **_measured_state().values_dict(),
        ),
    )
    pair = SourcePairRecord.create(logical_frame=3, first=first, second=second)
    reconstruction = ReconstructionBindingRecord.create(
        identity=_identity(),
        logical_frame=3,
        source_pair=pair,
        registration_binding_sha256="c" * 64,
        algorithm_signature={"algorithm": "film-base-core-pair", "version": 1},
        pitch_rows=5959,
        crop_start_row=1020,
        artifacts={
            "rgb": _artifact("rgb4x", "6"),
            "ir": _artifact("aligned_ir", "7"),
            "ir_valid_mask": _artifact("ir_valid_mask", "8"),
        },
    )
    manifest = FullNegativeProvenanceManifest.create(
        identity=_identity(),
        captures=(first, second),
        source_pair=pair,
        reconstruction=reconstruction,
    )

    assert FullNegativeProvenanceManifest.from_json(manifest.to_json()) == manifest
