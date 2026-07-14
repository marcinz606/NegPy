from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    ManualFrameApproval,
    RollFingerprintComparison,
    SelectedRollFingerprintComparison,
)
from negpy.infrastructure.scanners.ls5000_single_pass.continuation_plan import (
    CANONICAL_CONTINUATION_PLAN_SHA256,
)
from negpy.infrastructure.tiff_contract import has_linear_scanner_rgb_marker
from negpy.services.scanning.ls5000_single_pass_workflow import (
    LS5000SinglePassWorkflow,
    PackedCaptureContract,
    SinglePassAttempt,
    SinglePassIntegrityError,
    SinglePassSession,
    SinglePassWorkflowError,
)
from negpy.services.scanning import ls5000_single_pass_workflow as workflow_module
from negpy.services.scanning.quality import (
    FocusDetailTelemetry,
    ScanClippingTelemetry,
    StoppedTransportSmearAssessment,
)
from negpy.services.scanning.writer import write_full_negative_tiff


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _journal(
    stream: Path,
    payload: bytes,
    *,
    selected_slot: int = 39,
    boundary_offset_rows: int = 0,
) -> dict[str, Any]:
    origin = 109_060
    reviewed_fingerprint_sha256 = "c" * 64
    manual_approval = ManualFrameApproval(
        reviewed_fingerprint_sha256=reviewed_fingerprint_sha256,
        slot=selected_slot,
        boundary_offset_rows=boundary_offset_rows,
        thumbnail_sha256="e" * 64,
        reviewed_lookup_row=2_580 + boundary_offset_rows,
        reviewed_native_origin=origin,
        review_reasons=("transport-origin-inferred",),
    ).to_payload()
    roll_comparison = RollFingerprintComparison(
        matches=True,
        reason="matched",
        compared_frames=37,
        preview_height_delta_rows=2,
        visual_median_hamming=6.0,
        visual_p90_hamming=12,
        frame_start_median_delta_rows=2.0,
        frame_start_max_delta_rows=8,
        native_origin_median_delta=14.0,
        native_origin_max_delta=42,
        discriminative_frames=37,
        minimum_discriminative_frames=3,
        minimum_visual_log_span=0.5,
    ).to_payload()
    selected_comparison = SelectedRollFingerprintComparison(
        matches=True,
        reason="matched",
        slot=selected_slot,
        visual_hamming=8,
        maximum_visual_hamming=48,
        reviewed_visual_log_span=2.0,
        fresh_visual_log_span=1.9,
        minimum_visual_log_span=0.5,
    ).to_payload()
    exposures = {"1": 107_262, "2": 276_334, "3": 336_777, "9": 311_725}
    common_windows = [
        {
            "color_id": color,
            "resolution": [4000, 4000],
            "origin": [0, origin],
            "size": [2, 3],
            "samples": 4,
            "exposure_raw_10ns": exposures[str(color)],
        }
        for color in (9, 1, 2, 3)
    ]
    fine_windows = [{**window, "interleave": 64} for window in common_windows]
    return {
        "status": "complete",
        "capture_mode": "full",
        "expected_reads": 2,
        "completed_reads": 2,
        "expected_bytes": 8,
        "completed_bytes": 8,
        "disk_bytes": 8,
        "unit_released": True,
        "recovery_required": None,
        "output": str(stream.resolve()),
        "output_sha256": _sha256(payload),
        "scanner_identity": "Nikon LS-5000 ED 1.03",
        "requested_frame": selected_slot,
        "requested_boundary_offset_rows": boundary_offset_rows,
        "applied_boundary_offset_rows": boundary_offset_rows,
        "resolved_lookup_row": 2_580 + boundary_offset_rows,
        "resolved_native_origin": origin,
        # Old capture engines may record a label-derived hint.  It is evidence,
        # never authority for the explicit scanner slot selected by the user.
        "expected_frame_count": 36,
        "preview_geometry_validated_before_reads": True,
        "plan_sha256": "a" * 64,
        "continuation_plan_sha256": CANONICAL_CONTINUATION_PLAN_SHA256,
        "capture_engine_sha256": "b" * 64,
        "reviewed_roll_fingerprint_sha256": reviewed_fingerprint_sha256,
        "manual_review_approval": manual_approval,
        "live_frame_selection": {
            "frame": selected_slot,
            "frame_count": 37,
            "usable_rows": 5_200,
            "preview_sha256": "1" * 64,
            "table_sha256": "2" * 64,
            "roll_identity": {
                "reviewed_fingerprint_sha256": reviewed_fingerprint_sha256,
                "fresh_fingerprint_sha256": "d" * 64,
                "comparison": roll_comparison,
                "selected_slot_comparison": selected_comparison,
            },
            "selected": {
                "automatic": False,
                "manual_review": True,
                "frame": selected_slot,
                "native_origin": origin,
                "method": "explicit-addressable-slot",
                "selector": 145,
                "code": 264,
                "lookup_row": 2_580 + boundary_offset_rows,
            },
            "detection": {
                "confidence": "review",
                "frame_count": 37,
            },
            "transport_mapping": {"status": "manual-review-required"},
        },
        "fine_set_windows_preflight": common_windows,
        "fine_windows": fine_windows,
        "meter_controller_final_result": {
            "accepted": True,
            "final_exposures_raw_10ns": {
                "R": exposures["1"],
                "G": exposures["2"],
                "B": exposures["3"],
                "IR": exposures["9"],
            },
        },
        "meter_evidence_persisted_before_fine_arm": True,
        "meter_final_exposures": {
            "controller_channels_raw_10ns": {
                "R": exposures["1"],
                "G": exposures["2"],
                "B": exposures["3"],
                "IR": exposures["9"],
            },
            "wire_colors_raw_10ns": exposures,
        },
    }


def _attempt(
    tmp_path: Path,
    *,
    selected_slot: int = 39,
    boundary_offset_rows: int = 0,
) -> tuple[SinglePassAttempt, Path]:
    session = SinglePassSession(
        root=tmp_path,
        session_id="gold-roll",
        nominal_exposure_count=36,
    )
    directory = tmp_path / "attempts" / "full-slot39-attempt1"
    directory.mkdir(parents=True)
    stream = directory / "capture.bin"
    payload = b"12345678"
    stream.write_bytes(payload)
    journal = directory / "journal.json"
    journal.write_text(
        json.dumps(
            _journal(
                stream,
                payload,
                selected_slot=selected_slot,
                boundary_offset_rows=boundary_offset_rows,
            )
        ),
        encoding="utf-8",
    )
    return (
        SinglePassAttempt(
            session=session,
            attempt_id="full-slot39-attempt1",
            directory=directory,
            stream_path=stream,
            journal_path=journal,
            selected_slot=selected_slot,
            worker_returncode=0,
            boundary_offset_rows=boundary_offset_rows,
        ),
        stream,
    )


def _contract() -> PackedCaptureContract:
    return PackedCaptureContract(records=2, record_bytes=4, width=2, height=3)


def _decoded() -> np.ndarray:
    return np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)


def _smear_assessment(
    verdict: str = "clean",
) -> StoppedTransportSmearAssessment:
    return StoppedTransportSmearAssessment(
        verdict=verdict,
        start_row=None,
        suffix_rows=0,
        minimum_matches=64,
        tail_median_rms=None,
        tail_min_corr=None,
        pre_tail_median_rms=None,
        texture_span=1234.0,
        reason=f"unit-test {verdict}",
    )


def _workflow(**overrides: object) -> LS5000SinglePassWorkflow:
    decoded = _decoded()
    arguments: dict[str, object] = {
        "contract": _contract(),
        "decoder": lambda _path: (
            decoded,
            {
                "padding_validated_records": 2,
                "rgb_samples_decoded": 4,
                "ir_planes_transferred": 1,
            },
        ),
        # The production assessor correctly refuses this deliberately tiny
        # 3-by-2 unit fixture, so inject explicit clean evidence by default.
        "smear_assessor": lambda _rgb, *, dpi: _smear_assessment(),
    }
    arguments.update(overrides)
    return LS5000SinglePassWorkflow(**arguments)


def test_finalizes_explicit_tail_slot_without_treating_roll_count_as_a_gate(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)
    decoded = _decoded()
    workflow = _workflow()
    completed = workflow.finalize_attempt(attempt, delete_scratch=False)

    assert completed.resumed is False
    assert completed.scratch_deleted is False
    assert stream.is_file()
    assert completed.manifest_path.is_file()
    rgb_path = completed.output_paths["rgb"]
    ir_path = completed.output_paths["ir"]
    np.testing.assert_array_equal(tifffile.imread(rgb_path), np.rot90(decoded, k=1)[..., :3])
    np.testing.assert_array_equal(tifffile.imread(ir_path), np.rot90(decoded, k=1)[..., 3])
    with tifffile.TiffFile(rgb_path) as tiff:
        assert has_linear_scanner_rgb_marker(tiff.pages[0].tags)

    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"]["selected_slot"] == 39
    assert manifest["identity"]["boundary_offset_rows"] == 0
    assert manifest["roll_metadata"]["nominal_exposure_count"] == 36
    assert manifest["roll_metadata"]["observed_candidate_count"] == 37
    assert manifest["roll_metadata"]["count_is_advisory"] is True
    assert manifest["frame_evidence"]["selected"]["manual_review"] is True
    assert manifest["frame_evidence"]["roll_identity"]["comparison"]["matches"] is True
    assert manifest["frame_evidence"]["manual_review_approval"]["slot"] == 39
    assert manifest["orientation"]["source_to_storage"] == "numpy rot90(k=1, axes=(0,1))"
    smear_qc = manifest["quality_control"]["stopped_transport_smear"]
    assert smear_qc["coordinate_space"] == "scanner-native RGB before storage rotation"
    assert smear_qc["required_verdict"] == "clean"
    assert smear_qc["assessment"]["verdict"] == "clean"


def test_finalization_records_capture_clipping_and_focus_telemetry(
    tmp_path: Path,
) -> None:
    attempt, _stream = _attempt(tmp_path)
    workflow = _workflow(
        clipping_measurer=lambda _rgb: ScanClippingTelemetry(
            fractions=(0.02, 0.001, 0.0),
            clip_level=0.99,
            warning_fraction=0.01,
            warning=True,
        ),
        focus_measurer=lambda _rgb: FocusDetailTelemetry(
            method="normalized-gradient-v1",
            verdict="measured",
            score=0.03125,
            texture_span=12_345.0,
        ),
    )

    completed = workflow.finalize_attempt(attempt, delete_scratch=False)

    quality = completed.manifest["quality_control"]
    assert quality["capture_clipping"] == {
        "fractions": [0.02, 0.001, 0.0],
        "clip_level": 0.99,
        "warning_fraction": 0.01,
        "warning": True,
    }
    assert quality["focus_detail"] == {
        "method": "normalized-gradient-v1",
        "verdict": "measured",
        "score": 0.03125,
        "texture_span": 12_345.0,
    }


def test_finalizes_explicit_batch_frame_while_shared_reservation_is_retained(
    tmp_path: Path,
) -> None:
    attempt, stream = _attempt(tmp_path)
    batch_session_id = "batch-slot39-slot40-session"
    attempt = replace(
        attempt,
        batch_session_id=batch_session_id,
        batch_frame_index=1,
        batch_frame_total=2,
        batch_selected_slots=(39, 40),
    )
    journal = json.loads(attempt.journal_path.read_text(encoding="utf-8"))
    journal.update(
        status="frame-complete",
        frame_complete=True,
        unit_released=False,
        session_reservation_retained=True,
        batch_session={
            "frame_index": 1,
            "frame_total": 2,
            "selected_slots": [39, 40],
            "session_id": batch_session_id,
        },
    )
    attempt.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    completed = _workflow().finalize_attempt(attempt)

    assert completed.scratch_deleted is True
    assert not stream.exists()
    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture"]["unit_released"] is False
    assert manifest["capture"]["session_reservation_retained"] is True
    assert manifest["capture"]["batch_session"]["session_id"] == batch_session_id
    assert (
        manifest["capture"]["continuation_plan_sha256"]
        == CANONICAL_CONTINUATION_PLAN_SHA256
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda journal: journal["live_frame_selection"]["roll_identity"][
                "comparison"
            ].update(matches=False, reason="visual-content-mismatch"),
            "roll fingerprint comparison",
        ),
        (
            lambda journal: journal["live_frame_selection"]["roll_identity"].update(
                fresh_fingerprint_sha256="bad"
            ),
            "fresh roll fingerprint",
        ),
        (
            lambda journal: journal["live_frame_selection"]["roll_identity"][
                "selected_slot_comparison"
            ].update(matches=False, reason="selected-visual-content-mismatch"),
            "selected-slot roll fingerprint",
        ),
        (
            lambda journal: journal.__setitem__("manual_review_approval", None),
            "manual review approval",
        ),
    ],
)
def test_batch_finalization_refuses_unproven_roll_or_manual_review_evidence(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    attempt, stream = _attempt(tmp_path)
    batch_session_id = "batch-slot39-session"
    attempt = replace(
        attempt,
        batch_session_id=batch_session_id,
        batch_frame_index=1,
        batch_frame_total=1,
        batch_selected_slots=(39,),
    )
    journal = json.loads(attempt.journal_path.read_text(encoding="utf-8"))
    journal.update(
        status="frame-complete",
        frame_complete=True,
        unit_released=False,
        session_reservation_retained=True,
        batch_session={
            "frame_index": 1,
            "frame_total": 1,
            "selected_slots": [39],
            "session_id": batch_session_id,
        },
    )
    mutate(journal)
    attempt.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(SinglePassIntegrityError, match=message):
        _workflow().finalize_attempt(attempt)

    assert stream.is_file()


def test_batch_frame_refuses_missing_continuation_plan_provenance(
    tmp_path: Path,
) -> None:
    attempt, stream = _attempt(tmp_path)
    batch_session_id = "batch-slot39-slot40-session"
    attempt = replace(
        attempt,
        batch_session_id=batch_session_id,
        batch_frame_index=1,
        batch_frame_total=2,
        batch_selected_slots=(39, 40),
    )
    journal = json.loads(attempt.journal_path.read_text(encoding="utf-8"))
    journal.update(
        status="frame-complete",
        frame_complete=True,
        unit_released=False,
        session_reservation_retained=True,
        batch_session={
            "frame_index": 1,
            "frame_total": 2,
            "selected_slots": [39, 40],
            "session_id": batch_session_id,
        },
    )
    del journal["continuation_plan_sha256"]
    attempt.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(
        SinglePassIntegrityError,
        match="continuation_plan_sha256",
    ):
        _workflow().finalize_attempt(attempt)

    assert stream.is_file()


def test_single_frame_attempt_refuses_a_retained_unreleased_journal(
    tmp_path: Path,
) -> None:
    attempt, stream = _attempt(tmp_path)
    journal = json.loads(attempt.journal_path.read_text(encoding="utf-8"))
    journal.update(
        unit_released=False,
        session_reservation_retained=True,
        frame_complete=True,
        batch_session={
            "frame_index": 1,
            "frame_total": 1,
            "selected_slots": [39],
            "session_id": "unbound-batch",
        },
    )
    attempt.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(SinglePassIntegrityError, match="unit_released"):
        _workflow().finalize_attempt(attempt)

    assert stream.is_file()


@pytest.mark.parametrize(
    "batch_session_mutation",
    [
        lambda batch: batch.__setitem__("session_id", "different-session"),
        lambda batch: batch.__setitem__("selected_slots", [40, 39]),
        lambda batch: batch.__setitem__("frame_index", 2),
    ],
)
def test_batch_frame_refuses_session_identity_or_frame_order_mismatch(
    tmp_path: Path,
    batch_session_mutation: Any,
) -> None:
    attempt, stream = _attempt(tmp_path)
    batch_session_id = "batch-slot39-slot40-session"
    attempt = replace(
        attempt,
        batch_session_id=batch_session_id,
        batch_frame_index=1,
        batch_frame_total=2,
        batch_selected_slots=(39, 40),
    )
    journal = json.loads(attempt.journal_path.read_text(encoding="utf-8"))
    journal.update(
        status="frame-complete",
        frame_complete=True,
        unit_released=False,
        session_reservation_retained=True,
        batch_session={
            "frame_index": 1,
            "frame_total": 2,
            "selected_slots": [39, 40],
            "session_id": batch_session_id,
        },
    )
    batch_session_mutation(journal["batch_session"])
    attempt.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(SinglePassIntegrityError, match="batch_session"):
        _workflow().finalize_attempt(attempt)

    assert stream.is_file()


def test_batch_resume_is_bound_to_the_original_parent_session(
    tmp_path: Path,
) -> None:
    attempt, _stream = _attempt(tmp_path)
    batch_session_id = "batch-slot39-slot40-session"
    attempt = replace(
        attempt,
        batch_session_id=batch_session_id,
        batch_frame_index=1,
        batch_frame_total=2,
        batch_selected_slots=(39, 40),
    )
    journal = json.loads(attempt.journal_path.read_text(encoding="utf-8"))
    journal.update(
        status="frame-complete",
        frame_complete=True,
        unit_released=False,
        session_reservation_retained=True,
        batch_session={
            "frame_index": 1,
            "frame_total": 2,
            "selected_slots": [39, 40],
            "session_id": batch_session_id,
        },
    )
    attempt.journal_path.write_text(json.dumps(journal), encoding="utf-8")
    _workflow().finalize_attempt(attempt)

    mismatched = replace(attempt, batch_session_id="another-batch-session")
    with pytest.raises(SinglePassIntegrityError, match="identify this attempt"):
        _workflow().finalize_attempt(mismatched)


@pytest.mark.parametrize("verdict", ["smear", "indeterminate"])
def test_smear_qc_refuses_before_write_and_retains_scratch(
    tmp_path: Path,
    verdict: str,
) -> None:
    attempt, stream = _attempt(tmp_path)
    writer_called = False

    def forbidden_writer(*_args: object, **_kwargs: object) -> dict[str, dict[str, object]]:
        nonlocal writer_called
        writer_called = True
        raise AssertionError("refused RGB must not reach the writer")

    with pytest.raises(
        SinglePassIntegrityError,
        match=rf"smear QC refused decoded RGB: {verdict}",
    ):
        _workflow(
            writer=forbidden_writer,
            smear_assessor=lambda _rgb, *, dpi: _smear_assessment(verdict),
        ).finalize_attempt(attempt)

    assert writer_called is False
    assert stream.is_file()
    assert not (attempt.directory / "final" / "manifest.json").exists()


def test_boundary_offset_is_bound_from_request_through_completion_manifest(tmp_path: Path) -> None:
    attempt, _stream = _attempt(tmp_path, selected_slot=7, boundary_offset_rows=-23)

    completed = _workflow().finalize_attempt(attempt, delete_scratch=False)

    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"]["boundary_offset_rows"] == -23
    assert manifest["frame_evidence"]["boundary_offset_rows"] == -23
    assert manifest["frame_evidence"]["selected"]["lookup_row"] == 2_557


def test_smear_qc_receives_scanner_native_rgb_before_rotation(tmp_path: Path) -> None:
    attempt, _stream = _attempt(tmp_path)
    decoded = _decoded()
    observed: dict[str, object] = {}

    def assessor(rgb: np.ndarray, *, dpi: int) -> StoppedTransportSmearAssessment:
        observed["rgb"] = rgb.copy()
        observed["dpi"] = dpi
        return _smear_assessment()

    _workflow(smear_assessor=assessor).finalize_attempt(
        attempt,
        delete_scratch=False,
    )

    np.testing.assert_array_equal(observed["rgb"], decoded[..., :3])
    assert observed["dpi"] == 4000


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda journal: journal.__setitem__("status", "failed"), "status"),
        (lambda journal: journal.__setitem__("scanner_identity", "Other scanner"), "identity"),
        (lambda journal: journal.__setitem__("requested_frame", 38), "requested_frame"),
        (
            lambda journal: journal["fine_set_windows_preflight"][0].__setitem__("exposure_raw_10ns", 1),
            "changed between SET_WINDOW and GET_WINDOW",
        ),
        (
            lambda journal: journal["meter_final_exposures"]["wire_colors_raw_10ns"].__setitem__("9", 1),
            "exposure echo",
        ),
    ],
)
def test_invalid_worker_journal_retains_scratch_and_commits_nothing(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    attempt, stream = _attempt(tmp_path)
    journal = json.loads(attempt.journal_path.read_text(encoding="utf-8"))
    mutate(journal)
    attempt.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(SinglePassIntegrityError, match=message):
        _workflow().finalize_attempt(attempt)

    assert stream.is_file()
    assert not (attempt.directory / "final" / "manifest.json").exists()
    final_dir = attempt.directory / "final"
    if final_dir.exists():
        assert not list(final_dir.glob("*.tif"))


def test_stream_hash_mismatch_retains_scratch_before_decode(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)
    stream.write_bytes(b"87654321")
    decode_called = False

    def decoder(_path: Path) -> tuple[np.ndarray, dict[str, object]]:
        nonlocal decode_called
        decode_called = True
        raise AssertionError("must not decode an unbound stream")

    with pytest.raises(SinglePassIntegrityError, match="SHA-256"):
        _workflow(decoder=decoder).finalize_attempt(attempt)

    assert decode_called is False
    assert stream.read_bytes() == b"87654321"


def test_decode_failure_retains_scratch(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)

    def broken_decoder(_path: Path) -> tuple[np.ndarray, dict[str, object]]:
        raise ValueError("padding counter mismatch")

    with pytest.raises(SinglePassIntegrityError, match="decode refused.*padding counter mismatch"):
        _workflow(decoder=broken_decoder).finalize_attempt(attempt)

    assert stream.is_file()
    assert not (attempt.directory / "final" / "manifest.json").exists()


def test_writer_failure_retains_scratch(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)

    def broken_writer(*_args: object, **_kwargs: object) -> dict[str, dict[str, object]]:
        raise OSError("disk full")

    with pytest.raises(SinglePassWorkflowError, match="writer failed.*disk full"):
        _workflow(writer=broken_writer).finalize_attempt(attempt)

    assert stream.is_file()
    assert not (attempt.directory / "final" / "manifest.json").exists()


def test_output_reverification_failure_retains_scratch_and_withholds_manifest(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)

    def corrupting_writer(*args: object, **kwargs: object) -> dict[str, dict[str, object]]:
        evidence = write_full_negative_tiff(*args, **kwargs)
        rgb_path = attempt.directory / "final" / "frame039.tif"
        rgb_path.write_bytes(rgb_path.read_bytes() + b"changed-after-writer-evidence")
        return evidence

    with pytest.raises(SinglePassIntegrityError, match="RGB|rgb|hash"):
        _workflow(writer=corrupting_writer).finalize_attempt(attempt)

    assert stream.is_file()
    assert not (attempt.directory / "final" / "manifest.json").exists()


def test_manifest_failure_retains_scratch_after_tiffs_are_durable(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)

    def broken_manifest(_path: Path, _payload: bytes) -> None:
        raise OSError("manifest fsync failed")

    with pytest.raises(OSError, match="manifest fsync failed"):
        _workflow(manifest_writer=broken_manifest).finalize_attempt(attempt)

    assert stream.is_file()
    assert len(list((attempt.directory / "final").glob("*.tif"))) == 3
    assert not (attempt.directory / "final" / "manifest.json").exists()


def test_cleanup_failure_keeps_verified_archive_and_retains_scratch(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)

    def broken_cleanup(_path: Path) -> None:
        raise OSError("unlink refused")

    with pytest.raises(OSError, match="unlink refused"):
        _workflow(stream_deleter=broken_cleanup).finalize_attempt(attempt)

    assert stream.is_file()
    assert (attempt.directory / "final" / "manifest.json").is_file()
    assert len(list((attempt.directory / "final").glob("*.tif"))) == 3


def test_success_deletes_scratch_only_after_checkpoint_then_resume_skips_decode_and_write(tmp_path: Path) -> None:
    attempt, stream = _attempt(tmp_path)
    completed = _workflow().finalize_attempt(attempt)

    assert completed.scratch_deleted is True
    assert not stream.exists()
    checkpoint = json.loads((attempt.directory / "final" / "scratch-deletion.json").read_text(encoding="utf-8"))
    assert checkpoint["source_sha256"] == _sha256(b"12345678")
    assert checkpoint["source_bytes"] == 8

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a verified completed attempt must be skipped")

    resumed = _workflow(decoder=forbidden, writer=forbidden).finalize_attempt(attempt)
    assert resumed.resumed is True
    assert resumed.scratch_deleted is True


def test_resume_does_not_skip_when_a_committed_output_hash_no_longer_verifies(tmp_path: Path) -> None:
    attempt, _stream = _attempt(tmp_path)
    completed = _workflow().finalize_attempt(attempt)
    completed.output_paths["ir"].write_bytes(completed.output_paths["ir"].read_bytes() + b"tampered")

    with pytest.raises(SinglePassIntegrityError, match="hash"):
        _workflow().finalize_attempt(attempt)


def test_resume_rejects_semantically_tampered_manifest_even_when_outputs_are_unchanged(tmp_path: Path) -> None:
    attempt, _stream = _attempt(tmp_path)
    completed = _workflow().finalize_attempt(attempt)
    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    manifest["roll_metadata"]["warnings"] = []
    completed.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SinglePassIntegrityError, match="binding"):
        _workflow().finalize_attempt(attempt)


@pytest.mark.parametrize("missing", ["capture_clipping", "focus_detail"])
def test_resume_rejects_legacy_manifest_missing_required_quality_evidence(
    tmp_path: Path,
    missing: str,
) -> None:
    attempt, _stream = _attempt(tmp_path)
    completed = _workflow().finalize_attempt(attempt)
    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    manifest["quality_control"].pop(missing)
    manifest["binding_sha256"] = workflow_module._manifest_binding(manifest)
    completed.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SinglePassIntegrityError, match=missing.replace("_", " ")):
        _workflow().finalize_attempt(attempt)
