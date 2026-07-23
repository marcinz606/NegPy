from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from negpy.services.roll import live_evidence
from negpy.services.roll.live_evidence import (
    bind_capture_evidence_inventory,
    snapshot_capture_evidence,
    validate_six_frame_batch_capture_evidence,
)
from negpy.services.roll.live_reservation import (
    FixedOutputLease,
    InventoryConflict,
)


def _lease(root: Path) -> FixedOutputLease:
    return FixedOutputLease.acquire(
        root,
        {"schema": "test.capture-evidence-owner.v1"},
        require_empty=True,
    )


_SLOTS = (1, 2, 3, 4, 5, 6)
_METER_BYTES = 3_264_000
_METER_SHA256 = "2d417c2ed40641cd243f33989601b5a06c7a7c5b893c092c1868e0b9addd03e1"
_PREVIEW_BYTES = 6_250_496
_PREVIEW_SHA256 = "690563a295100f3bb51b5cedbfc3e4a3df467d171d96483420810fd63e75a380"
_METER_LAYOUT = {
    "passes": 3,
    "rows_per_pass": 425,
    "columns": 281,
    "decoded_raster_channel_order": ["R", "G", "B", "IR"],
    "wire_window_color_order": [9, 1, 2, 3],
    "wire_color_to_controller_channel": {
        "9": "IR",
        "1": "R",
        "2": "G",
        "3": "B",
    },
    "sample_byte_order": "big-endian-u16",
    "row_core_bytes": 2_248,
    "row_stride_bytes": 2_560,
    "row_tail_bytes": 312,
}


def _write_sparse_zeros(path: Path, size: int) -> None:
    with path.open("xb") as stream:
        stream.truncate(size)


def _write_completed_batch(root: Path) -> dict[str, object]:
    """Materialize Coolscan's durable tree after raw fine-stream cleanup."""

    batch = root / "batch-slot01-slot06-fixture"
    batch.mkdir()
    session_id = batch.name
    reviewed_sha256 = "a" * 64
    fresh_sha256 = "b" * 64
    from coolscanpy.protocol.ls5000_single_pass.bundle import (
        CAPTURE_BUNDLE_SHA256,
        CAPTURE_WORKER_SHA256,
    )

    engine_sha256 = CAPTURE_WORKER_SHA256
    bundle_sha256 = CAPTURE_BUNDLE_SHA256
    plan_payload = b"pinned first-frame plan\n"
    continuation_payload = b'{"kind":"pinned-continuation"}\n'
    plan_sha256 = hashlib.sha256(plan_payload).hexdigest()
    continuation_sha256 = hashlib.sha256(continuation_payload).hexdigest()
    (batch / "replay-first-rgbi4-plan.jsonl").write_bytes(plan_payload)
    (batch / "replay-next-rgbi4-plan.json").write_bytes(continuation_payload)
    (batch / "replay-first-rgbi4-manifest.json").write_text(
        json.dumps({"plan_sha256": plan_sha256}),
        encoding="utf-8",
    )
    (batch / "stdout.txt").write_bytes(b"")
    (batch / "stderr.txt").write_bytes(b"")

    approvals = {
        1: {"binding_sha256": "1" * 64},
        6: {"binding_sha256": "6" * 64},
    }
    frames = [
        {
            "ack": f"frame-{slot:03d}/parent-ack.json",
            "boundary_offset_rows": 0,
            "journal": f"frame-{slot:03d}/journal.json",
            "manual_review_approval": approvals.get(slot),
            "output": f"frame-{slot:03d}/capture.bin",
            "slot": slot,
        }
        for slot in _SLOTS
    ]
    job = {
        "apply_all_boundary_offsets_before_first_frame": True,
        "capture_plan_sha256": plan_sha256,
        "continuation_plan_sha256": continuation_sha256,
        "expected_usb_address": 7,
        "expected_usb_bus": 3,
        "frames": frames,
        "parent_ack_required_after_every_frame": True,
        "release_once_after_last_frame": True,
        "reviewed_roll_fingerprint": {"binding_sha256": reviewed_sha256},
        "schema_version": 3,
        "session_id": session_id,
        "session_contract": "one-process-one-reservation",
    }
    job_payload = json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    (batch / "batch-job.json").write_bytes(job_payload)
    job_sha256 = hashlib.sha256(job_payload).hexdigest()

    density_calibration = {"session_id": session_id, "fixture": "validated"}
    (batch / "session-journal.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "session_id": session_id,
                "density_calibration_session_id": session_id,
                "nikon_density_calibration": density_calibration,
                "selected_slots": list(_SLOTS),
                "completed_slots": list(_SLOTS),
                "active_frame_index": None,
                "active_slot": None,
                "batch_job_sha256": job_sha256,
                "capture_engine_sha256": engine_sha256,
                "capture_bundle_sha256": bundle_sha256,
                "plan_sha256": plan_sha256,
                "continuation_plan_sha256": continuation_sha256,
                "manual_review_approval_sha256_by_slot": {
                    str(slot): (None if slot not in approvals else approvals[slot]["binding_sha256"]) for slot in _SLOTS
                },
                "reviewed_roll_fingerprint_sha256": reviewed_sha256,
                "expected_usb_bus": 3,
                "expected_usb_address": 7,
                "actual_usb_bus": 3,
                "actual_usb_address": 7,
                "reservation_acquired": True,
                "unit_release_attempts": 1,
                "unit_released": True,
                "recovery_required": "none",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_sha256_by_slot: dict[str, str] = {}
    for frame_index, slot in enumerate(_SLOTS, start=1):
        frame = batch / f"frame-{slot:03d}"
        frame.mkdir()
        meter = frame / "capture-meter.bin"
        _write_sparse_zeros(meter, _METER_BYTES)
        nonce = f"{slot:032x}"
        selection = {
            "frame": slot,
            "boundary_offset": {
                "requested_rows": 0,
                "applied_rows": 0,
            },
            "roll_identity": {
                "reviewed_fingerprint_sha256": reviewed_sha256,
                "fresh_fingerprint_sha256": fresh_sha256,
                "comparison": {"matches": True, "reason": "matched"},
                "selected_slot_comparison": {
                    "matches": True,
                    "reason": "matched",
                    "slot": slot,
                },
            },
        }
        output_sha256 = hashlib.sha256(f"capture-{slot}".encode()).hexdigest()
        output_sha256_by_slot[str(slot)] = output_sha256
        journal: dict[str, object] = {
            "status": "frame-complete",
            "frame_complete": True,
            "session_reservation_retained": True,
            "unit_released": False,
            "recovery_required": None,
            "batch_session": {
                "session_id": session_id,
                "frame_index": frame_index,
                "frame_total": len(_SLOTS),
                "selected_slots": list(_SLOTS),
            },
            "capture_mode": "full",
            "requested_frame": slot,
            "requested_boundary_offset_rows": 0,
            "expected_reads": 2_980,
            "completed_reads": 2_980,
            "expected_bytes": 619_458_560,
            "completed_bytes": 619_458_560,
            "disk_bytes": 619_458_560,
            "output": str(frame / "capture.bin"),
            "output_sha256": output_sha256,
            "plan_sha256": plan_sha256,
            "continuation_plan_sha256": continuation_sha256,
            "capture_engine_sha256": engine_sha256,
            "capture_bundle_sha256": bundle_sha256,
            "manual_review_approval": approvals.get(slot),
            "reviewed_roll_fingerprint_sha256": reviewed_sha256,
            "expected_usb_bus": 3,
            "expected_usb_address": 7,
            "actual_usb_bus": 3,
            "actual_usb_address": 7,
            "ack_nonce": nonce,
            "density_calibration_session_id": session_id,
            "nikon_density_calibration": density_calibration,
            "live_frame_selection": selection,
            "meter_evidence": {
                "path": str(meter),
                "bytes": _METER_BYTES,
                "sha256": _METER_SHA256,
                "complete": True,
                "durable_completed_passes": 3,
            },
            "meter_evidence_persisted_before_fine_arm": True,
            "meter_group_bytes": [1_088_000, 1_088_000, 1_088_000],
            "meter_group_offsets": [0, 1_088_000, 2_176_000],
            "meter_completed_reads": 15,
            "meter_completed_bytes": _METER_BYTES,
            "meter_layout": _METER_LAYOUT,
        }
        if slot == 1:
            preview = frame / "capture-preview.bin"
            _write_sparse_zeros(preview, _PREVIEW_BYTES)
            table_payload = b"six-strip-transport-table"
            table = frame / "capture-008e.bin"
            table.write_bytes(table_payload)
            mapping = frame / "capture-frame-map.json"
            mapping.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
            journal["live_index_artifacts"] = {
                "preview": str(preview),
                "table": str(table),
                "mapping": str(mapping),
            }
            journal["live_index_evidence"] = {
                "status": "persisted-before-frame-detection",
                "preview_bytes": _PREVIEW_BYTES,
                "preview_sha256": _PREVIEW_SHA256,
                "table_bytes": len(table_payload),
                "table_sha256": hashlib.sha256(table_payload).hexdigest(),
            }
        (frame / "journal.json").write_text(
            json.dumps(journal, sort_keys=True),
            encoding="utf-8",
        )
        (frame / "parent-ack.json").write_text(
            json.dumps(
                {
                    "ack_nonce": nonce,
                    "action": "continue",
                    "frame_index": frame_index,
                    "schema_version": 1,
                    "session_id": session_id,
                    "slot": slot,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return {
        "batch": batch,
        "output_sha256_by_slot": output_sha256_by_slot,
        "reviewed_sha256": reviewed_sha256,
    }


def test_completed_post_finalization_batch_binds_all_six_capture_records(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    expected = _write_completed_batch(root)

    snapshot = snapshot_capture_evidence(root)
    binding = validate_six_frame_batch_capture_evidence(snapshot)

    assert snapshot.manifest["file_count"] == 28
    assert binding["frame_output_sha256_by_slot"] == expected["output_sha256_by_slot"]
    assert binding["reviewed_fingerprint_sha256"] == expected["reviewed_sha256"]


def test_finder_metadata_is_ignored_without_weakening_capture_evidence_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    lease = _lease(root)
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    (root / ".DS_Store").write_bytes(b"finder root metadata")
    (batch / ".DS_Store").write_bytes(b"finder batch metadata")

    try:
        snapshot = snapshot_capture_evidence(root)
        inventory = lease.assert_inventory(snapshot.files)
        binding = validate_six_frame_batch_capture_evidence(snapshot)
        receipt_evidence = bind_capture_evidence_inventory(snapshot, inventory)
    finally:
        lease.release()

    assert snapshot.manifest["file_count"] == 28
    assert all(not row["path"].endswith(".DS_Store") for row in snapshot.manifest["files"])
    assert binding["session_id"] == batch.name
    assert receipt_evidence["retained"] is True


def test_finder_metadata_name_does_not_bypass_symlink_rejection(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"must not be followed")
    (root / ".DS_Store").symlink_to(outside)

    with pytest.raises(InventoryConflict, match="exclusively owned regular file"):
        snapshot_capture_evidence(root)


def _rewrite_json(path: Path, mutate) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def test_synthetic_journal_skeleton_without_meter_or_ack_artifacts_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    for slot in _SLOTS:
        frame = batch / f"frame-{slot:03d}"
        (frame / "capture-meter.bin").unlink()
        (frame / "parent-ack.json").unlink()

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match="capture evidence is incomplete"):
        validate_six_frame_batch_capture_evidence(snapshot)


def test_nonempty_second_attempt_tree_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    _write_completed_batch(root)
    partial = root / "batch-slot01-slot06-aborted" / "frame-001"
    partial.mkdir(parents=True)
    (partial / "partial.bin").write_bytes(b"partial")

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match="outside the completed six-frame batch"):
        validate_six_frame_batch_capture_evidence(snapshot)


def test_additional_diagnostic_inside_completed_batch_is_allowed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    diagnostic = batch / "diagnostics" / "capture-timing.txt"
    diagnostic.parent.mkdir()
    diagnostic.write_text("retained diagnostic\n", encoding="utf-8")

    snapshot = snapshot_capture_evidence(root)
    binding = validate_six_frame_batch_capture_evidence(snapshot)

    assert binding["session_id"] == batch.name
    assert any(row["path"].endswith("capture-timing.txt") for row in snapshot.manifest["files"])


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        (
            "frame-003/journal.json",
            lambda document: document.__setitem__("completed_bytes", 1),
            "frame 3 journal",
        ),
        (
            "frame-004/parent-ack.json",
            lambda document: document.__setitem__("action", "stop"),
            "frame 4 parent ACK",
        ),
        (
            "session-journal.json",
            lambda document: document.__setitem__("actual_usb_address", 8),
            "completed six-frame batch",
        ),
        (
            "session-journal.json",
            lambda document: document.pop("batch_job_sha256"),
            "completed six-frame batch",
        ),
        (
            "session-journal.json",
            lambda document: document.__setitem__("batch_job_sha256", "0" * 64),
            "completed six-frame batch",
        ),
        (
            "frame-001/capture-frame-map.json",
            lambda document: document.__setitem__("frame", 2),
            "live preview, transport table, or frame map",
        ),
        (
            "frame-005/journal.json",
            lambda document: document["live_frame_selection"]["boundary_offset"].__setitem__("applied_rows", 1),
            "frame 5 journal",
        ),
    ],
)
def test_mutated_capture_contract_is_rejected(
    tmp_path: Path,
    target: str,
    mutation,
    message: str,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    _rewrite_json(batch / target, mutation)

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match=message):
        validate_six_frame_batch_capture_evidence(snapshot)


def test_meter_payload_must_match_its_frame_journal_digest(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    meter = batch / "frame-005" / "capture-meter.bin"
    with meter.open("r+b") as stream:
        stream.write(b"changed")

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match="frame 5 meter sidecar"):
        validate_six_frame_batch_capture_evidence(snapshot)


def test_parent_ack_nonce_must_use_the_worker_safe_alphabet(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    _rewrite_json(
        batch / "frame-002" / "journal.json",
        lambda document: document.__setitem__("ack_nonce", "../unsafe"),
    )
    _rewrite_json(
        batch / "frame-002" / "parent-ack.json",
        lambda document: document.__setitem__("ack_nonce", "../unsafe"),
    )

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match="frame 2 parent ACK"):
        validate_six_frame_batch_capture_evidence(snapshot)


@pytest.mark.parametrize("name", ["capture-preview.bin", "capture-008e.bin"])
def test_live_index_payload_must_match_first_frame_journal(
    tmp_path: Path,
    name: str,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    artifact = batch / "frame-001" / name
    with artifact.open("r+b") as stream:
        stream.write(b"changed")

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(
        InventoryConflict,
        match="live preview, transport table, or frame map",
    ):
        validate_six_frame_batch_capture_evidence(snapshot)


def test_retained_plan_bytes_must_match_job_and_journals(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    (batch / "replay-first-rgbi4-plan.jsonl").write_bytes(b"changed plan\n")

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match="retained plans"):
        validate_six_frame_batch_capture_evidence(snapshot)


def test_snapshot_hashes_and_binds_retained_attempt_tree(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    lease = _lease(root)
    try:
        frame = root / "batch-a" / "frame-001"
        frame.mkdir(parents=True)
        preview = frame / "capture-preview.bin"
        table = frame / "capture-008e.bin"
        preview.write_bytes(b"preview")
        table.write_bytes(b"table")

        snapshot = snapshot_capture_evidence(root)
        inventory = lease.assert_inventory(snapshot.files)
        receipt = bind_capture_evidence_inventory(snapshot, inventory)

        assert receipt["file_count"] == 2
        assert receipt["directory_count"] == 2
        assert receipt["total_bytes"] == 12
        assert receipt["files"] == [
            {
                "path": "batch-a/frame-001/capture-008e.bin",
                "bytes": 5,
                "sha256": hashlib.sha256(b"table").hexdigest(),
            },
            {
                "path": "batch-a/frame-001/capture-preview.bin",
                "bytes": 7,
                "sha256": hashlib.sha256(b"preview").hexdigest(),
            },
        ]
    finally:
        lease.release()


def test_snapshot_rejects_symbolic_link(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    (root / "linked.bin").symlink_to(target)

    with pytest.raises(InventoryConflict, match="regular file"):
        snapshot_capture_evidence(root)


def test_inventory_binding_rejects_post_hash_mutation(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    lease = _lease(root)
    try:
        artifact = root / "journal.json"
        artifact.write_bytes(b"one")
        snapshot = snapshot_capture_evidence(root)
        artifact.write_bytes(b"two")
        inventory = lease.assert_inventory(snapshot.files)

        with pytest.raises(InventoryConflict, match="changed between hashing"):
            bind_capture_evidence_inventory(snapshot, inventory)
    finally:
        lease.release()


def test_snapshot_refuses_manifest_that_cannot_fit_in_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    (root / "capture-preview.bin").write_bytes(b"preview")
    monkeypatch.setattr(live_evidence, "_MAX_MANIFEST_BYTES", 1)

    with pytest.raises(InventoryConflict, match="receipt-safe size limit"):
        snapshot_capture_evidence(root)


def test_duplicate_json_keys_in_capture_evidence_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    session_path = batch / "session-journal.json"
    document = json.loads(session_path.read_text(encoding="utf-8"))
    body = json.dumps(document, sort_keys=True)
    assert body.startswith("{")
    duplicated = '{"batch_job_sha256": "' + "0" * 64 + '", ' + body[1:]
    assert json.loads(duplicated)["batch_job_sha256"] == document["batch_job_sha256"]
    session_path.write_text(duplicated, encoding="utf-8")

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match="repeats key"):
        validate_six_frame_batch_capture_evidence(snapshot)


def test_session_engine_identity_must_match_the_pinned_runtime(tmp_path: Path) -> None:
    from coolscanpy.protocol.ls5000_single_pass.bundle import CAPTURE_BUNDLE_SHA256

    root = tmp_path / "attempts"
    root.mkdir()
    completed = _write_completed_batch(root)
    batch = completed["batch"]
    assert isinstance(batch, Path)
    foreign = "f" * 64
    assert foreign != CAPTURE_BUNDLE_SHA256
    for name in ("session-journal.json", *(f"frame-{slot:03d}/journal.json" for slot in _SLOTS)):
        _rewrite_json(
            batch / name,
            lambda document: document.__setitem__("capture_bundle_sha256", foreign),
        )

    snapshot = snapshot_capture_evidence(root)

    with pytest.raises(InventoryConflict, match="completed six-frame batch"):
        validate_six_frame_batch_capture_evidence(snapshot)
