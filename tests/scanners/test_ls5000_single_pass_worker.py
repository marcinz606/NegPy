"""Hardware-free contracts for the packaged LS-5000 capture worker."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from negpy.infrastructure.scanners.ls5000_single_pass import worker as worker_module
from negpy.infrastructure.scanners.ls5000_single_pass.roll_index import (
    NativeFrameOrigin,
    TransportMapping,
)
from negpy.infrastructure.scanners.ls5000_single_pass.plan import (
    CANONICAL_PLAN_SHA256,
    load_canonical_plan,
)
from negpy.infrastructure.scanners.ls5000_single_pass.continuation_plan import (
    load_canonical_continuation_plan,
)
from negpy.infrastructure.scanners.ls5000_single_pass.worker import (
    DATA_PACKAGE,
    FRAME_TABLE_SEND_BYTES,
    FRAME_TABLE_SEND_RECORDS,
    ProtocolError,
    TransactionResult,
    _cleanup_synchronized,
    _bind_plan_to_live_selection,
    _derive_index_geometry,
    compile_continuation_steps,
    load_validated_batch_job,
    main,
    wait_for_parent_ack,
    _meter_controller_sha256,
    apply_boundary_offset,
    apply_batch_boundary_offsets,
    build_live_frame_table_payload,
)
from importlib.resources import files
from negpy.infrastructure.scanners.ls5000_single_pass.roll_index import TransportRecord
from negpy.infrastructure.scanners.ls5000_single_pass.window import decode_window_block


LIVE8_TRANSPORT_FIELDS = (
    (6202, 8, 22),
    (12250, 16, 22),
    (18326, 24, 26),
    (24360, 32, 24),
    (30436, 40, 28),
    (36484, 48, 28),
    (42532, 56, 28),
    (48608, 64, 32),
    (54614, 72, 26),
    (60676, 80, 28),
    (66710, 88, 26),
    (72758, 96, 26),
    (78778, 104, 22),
    (84896, 112, 32),
    (90916, 120, 28),
    (96992, 128, 32),
    (103026, 136, 30),
    (109046, 144, 26),
    (115136, 152, 32),
    (121156, 160, 28),
    (127176, 168, 24),
    (133280, 176, 32),
    (139286, 184, 26),
    (145362, 192, 30),
    (151396, 200, 28),
    (157416, 208, 24),
    (163464, 216, 24),
    (169526, 224, 26),
    (175560, 232, 24),
    (181636, 240, 28),
    (187684, 248, 28),
    (193746, 256, 30),
    (199794, 264, 30),
    (205842, 272, 30),
    (211904, 280, 32),
    (217924, 288, 28),
    (223958, 296, 260),
)


def test_frozen_worker_uses_pinned_meter_identity_without_loose_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_source = Path("/frozen/NegPy.app/Contents/Resources/meter.py")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        "negpy.infrastructure.scanners.ls5000_single_pass.worker.meter_module.__file__",
        str(missing_source),
    )


def test_continuation_compiler_emits_the_pinned_89_steps_without_session_commands() -> None:
    steps = compile_continuation_steps(
        load_canonical_plan(),
        load_canonical_continuation_plan(),
    )

    assert len(steps) == 89
    assert [step.code for step in steps] == [
        raw[0]
        for raw in load_canonical_continuation_plan()["trace_equivalence"][
            "semantic_steps"
        ]
    ]
    sequences = [entry["seq"] for step in steps for entry in step.entries]
    assert sequences[0] == 225
    assert sequences[-1] == 606
    assert 500 not in sequences
    assert not {
        "RESERVE_UNIT",
        "RELEASE_UNIT",
        "VENDOR_E0:EJECT",
        "SEND:sub_008f",
    }.intersection(entry["name"] for step in steps for entry in step.entries)
    assert [step.code for step in steps].count("R") == 15


def test_batch_job_loader_binds_ordered_frame_paths_and_parent_ack_contract(
    tmp_path: Path,
) -> None:
    session_id = "batch-slot17-slot19-session"
    job_path = tmp_path / "batch-job.json"
    job_path.write_text(
        json.dumps(
            {
                "apply_all_boundary_offsets_before_first_frame": True,
                "capture_plan_sha256": "a" * 64,
                "continuation_plan_sha256": "b" * 64,
                "frames": [
                    {
                        "ack": "frame-017/parent-ack.json",
                        "boundary_offset_rows": -12,
                        "journal": "frame-017/journal.json",
                        "output": "frame-017/capture.bin",
                        "slot": 17,
                    },
                    {
                        "ack": "frame-019/parent-ack.json",
                        "boundary_offset_rows": 8,
                        "journal": "frame-019/journal.json",
                        "output": "frame-019/capture.bin",
                        "slot": 19,
                    },
                ],
                "parent_ack_required_after_every_frame": True,
                "release_once_after_last_frame": True,
                "schema_version": 1,
                "session_contract": "one-process-one-reservation",
                "session_id": session_id,
            }
        ),
        encoding="utf-8",
    )

    job = load_validated_batch_job(
        job_path,
        expected_plan_sha256="a" * 64,
        expected_continuation_sha256="b" * 64,
    )

    assert job.session_id == session_id
    assert job.selected_slots == (17, 19)
    assert [frame.boundary_offset_rows for frame in job.frames] == [-12, 8]
    assert job.job_sha256 == hashlib.sha256(job_path.read_bytes()).hexdigest()
    assert job.frames[0].output == tmp_path / "frame-017" / "capture.bin"
    assert job.frames[0].journal == tmp_path / "frame-017" / "journal.json"
    assert job.frames[0].ack == tmp_path / "frame-017" / "parent-ack.json"


def test_parent_ack_is_bound_to_session_frame_slot_and_fresh_nonce(
    tmp_path: Path,
) -> None:
    ack_path = tmp_path / "parent-ack.json"
    ack_path.write_text(
        json.dumps(
            {
                "ack_nonce": "fresh-nonce-17",
                "action": "continue",
                "frame_index": 1,
                "schema_version": 1,
                "session_id": "batch-session",
                "slot": 17,
            }
        ),
        encoding="utf-8",
    )

    action = wait_for_parent_ack(
        ack_path,
        session_id="batch-session",
        frame_index=1,
        slot=17,
        nonce="fresh-nonce-17",
        timeout_seconds=0,
        poll_seconds=0,
    )

    assert action == "continue"

    assert _meter_controller_sha256() == (
        "a1ffc58195cfe5335b44e154b42f6e9ece7fd7ad647975c1c0f213b67651c81e"
    )


def test_batch_cli_dry_run_validates_one_session_without_single_frame_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = files(DATA_PACKAGE)
    plan_path = Path(data.joinpath("replay-first-rgbi4-plan.jsonl"))
    manifest_path = Path(data.joinpath("replay-first-rgbi4-manifest.json"))
    continuation_path = Path(data.joinpath("replay-next-rgbi4-plan.json"))
    job_path = tmp_path / "batch-job.json"
    job_path.write_text(
        json.dumps(
            {
                "apply_all_boundary_offsets_before_first_frame": True,
                "capture_plan_sha256": hashlib.sha256(
                    plan_path.read_bytes()
                ).hexdigest(),
                "continuation_plan_sha256": hashlib.sha256(
                    continuation_path.read_bytes()
                ).hexdigest(),
                "frames": [
                    {
                        "ack": "frame-017/parent-ack.json",
                        "boundary_offset_rows": -12,
                        "journal": "frame-017/journal.json",
                        "output": "frame-017/capture.bin",
                        "slot": 17,
                    },
                    {
                        "ack": "frame-019/parent-ack.json",
                        "boundary_offset_rows": 8,
                        "journal": "frame-019/journal.json",
                        "output": "frame-019/capture.bin",
                        "slot": 19,
                    },
                ],
                "parent_ack_required_after_every_frame": True,
                "release_once_after_last_frame": True,
                "schema_version": 1,
                "session_contract": "one-process-one-reservation",
                "session_id": "batch-dry-run",
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "--batch-job",
            str(job_path),
            "--plan",
            str(plan_path),
            "--continuation-plan",
            str(continuation_path),
            "--manifest",
            str(manifest_path),
            "--session-journal",
            str(tmp_path / "session-journal.json"),
        ]
    )

    output = capsys.readouterr().out
    assert "validated RGBI4x batch" in output
    assert "slots 17, 19" in output
    assert "dry run only; scanner was not accessed" in output
    assert not (tmp_path / "frame-017").exists()


def test_synchronized_cleanup_receipt_records_the_single_release_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[tuple[object, object]] = []

    def release(ep_out: object, ep_in: object) -> TransactionResult:
        releases.append((ep_out, ep_in))
        return TransactionResult(1, b"", bytes.fromhex("0000000000000000"), "000000", 2)

    monkeypatch.setattr(
        "negpy.infrastructure.scanners.ls5000_single_pass.worker._release_unit",
        release,
    )

    receipt = _cleanup_synchronized(
        "out",
        "in",
        scan_active=False,
        ready_required=False,
        reserved=True,
    )

    assert releases == [("out", "in")]
    assert receipt == {
        "attempted": True,
        "complete": True,
        "release_attempted": True,
        "release_status": "0000000000000000",
        "release_succeeded": True,
    }


def test_live_batch_connect_failure_records_no_reservation_and_no_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "batch-connect-failure"
    root.mkdir()
    frame = worker_module.BatchFrameSpec(
        slot=17,
        boundary_offset_rows=0,
        output=root / "frame-017" / "capture.bin",
        journal=root / "frame-017" / "journal.json",
        ack=root / "frame-017" / "parent-ack.json",
    )
    frame.output.parent.mkdir()
    batch = worker_module.LiveBatchJob(
        session_id="batch-connect-failure",
        root=root,
        frames=(frame,),
        plan_sha256=CANONICAL_PLAN_SHA256,
        continuation_plan_sha256=(
            worker_module.CANONICAL_CONTINUATION_PLAN_SHA256
        ),
        job_sha256="c" * 64,
    )
    session_journal = root / "session-journal.json"
    monkeypatch.setattr(
        worker_module,
        "_connect_device",
        lambda: (_ for _ in ()).throw(ProtocolError("USB device absent")),
    )

    with pytest.raises(ProtocolError, match="USB device absent"):
        worker_module.run_live_capture(
            load_canonical_plan(),
            tmp_path / "plan.jsonl",
            CANONICAL_PLAN_SHA256,
            frame.output,
            frame.journal,
            worker_module.EXPECTED_FINE_READS,
            frame=17,
            boundary_offset_rows=0,
            batch_job=batch,
            continuation_plan=load_canonical_continuation_plan(),
            continuation_plan_sha256=(
                worker_module.CANONICAL_CONTINUATION_PLAN_SHA256
            ),
            session_journal_path=session_journal,
        )

    receipt = json.loads(session_journal.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["reservation_acquired"] is False
    assert receipt["unit_release_attempts"] == 0
    assert receipt["unit_released"] is False
    assert receipt["recovery_required"] == "none"


def _mapping(fields: tuple[tuple[int, int, int], ...]) -> TransportMapping:
    origins = tuple(
        NativeFrameOrigin(
            frame=frame,
            boundary_index=frame - 1,
            boundary_output_row=frame * 144,
            lookup_row=frame * 144,
            code=code,
            selector=selector,
            native_origin=native_origin,
            method="test-fixture",
            automatic=True,
            manual_review=False,
            review_reasons=(),
            affine_residual_rows=0.0,
        )
        for frame, (native_origin, selector, code) in enumerate(fields, start=1)
    )
    return TransportMapping(
        record_count=len(origins),
        native_intercept=0.0,
        native_units_per_preview_row=42.0,
        anchor_mae_rows=0.0,
        anchor_max_error_rows=0.0,
        origins=origins,
    )


def test_live8_frame_table_is_the_exact_firmware_accepted_payload() -> None:
    payload = build_live_frame_table_payload(_mapping(LIVE8_TRANSPORT_FIELDS))
    send = next(entry for entry in load_canonical_plan() if entry["seq"] == 174)

    assert FRAME_TABLE_SEND_RECORDS == 37
    assert FRAME_TABLE_SEND_BYTES == 300
    assert send["cdb"] == "2a008f00000300012c00"
    assert len(payload) == FRAME_TABLE_SEND_BYTES
    assert payload[:4] == bytes.fromhex("012a2500")
    assert hashlib.sha256(payload).hexdigest() == "b78f6d8a1df1e0d5b242eda27eca88d121a6db2d2e64cf55ae9305142e39fc08"


def test_frame_table_refuses_to_guess_when_fewer_than_37_origins_are_proven() -> None:
    with pytest.raises(ProtocolError, match="fewer than 37"):
        build_live_frame_table_payload(_mapping(LIVE8_TRANSPORT_FIELDS[:36]))


def test_frame_table_ignores_advisory_slots_after_the_fixed_37_records() -> None:
    extra = (230000, 304, 24)
    payload = build_live_frame_table_payload(_mapping((*LIVE8_TRANSPORT_FIELDS, extra)))

    assert len(payload) == FRAME_TABLE_SEND_BYTES
    assert hashlib.sha256(payload).hexdigest() == "b78f6d8a1df1e0d5b242eda27eca88d121a6db2d2e64cf55ae9305142e39fc08"


def test_boundary_offset_resolves_raw_identity_from_the_same_transport_table() -> None:
    mapping = _mapping(LIVE8_TRANSPORT_FIELDS)
    records = tuple(
        TransportRecord(
            row=row,
            code=22 + 2 * (row % 4),
            selector=row,
            native_origin=756 * row + 7 * (22 + 2 * (row % 4)),
        )
        for row in range(6_000)
    )

    adjusted, selected = apply_boundary_offset(
        mapping,
        records,
        frame=18,
        offset_rows=-73,
    )

    assert selected.lookup_row == mapping.origins[17].lookup_row - 73
    source_record = records[selected.lookup_row]
    assert (selected.code, selected.selector, selected.native_origin) == (
        source_record.code,
        source_record.selector,
        source_record.native_origin,
    )
    assert adjusted.origins[17] is selected
    assert selected.automatic is True
    assert adjusted.origins[:17] == mapping.origins[:17]
    assert adjusted.origins[18:] == mapping.origins[18:]


def test_internal_window_decoder_reads_the_fields_the_worker_patches() -> None:
    payload = bytearray(58)
    payload[7] = 50
    payload[8] = 9
    payload[10:14] = bytes.fromhex("0fa00fa0")
    payload[14:18] = (12).to_bytes(4, "big")
    payload[18:22] = (109_060).to_bytes(4, "big")
    payload[22:26] = (3_946).to_bytes(4, "big")
    payload[26:30] = (5_959).to_bytes(4, "big")
    payload[34] = 16
    payload[48] = 0x30
    payload[50] = 0x01
    payload[51] = 0x10
    payload[54:58] = (120_000).to_bytes(4, "big")

    decoded = decode_window_block(payload)

    assert decoded is not None
    assert decoded["color_name"] == "IR"
    assert decoded["upper_left_y"] == 109_060
    assert decoded["width"] == 3_946
    assert decoded["height"] == 5_959
    assert decoded["samples_per_scan_minus1_nibble"] == 3
    assert decoded["is_multi_sample"] is True
    assert decoded["exposure_raw_10ns"] == 120_000
    assert decode_window_block(payload[:-1]) is None


def test_resolved_offset_is_encoded_into_the_selected_fixed_table_record() -> None:
    records = tuple(
        TransportRecord(
            row=row,
            code=6 * (row % 18),
            selector=row // 18,
            native_origin=42 * row,
        )
        for row in range(6_000)
    )
    lookup_rows = tuple(100 + 143 * index for index in range(37))
    origins = tuple(
        NativeFrameOrigin(
            frame=frame,
            boundary_index=frame - 1,
            boundary_output_row=row - 4,
            lookup_row=row,
            code=records[row].code,
            selector=records[row].selector,
            native_origin=records[row].native_origin,
            method="direct-gap-trailing-row",
            automatic=True,
            manual_review=False,
            review_reasons=(),
            affine_residual_rows=0.0,
        )
        for frame, row in enumerate(lookup_rows, start=1)
    )
    mapping = TransportMapping(6_000, 0.0, 42.0, 0.0, 0.0, origins)

    adjusted, selected = apply_boundary_offset(
        mapping,
        records,
        frame=18,
        offset_rows=10,
    )
    payload = build_live_frame_table_payload(adjusted)

    encoded = struct.unpack_from(">IHH", payload, 4 + 17 * 8)
    assert encoded == (selected.native_origin, selected.selector, selected.code)
    assert selected.lookup_row == lookup_rows[17] + 10


def test_batch_offsets_share_the_one_retained_table_and_later_frame_origins() -> None:
    records = tuple(
        TransportRecord(
            row=row,
            code=6 * (row % 18),
            selector=row // 18,
            native_origin=42 * row,
        )
        for row in range(6_000)
    )
    lookup_rows = tuple(100 + 143 * index for index in range(37))
    origins = tuple(
        NativeFrameOrigin(
            frame=frame,
            boundary_index=frame - 1,
            boundary_output_row=row - 4,
            lookup_row=row,
            code=records[row].code,
            selector=records[row].selector,
            native_origin=records[row].native_origin,
            method="direct-gap-trailing-row",
            automatic=True,
            manual_review=False,
            review_reasons=(),
            affine_residual_rows=0.0,
        )
        for frame, row in enumerate(lookup_rows, start=1)
    )
    mapping = TransportMapping(6_000, 0.0, 42.0, 0.0, 0.0, origins)

    combined, resolved = apply_batch_boundary_offsets(
        mapping,
        records,
        ((7, 9), (18, -11), (23, 6)),
    )

    assert [selected.lookup_row for _base, selected in resolved] == [
        lookup_rows[6] + 9,
        lookup_rows[17] - 11,
        lookup_rows[22] + 6,
    ]
    plan = load_canonical_plan()
    geometry = _derive_index_geometry(plan)
    bindings = []
    for slot in (7, 18, 23):
        selection = SimpleNamespace(
            frame=slot,
            frame_count=len(combined.origins),
            geometry=geometry,
            mapping=combined,
            selected=combined.origins[slot - 1],
        )
        bindings.append(_bind_plan_to_live_selection(plan, selection))

    # The first and only transmitted frame table already contains every
    # selected frame's operator-adjusted raw transport identity.
    retained_table = bytes.fromhex(bindings[0][173]["data_out"])
    for slot in (7, 18, 23):
        selected = combined.origins[slot - 1]
        assert struct.unpack_from(">IHH", retained_table, 4 + (slot - 1) * 8) == (
            selected.native_origin,
            selected.selector,
            selected.code,
        )

    # Every later continuation binds autofocus and all RGBI windows to the
    # exact same origin already stored in that retained table.
    for slot, bound in zip((7, 18, 23), bindings, strict=True):
        origin = combined.origins[slot - 1].native_origin
        autofocus = bytes.fromhex(bound[230]["data_out"])
        assert int.from_bytes(autofocus[5:9], "big") == origin + 2_979
        for sequence in (503, 504, 505, 506, 530, 531, 532, 533, 556, 557, 558, 559, 581, 582, 583, 584):
            window = decode_window_block(bytes.fromhex(bound[sequence - 1]["data_out"]))
            assert window is not None
            assert window["upper_left_y"] == origin


def test_continuation_executor_runs_all_89_steps_with_fake_usb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(
        TransportRecord(
            row=row,
            code=6 * (row % 18),
            selector=row // 18,
            native_origin=42 * row,
        )
        for row in range(6_000)
    )
    origins = tuple(
        NativeFrameOrigin(
            frame=frame,
            boundary_index=frame - 1,
            boundary_output_row=row - 4,
            lookup_row=row,
            code=records[row].code,
            selector=records[row].selector,
            native_origin=records[row].native_origin,
            method="direct-gap-trailing-row",
            automatic=True,
            manual_review=False,
            review_reasons=(),
            affine_residual_rows=0.0,
        )
        for frame, row in enumerate(
            (100 + 143 * index for index in range(37)),
            start=1,
        )
    )
    mapping = TransportMapping(6_000, 0.0, 42.0, 0.0, 0.0, origins)
    combined, _resolved = apply_batch_boundary_offsets(
        mapping,
        records,
        ((7, 9), (18, -11)),
    )
    plan = load_canonical_plan()
    geometry = _derive_index_geometry(plan)
    selection = SimpleNamespace(
        frame=18,
        frame_count=len(combined.origins),
        geometry=geometry,
        mapping=combined,
        selected=combined.origins[17],
        requested_boundary_offset_rows=-11,
        applied_boundary_offset_rows=-11,
        diagnostics=lambda: {"frame": 18, "prevalidated": True},
    )
    root = tmp_path / "continuation"
    root.mkdir()
    first = worker_module.BatchFrameSpec(
        7,
        9,
        root / "frame-007" / "capture.bin",
        root / "frame-007" / "journal.json",
        root / "frame-007" / "parent-ack.json",
    )
    second = worker_module.BatchFrameSpec(
        18,
        -11,
        root / "frame-018" / "capture.bin",
        root / "frame-018" / "journal.json",
        root / "frame-018" / "parent-ack.json",
    )
    batch = worker_module.LiveBatchJob(
        "continuation-session",
        root,
        (first, second),
        CANONICAL_PLAN_SHA256,
        worker_module.CANONICAL_CONTINUATION_PLAN_SHA256,
        "c" * 64,
    )
    canonical_target = worker_module.validate_plan(plan)
    tiny_target = {
        **canonical_target,
        "repeat": 1,
        "request_len": 1,
        "request_parts": [1],
    }
    monkeypatch.setattr(worker_module, "EXPECTED_FINE_READS", 1)
    monkeypatch.setattr(worker_module, "METER_GROUP_BYTES", 5)
    monkeypatch.setattr(worker_module, "METER_CAPTURE_BYTES", 15)
    monkeypatch.setattr(worker_module, "validate_plan", lambda _plan: tiny_target)
    ready_groups: list[tuple[int, ...]] = []
    transactions: list[int] = []

    def ready(
        _ep_out: object,
        _ep_in: object,
        entries: list[dict],
        **_kwargs: object,
    ) -> tuple[int, int]:
        ready_groups.append(tuple(entry["seq"] for entry in entries))
        return 1, 0

    def perform(
        _ep_out: object,
        _ep_in: object,
        entry: dict,
        **_kwargs: object,
    ) -> TransactionResult:
        sequence = entry["seq"]
        transactions.append(sequence)
        if sequence in (
            *worker_module.METER_GET_WINDOW_SEQUENCES,
            *worker_module.FINE_GET_WINDOW_SEQUENCES,
        ):
            payload = bytes.fromhex(entry["expected_data_in"])
        elif sequence in worker_module.METER_READ_SEQUENCES or sequence == 607:
            payload = b"x"
        else:
            payload = bytes.fromhex(entry.get("expected_data_in", ""))
        return TransactionResult(
            phase=entry.get("expected_phase", 1),
            payload=payload,
            status=bytes.fromhex(entry.get("expected_status", "00" * 8)),
            sense=entry.get("expected_sense", "000000"),
            stall_recoveries=0,
        )

    accepted_proposal = SimpleNamespace(
        accepted=True,
        proposed_exposures=dict(worker_module.DEFAULT_EXPOSURES),
        refusals=(),
        to_dict=lambda: {"accepted": True},
    )
    accepted_final = SimpleNamespace(
        accepted=True,
        final_exposures=dict(worker_module.DEFAULT_EXPOSURES),
        refusals=(),
        to_dict=lambda: {"accepted": True},
    )
    monkeypatch.setattr(worker_module, "_perform_ready_group", ready)
    monkeypatch.setattr(worker_module, "_perform_with_busy_retry", perform)
    monkeypatch.setattr(
        worker_module,
        "observe_meter_pass",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        worker_module,
        "propose_next_exposures",
        lambda *_args, **_kwargs: accepted_proposal,
    )
    monkeypatch.setattr(
        worker_module,
        "verify_final_convergence",
        lambda *_args, **_kwargs: accepted_final,
    )
    monkeypatch.setattr(
        worker_module,
        "_wait_post_scan_ready",
        lambda *_args, **_kwargs: (1, 0),
    )

    journal = worker_module._run_live_continuation_frame(
        "out",
        "in",
        plan,
        tmp_path / "plan.jsonl",
        CANONICAL_PLAN_SHA256,
        load_canonical_continuation_plan(),
        worker_module.CANONICAL_CONTINUATION_PLAN_SHA256,
        second,
        selection,
        batch_job=batch,
        frame_index=2,
        lifecycle=worker_module.SessionLifecycle(),
    )

    assert len(ready_groups) == 15
    assert len(transactions) - 1 + len(ready_groups) == 89
    assert 174 not in transactions
    assert 500 not in transactions
    assert transactions[-1] == 607
    assert journal["status"] == "frame-complete"
    assert journal["frame_complete"] is True
    assert journal["session_reservation_retained"] is True
    assert journal["unit_released"] is False
    assert second.output.read_bytes() == b"x"
    assert second.journal.read_text(encoding="utf-8") == (
        json.dumps(journal, indent=2, sort_keys=True) + "\n"
    )


def test_live_two_frame_batch_uses_one_combined_table_and_one_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = load_canonical_plan()
    canonical_target = worker_module.validate_plan(plan)
    tiny_target = {
        **canonical_target,
        "repeat": 1,
        "request_len": 1,
        "request_parts": [1],
    }
    for sequence in worker_module.PREVIEW_READ_SEQUENCES:
        entry = plan[sequence - 1]
        entry["request_len"] = 1
        entry["request_parts"] = [1]

    records = tuple(
        TransportRecord(
            row=row,
            code=6 * (row % 18),
            selector=row // 18,
            native_origin=42 * row,
        )
        for row in range(6_000)
    )
    origins = tuple(
        NativeFrameOrigin(
            frame=frame,
            boundary_index=frame - 1,
            boundary_output_row=row - 4,
            lookup_row=row,
            code=records[row].code,
            selector=records[row].selector,
            native_origin=records[row].native_origin,
            method="direct-gap-trailing-row",
            automatic=True,
            manual_review=False,
            review_reasons=(),
            affine_residual_rows=0.0,
        )
        for frame, row in enumerate(
            (100 + 143 * index for index in range(37)),
            start=1,
        )
    )
    base_mapping = TransportMapping(6_000, 0.0, 42.0, 0.0, 0.0, origins)
    combined, resolved = apply_batch_boundary_offsets(
        base_mapping,
        records,
        ((7, 9), (18, -11)),
    )
    geometry = SimpleNamespace(
        requested_resolution=97,
        native_resolution=4_000,
        pitch=41,
        native_width=3_946,
        native_height=250_278,
        width=96,
        height=len(worker_module.PREVIEW_READ_SEQUENCES),
        block_bytes=1,
        expected_stream_bytes=len(worker_module.PREVIEW_READ_SEQUENCES),
    )

    def selection_for(
        slot: int,
        offset: int,
        pair: tuple[NativeFrameOrigin, NativeFrameOrigin],
    ) -> SimpleNamespace:
        base, selected = pair
        return SimpleNamespace(
            frame=slot,
            frame_count=len(combined.origins),
            geometry=geometry,
            mapping=combined,
            base_selected=base,
            selected=selected,
            requested_boundary_offset_rows=offset,
            applied_boundary_offset_rows=offset,
            diagnostics=lambda: {
                "frame": slot,
                "boundary_offset": {
                    "requested_rows": offset,
                    "applied_rows": offset,
                },
                "selected": {
                    "frame": slot,
                    "lookup_row": selected.lookup_row,
                    "native_origin": selected.native_origin,
                },
            },
        )

    selections = (
        selection_for(7, 9, resolved[0]),
        selection_for(18, -11, resolved[1]),
    )
    root = tmp_path / "successful-batch"
    root.mkdir()
    first = worker_module.BatchFrameSpec(
        7,
        9,
        root / "frame-007" / "capture.bin",
        root / "frame-007" / "journal.json",
        root / "frame-007" / "parent-ack.json",
    )
    second = worker_module.BatchFrameSpec(
        18,
        -11,
        root / "frame-018" / "capture.bin",
        root / "frame-018" / "journal.json",
        root / "frame-018" / "parent-ack.json",
    )
    first.output.parent.mkdir()
    batch = worker_module.LiveBatchJob(
        "successful-batch",
        root,
        (first, second),
        CANONICAL_PLAN_SHA256,
        worker_module.CANONICAL_CONTINUATION_PLAN_SHA256,
        "c" * 64,
    )
    nonce = "offline-batch-nonce"
    first.ack.write_text(
        json.dumps(
            {
                "ack_nonce": nonce,
                "action": "continue",
                "frame_index": 1,
                "schema_version": 1,
                "session_id": batch.session_id,
                "slot": first.slot,
            }
        ),
        encoding="utf-8",
    )

    startup = bytearray(10 + 37 * 8)
    startup[:4] = b"\x8f\0\0\0"
    startup[4:6] = (len(startup) - 6).to_bytes(2, "big")
    startup[6:8] = (len(startup) - 8).to_bytes(2, "big")
    startup[8] = 37
    header_8e = b"\0\x8e\0\0\0\x06"
    prevalidated = False
    reserves: list[int] = []
    sent_tables: list[bytes] = []
    transactions: list[int] = []
    fine_reads: list[int] = []
    ready_groups: list[tuple[int, ...]] = []
    ack_boundaries: list[tuple[int, int, int, int]] = []
    releases: list[tuple[object, object]] = []
    usb_events: list[str] = []

    class USBUtil:
        @staticmethod
        def release_interface(_device: object, _number: int) -> None:
            usb_events.append("interface-released")

        @staticmethod
        def dispose_resources(_device: object) -> None:
            usb_events.append("resources-disposed")

    ep_out = SimpleNamespace(bEndpointAddress=0x01)
    ep_in = SimpleNamespace(bEndpointAddress=0x82)
    interface = SimpleNamespace(bInterfaceNumber=0)

    def derive_batch(
        _plan: list[dict],
        preview: bytes,
        table: bytes,
        frames: tuple[worker_module.BatchFrameSpec, ...],
    ) -> tuple[SimpleNamespace, ...]:
        nonlocal prevalidated
        assert len(preview) == len(worker_module.PREVIEW_READ_SEQUENCES)
        assert table == header_8e
        assert frames == batch.frames
        prevalidated = True
        return selections

    def perform(
        _ep_out: object,
        _ep_in: object,
        entry: dict,
        **_kwargs: object,
    ) -> TransactionResult:
        sequence = entry["seq"]
        transactions.append(sequence)
        if sequence == 17:
            reserves.append(sequence)
        if sequence == 174:
            assert prevalidated is True
            sent_tables.append(bytes.fromhex(entry["data_out"]))
        if sequence in (115, 116, 117):
            payload = b"window"
        elif sequence in worker_module.PREVIEW_READ_SEQUENCES:
            payload = b"p"
        elif sequence == 171:
            payload = header_8e
        elif sequence == 172:
            payload = header_8e
        elif sequence in (
            *worker_module.METER_GET_WINDOW_SEQUENCES,
            *worker_module.FINE_GET_WINDOW_SEQUENCES,
        ):
            payload = bytes.fromhex(entry["expected_data_in"])
        elif sequence in worker_module.METER_READ_SEQUENCES:
            payload = b"m"
        elif sequence == 607:
            fine_reads.append(sequence)
            payload = b"f"
        else:
            payload = bytes.fromhex(entry.get("expected_data_in", ""))
        return TransactionResult(
            phase=entry.get("expected_phase", 1),
            payload=payload,
            status=bytes.fromhex(entry.get("expected_status", "00" * 8)),
            sense=entry.get("expected_sense", "000000"),
            stall_recoveries=0,
        )

    def perform_startup(
        _ep_out: object,
        _ep_in: object,
        entry: dict,
        **_kwargs: object,
    ) -> TransactionResult:
        assert entry["seq"] == worker_module.VARIABLE_FRAME_TABLE_SEQUENCE
        return TransactionResult(
            phase=3,
            payload=bytes(startup),
            status=bytes(8),
            sense="000000",
            stall_recoveries=0,
        )

    def ready(
        _ep_out: object,
        _ep_in: object,
        entries: list[dict],
        **_kwargs: object,
    ) -> tuple[int, int]:
        ready_groups.append(tuple(entry["seq"] for entry in entries))
        return 1, 0

    real_wait_for_parent_ack = worker_module.wait_for_parent_ack

    def acknowledge(
        path: Path,
        *,
        session_id: str,
        frame_index: int,
        slot: int,
        nonce: str,
        timeout_seconds: float = 1_800.0,
        poll_seconds: float = 0.1,
    ) -> str:
        del timeout_seconds, poll_seconds
        ack_boundaries.append(
            (frame_index, slot, len(transactions), len(ready_groups))
        )
        if frame_index == 2:
            path.write_text(
                json.dumps(
                    {
                        "ack_nonce": nonce,
                        "action": "continue",
                        "frame_index": frame_index,
                        "schema_version": 1,
                        "session_id": session_id,
                        "slot": slot,
                    }
                ),
                encoding="utf-8",
            )
        return real_wait_for_parent_ack(
            path,
            session_id=session_id,
            frame_index=frame_index,
            slot=slot,
            nonce=nonce,
            timeout_seconds=0,
            poll_seconds=0,
        )

    def release(ep_out_value: object, ep_in_value: object) -> TransactionResult:
        releases.append((ep_out_value, ep_in_value))
        return TransactionResult(1, b"", bytes(8), "000000", 0)

    preview_windows = [
        {
            "color_id": color,
            "resx": 97,
            "resy": 97,
            "upper_left_x": 0,
            "upper_left_y": 0,
            "width": 3_946,
            "height": 250_278,
            "bit_depth": 16,
        }
        for color in (1, 2, 3)
    ]
    accepted_proposal = SimpleNamespace(
        accepted=True,
        proposed_exposures=dict(worker_module.DEFAULT_EXPOSURES),
        refusals=(),
        to_dict=lambda: {"accepted": True},
    )
    accepted_final = SimpleNamespace(
        accepted=True,
        final_exposures=dict(worker_module.DEFAULT_EXPOSURES),
        refusals=(),
        to_dict=lambda: {"accepted": True},
    )
    monkeypatch.setattr(worker_module, "EXPECTED_FINE_READS", 1)
    monkeypatch.setattr(
        worker_module,
        "EXPECTED_PREVIEW_BYTES",
        len(worker_module.PREVIEW_READ_SEQUENCES),
    )
    monkeypatch.setattr(worker_module, "METER_GROUP_BYTES", 5)
    monkeypatch.setattr(worker_module, "METER_CAPTURE_BYTES", 15)
    monkeypatch.setattr(worker_module, "validate_plan", lambda _plan: tiny_target)
    monkeypatch.setattr(worker_module, "_derive_index_geometry", lambda _plan: geometry)
    monkeypatch.setattr(worker_module, "_validate_scanner_identity", lambda _payload: None)
    monkeypatch.setattr(worker_module, "_validate_live_preview_windows", lambda *_args: preview_windows)
    monkeypatch.setattr(worker_module, "_derive_live_batch_selections", derive_batch)
    monkeypatch.setattr(worker_module, "_perform_with_busy_retry", perform)
    monkeypatch.setattr(worker_module, "_perform_variable_frame_table_transaction", perform_startup)
    monkeypatch.setattr(worker_module, "_perform_ready_group", ready)
    monkeypatch.setattr(worker_module, "_wait_post_scan_ready", lambda *_args, **_kwargs: (1, 0))
    monkeypatch.setattr(worker_module, "observe_meter_pass", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(worker_module, "propose_next_exposures", lambda *_args, **_kwargs: accepted_proposal)
    monkeypatch.setattr(worker_module, "verify_final_convergence", lambda *_args, **_kwargs: accepted_final)
    monkeypatch.setattr(worker_module, "wait_for_parent_ack", acknowledge)
    monkeypatch.setattr(worker_module, "_release_unit", release)
    monkeypatch.setattr(worker_module.secrets, "token_hex", lambda _size: nonce)
    monkeypatch.setattr(
        worker_module,
        "_connect_device",
        lambda: (object(), interface, ep_out, ep_in, USBUtil),
    )

    session_journal_path = root / "session-journal.json"
    worker_module.run_live_capture(
        plan,
        tmp_path / "plan.jsonl",
        CANONICAL_PLAN_SHA256,
        first.output,
        first.journal,
        1,
        frame=first.slot,
        boundary_offset_rows=first.boundary_offset_rows,
        batch_job=batch,
        continuation_plan=load_canonical_continuation_plan(),
        continuation_plan_sha256=(
            worker_module.CANONICAL_CONTINUATION_PLAN_SHA256
        ),
        session_journal_path=session_journal_path,
    )

    assert reserves == [17]
    assert len(sent_tables) == 1
    retained_table = sent_tables[0]
    for slot in (7, 18):
        origin = combined.origins[slot - 1]
        assert struct.unpack_from(">IHH", retained_table, 4 + (slot - 1) * 8) == (
            origin.native_origin,
            origin.selector,
            origin.code,
        )
    assert fine_reads == [607, 607]
    assert [(frame, slot) for frame, slot, _tx, _ready in ack_boundaries] == [
        (1, 7),
        (2, 18),
    ]
    first_boundary = ack_boundaries[0]
    second_boundary = ack_boundaries[1]
    continuation_transactions = (
        second_boundary[2] - first_boundary[2]
    )
    continuation_ready_groups = second_boundary[3] - first_boundary[3]
    assert continuation_transactions - 1 + continuation_ready_groups == 89
    assert 174 not in transactions[first_boundary[2] : second_boundary[2]]
    assert releases == [(ep_out, ep_in)]
    first_receipt = json.loads(first.journal.read_text(encoding="utf-8"))
    assert first_receipt["status"] == "frame-complete"
    assert first_receipt["session_reservation_retained"] is True
    assert first_receipt["unit_released"] is False
    second_receipt = json.loads(second.journal.read_text(encoding="utf-8"))
    assert second_receipt["status"] == "frame-complete"
    assert second_receipt["batch_session"] == {
        "frame_index": 2,
        "frame_total": 2,
        "selected_slots": [7, 18],
        "session_id": batch.session_id,
    }
    assert second_receipt["continuation_plan_sha256"] == (
        worker_module.CANONICAL_CONTINUATION_PLAN_SHA256
    )
    assert second_receipt["session_reservation_retained"] is True
    assert second_receipt["unit_released"] is False
    assert second.output.read_bytes() == b"f"
    session = json.loads(session_journal_path.read_text(encoding="utf-8"))
    assert session["status"] == "complete"
    assert session["completed_slots"] == [7, 18]
    assert session["reservation_acquired"] is True
    assert session["unit_release_attempts"] == 1
    assert session["unit_released"] is True
    assert session["recovery_required"] == "none"
    assert session["batch_job_sha256"] == batch.job_sha256
    assert session["capture_engine_sha256"] == worker_module.CAPTURE_WORKER_SHA256
    assert session["capture_bundle_sha256"] == worker_module.CAPTURE_BUNDLE_SHA256
    assert usb_events == ["interface-released", "resources-disposed"]
