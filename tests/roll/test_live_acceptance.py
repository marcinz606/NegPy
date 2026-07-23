from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from negpy.infrastructure.roll.repair import RepairMode
from negpy.services.roll.exact_color import PositiveColorMode
from negpy.services.roll.live_acceptance import (
    LiveAcceptanceError,
    LiveAcceptanceRequest,
    run_live_acceptance,
)
from negpy.services.roll.live_reservation import FixedOutputLease, OUTPUT_LOCK_NAME
from negpy.services.roll.live_review import (
    REVIEW_BASIS,
    SCHEMA as REVIEW_SCHEMA,
    ValidatedReviewedApproval,
    thumbnail_sha256,
)
from negpy.services.roll.service import RollFrameOutput


_SLOTS = (1, 2, 3, 4, 5, 6)
_APPROVED_SLOTS = (1, 6)
_OUTPUT_FIELDS = (
    "rgb_path",
    "ir_path",
    "repaired_rgb_path",
    "repaired_ir_path",
    "positive_path",
    "receipt_path",
    "synthesis_mask_path",
    "native_synthesis_mask_path",
    "hybrid_receipt_path",
)
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


class _Runtime:
    def __init__(self) -> None:
        self.validated = False

    def validate_files(self) -> None:
        self.validated = True


@dataclasses.dataclass(frozen=True)
class _Approval:
    payload: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


@dataclasses.dataclass(frozen=True)
class _AcceptanceFixture:
    request: LiveAcceptanceRequest
    review: ValidatedReviewedApproval
    thumbnails: dict[int, np.ndarray]


def _json_sha256(document: object) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _approval(slot: int, image: np.ndarray, fingerprint: str) -> _Approval:
    payload: dict[str, object] = {
        "boundary_offset_rows": 0,
        "review_reasons": [f"slot-{slot}-edge-review"],
        "reviewed_fingerprint_sha256": fingerprint,
        "reviewed_lookup_row": (slot - 1) * 143,
        "reviewed_native_origin": (slot - 1) * 5_959,
        "schema_version": 1,
        "slot": slot,
        "thumbnail_sha256": thumbnail_sha256(image),
    }
    return _Approval({**payload, "binding_sha256": _json_sha256(payload)})


def _fixture(
    tmp_path: Path,
    *,
    confirm_live: bool = True,
) -> _AcceptanceFixture:
    session_path = tmp_path / "preview-session.json"
    session_path.write_text('{"version":1}', encoding="utf-8")
    session_bytes = session_path.read_bytes()

    thumbnails = {slot: (np.arange(60, dtype=np.uint16).reshape(4, 5, 3) + np.uint16(slot * 1_000)) for slot in _SLOTS}
    fingerprint = "c" * 64
    approvals = {slot: _approval(slot, thumbnails[slot], fingerprint) for slot in _APPROVED_SLOTS}

    contact_path = tmp_path / "reviewed-contact-sheet.png"
    contact = np.concatenate(
        tuple((thumbnails[slot] >> 8).astype(np.uint8) for slot in _SLOTS),
        axis=1,
    )
    Image.fromarray(contact).save(contact_path)
    contact_bytes = contact_path.read_bytes()

    review_document = {
        "approvals": [approvals[slot].to_payload() for slot in _APPROVED_SLOTS],
        "contact_sheet": {
            "bytes": len(contact_bytes),
            "path": str(contact_path),
            "sha256": hashlib.sha256(contact_bytes).hexdigest(),
        },
        "preview_session": {
            "bytes": len(session_bytes),
            "path": str(session_path),
            "sha256": hashlib.sha256(session_bytes).hexdigest(),
        },
        "review_basis": REVIEW_BASIS,
        "reviewed_fingerprint_sha256": fingerprint,
        "schema": REVIEW_SCHEMA,
    }
    review_path = tmp_path / "reviewed-approval.json"
    review_path.write_text(
        json.dumps(
            review_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    review_bytes = review_path.read_bytes()
    review = ValidatedReviewedApproval(
        sha256=hashlib.sha256(review_bytes).hexdigest(),
        byte_length=len(review_bytes),
        reviewed_fingerprint_sha256=fingerprint,
        contact_sheet_path=contact_path,
        contact_sheet_sha256=hashlib.sha256(contact_bytes).hexdigest(),
        approvals=approvals,
    )

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    attempts_root = tmp_path / "scanner-attempts"
    attempts_root.mkdir()
    request = LiveAcceptanceRequest(
        device_id="usb:2:7",
        preview_session_path=session_path,
        preview_session_sha256=hashlib.sha256(session_bytes).hexdigest(),
        reviewed_approval_path=review_path,
        reviewed_approval_sha256=review.sha256,
        output_dir=output_dir,
        run_receipt_path=tmp_path / "run-receipt.json",
        attempts_root_path=attempts_root,
        confirm_live=confirm_live,
    )
    return _AcceptanceFixture(
        request=request,
        review=review,
        thumbnails=thumbnails,
    )


def _write_sparse_zeros(path: Path, size: int) -> None:
    with path.open("xb") as stream:
        stream.truncate(size)


def _write_completed_attempt_evidence(
    attempts_root: Path,
    *,
    reviewed_fingerprint: str,
    approvals: dict[int, _Approval],
) -> Path:
    """Mirror Coolscan's durable batch tree after fine-stream cleanup."""

    batch = attempts_root / "batch-slot01-slot06-test"
    batch.mkdir()
    session_id = batch.name
    fresh_fingerprint = "e" * 64
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

    approval_payloads = {slot: approval.to_payload() for slot, approval in approvals.items()}
    frames = [
        {
            "ack": f"frame-{slot:03d}/parent-ack.json",
            "boundary_offset_rows": 0,
            "journal": f"frame-{slot:03d}/journal.json",
            "manual_review_approval": approval_payloads.get(slot),
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
        "reviewed_roll_fingerprint": {
            "binding_sha256": reviewed_fingerprint,
        },
        "schema_version": 3,
        "session_id": session_id,
        "session_contract": "one-process-one-reservation",
    }
    job_bytes = json.dumps(
        job,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (batch / "batch-job.json").write_bytes(job_bytes)
    job_sha256 = hashlib.sha256(job_bytes).hexdigest()
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
                    str(slot): (None if slot not in approval_payloads else approval_payloads[slot]["binding_sha256"]) for slot in _SLOTS
                },
                "reviewed_roll_fingerprint_sha256": reviewed_fingerprint,
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

    for frame_index, slot in enumerate(_SLOTS, start=1):
        frame = batch / f"frame-{slot:03d}"
        frame.mkdir()
        meter = frame / "capture-meter.bin"
        _write_sparse_zeros(meter, _METER_BYTES)
        nonce = f"{slot:032x}"
        selection = {
            "frame": slot,
            "requested_boundary_offset_rows": 0,
            "applied_boundary_offset_rows": 0,
            "roll_identity": {
                "reviewed_fingerprint_sha256": reviewed_fingerprint,
                "fresh_fingerprint_sha256": fresh_fingerprint,
                "comparison": {"matches": True, "reason": "matched"},
                "selected_slot_comparison": {
                    "matches": True,
                    "reason": "matched",
                    "slot": slot,
                },
            },
        }
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
            "output_sha256": hashlib.sha256(f"capture-{slot}".encode()).hexdigest(),
            "plan_sha256": plan_sha256,
            "continuation_plan_sha256": continuation_sha256,
            "capture_engine_sha256": engine_sha256,
            "capture_bundle_sha256": bundle_sha256,
            "manual_review_approval": approval_payloads.get(slot),
            "reviewed_roll_fingerprint_sha256": reviewed_fingerprint,
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
    return batch


class _Service:
    def __init__(
        self,
        fixture: _AcceptanceFixture,
        *,
        restored_slots: tuple[int, ...] = _SLOTS,
        approval_slots: tuple[int, ...] = _APPROVED_SLOTS,
        wrong_approval_slot: int | None = None,
        scan_error: Exception | None = None,
        write_error_slot: int | None = None,
    ) -> None:
        self.output_dir = fixture.request.output_dir
        self.thumbnails = fixture.thumbnails
        self.approvals = fixture.review.approvals
        self.reviewed_fingerprint = fixture.review.reviewed_fingerprint_sha256
        self.restored_slots = restored_slots
        self.approval_slots = approval_slots
        self.wrong_approval_slot = wrong_approval_slot
        self.scan_error = scan_error
        self.write_error_slot = write_error_slot
        self.calls: list[object] = []
        self.approved: set[int] = set()
        self.closed = False
        self.safe_stop_calls = 0
        self.eject_calls = 0

    def open_roll(self, device_id: str, *, attempts_root=None) -> None:
        self.calls.append(("open_roll", device_id, attempts_root))
        _write_completed_attempt_evidence(
            Path(attempts_root),
            reviewed_fingerprint=self.reviewed_fingerprint,
            approvals=self.approvals,
        )

    def restore_preview_session(self, payload: str, slots=None):
        self.calls.append(("restore_preview_session", payload, slots))
        return [
            SimpleNamespace(
                slot=slot,
                image=np.array(
                    self.thumbnails.get(
                        slot,
                        np.full((4, 5, 3), slot, dtype=np.uint16),
                    ),
                    copy=True,
                ),
            )
            for slot in self.restored_slots
        ]

    def needs_approval(self, slot: int) -> bool:
        return slot in self.approval_slots

    def approve(self, slot: int) -> object:
        self.calls.append(("approve", slot))
        self.approved.add(slot)
        approval = self.approvals[slot]
        if slot != self.wrong_approval_slot:
            return approval
        wrong = approval.to_payload()
        wrong["binding_sha256"] = "0" * 64
        return _Approval(wrong)

    def prepare_batch(self) -> None:
        self.calls.append("prepare_batch")

    def scan_many(self, slots, *, on_progress=None):
        ordered = tuple(slots)
        self.calls.append(("scan_many", ordered))
        if self.scan_error is not None:
            raise self.scan_error
        for index, slot in enumerate(ordered, start=1):
            if on_progress is not None:
                on_progress(
                    SimpleNamespace(
                        stage="fine-scan",
                        slot=slot,
                        index=index,
                        total=len(ordered),
                        fraction=index / len(ordered),
                        message=f"slot {slot} complete",
                    )
                )
            yield SimpleNamespace(slot=slot)

    def write_frame(
        self,
        frame,
        output_folder: str,
        filename_pattern: str,
        **kwargs,
    ) -> RollFrameOutput:
        self.calls.append(("write_frame", frame.slot, output_folder, filename_pattern, kwargs))
        on_repair_progress = kwargs.get("on_repair_progress")
        if callable(on_repair_progress):
            on_repair_progress(0.5)
        if frame.slot == self.write_error_slot:
            raise RuntimeError(f"write failed for slot {frame.slot}")
        base = self.output_dir / f"acceptance_slot{frame.slot:02d}"
        paths = {
            "rgb_path": base.with_suffix(".tif"),
            "ir_path": base.with_name(base.name + "_IR.tif"),
            "repaired_rgb_path": base.with_name(base.name + "_repaired.tif"),
            "repaired_ir_path": base.with_name(base.name + "_repaired_IR.tif"),
            "positive_path": base.with_name(base.name + "_positive.tif"),
            "synthesis_mask_path": base.with_name(base.name + "_repaired_SYNTH.png"),
            "native_synthesis_mask_path": base.with_name(base.name + "_native_SYNTH.png"),
            "hybrid_receipt_path": base.with_name(base.name + "_hybrid_receipt.json"),
        }
        for path in paths.values():
            path.write_bytes(b"artifact")
        receipt_path = base.with_name(base.name + "_receipt.json")
        receipt_path.write_text(
            json.dumps(
                {
                    "depth": 16,
                    "dpi": 4_000,
                    "slot": frame.slot,
                    "outputs": {
                        "unrepaired": {"written": True},
                        "repair_acquisition_evidence": {
                            "retained": True,
                            "replayable": True,
                        },
                        "native_color_evidence": {"retained": True},
                        "repaired": {
                            "written": True,
                            "mode_requested": "hybrid",
                            "mode_resolved": "hybrid",
                            "degraded": False,
                        },
                        "positive": {
                            "written": True,
                            "color_mode": "nikon-exact",
                            "exact_nikon_color": True,
                            "native_per_acquisition_builder": True,
                            "builder_validated": True,
                            "cms_verified": True,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return RollFrameOutput(
            slot=frame.slot,
            receipt_path=str(receipt_path),
            **{name: str(path) for name, path in paths.items()},
        )

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True

    def safe_stop(self) -> None:
        self.safe_stop_calls += 1

    def eject(self) -> None:
        self.eject_calls += 1


class _ExplodingProgress:
    @property
    def stage(self) -> str:
        raise RuntimeError("progress extraction exploded")


class _UnrepresentableProgressValue:
    def __repr__(self) -> str:
        raise RuntimeError("progress repr exploded")


class _PoisonTelemetryService(_Service):
    def scan_many(self, slots, *, on_progress=None):
        ordered = tuple(slots)
        self.calls.append(("scan_many", ordered))
        for slot in ordered:
            if on_progress is not None:
                on_progress(_ExplodingProgress())
            yield SimpleNamespace(slot=slot)

    def write_frame(
        self,
        frame,
        output_folder: str,
        filename_pattern: str,
        **kwargs,
    ) -> RollFrameOutput:
        callback = kwargs.get("on_repair_progress")
        if callable(callback):
            callback(_UnrepresentableProgressValue())
        without_callback = dict(kwargs)
        without_callback["on_repair_progress"] = None
        return super().write_frame(
            frame,
            output_folder,
            filename_pattern,
            **without_callback,
        )


class _MutatingEvidenceLease:
    def __init__(self, delegate: FixedOutputLease, target: Path) -> None:
        self._delegate = delegate
        self._target = target

    @property
    def released(self) -> bool:
        return self._delegate.released

    def assert_inventory(self, *args, **kwargs):
        return self._delegate.assert_inventory(*args, **kwargs)

    def release(self) -> None:
        self._delegate.release()

    def release_verified(self, owned_files, *, previous, finalize):
        def finalize_then_mutate() -> None:
            finalize()
            self._target.write_bytes(b"mutated-after-receipt-publication")

        return self._delegate.release_verified(
            owned_files,
            previous=previous,
            finalize=finalize_then_mutate,
        )


def _mutating_evidence_lease_factory(target: Path):
    def acquire(root, lock_document, *, require_empty):
        return _MutatingEvidenceLease(
            FixedOutputLease.acquire(
                root,
                lock_document,
                require_empty=require_empty,
            ),
            target,
        )

    return acquire


def _output_paths(output: RollFrameOutput) -> list[Path]:
    paths = [Path(getattr(output, field)) for field in _OUTPUT_FIELDS]
    assert all(path.is_absolute() and path.is_file() for path in paths)
    return sorted(paths)


class _ValidationHarness:
    def __init__(self, service: _Service) -> None:
        self.service = service
        self.inventory_closed_states: list[bool] = []
        self.batch_closed_states: list[bool] = []

    def collect(
        self,
        output: RollFrameOutput,
        *,
        output_dir: Path,
        expected_slot: int,
    ) -> list[str]:
        assert output.slot == expected_slot
        assert Path(output_dir) == self.service.output_dir.resolve()
        self.inventory_closed_states.append(self.service.closed)
        return [str(path) for path in _output_paths(output)]

    def validate_batch(
        self,
        outputs,
        **kwargs,
    ) -> dict[str, Any]:
        self.batch_closed_states.append(self.service.closed)
        assert self.service.closed is True
        output_dir = Path(kwargs["output_dir"])
        assert output_dir == self.service.output_dir.resolve()
        assert kwargs["allowed_output_lock_name"] == OUTPUT_LOCK_NAME
        ordered = tuple(outputs)
        assert tuple(output.slot for output in ordered) == _SLOTS
        frames = []
        all_paths: set[Path] = set()
        for output in ordered:
            paths = _output_paths(output)
            all_paths.update(paths)
            receipt_path = Path(output.receipt_path)
            receipt_bytes = receipt_path.read_bytes()
            frames.append(
                {
                    "slot": output.slot,
                    "frame_receipt": {
                        "path": str(receipt_path),
                        "bytes": len(receipt_bytes),
                        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                    },
                    "referenced_file_count": len(paths),
                    "referenced_files": [str(path) for path in paths],
                }
            )
        referenced = [str(path) for path in sorted(all_paths)]
        approval_payloads = {slot: self.service.approvals[slot].to_payload() for slot in _APPROVED_SLOTS}
        return {
            "schema": "negpy.ls5000-deep-acceptance.v1",
            "status": "passed",
            "scope": "six-frame-batch",
            "output_dir": str(output_dir),
            "slots": list(_SLOTS),
            "approved_slots": list(_APPROVED_SLOTS),
            "device_id": "usb:2:7",
            "device_model": "Nikon LS-5000 ED 1.03",
            "reviewed_fingerprint_sha256": approval_payloads[1]["reviewed_fingerprint_sha256"],
            "manual_approval_bindings": [
                {
                    "slot": slot,
                    "binding_sha256": approval_payloads[slot]["binding_sha256"],
                }
                for slot in _APPROVED_SLOTS
            ],
            "frames": frames,
            "referenced_files": referenced,
            "inventory": {
                "regular_file_count": len(referenced) + 1,
                "visible_file_count": len(referenced),
                "allowed_output_lock_path": str(output_dir / OUTPUT_LOCK_NAME),
                "exact": True,
            },
        }


def _review_loader(fixture: _AcceptanceFixture):
    def load(request: LiveAcceptanceRequest, session) -> ValidatedReviewedApproval:
        assert request.reviewed_approval_path == fixture.request.reviewed_approval_path
        assert request.reviewed_approval_sha256 == fixture.review.sha256
        assert session.sha256 == fixture.request.preview_session_sha256
        return fixture.review

    return load


def _invoke(
    fixture: _AcceptanceFixture,
    service: _Service,
    runtime: _Runtime,
    **overrides,
) -> dict[str, Any]:
    dependencies: dict[str, Any] = {
        "service_factory": lambda *, hybrid_runtime: service,
        "hybrid_runtime_loader": lambda _path: runtime,
        "review_loader": _review_loader(fixture),
        "emit": lambda _event: None,
    }
    dependencies.update(overrides)
    return run_live_acceptance(fixture.request, **dependencies)


def _assert_no_recovery_or_eject(
    service: _Service,
    receipt: dict[str, Any],
) -> None:
    assert service.safe_stop_calls == 0
    assert service.eject_calls == 0
    assert receipt["retry_count"] == 0
    assert receipt["eject_requested"] is False


def test_runs_one_six_slot_hybrid_nikon_exact_batch_and_records_truthful_state(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)
    events: list[dict[str, Any]] = []

    receipt = _invoke(
        fixture,
        service,
        runtime,
        frame_inventory_collector=harness.collect,
        batch_validator=harness.validate_batch,
        emit=events.append,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["slots"] == list(_SLOTS)
    assert receipt["approved_slots"] == list(_APPROVED_SLOTS)
    assert [frame["slot"] for frame in receipt["frames"]] == list(_SLOTS)
    assert all(frame["deep_acceptance"]["slot"] == frame["slot"] for frame in receipt["frames"])
    assert receipt["deep_acceptance"]["status"] == "passed"
    assert runtime.validated is True
    assert service.calls[:5] == [
        (
            "open_roll",
            "usb:2:7",
            fixture.request.attempts_root_path.resolve(),
        ),
        ("restore_preview_session", '{"version":1}', _SLOTS),
        ("approve", 1),
        ("approve", 6),
        "prepare_batch",
    ]
    assert [call for call in service.calls if isinstance(call, tuple) and call[0] == "scan_many"] == [("scan_many", _SLOTS)]
    writes = [call for call in service.calls if isinstance(call, tuple) and call[0] == "write_frame"]
    assert [call[1] for call in writes] == list(_SLOTS)
    assert all(call[4]["write_unrepaired"] is True for call in writes)
    assert all(call[4]["write_repaired"] is True for call in writes)
    assert all(call[4]["write_positive"] is True for call in writes)
    assert all(call[4]["repair_mode"] == RepairMode.HYBRID.value for call in writes)
    assert all(call[4]["positive_mode"] == PositiveColorMode.NIKON_EXACT.value for call in writes)
    assert harness.inventory_closed_states == [False] * 6
    assert harness.batch_closed_states == [True]
    assert service.closed is True
    assert service.calls[-1] == "close"
    state = receipt["operation_state"]
    assert state == {
        "phase": "succeeded",
        "service_constructed": True,
        "roll_opened": True,
        "requested_slots": list(_SLOTS),
        "restored_slots": list(_SLOTS),
        "required_approval_slots": list(_APPROVED_SLOTS),
        "approved_slots": list(_APPROVED_SLOTS),
        "batch_prepared": True,
        "yielded_slots": list(_SLOTS),
        "committed_slots": list(_SLOTS),
        "verified_slots": list(_SLOTS),
        "processing_slot": None,
        "batch_exhausted": True,
        "last_progress": {
            "event": "repair_progress",
            "slot": 6,
            "fraction": 0.5,
        },
        "transport_may_have_advanced_beyond_yielded": False,
    }
    assert receipt["output_lease"]["released"] is True
    assert receipt["capture_evidence_lease"]["released"] is True
    evidence = receipt["capture_evidence"]
    assert evidence["retained"] is True
    assert evidence["file_count"] == 28
    assert evidence["snapshot_error"] is None
    assert {Path(row["path"]).name for row in evidence["files"]} == {
        "batch-job.json",
        "capture-preview.bin",
        "capture-008e.bin",
        "capture-frame-map.json",
        "capture-meter.bin",
        "journal.json",
        "parent-ack.json",
        "replay-first-rgbi4-manifest.json",
        "replay-first-rgbi4-plan.jsonl",
        "replay-next-rgbi4-plan.json",
        "session-journal.json",
        "stderr.txt",
        "stdout.txt",
    }
    assert not any(Path(row["path"]).name == "capture.bin" for row in evidence["files"])
    assert evidence["batch_binding"]["selected_slots"] == list(_SLOTS)
    assert evidence["batch_binding"]["first_frame_directory"].endswith("/frame-001")
    assert not (fixture.request.attempts_root_path / OUTPUT_LOCK_NAME).exists()
    assert not (fixture.request.output_dir / OUTPUT_LOCK_NAME).exists()
    assert json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8")) == receipt
    assert any(event["event"] == "scan_progress" for event in events)
    _assert_no_recovery_or_eject(service, receipt)


def test_telemetry_callback_exceptions_never_abort_the_live_run(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)

    def broken_emit(_event: dict[str, Any]) -> None:
        raise RuntimeError("telemetry sink is unavailable")

    receipt = _invoke(
        fixture,
        service,
        runtime,
        frame_inventory_collector=harness.collect,
        batch_validator=harness.validate_batch,
        emit=broken_emit,
    )

    assert receipt["status"] == "succeeded"
    assert len(receipt["telemetry_errors"]) == 16
    assert all(error["type"] == "RuntimeError" for error in receipt["telemetry_errors"])
    assert all(error["message"] == "telemetry sink is unavailable" for error in receipt["telemetry_errors"])
    _assert_no_recovery_or_eject(service, receipt)


def test_poison_progress_extraction_and_repr_never_abort_the_live_run(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _PoisonTelemetryService(fixture)
    harness = _ValidationHarness(service)

    receipt = _invoke(
        fixture,
        service,
        runtime,
        frame_inventory_collector=harness.collect,
        batch_validator=harness.validate_batch,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["telemetry_error_count"] == 6
    assert all(
        error
        == {
            "event": "scan_progress",
            "type": "RuntimeError",
            "message": "progress extraction exploded",
        }
        for error in receipt["telemetry_errors"]
    )
    assert receipt["operation_state"]["last_progress"] == {
        "event": "repair_progress",
        "slot": 6,
        "fraction": "<unrepresentable _UnrepresentableProgressValue>",
    }
    _assert_no_recovery_or_eject(service, receipt)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("device_id", "usb:another-scanner"),
        ("device_model", "Nikon LS-4000 ED 1.10"),
        ("reviewed_fingerprint_sha256", "d" * 64),
        (
            "manual_approval_bindings",
            [
                {"slot": 1, "binding_sha256": "0" * 64},
                {"slot": 6, "binding_sha256": "1" * 64},
            ],
        ),
    ],
)
def test_deep_batch_identity_mismatch_can_never_publish_success(
    tmp_path: Path,
    field: str,
    wrong_value: object,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)

    def mismatched_batch(outputs, **kwargs):
        result = harness.validate_batch(outputs, **kwargs)
        result[field] = wrong_value
        return result

    with pytest.raises(LiveAcceptanceError, match="invalid result") as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=mismatched_batch,
        )

    assert raised.value.receipt["status"] == "failed"
    assert raised.value.receipt["operation_state"]["verified_slots"] == []
    assert json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8"))["status"] == "failed"
    _assert_no_recovery_or_eject(service, raised.value.receipt)


@pytest.mark.parametrize(
    "trigger_event",
    ["six_frame_batch_verified", "run_finished"],
)
def test_output_path_swap_during_finalization_overwrites_any_success_with_failure(
    tmp_path: Path,
    trigger_event: str,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)
    moved = tmp_path / "moved-outputs"
    swapped = False

    def swapping_emit(event: dict[str, Any]) -> None:
        nonlocal swapped
        if event.get("event") != trigger_event or swapped:
            return
        swapped = True
        fixture.request.output_dir.rename(moved)
        fixture.request.output_dir.mkdir()

    with pytest.raises(
        LiveAcceptanceError,
        match="pathname no longer identifies",
    ) as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=harness.validate_batch,
            emit=swapping_emit,
        )

    assert swapped is True
    assert raised.value.receipt["status"] == "failed"
    assert json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert not (moved / OUTPUT_LOCK_NAME).exists()
    assert list(fixture.request.output_dir.iterdir()) == []
    assert len(list(moved.iterdir())) == len(_SLOTS) * len(_OUTPUT_FIELDS)
    _assert_no_recovery_or_eject(service, raised.value.receipt)


def test_capture_evidence_mutation_after_success_publication_is_truthfully_failed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)
    target = fixture.request.attempts_root_path / "batch-slot01-slot06-test" / "frame-001" / "capture-preview.bin"

    with pytest.raises(LiveAcceptanceError, match="previously owned output changed") as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=harness.validate_batch,
            evidence_lease_factory=_mutating_evidence_lease_factory(target),
        )

    receipt = raised.value.receipt
    assert receipt["status"] == "failed"
    assert receipt["capture_evidence"]["retained"] is False
    assert receipt["capture_evidence"]["snapshot_error"] is not None
    assert json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8")) == receipt
    _assert_no_recovery_or_eject(service, receipt)


def test_failed_scan_evidence_mutation_is_truthfully_failed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture, scan_error=RuntimeError("scan failed"))
    target = fixture.request.attempts_root_path / "batch-slot01-slot06-test" / "frame-001" / "capture-preview.bin"

    with pytest.raises(LiveAcceptanceError, match="scan failed") as raised:
        _invoke(
            fixture,
            service,
            runtime,
            evidence_lease_factory=_mutating_evidence_lease_factory(target),
        )

    receipt = raised.value.receipt
    assert receipt["status"] == "failed"
    assert receipt["capture_evidence"]["retained"] is False
    assert receipt["capture_evidence"]["snapshot_error"] is not None
    assert json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8")) == receipt
    _assert_no_recovery_or_eject(service, receipt)


def test_required_capture_basenames_split_across_directories_never_pass(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)
    original_open = service.open_roll

    def open_with_split_evidence(device_id: str, *, attempts_root=None) -> None:
        original_open(device_id, attempts_root=attempts_root)
        frame = Path(attempts_root) / "batch-slot01-slot06-test" / "frame-001"
        for index, name in enumerate(("capture-preview.bin", "capture-008e.bin", "capture-frame-map.json")):
            decoy = Path(attempts_root) / f"decoy-{index}"
            decoy.mkdir()
            (frame / name).rename(decoy / name)

    service.open_roll = open_with_split_evidence  # type: ignore[method-assign]

    with pytest.raises(LiveAcceptanceError, match="accepted batch capture evidence"):
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=harness.validate_batch,
        )

    receipt = json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    _assert_no_recovery_or_eject(service, receipt)


def test_missing_sixth_frame_journal_never_passes_completed_batch_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)
    original_open = service.open_roll

    def open_without_sixth_journal(device_id: str, *, attempts_root=None) -> None:
        original_open(device_id, attempts_root=attempts_root)
        missing = Path(attempts_root) / "batch-slot01-slot06-test" / "frame-006" / "journal.json"
        missing.unlink()

    service.open_roll = open_without_sixth_journal  # type: ignore[method-assign]

    with pytest.raises(LiveAcceptanceError, match="accepted batch capture evidence"):
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=harness.validate_batch,
        )


def test_session_journal_must_bind_hashed_batch_job(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)
    original_open = service.open_roll

    def open_with_wrong_job_hash(device_id: str, *, attempts_root=None) -> None:
        original_open(device_id, attempts_root=attempts_root)
        session_path = Path(attempts_root) / "batch-slot01-slot06-test" / "session-journal.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["batch_job_sha256"] = "0" * 64
        session_path.write_text(json.dumps(session), encoding="utf-8")

    service.open_roll = open_with_wrong_job_hash  # type: ignore[method-assign]

    with pytest.raises(LiveAcceptanceError, match="completed six-frame batch"):
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=harness.validate_batch,
        )


def _never_constructed_factory(calls: list[str]):
    def factory(*, hybrid_runtime):
        calls.append("constructed")
        raise AssertionError("service must not be constructed")

    return factory


def test_missing_confirmation_fails_before_constructing_service(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, confirm_live=False)
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="--confirm-live is required") as raised:
        run_live_acceptance(
            fixture.request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    assert raised.value.receipt["status"] == "failed"
    assert raised.value.receipt["operation_state"]["roll_opened"] is False
    assert not fixture.request.run_receipt_path.exists()


def test_session_hash_mismatch_fails_before_constructing_service(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = dataclasses.replace(
        fixture.request,
        preview_session_sha256="0" * 64,
    )
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="SHA-256 mismatch"):
        run_live_acceptance(
            request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    receipt = json.loads(request.run_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["operation_state"]["roll_opened"] is False


def test_missing_session_fails_before_constructing_service(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.request.preview_session_path.unlink()
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="preview session is missing"):
        run_live_acceptance(
            fixture.request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    receipt = json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["operation_state"]["roll_opened"] is False


def test_preexisting_output_collision_is_preserved_without_opening_roll(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    existing = fixture.request.output_dir / "existing.tif"
    existing.write_bytes(b"do not overwrite")
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="unexpected files"):
        run_live_acceptance(
            fixture.request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    assert existing.read_bytes() == b"do not overwrite"
    assert sorted(path.name for path in fixture.request.output_dir.iterdir()) == ["existing.tif"]
    receipt = json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["operation_state"]["roll_opened"] is False


def test_preexisting_capture_evidence_is_preserved_without_opening_roll(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    existing = fixture.request.attempts_root_path / "existing.bin"
    existing.write_bytes(b"do not overwrite")
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="unexpected files"):
        run_live_acceptance(
            fixture.request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    assert existing.read_bytes() == b"do not overwrite"
    assert sorted(path.name for path in fixture.request.attempts_root_path.iterdir()) == ["existing.bin"]
    receipt = json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["operation_state"]["roll_opened"] is False


def test_capture_evidence_root_must_not_overlap_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = dataclasses.replace(
        fixture.request,
        attempts_root_path=fixture.request.output_dir,
    )
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="disjoint"):
        run_live_acceptance(
            request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    assert not request.run_receipt_path.exists()


def test_capture_evidence_root_must_not_be_a_symlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    linked = tmp_path / "linked-attempts"
    linked.symlink_to(fixture.request.attempts_root_path, target_is_directory=True)
    request = dataclasses.replace(fixture.request, attempts_root_path=linked)
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="non-symlink"):
        run_live_acceptance(
            request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    assert not request.run_receipt_path.exists()


def test_preexisting_run_receipt_is_never_overwritten_or_live_opened(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.request.run_receipt_path.write_bytes(b"existing receipt")
    before = fixture.request.run_receipt_path.stat()
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="already exists") as raised:
        run_live_acceptance(
            fixture.request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    after = fixture.request.run_receipt_path.stat()
    assert factory_calls == []
    assert fixture.request.run_receipt_path.read_bytes() == b"existing receipt"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert raised.value.receipt["operation_state"]["roll_opened"] is False


def test_wrong_restored_slots_fails_without_scanning_and_closes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture, restored_slots=(1, 2, 3, 4, 5, 7))

    with pytest.raises(LiveAcceptanceError, match="restored slots") as raised:
        _invoke(fixture, service, runtime)

    assert not any(isinstance(call, tuple) and call[0] == "scan_many" for call in service.calls)
    assert service.closed is True
    _assert_no_recovery_or_eject(service, raised.value.receipt)


def test_missing_edge_approval_fails_without_scanning_and_closes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture, approval_slots=(1,))

    with pytest.raises(LiveAcceptanceError, match="approval set") as raised:
        _invoke(fixture, service, runtime)

    assert not any(isinstance(call, tuple) and call[0] == "scan_many" for call in service.calls)
    assert service.closed is True
    _assert_no_recovery_or_eject(service, raised.value.receipt)


def test_wrong_service_approval_return_fails_before_scan_with_truthful_partial_state(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture, wrong_approval_slot=6)

    with pytest.raises(LiveAcceptanceError, match="differs from reviewed evidence") as raised:
        _invoke(fixture, service, runtime)

    assert [call for call in service.calls if isinstance(call, tuple) and call[0] == "approve"] == [("approve", 1), ("approve", 6)]
    assert "prepare_batch" not in service.calls
    assert not any(isinstance(call, tuple) and call[0] == "scan_many" for call in service.calls)
    state = raised.value.receipt["operation_state"]
    assert state["required_approval_slots"] == list(_APPROVED_SLOTS)
    assert state["approved_slots"] == [1]
    assert state["yielded_slots"] == []
    assert service.closed is True
    _assert_no_recovery_or_eject(service, raised.value.receipt)


def test_write_failure_records_yielded_committed_verified_and_transport_state(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture, write_error_slot=3)
    harness = _ValidationHarness(service)

    def batch_must_not_run(*_args, **_kwargs):
        raise AssertionError("deep batch validation must not run after write failure")

    with pytest.raises(LiveAcceptanceError, match="write failed for slot 3") as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=batch_must_not_run,
        )

    receipt = raised.value.receipt
    state = receipt["operation_state"]
    assert state["yielded_slots"] == [1, 2, 3]
    assert state["committed_slots"] == [1, 2]
    assert state["verified_slots"] == []
    assert state["processing_slot"] == 3
    assert state["batch_exhausted"] is False
    assert state["transport_may_have_advanced_beyond_yielded"] is True
    assert [frame["slot"] for frame in receipt["frames"]] == [1, 2]
    assert harness.inventory_closed_states == [False, False]
    assert harness.batch_closed_states == []
    assert service.closed is True
    assert json.loads(fixture.request.run_receipt_path.read_text(encoding="utf-8")) == receipt
    _assert_no_recovery_or_eject(service, receipt)


def test_scan_exception_closes_without_retry_safe_stop_or_eject(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture, scan_error=RuntimeError("scan failed"))
    harness = _ValidationHarness(service)

    with pytest.raises(LiveAcceptanceError, match="scan failed") as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=harness.validate_batch,
        )

    state = raised.value.receipt["operation_state"]
    assert state["yielded_slots"] == []
    assert state["committed_slots"] == []
    assert state["verified_slots"] == []
    assert state["batch_exhausted"] is False
    assert state["transport_may_have_advanced_beyond_yielded"] is True
    assert service.closed is True
    assert raised.value.receipt["capture_evidence"]["retained"] is True
    assert raised.value.receipt["capture_evidence"]["file_count"] == 28
    assert raised.value.receipt["capture_evidence_lease"]["released"] is True
    _assert_no_recovery_or_eject(service, raised.value.receipt)


def test_default_deep_batch_validator_rejects_literal_fake_artifacts_after_close(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)

    with pytest.raises(LiveAcceptanceError) as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
        )

    receipt = raised.value.receipt
    state = receipt["operation_state"]
    assert service.closed is True
    assert state["yielded_slots"] == list(_SLOTS)
    assert state["committed_slots"] == list(_SLOTS)
    assert state["verified_slots"] == []
    assert state["batch_exhausted"] is True
    assert state["transport_may_have_advanced_beyond_yielded"] is False
    assert receipt["deep_acceptance"] is None
    assert receipt["error"]["type"] == "DeepAcceptanceError"
    _assert_no_recovery_or_eject(service, receipt)


def test_injected_deep_batch_failure_fails_only_after_scanner_close(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(fixture)
    harness = _ValidationHarness(service)
    close_states: list[bool] = []

    def reject_batch(_outputs, **_kwargs):
        close_states.append(service.closed)
        raise RuntimeError("deep six-frame audit rejected")

    with pytest.raises(
        LiveAcceptanceError,
        match="deep six-frame audit rejected",
    ) as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=reject_batch,
        )

    assert close_states == [True]
    state = raised.value.receipt["operation_state"]
    assert state["yielded_slots"] == list(_SLOTS)
    assert state["committed_slots"] == list(_SLOTS)
    assert state["verified_slots"] == []
    assert state["batch_exhausted"] is True
    assert state["transport_may_have_advanced_beyond_yielded"] is False
    assert service.closed is True
    _assert_no_recovery_or_eject(service, raised.value.receipt)


def test_sane_style_device_id_fails_before_constructing_service(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = dataclasses.replace(
        fixture.request,
        device_id="coolscan3:usb:libusb:002:007",
    )
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="usb:BUS:ADDRESS") as raised:
        run_live_acceptance(
            request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []
    assert raised.value.receipt["operation_state"]["roll_opened"] is False


def test_receipt_inside_attempts_root_fails_before_constructing_service(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = dataclasses.replace(
        fixture.request,
        run_receipt_path=fixture.request.attempts_root_path / "receipt.json",
    )
    factory_calls: list[str] = []

    with pytest.raises(LiveAcceptanceError, match="outside the attempts root"):
        run_live_acceptance(
            request,
            service_factory=_never_constructed_factory(factory_calls),
            emit=lambda _event: None,
        )

    assert factory_calls == []


def test_parser_requires_attempts_root() -> None:
    from negpy.services.roll import live_acceptance as module

    argv = [
        "--device-id", "usb:2:7",
        "--preview-session", "session.json",
        "--preview-session-sha256", "0" * 64,
        "--reviewed-approval", "review.json",
        "--reviewed-approval-sha256", "0" * 64,
        "--output-dir", "outputs",
        "--run-receipt", "receipt.json",
        "--confirm-live",
    ]
    with pytest.raises(SystemExit):
        module._parser().parse_args(argv)
    parsed = module._parser().parse_args([*argv, "--attempts-root", "attempts"])
    assert parsed.attempts_root == Path("attempts")


def test_interrupt_handlers_route_signals_to_truthful_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal as signal_module

    from negpy.services.roll import live_acceptance as module

    recorded: dict[int, object] = {}
    monkeypatch.setattr(
        module.signal,
        "signal",
        lambda number, handler: recorded.__setitem__(number, handler),
    )

    assert module.install_interrupt_handlers() is True
    assert set(recorded) == {signal_module.SIGINT, signal_module.SIGTERM}

    handler = recorded[signal_module.SIGTERM]
    assert callable(handler)
    with pytest.raises(module.LiveAcceptanceInterrupted, match="signal"):
        handler(signal_module.SIGTERM, None)
    assert recorded[signal_module.SIGINT] is signal_module.SIG_IGN
    assert recorded[signal_module.SIGTERM] is signal_module.SIG_IGN


def test_interrupt_mid_scan_closes_and_writes_truthful_failed_receipt(
    tmp_path: Path,
) -> None:
    from negpy.services.roll.live_acceptance import LiveAcceptanceInterrupted

    fixture = _fixture(tmp_path)
    runtime = _Runtime()
    service = _Service(
        fixture,
        scan_error=LiveAcceptanceInterrupted("received signal 15"),
    )
    harness = _ValidationHarness(service)

    with pytest.raises(LiveAcceptanceError, match="received signal 15") as raised:
        _invoke(
            fixture,
            service,
            runtime,
            frame_inventory_collector=harness.collect,
            batch_validator=harness.validate_batch,
        )

    state = raised.value.receipt["operation_state"]
    assert state["committed_slots"] == []
    assert service.closed is True
    assert raised.value.receipt["status"] == "failed"
    _assert_no_recovery_or_eject(service, raised.value.receipt)
