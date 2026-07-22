"""One-shot, fail-closed LS-5000 six-frame live acceptance runner."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from negpy.infrastructure.roll.repair import RepairMode
from negpy.services.repair.hybrid_runtime_manifest import (
    load_default_hybrid_runtime_manifest,
)
from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig
from negpy.services.roll.exact_color import PositiveColorMode
from negpy.services.roll.deep_acceptance import (
    collect_completed_frame_files,
    validate_six_frame_batch,
)
from negpy.services.roll.live_reservation import (
    OUTPUT_LOCK_NAME,
    ExclusiveReceiptReservation,
    FixedOutputLease,
    ReservationConflict,
)
from negpy.services.roll.live_review import (
    ValidatedReviewedApproval,
    approval_payload,
    load_reviewed_approval,
    validate_restored_thumbnails,
)
from negpy.services.roll.service import RollFrameOutput, RollScanningService


SCHEMA = "negpy.ls5000-live-acceptance.v2"
SLOTS = (1, 2, 3, 4, 5, 6)
APPROVED_SLOTS = (1, 6)
EXPECTED_DEVICE_MODEL = "Nikon LS-5000 ED 1.03"
FILENAME_PATTERN = 'acceptance_slot{{ "%02d" % seq }}'
_SESSION_MAX_BYTES = 64 * 1024
_FRAME_RECEIPT_MAX_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_TELEMETRY_ERRORS = 16


class LiveAcceptanceError(RuntimeError):
    """The one-shot live acceptance run stopped without passing."""

    def __init__(self, message: str, *, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        super().__init__(message)


@dataclasses.dataclass(frozen=True)
class LiveAcceptanceRequest:
    device_id: str
    preview_session_path: Path
    preview_session_sha256: str
    reviewed_approval_path: Path
    reviewed_approval_sha256: str
    output_dir: Path
    run_receipt_path: Path
    confirm_live: bool
    hybrid_runtime_manifest_path: Path | None = None


@dataclasses.dataclass(frozen=True)
class _SessionPayload:
    text: str
    sha256: str
    size: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_event(event: dict[str, Any]) -> None:
    print(
        json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _read_stable_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    descriptor: int | None = None
    try:
        linked = path.lstat()
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if linked.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds its safe size limit")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino, opened.st_size) != (
                linked.st_dev,
                linked.st_ino,
                linked.st_size,
            ):
                raise ValueError(f"{label} changed while opening")
            payload = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
        final = path.lstat()

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if (
            len(payload) != opened.st_size
            or len(payload) > maximum_bytes
            or identity(opened) != identity(after)
            or identity(opened) != identity(final)
        ):
            raise ValueError(f"{label} changed while reading")
        return payload
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    except OSError as error:
        raise ValueError(f"could not read {label}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_session(path: Path, expected_sha256: str) -> _SessionPayload:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("preview-session pin must be a lowercase SHA-256 digest")
    payload = _read_stable_regular_file(
        path,
        maximum_bytes=_SESSION_MAX_BYTES,
        label="preview session",
    )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError(f"preview-session SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("preview session is not UTF-8") from error
    return _SessionPayload(text=text, sha256=actual_sha256, size=len(payload))


def _default_review_loader(
    request: LiveAcceptanceRequest,
    session: _SessionPayload,
) -> ValidatedReviewedApproval:
    return load_reviewed_approval(
        request.reviewed_approval_path,
        request.reviewed_approval_sha256,
        preview_session_path=request.preview_session_path,
        preview_session_payload=session.text,
        preview_session_sha256=session.sha256,
    )


def _preflight_paths(request: LiveAcceptanceRequest) -> tuple[Path, Path]:
    output = request.output_dir.resolve(strict=True)
    linked_output = request.output_dir.lstat()
    if stat.S_ISLNK(linked_output.st_mode) or not stat.S_ISDIR(linked_output.st_mode):
        raise ValueError("output directory must be an existing non-symlink directory")
    receipt = request.run_receipt_path.absolute()
    receipt_parent = receipt.parent.resolve(strict=True)
    if not receipt_parent.is_dir():
        raise ValueError("run-receipt parent must be a directory")
    receipt = receipt_parent / receipt.name
    if receipt == output or receipt.is_relative_to(output):
        raise ValueError("run receipt must be outside the output directory")
    return output, receipt


def _require_artifact(path_value: str | None, *, output_dir: Path, label: str) -> str:
    if not path_value:
        raise ValueError(f"{label} was not produced")
    path = Path(path_value)
    linked = path.lstat()
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(output_dir):
        raise ValueError(f"{label} escaped the output directory")
    return str(resolved)


def _require_true(document: dict[str, Any], key: str, *, label: str) -> None:
    if document.get(key) is not True:
        raise ValueError(f"{label} is not verified")


def _validate_frame_output(
    output: RollFrameOutput,
    *,
    expected_slot: int,
    output_dir: Path,
) -> dict[str, Any]:
    if output.slot != expected_slot:
        raise ValueError(f"writer returned slot {output.slot}; expected {expected_slot}")
    artifacts = {
        field: _require_artifact(
            getattr(output, field),
            output_dir=output_dir,
            label=f"slot {expected_slot} {field}",
        )
        for field in (
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
    }
    receipt_path = Path(artifacts["receipt_path"])
    receipt_bytes = _read_stable_regular_file(
        receipt_path,
        maximum_bytes=_FRAME_RECEIPT_MAX_BYTES,
        label=f"slot {expected_slot} frame receipt",
    )
    try:
        frame_receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"slot {expected_slot} frame receipt is invalid JSON") from error
    if not isinstance(frame_receipt, dict) or frame_receipt.get("slot") != expected_slot:
        raise ValueError(f"slot {expected_slot} frame receipt identity is wrong")
    outputs = frame_receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"slot {expected_slot} frame receipt has no outputs object")
    for tier in ("unrepaired", "repaired", "positive"):
        entry = outputs.get(tier)
        if not isinstance(entry, dict) or entry.get("written") is not True:
            raise ValueError(f"slot {expected_slot} {tier} tier was not written")
    repaired = outputs["repaired"]
    if (
        repaired.get("mode_requested") != RepairMode.HYBRID.value
        or repaired.get("mode_resolved") != RepairMode.HYBRID.value
        or repaired.get("degraded") is not False
    ):
        raise ValueError(f"slot {expected_slot} did not complete non-degraded Hybrid repair")
    positive = outputs["positive"]
    if positive.get("color_mode") != PositiveColorMode.NIKON_EXACT.value:
        raise ValueError(f"slot {expected_slot} positive is not Nikon exact")
    for key in (
        "exact_nikon_color",
        "native_per_acquisition_builder",
        "builder_validated",
        "cms_verified",
    ):
        _require_true(positive, key, label=f"slot {expected_slot} positive {key}")
    for evidence_key, truth_keys in (
        ("repair_acquisition_evidence", ("retained", "replayable")),
        ("native_color_evidence", ("retained",)),
    ):
        evidence = outputs.get(evidence_key)
        if not isinstance(evidence, dict):
            raise ValueError(f"slot {expected_slot} has no {evidence_key}")
        for key in truth_keys:
            _require_true(
                evidence,
                key,
                label=f"slot {expected_slot} {evidence_key} {key}",
            )
    return {
        "slot": expected_slot,
        "artifacts": artifacts,
        "frame_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }


def _event_value(value: object) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    try:
        return repr(value)[:512]
    except Exception:
        try:
            name = type(value).__name__
        except Exception:
            name = "object"
        return f"<unrepresentable {name}>"


def _progress_event(progress: object) -> dict[str, Any]:
    event: dict[str, Any] = {"event": "scan_progress"}
    for field in ("stage", "slot", "index", "total", "fraction", "message"):
        value = getattr(progress, field, None)
        if value is not None:
            event[field] = _event_value(value)
    return event


def _error_payload(error: BaseException) -> dict[str, str]:
    try:
        message = str(error)[:4096]
    except Exception:
        message = "<exception message unavailable>"
    return {
        "type": type(error).__name__,
        "message": message,
    }


def _combine_error(
    current: BaseException | None,
    later: BaseException,
    *,
    label: str,
) -> BaseException:
    if current is None:
        return later
    current_payload = _error_payload(current)
    later_payload = _error_payload(later)
    return RuntimeError(
        f"{current_payload['type']}: {current_payload['message']}; {label}: {later_payload['type']}: {later_payload['message']}"
    )


def _default_service_factory(
    *,
    hybrid_runtime: HybridRuntimeConfig,
) -> RollScanningService:
    return RollScanningService(hybrid_runtime=hybrid_runtime)


def run_live_acceptance(
    request: LiveAcceptanceRequest,
    *,
    service_factory: Callable[..., Any] = _default_service_factory,
    hybrid_runtime_loader: Callable[[Path | None], Any] = (load_default_hybrid_runtime_manifest),
    review_loader: Callable[[LiveAcceptanceRequest, _SessionPayload], ValidatedReviewedApproval] = _default_review_loader,
    frame_inventory_collector: Callable[..., list[str]] = (collect_completed_frame_files),
    batch_validator: Callable[..., dict[str, Any]] = validate_six_frame_batch,
    receipt_reserver: Callable[..., Any] = ExclusiveReceiptReservation.reserve,
    output_lease_factory: Callable[..., Any] = FixedOutputLease.acquire,
    emit: Callable[[dict[str, Any]], None] = _json_event,
) -> dict[str, Any]:
    """Run one exact six-slot batch; any discrepancy ends it without retry."""

    started_at = _utc_now()
    run_id = uuid.uuid4().hex
    phase = "starting"
    frames: list[dict[str, Any]] = []
    outputs: list[RollFrameOutput] = []
    restored_slots: list[object] = []
    required_approval_slots: list[int] = []
    approved_slots: list[int] = []
    yielded_slots: list[object] = []
    committed_slots: list[object] = []
    verified_slots: list[int] = []
    telemetry_errors: list[dict[str, str]] = []
    telemetry_error_count = 0
    last_progress: dict[str, Any] | None = None
    processing_slot: int | None = None
    batch_prepared = False
    batch_exhausted = False
    transport_may_have_advanced_beyond_yielded = False

    session: _SessionPayload | None = None
    review: ValidatedReviewedApproval | None = None
    runtime: Any = None
    deep_batch: dict[str, Any] | None = None
    output_dir: Path | None = None
    receipt_path = request.run_receipt_path.absolute()
    receipt_reservation: Any = None
    receipt_inode: tuple[int, int] | None = None
    receipt_reserved = False
    output_lease: Any = None
    lease_acquired = False
    lease_release_attempted = False
    lease_released = False
    inventory_snapshot: Any = None
    owned_files: set[Path] = set()
    service: Any = None
    service_constructed = False
    roll_opened = False
    iterator: Any = None
    iterator_created = False
    iterator_close_attempted = False
    iterator_close_succeeded = False
    close_attempted = False
    close_succeeded = False
    operation_error: BaseException | None = None

    def record_telemetry_error(event_name: str, error: BaseException) -> None:
        nonlocal telemetry_error_count
        telemetry_error_count += 1
        if len(telemetry_errors) < _MAX_TELEMETRY_ERRORS:
            telemetry_errors.append(
                {
                    "event": event_name[:128],
                    **_error_payload(error),
                }
            )

    def safe_emit(event: dict[str, Any]) -> None:
        try:
            emit(event)
        except Exception as error:
            record_telemetry_error(
                str(event.get("event", "unknown")),
                error,
            )

    def operation_state() -> dict[str, Any]:
        return {
            "phase": phase,
            "service_constructed": service_constructed,
            "roll_opened": roll_opened,
            "requested_slots": list(SLOTS),
            "restored_slots": list(restored_slots),
            "required_approval_slots": list(required_approval_slots),
            "approved_slots": list(approved_slots),
            "batch_prepared": batch_prepared,
            "yielded_slots": list(yielded_slots),
            "committed_slots": list(committed_slots),
            "verified_slots": list(verified_slots),
            "processing_slot": processing_slot,
            "batch_exhausted": batch_exhausted,
            "last_progress": last_progress,
            "transport_may_have_advanced_beyond_yielded": (transport_may_have_advanced_beyond_yielded),
        }

    def receipt_document(
        status: str,
        *,
        finished_at: str | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "status": status,
            "phase": phase,
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "device_id": request.device_id,
            "preview_session": {
                "path": str(request.preview_session_path.absolute()),
                "expected_sha256": request.preview_session_sha256,
                "verified_sha256": session.sha256 if session is not None else None,
                "bytes": session.size if session is not None else None,
            },
            "reviewed_approval": {
                "path": str(request.reviewed_approval_path.absolute()),
                "expected_sha256": request.reviewed_approval_sha256,
                "verified_sha256": review.sha256 if review is not None else None,
                "bytes": review.byte_length if review is not None else None,
                "reviewed_fingerprint_sha256": (review.reviewed_fingerprint_sha256 if review is not None else None),
                "contact_sheet": (
                    {
                        "path": str(review.contact_sheet_path),
                        "sha256": review.contact_sheet_sha256,
                    }
                    if review is not None
                    else None
                ),
            },
            "output_dir": str(output_dir or request.output_dir.absolute()),
            "run_receipt": str(receipt_path),
            "receipt_reservation": {
                "reserved": receipt_reserved,
                "inode": list(receipt_inode) if receipt_inode is not None else None,
            },
            "output_lease": {
                "acquired": lease_acquired,
                "lock_name": OUTPUT_LOCK_NAME,
                "release_attempted": lease_release_attempted,
                "released": lease_released,
            },
            "slots": list(SLOTS),
            "approved_slots": list(approved_slots),
            "operation_state": operation_state(),
            "settings": {
                "write_unrepaired": True,
                "write_repaired": True,
                "write_positive": True,
                "repair_mode": RepairMode.HYBRID.value,
                "positive_mode": PositiveColorMode.NIKON_EXACT.value,
                "filename_pattern": FILENAME_PATTERN,
            },
            "frames": frames,
            "deep_acceptance": deep_batch,
            "close": {
                "iterator": {
                    "attempted": iterator_close_attempted,
                    "succeeded": iterator_close_succeeded,
                },
                "roll": {
                    "attempted": close_attempted,
                    "succeeded": close_succeeded,
                },
            },
            "telemetry_errors": list(telemetry_errors),
            "telemetry_error_count": telemetry_error_count,
            "telemetry_errors_truncated": (telemetry_error_count > len(telemetry_errors)),
            "retry_count": 0,
            "eject_requested": False,
        }
        if error is not None:
            document["error"] = _error_payload(error)
        return document

    def on_scan_progress(progress: object) -> None:
        nonlocal last_progress
        try:
            event = _progress_event(progress)
        except Exception as error:
            record_telemetry_error("scan_progress", error)
            return
        last_progress = event
        safe_emit(event)

    try:
        safe_emit({"event": "run_started", "slots": list(SLOTS), "run_id": run_id})
        if request.confirm_live is not True:
            raise ValueError("--confirm-live is required")
        if not request.device_id.strip():
            raise ValueError("device id must be non-empty")
        output_dir, receipt_path = _preflight_paths(request)

        phase = "reserving_receipt"
        receipt_reservation = receipt_reserver(
            receipt_path,
            receipt_document("in_progress"),
        )
        receipt_reserved = True
        receipt_inode = tuple(receipt_reservation.inode)

        phase = "reserving_output"
        output_lease = output_lease_factory(
            output_dir,
            {
                "schema": "negpy.ls5000-live-output-lock.v1",
                "run_id": run_id,
                "run_receipt": str(receipt_path),
                "created_at": started_at,
            },
            require_empty=True,
        )
        lease_acquired = True
        inventory_snapshot = output_lease.assert_inventory(())
        phase = "output_reserved"
        receipt_reservation.publish(receipt_document("in_progress"))

        phase = "validating_offline_inputs"
        session = _load_session(
            request.preview_session_path,
            request.preview_session_sha256,
        )
        review = review_loader(request, session)
        runtime = hybrid_runtime_loader(request.hybrid_runtime_manifest_path)
        if runtime is None:
            raise ValueError("the pinned Hybrid runtime is not installed")
        runtime.validate_files()
        receipt_reservation.assert_owned()
        inventory_snapshot = output_lease.assert_inventory(
            (),
            previous=inventory_snapshot,
        )

        phase = "ready_to_open"
        receipt_reservation.publish(receipt_document("in_progress"))
        safe_emit(
            {
                "event": "preflight_passed",
                "preview_session_sha256": session.sha256,
                "reviewed_approval_sha256": review.sha256,
            }
        )

        service = service_factory(hybrid_runtime=runtime)
        service_constructed = True
        receipt_reservation.assert_owned()
        inventory_snapshot = output_lease.assert_inventory(
            (),
            previous=inventory_snapshot,
        )
        phase = "opening_roll"
        service.open_roll(request.device_id)
        roll_opened = True
        safe_emit({"event": "roll_opened", "device_id": request.device_id})

        phase = "restoring_review"
        thumbnails = service.restore_preview_session(session.text, slots=SLOTS)
        restored_slots.extend(getattr(thumbnail, "slot", None) for thumbnail in thumbnails)
        if tuple(restored_slots) != SLOTS:
            raise ValueError(f"restored slots are {tuple(restored_slots)}; expected exactly {SLOTS}")
        validate_restored_thumbnails(thumbnails, review)
        required_approval_slots.extend(slot for slot in SLOTS if service.needs_approval(slot))
        if tuple(required_approval_slots) != APPROVED_SLOTS:
            raise ValueError(f"restored session approval set is {tuple(required_approval_slots)}; expected {APPROVED_SLOTS}")
        safe_emit({"event": "preview_session_restored", "slots": list(restored_slots)})

        phase = "applying_reviewed_approvals"
        for slot in APPROVED_SLOTS:
            returned = service.approve(slot)
            expected_payload = approval_payload(review.approvals[slot])
            if approval_payload(returned) != expected_payload:
                raise ValueError(f"service approval for slot {slot} differs from reviewed evidence")
            approved_slots.append(slot)
            safe_emit({"event": "slot_approved", "slot": slot})
        receipt_reservation.publish(receipt_document("in_progress"))

        phase = "preparing_batch"
        service.prepare_batch()
        batch_prepared = True
        receipt_reservation.assert_owned()
        inventory_snapshot = output_lease.assert_inventory(
            owned_files,
            previous=inventory_snapshot,
        )
        phase = "scanning"
        receipt_reservation.publish(receipt_document("in_progress"))
        iterator = iter(
            service.scan_many(
                SLOTS,
                on_progress=on_scan_progress,
            )
        )
        iterator_created = True
        while True:
            try:
                frame = next(iterator)
            except StopIteration:
                batch_exhausted = True
                break
            actual_slot = getattr(frame, "slot", None)
            yielded_slots.append(_event_value(actual_slot))
            if len(yielded_slots) > len(SLOTS):
                raise ValueError("scanner yielded more than six frames")
            expected_slot = SLOTS[len(yielded_slots) - 1]
            processing_slot = expected_slot
            if actual_slot != expected_slot:
                raise ValueError(f"scanner yielded slot {actual_slot}; expected {expected_slot}")
            receipt_reservation.assert_owned()
            inventory_snapshot = output_lease.assert_inventory(
                owned_files,
                previous=inventory_snapshot,
            )

            def on_repair_progress(
                fraction: object,
                *,
                slot: int = expected_slot,
            ) -> None:
                nonlocal last_progress
                try:
                    event = {
                        "event": "repair_progress",
                        "slot": slot,
                        "fraction": _event_value(fraction),
                    }
                except Exception as error:
                    record_telemetry_error("repair_progress", error)
                    return
                last_progress = event
                safe_emit(event)

            output = service.write_frame(
                frame,
                str(output_dir),
                FILENAME_PATTERN,
                write_unrepaired=True,
                write_repaired=True,
                write_positive=True,
                repair_mode=RepairMode.HYBRID.value,
                positive_mode=PositiveColorMode.NIKON_EXACT.value,
                on_repair_progress=on_repair_progress,
            )
            outputs.append(output)
            returned_slot = _event_value(output.slot)
            committed_slots.append(returned_slot)
            frame_entry: dict[str, Any] = {
                "slot": returned_slot,
                "expected_slot": expected_slot,
                "returned_output": {
                    field: getattr(output, field)
                    for field in (
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
                },
            }
            frames.append(frame_entry)
            safe_emit({"event": "frame_committed", "slot": expected_slot})

            frame_entry.update(
                _validate_frame_output(
                    output,
                    expected_slot=expected_slot,
                    output_dir=output_dir,
                )
            )
            referenced = frame_inventory_collector(
                output,
                output_dir=output_dir,
                expected_slot=expected_slot,
            )
            if type(referenced) is not list or not referenced or any(type(path) is not str for path in referenced):
                raise ValueError(f"slot {expected_slot} publication inventory is invalid")
            owned_files.update(Path(path) for path in referenced)
            inventory_snapshot = output_lease.assert_inventory(
                owned_files,
                previous=inventory_snapshot,
            )
            frame_entry["publication_inventory"] = list(referenced)
            processing_slot = None
            receipt_reservation.publish(receipt_document("in_progress"))
            safe_emit({"event": "frame_checkpointed", "slot": expected_slot})

        if tuple(yielded_slots) != SLOTS:
            raise ValueError(f"scanner yielded {tuple(yielded_slots)}; expected exactly {SLOTS}")
        if tuple(committed_slots) != SLOTS:
            raise ValueError("not every yielded frame was durably committed")
        phase = "batch_exhausted"
        receipt_reservation.publish(receipt_document("in_progress"))
    except ReservationConflict as error:
        operation_error = error
    except BaseException as error:
        operation_error = error
    finally:
        if iterator is not None:
            close_iterator = getattr(iterator, "close", None)
            if callable(close_iterator):
                iterator_close_attempted = True
                try:
                    close_iterator()
                except BaseException as iterator_error:
                    operation_error = _combine_error(
                        operation_error,
                        iterator_error,
                        label="scan iterator close failed",
                    )
                else:
                    iterator_close_succeeded = True
        if service is not None:
            close_attempted = True
            try:
                service.close()
            except BaseException as close_error:
                operation_error = _combine_error(
                    operation_error,
                    close_error,
                    label="scanner close failed",
                )
            else:
                close_succeeded = True
                safe_emit({"event": "roll_closed"})

    if iterator_created and not batch_exhausted:
        transport_may_have_advanced_beyond_yielded = True

    if operation_error is None:
        try:
            phase = "validating_six_frame_batch"
            if output_dir is None or output_lease is None or runtime is None:
                raise RuntimeError("live acceptance state is incomplete")
            if not close_succeeded:
                raise RuntimeError("scanner was not closed before deep acceptance")
            receipt_reservation.publish(receipt_document("in_progress"))
            deep_batch = batch_validator(
                outputs,
                output_dir=output_dir,
                allowed_output_lock_name=OUTPUT_LOCK_NAME,
                hybrid_runtime=runtime,
            )
            if review is None:
                raise RuntimeError("reviewed approval disappeared before deep acceptance")
            expected_manual_approval_bindings: list[dict[str, Any]] = []
            for slot in APPROVED_SLOTS:
                binding = approval_payload(review.approvals[slot]).get("binding_sha256")
                if type(binding) is not str or _SHA256_RE.fullmatch(binding) is None:
                    raise ValueError(f"reviewed approval binding for slot {slot} is invalid")
                expected_manual_approval_bindings.append(
                    {
                        "slot": slot,
                        "binding_sha256": binding,
                    }
                )
            if (
                type(deep_batch) is not dict
                or deep_batch.get("status") != "passed"
                or deep_batch.get("slots") != list(SLOTS)
                or deep_batch.get("approved_slots") != list(APPROVED_SLOTS)
                or deep_batch.get("device_id") != request.device_id
                or deep_batch.get("device_model") != EXPECTED_DEVICE_MODEL
                or deep_batch.get("reviewed_fingerprint_sha256") != review.reviewed_fingerprint_sha256
                or deep_batch.get("manual_approval_bindings") != expected_manual_approval_bindings
                or type(deep_batch.get("referenced_files")) is not list
                or any(type(path) is not str for path in deep_batch["referenced_files"])
                or type(deep_batch.get("frames")) is not list
                or len(deep_batch["frames"]) != len(SLOTS)
            ):
                raise ValueError("six-frame deep acceptance returned an invalid result")
            for expected_slot, frame_entry, deep_frame in zip(
                SLOTS,
                frames,
                deep_batch["frames"],
                strict=True,
            ):
                if (
                    type(deep_frame) is not dict
                    or deep_frame.get("slot") != expected_slot
                    or type(deep_frame.get("referenced_files")) is not list
                    or any(type(path) is not str for path in deep_frame["referenced_files"])
                    or set(deep_frame["referenced_files"]) != set(frame_entry["publication_inventory"])
                ):
                    raise ValueError(f"slot {expected_slot} deep inventory differs from its live checkpoint")
                frame_entry["deep_acceptance"] = deep_frame
                verified_slots.append(expected_slot)
                safe_emit({"event": "frame_verified", "slot": expected_slot})
            batch_files = {Path(path) for path in deep_batch["referenced_files"]}
            if batch_files != owned_files:
                raise ValueError("six-frame deep acceptance inventory differs from frame checkpoints")
            if tuple(verified_slots) != SLOTS:
                raise ValueError("not every committed frame was deeply verified")
            inventory_snapshot = output_lease.assert_inventory(
                batch_files,
                previous=inventory_snapshot,
            )
            receipt_reservation.assert_owned()
            phase = "six_frame_batch_verified"
            receipt_reservation.publish(receipt_document("in_progress"))
            safe_emit({"event": "six_frame_batch_verified", "slots": list(SLOTS)})
        except BaseException as validation_error:
            operation_error = validation_error

    receipt: dict[str, Any] | None = None
    if operation_error is None:
        if output_lease is None or inventory_snapshot is None:
            operation_error = RuntimeError("output lease state is incomplete at finalization")
        elif receipt_reservation is None:
            operation_error = RuntimeError("run receipt was never reserved")

    if operation_error is None:
        lease_release_attempted = True
        phase = "finalizing_success"
        safe_emit({"event": "run_finalizing", "status": "succeeded"})

        def publish_success_while_directory_is_held() -> None:
            nonlocal lease_released, phase, receipt
            # release_verified invokes this only after unlinking the fixed
            # lock, while retaining the directory descriptor for a second
            # pathname + inventory check after publication.
            lease_released = True
            phase = "succeeded"
            safe_emit(
                {
                    "event": "run_finished",
                    "status": "succeeded",
                    "run_receipt": str(receipt_path),
                }
            )
            receipt = receipt_document(
                "succeeded",
                finished_at=_utc_now(),
            )
            receipt_reservation.publish(receipt)

        try:
            output_lease.release_verified(
                owned_files,
                previous=inventory_snapshot,
                finalize=publish_success_while_directory_is_held,
            )
        except BaseException as lease_error:
            operation_error = _combine_error(
                operation_error,
                lease_error,
                label="verified output lease release failed",
            )
        finally:
            lease_released = bool(output_lease.released)

    if operation_error is not None:
        if output_lease is not None and not output_lease.released:
            lease_release_attempted = True
            try:
                output_lease.release()
            except BaseException as lease_error:
                operation_error = _combine_error(
                    operation_error,
                    lease_error,
                    label="output lease cleanup failed",
                )
            finally:
                lease_released = bool(output_lease.released)
        phase = "failed"
        safe_emit({"event": "run_finalizing", "status": "failed"})
        safe_emit(
            {
                "event": "run_finished",
                "status": "failed",
                "run_receipt": str(receipt_path),
            }
        )
        receipt = receipt_document(
            "failed",
            finished_at=_utc_now(),
            error=operation_error,
        )

    assert receipt is not None
    if receipt_reservation is None:
        message = _error_payload(operation_error or RuntimeError("run receipt reservation failed"))["message"]
        raise LiveAcceptanceError(message, receipt=receipt) from operation_error

    if receipt.get("status") == "failed":
        try:
            receipt_reservation.publish(receipt)
        except BaseException as receipt_error:
            failed_receipt = dict(receipt)
            failed_receipt["status"] = "failed"
            failed_receipt["phase"] = "failed"
            failed_receipt["error"] = _error_payload(receipt_error)
            raise LiveAcceptanceError(
                f"could not publish the final run receipt: {_error_payload(receipt_error)['message']}",
                receipt=failed_receipt,
            ) from receipt_error

    try:
        receipt_reservation.close()
    except Exception:
        # The document was already fsynced into the held inode. A close(2)
        # housekeeping error must not replace that durable outcome.
        pass

    if operation_error is not None:
        raise LiveAcceptanceError(
            _error_payload(operation_error)["message"],
            receipt=receipt,
        ) from operation_error
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run the pinned six-frame LS-5000 live acceptance exactly once. The output directory must already exist and be empty.")
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--preview-session", required=True, type=Path)
    parser.add_argument("--preview-session-sha256", required=True)
    parser.add_argument("--reviewed-approval", required=True, type=Path)
    parser.add_argument("--reviewed-approval-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-receipt", required=True, type=Path)
    parser.add_argument("--hybrid-runtime-manifest", type=Path)
    parser.add_argument("--confirm-live", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = LiveAcceptanceRequest(
        device_id=args.device_id,
        preview_session_path=args.preview_session,
        preview_session_sha256=args.preview_session_sha256,
        reviewed_approval_path=args.reviewed_approval,
        reviewed_approval_sha256=args.reviewed_approval_sha256,
        output_dir=args.output_dir,
        run_receipt_path=args.run_receipt,
        confirm_live=args.confirm_live,
        hybrid_runtime_manifest_path=args.hybrid_runtime_manifest,
    )
    try:
        run_live_acceptance(request)
    except LiveAcceptanceError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
