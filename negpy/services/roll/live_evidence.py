"""Stable, receipt-ready inventory of retained Coolscan attempt evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from pathlib import PurePosixPath

from negpy.services.roll.live_reservation import (
    OUTPUT_LOCK_NAME,
    FileIdentity,
    InventoryConflict,
    InventorySnapshot,
)


CAPTURE_EVIDENCE_MANIFEST_SCHEMA = "negpy.coolscan-attempts-manifest.v1"
_MAX_FILES = 4_096
_MAX_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 128 * 1024
_MAX_BATCH_JSON_BYTES = 256 * 1024
_READ_CHUNK_BYTES = 4 * 1024 * 1024
_SIX_FRAME_SLOTS = (1, 2, 3, 4, 5, 6)
_SIX_FRAME_BATCH_ROOT_RE = re.compile(r"batch-slot01-slot06-[^/]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# The batch worker mints every parent-ACK nonce with secrets.token_hex(16).
_ACK_NONCE_RE = re.compile(r"[0-9a-f]{32}")
_FIRST_PLAN_NAME = "replay-first-rgbi4-plan.jsonl"
_CONTINUATION_PLAN_NAME = "replay-next-rgbi4-plan.json"
_CAPTURE_MANIFEST_NAME = "replay-first-rgbi4-manifest.json"
_FINE_READS = 2_980
_FINE_BYTES = 619_458_560
_METER_PASS_BYTES = 1_088_000
_METER_BYTES = _METER_PASS_BYTES * 3
_PREVIEW_BYTES = 6_250_496
_BATCH_JOB_KEYS = {
    "apply_all_boundary_offsets_before_first_frame",
    "capture_plan_sha256",
    "continuation_plan_sha256",
    "expected_usb_address",
    "expected_usb_bus",
    "frames",
    "parent_ack_required_after_every_frame",
    "release_once_after_last_frame",
    "reviewed_roll_fingerprint",
    "schema_version",
    "session_id",
    "session_contract",
}
_BATCH_FRAME_KEYS = {
    "ack",
    "boundary_offset_rows",
    "journal",
    "manual_review_approval",
    "output",
    "slot",
}
_ACK_KEYS = {
    "ack_nonce",
    "action",
    "frame_index",
    "schema_version",
    "session_id",
    "slot",
}
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


@dataclasses.dataclass(frozen=True)
class CaptureEvidenceSnapshot:
    root: Path
    files: tuple[Path, ...]
    identities: tuple[tuple[str, FileIdentity], ...]
    manifest: dict[str, object]


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity.from_stat(metadata)


def _stable_file_digest(path: Path) -> tuple[int, str, FileIdentity]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise InventoryConflict(f"capture evidence must be an exclusively owned regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise InventoryConflict(f"capture evidence changed while opening: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > _MAX_TOTAL_BYTES:
                raise InventoryConflict("capture evidence exceeds its safe size limit")
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    opened_identity = _identity(opened)
    if total != opened.st_size or opened_identity != _identity(after_read) or opened_identity != _identity(after_path):
        raise InventoryConflict(f"capture evidence changed while hashing: {path}")
    return total, digest.hexdigest(), opened_identity


def _stable_file_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one small evidence file without following or racing a pathname."""

    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise InventoryConflict(f"capture evidence must be an exclusively owned regular file: {path}")
    if before.st_size > maximum_bytes:
        raise InventoryConflict(f"capture evidence JSON is too large: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise InventoryConflict(f"capture evidence changed while opening: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise InventoryConflict(f"capture evidence JSON is too large: {path}")
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    opened_identity = _identity(opened)
    if total != opened.st_size or opened_identity != _identity(after_read) or opened_identity != _identity(after_path):
        raise InventoryConflict(f"capture evidence changed while reading: {path}")
    return b"".join(chunks)


def snapshot_capture_evidence(root: Path) -> CaptureEvidenceSnapshot:
    """Hash one retained attempts tree without following links or special files."""

    canonical = root.resolve(strict=True)
    linked_root = root.lstat()
    if stat.S_ISLNK(linked_root.st_mode) or not stat.S_ISDIR(linked_root.st_mode):
        raise InventoryConflict("capture evidence root must be an existing non-symlink directory")

    paths: list[Path] = []
    for current_text, directory_names, file_names in os.walk(
        canonical,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_text)
        current_metadata = current.lstat()
        if stat.S_ISLNK(current_metadata.st_mode) or not stat.S_ISDIR(current_metadata.st_mode):
            raise InventoryConflict(f"capture evidence entry is not a real directory: {current}")
        for name in directory_names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise InventoryConflict(f"capture evidence entry is not a real directory: {child}")
        for name in file_names:
            path = current / name
            if current == canonical and name == OUTPUT_LOCK_NAME:
                continue
            paths.append(path)
            if len(paths) > _MAX_FILES:
                raise InventoryConflict("capture evidence contains too many files")

    rows: list[dict[str, object]] = []
    identities: list[tuple[str, FileIdentity]] = []
    total_bytes = 0
    ordered_paths = sorted(paths, key=lambda item: item.relative_to(canonical).as_posix())
    for path in ordered_paths:
        relative = path.relative_to(canonical).as_posix()
        size, digest, identity = _stable_file_digest(path)
        total_bytes += size
        if total_bytes > _MAX_TOTAL_BYTES:
            raise InventoryConflict("capture evidence exceeds its safe size limit")
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        identities.append((relative, identity))

    manifest_document = {
        "schema": CAPTURE_EVIDENCE_MANIFEST_SCHEMA,
        "files": rows,
    }
    manifest_payload = (
        json.dumps(
            manifest_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if len(manifest_payload) > _MAX_MANIFEST_BYTES:
        raise InventoryConflict("capture evidence manifest exceeds its receipt-safe size limit")
    manifest = {
        **manifest_document,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "file_count": len(rows),
        "total_bytes": total_bytes,
    }
    return CaptureEvidenceSnapshot(
        root=canonical,
        files=tuple(ordered_paths),
        identities=tuple(identities),
        manifest=manifest,
    )


def _snapshot_json(
    snapshot: CaptureEvidenceSnapshot,
    relative: str,
) -> dict[str, object]:
    rows = snapshot.manifest.get("files")
    if not isinstance(rows, list):
        raise InventoryConflict("capture evidence manifest has no file inventory")
    matches = [row for row in rows if isinstance(row, dict) and row.get("path") == relative]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise InventoryConflict(f"capture evidence manifest does not bind {relative}")
    payload = _stable_file_bytes(
        snapshot.root / Path(relative),
        maximum_bytes=_MAX_BATCH_JSON_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != matches[0]["sha256"]:
        raise InventoryConflict(f"capture evidence changed after hashing: {relative}")
    def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        merged: dict[str, object] = {}
        for key, value in pairs:
            if key in merged:
                raise InventoryConflict(
                    f"capture evidence JSON repeats key {key!r}: {relative}"
                )
            merged[key] = value
        return merged

    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InventoryConflict(f"capture evidence JSON is invalid: {relative}") from error
    if not isinstance(document, dict):
        raise InventoryConflict(f"capture evidence JSON is not an object: {relative}")
    return document


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _approval_binding(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InventoryConflict("accepted batch has a malformed manual approval")
    binding = value.get("binding_sha256")
    if not isinstance(binding, str) or not _is_sha256(binding):
        raise InventoryConflict("accepted batch has a malformed manual approval")
    return binding


def _usb_component(value: object, *, address: bool) -> int:
    lower, upper = (1, 127) if address else (0, 999)
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        label = "address" if address else "bus"
        raise InventoryConflict(f"accepted batch USB {label} is malformed")
    return value


def _artifact_row(
    row_by_path: dict[str, dict[str, object]],
    relative: str,
) -> dict[str, object]:
    row = row_by_path.get(relative)
    if not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] < 0 or not _is_sha256(row.get("sha256")):
        raise InventoryConflict(f"capture evidence manifest does not bind {relative}")
    return row


def validate_six_frame_batch_capture_evidence(
    snapshot: CaptureEvidenceSnapshot,
) -> dict[str, object]:
    """Bind the durable post-finalization tree for one completed six-frame batch.

    Coolscan deletes each 619 MB ``capture.bin`` only after hashing, decoding,
    and verifying its private completion outputs.  The caller-owned tree keeps
    the six journals, three-pass meter sidecars, parent ACKs, first-frame live
    index artifacts, and sealed batch inputs.  Those are the artifacts this
    function requires and cross-binds; a deleted fine-stream scratch file is
    deliberately not part of the retained contract.
    """

    rows = snapshot.manifest.get("files")
    if not isinstance(rows, list):
        raise InventoryConflict("capture evidence manifest has no file inventory")
    typed_rows = [row for row in rows if isinstance(row, dict) and isinstance(row.get("path"), str)]
    if len(typed_rows) != len(rows):
        raise InventoryConflict("capture evidence manifest has a malformed file row")
    relative_paths = {row["path"] for row in typed_rows}
    if len(relative_paths) != len(typed_rows):
        raise InventoryConflict("capture evidence manifest has duplicate file paths")
    row_by_path = {row["path"]: row for row in typed_rows}
    session_journals: list[PurePosixPath] = []
    for relative in relative_paths:
        path = PurePosixPath(relative)
        if len(path.parts) == 2 and path.name == "session-journal.json" and _SIX_FRAME_BATCH_ROOT_RE.fullmatch(path.parts[0]) is not None:
            session_journals.append(path)
    if len(session_journals) != 1:
        raise InventoryConflict("capture evidence must contain exactly one slots-1-through-6 batch session")

    batch_root = session_journals[0].parent
    if any(PurePosixPath(relative).parts[:1] != batch_root.parts for relative in relative_paths):
        raise InventoryConflict("accepted batch capture evidence contains files outside the completed six-frame batch")
    first_frame = batch_root / "frame-001"
    required = {
        str(batch_root / "batch-job.json"),
        str(batch_root / _FIRST_PLAN_NAME),
        str(batch_root / _CONTINUATION_PLAN_NAME),
        str(batch_root / _CAPTURE_MANIFEST_NAME),
        str(batch_root / "session-journal.json"),
        str(batch_root / "stdout.txt"),
        str(batch_root / "stderr.txt"),
        str(first_frame / "capture-preview.bin"),
        str(first_frame / "capture-008e.bin"),
        str(first_frame / "capture-frame-map.json"),
        *(str(batch_root / f"frame-{slot:03d}" / "journal.json") for slot in _SIX_FRAME_SLOTS),
        *(str(batch_root / f"frame-{slot:03d}" / "capture-meter.bin") for slot in _SIX_FRAME_SLOTS),
        *(str(batch_root / f"frame-{slot:03d}" / "parent-ack.json") for slot in _SIX_FRAME_SLOTS),
    }
    missing = sorted(required - relative_paths)
    if missing:
        raise InventoryConflict("accepted batch capture evidence is incomplete: " + ", ".join(missing))

    job_path = str(batch_root / "batch-job.json")
    session_path = str(batch_root / "session-journal.json")
    job = _snapshot_json(snapshot, job_path)
    session = _snapshot_json(snapshot, session_path)
    if set(job) != _BATCH_JOB_KEYS:
        raise InventoryConflict("accepted batch job has an unexpected schema")

    session_id = job.get("session_id")
    if not isinstance(session_id, str) or session_id != batch_root.name:
        raise InventoryConflict("accepted batch job session does not name its batch root")
    if (
        job.get("schema_version") != 3
        or job.get("session_contract") != "one-process-one-reservation"
        or job.get("apply_all_boundary_offsets_before_first_frame") is not True
        or job.get("parent_ack_required_after_every_frame") is not True
        or job.get("release_once_after_last_frame") is not True
    ):
        raise InventoryConflict("accepted batch job does not preserve the one-shot contract")

    expected_usb_bus = _usb_component(job.get("expected_usb_bus"), address=False)
    expected_usb_address = _usb_component(job.get("expected_usb_address"), address=True)
    reviewed_fingerprint = job.get("reviewed_roll_fingerprint")
    reviewed_binding = reviewed_fingerprint.get("binding_sha256") if isinstance(reviewed_fingerprint, dict) else None
    if not _is_sha256(reviewed_binding):
        raise InventoryConflict("accepted batch job has no reviewed fingerprint")

    plan_path = str(batch_root / _FIRST_PLAN_NAME)
    continuation_path = str(batch_root / _CONTINUATION_PLAN_NAME)
    manifest_path = str(batch_root / _CAPTURE_MANIFEST_NAME)
    plan_row = _artifact_row(row_by_path, plan_path)
    continuation_row = _artifact_row(row_by_path, continuation_path)
    plan_sha256 = job.get("capture_plan_sha256")
    continuation_sha256 = job.get("continuation_plan_sha256")
    if (
        not _is_sha256(plan_sha256)
        or plan_sha256 != plan_row["sha256"]
        or not _is_sha256(continuation_sha256)
        or continuation_sha256 != continuation_row["sha256"]
    ):
        raise InventoryConflict("accepted batch job is not bound to its retained plans")
    capture_manifest = _snapshot_json(snapshot, manifest_path)
    if capture_manifest.get("plan_sha256") != plan_sha256:
        raise InventoryConflict("accepted batch capture manifest is not bound to its plan")

    frames = job.get("frames")
    if not isinstance(frames, list) or len(frames) != len(_SIX_FRAME_SLOTS):
        raise InventoryConflict("accepted batch job does not contain exactly six frames")
    frame_jobs: dict[int, dict[str, object]] = {}
    approval_sha256_by_slot: dict[str, str | None] = {}
    for expected_slot, frame in zip(_SIX_FRAME_SLOTS, frames, strict=True):
        if not isinstance(frame, dict) or set(frame) != _BATCH_FRAME_KEYS:
            raise InventoryConflict("accepted batch job has an invalid frame entry")
        expected_directory = f"frame-{expected_slot:03d}"
        offset = frame.get("boundary_offset_rows")
        minimum_offset = 0 if expected_slot == 1 else -144
        if (
            frame.get("slot") != expected_slot
            or frame.get("ack") != f"{expected_directory}/parent-ack.json"
            or frame.get("journal") != f"{expected_directory}/journal.json"
            or frame.get("output") != f"{expected_directory}/capture.bin"
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or not minimum_offset <= offset <= 144
        ):
            raise InventoryConflict("accepted batch job frame paths or order are invalid")
        approval_sha256_by_slot[str(expected_slot)] = _approval_binding(frame.get("manual_review_approval"))
        frame_jobs[expected_slot] = frame

    # session-journal.json and the frame journals are engine-owned documents:
    # their full key sets belong to the pinned Coolscan engine, so they are
    # deliberately not closed-world key-checked here (the NegPy-owned
    # batch-job/frame/ACK documents above are). What pins their schema is the
    # exact engine identity below — the session must carry the worker and
    # bundle hashes of the coolscanpy actually imported by this process — plus
    # duplicate-key rejection in _snapshot_json and strict per-field checks on
    # every load-bearing key.
    from coolscanpy.protocol.ls5000_single_pass.bundle import (
        CAPTURE_BUNDLE_SHA256,
        CAPTURE_WORKER_SHA256,
    )

    job_sha256 = _artifact_row(row_by_path, job_path)["sha256"]
    engine_sha256 = session.get("capture_engine_sha256")
    bundle_sha256 = session.get("capture_bundle_sha256")
    density_calibration = session.get("nikon_density_calibration")
    if (
        session.get("status") != "complete"
        or session.get("session_id") != session_id
        or session.get("density_calibration_session_id") != session_id
        or not isinstance(density_calibration, dict)
        or density_calibration.get("session_id") != session_id
        or session.get("batch_job_sha256") != job_sha256
        or session.get("selected_slots") != list(_SIX_FRAME_SLOTS)
        or session.get("completed_slots") != list(_SIX_FRAME_SLOTS)
        or session.get("plan_sha256") != plan_sha256
        or session.get("continuation_plan_sha256") != continuation_sha256
        or engine_sha256 != CAPTURE_WORKER_SHA256
        or bundle_sha256 != CAPTURE_BUNDLE_SHA256
        or session.get("manual_review_approval_sha256_by_slot") != approval_sha256_by_slot
        or session.get("reviewed_roll_fingerprint_sha256") != reviewed_binding
        or session.get("expected_usb_bus") != expected_usb_bus
        or session.get("expected_usb_address") != expected_usb_address
        or session.get("actual_usb_bus") != expected_usb_bus
        or session.get("actual_usb_address") != expected_usb_address
        or session.get("reservation_acquired") is not True
        or session.get("unit_release_attempts") != 1
        or session.get("unit_released") is not True
        or session.get("recovery_required") != "none"
        or session.get("active_frame_index") is not None
        or session.get("active_slot") is not None
    ):
        raise InventoryConflict("capture evidence is not bound to one completed six-frame batch")

    journal_sha256_by_slot: dict[str, object] = {}
    output_sha256_by_slot: dict[str, object] = {}
    meter_sha256_by_slot: dict[str, object] = {}
    ack_sha256_by_slot: dict[str, object] = {}
    first_journal: dict[str, object] | None = None
    for frame_index, slot in enumerate(_SIX_FRAME_SLOTS, start=1):
        frame_root = batch_root / f"frame-{slot:03d}"
        journal_path = str(frame_root / "journal.json")
        meter_path = str(frame_root / "capture-meter.bin")
        ack_path = str(frame_root / "parent-ack.json")
        journal = _snapshot_json(snapshot, journal_path)
        frame_job = frame_jobs[slot]
        batch_session = journal.get("batch_session")
        output_sha256 = journal.get("output_sha256")
        selection = journal.get("live_frame_selection")
        roll_identity = selection.get("roll_identity") if isinstance(selection, dict) else None
        selected_comparison = roll_identity.get("selected_slot_comparison") if isinstance(roll_identity, dict) else None
        comparison = roll_identity.get("comparison") if isinstance(roll_identity, dict) else None
        expected_output = str(snapshot.root / Path(str(batch_root / frame_job["output"])))
        if (
            journal.get("status") != "frame-complete"
            or journal.get("requested_frame") != slot
            or journal.get("requested_boundary_offset_rows") != frame_job["boundary_offset_rows"]
            or journal.get("frame_complete") is not True
            or journal.get("session_reservation_retained") is not True
            or journal.get("unit_released") is not False
            or journal.get("recovery_required") not in (None, "none")
            or journal.get("capture_mode") != "full"
            or journal.get("expected_reads") != _FINE_READS
            or journal.get("completed_reads") != _FINE_READS
            or journal.get("expected_bytes") != _FINE_BYTES
            or journal.get("completed_bytes") != _FINE_BYTES
            or journal.get("disk_bytes") != _FINE_BYTES
            or journal.get("output") != expected_output
            or not _is_sha256(output_sha256)
            or journal.get("plan_sha256") != plan_sha256
            or journal.get("continuation_plan_sha256") != continuation_sha256
            or journal.get("capture_engine_sha256") != engine_sha256
            or journal.get("capture_bundle_sha256") != bundle_sha256
            or journal.get("manual_review_approval") != frame_job["manual_review_approval"]
            or journal.get("reviewed_roll_fingerprint_sha256") != reviewed_binding
            or journal.get("expected_usb_bus") != expected_usb_bus
            or journal.get("expected_usb_address") != expected_usb_address
            or journal.get("actual_usb_bus") != expected_usb_bus
            or journal.get("actual_usb_address") != expected_usb_address
            or journal.get("density_calibration_session_id") != session_id
            or journal.get("nikon_density_calibration") != density_calibration
            or not isinstance(batch_session, dict)
            or batch_session.get("session_id") != session_id
            or batch_session.get("frame_index") != frame_index
            or batch_session.get("frame_total") != len(_SIX_FRAME_SLOTS)
            or batch_session.get("selected_slots") != list(_SIX_FRAME_SLOTS)
            or not isinstance(selection, dict)
            or selection.get("frame") != slot
            or selection.get("requested_boundary_offset_rows") != frame_job["boundary_offset_rows"]
            or selection.get("applied_boundary_offset_rows") != frame_job["boundary_offset_rows"]
            or not isinstance(roll_identity, dict)
            or roll_identity.get("reviewed_fingerprint_sha256") != reviewed_binding
            or not _is_sha256(roll_identity.get("fresh_fingerprint_sha256"))
            or not isinstance(comparison, dict)
            or comparison.get("matches") is not True
            or comparison.get("reason") != "matched"
            or not isinstance(selected_comparison, dict)
            or selected_comparison.get("matches") is not True
            or selected_comparison.get("reason") != "matched"
            or selected_comparison.get("slot") != slot
        ):
            raise InventoryConflict(f"capture evidence frame {slot} journal is not bound to the completed batch")

        meter_row = _artifact_row(row_by_path, meter_path)
        meter_evidence = journal.get("meter_evidence")
        if (
            meter_row["bytes"] != _METER_BYTES
            or not isinstance(meter_evidence, dict)
            or meter_evidence.get("path") != str(snapshot.root / Path(str(frame_root / "capture-meter.bin")))
            or meter_evidence.get("bytes") != _METER_BYTES
            or meter_evidence.get("sha256") != meter_row["sha256"]
            or meter_evidence.get("complete") is not True
            or meter_evidence.get("durable_completed_passes") != 3
            or journal.get("meter_evidence_persisted_before_fine_arm") is not True
            or journal.get("meter_group_bytes") != [_METER_PASS_BYTES] * 3
            or journal.get("meter_group_offsets") != [0, _METER_PASS_BYTES, _METER_PASS_BYTES * 2]
            or journal.get("meter_completed_reads") != 15
            or journal.get("meter_completed_bytes") != _METER_BYTES
            or journal.get("meter_layout") != _METER_LAYOUT
        ):
            raise InventoryConflict(f"capture evidence frame {slot} meter sidecar is not durably bound")

        ack = _snapshot_json(snapshot, ack_path)
        nonce = journal.get("ack_nonce")
        if (
            set(ack) != _ACK_KEYS
            or not isinstance(nonce, str)
            or _ACK_NONCE_RE.fullmatch(nonce) is None
            or ack.get("ack_nonce") != nonce
            or ack.get("action") != "continue"
            or ack.get("frame_index") != frame_index
            or ack.get("schema_version") != 1
            or ack.get("session_id") != session_id
            or ack.get("slot") != slot
        ):
            raise InventoryConflict(f"capture evidence frame {slot} parent ACK is not bound to its journal")

        journal_sha256_by_slot[str(slot)] = _artifact_row(row_by_path, journal_path)["sha256"]
        output_sha256_by_slot[str(slot)] = output_sha256
        meter_sha256_by_slot[str(slot)] = meter_row["sha256"]
        ack_sha256_by_slot[str(slot)] = _artifact_row(row_by_path, ack_path)["sha256"]
        if slot == 1:
            first_journal = journal

    assert first_journal is not None
    preview_path = str(first_frame / "capture-preview.bin")
    table_path = str(first_frame / "capture-008e.bin")
    mapping_path = str(first_frame / "capture-frame-map.json")
    preview_row = _artifact_row(row_by_path, preview_path)
    table_row = _artifact_row(row_by_path, table_path)
    mapping = _snapshot_json(snapshot, mapping_path)
    live_index_artifacts = first_journal.get("live_index_artifacts")
    live_index_evidence = first_journal.get("live_index_evidence")
    if (
        preview_row["bytes"] != _PREVIEW_BYTES
        or not isinstance(live_index_artifacts, dict)
        or live_index_artifacts
        != {
            "preview": str(snapshot.root / Path(preview_path)),
            "table": str(snapshot.root / Path(table_path)),
            "mapping": str(snapshot.root / Path(mapping_path)),
        }
        or not isinstance(live_index_evidence, dict)
        or live_index_evidence.get("status") != "persisted-before-frame-detection"
        or live_index_evidence.get("preview_bytes") != preview_row["bytes"]
        or live_index_evidence.get("preview_sha256") != preview_row["sha256"]
        or live_index_evidence.get("table_bytes") != table_row["bytes"]
        or live_index_evidence.get("table_sha256") != table_row["sha256"]
        or mapping != first_journal.get("live_frame_selection")
    ):
        raise InventoryConflict("capture evidence live preview, transport table, or frame map is not bound")

    return {
        "batch_root": str(batch_root),
        "session_id": session_id,
        "batch_job_sha256": job_sha256,
        "capture_plan_sha256": plan_sha256,
        "continuation_plan_sha256": continuation_sha256,
        "capture_engine_sha256": engine_sha256,
        "capture_bundle_sha256": bundle_sha256,
        "expected_usb_bus": expected_usb_bus,
        "expected_usb_address": expected_usb_address,
        "selected_slots": list(_SIX_FRAME_SLOTS),
        "reviewed_fingerprint_sha256": reviewed_binding,
        "first_frame_directory": str(first_frame),
        "session_journal_sha256": _artifact_row(row_by_path, session_path)["sha256"],
        "frame_journal_sha256_by_slot": journal_sha256_by_slot,
        "frame_output_sha256_by_slot": output_sha256_by_slot,
        "frame_meter_sha256_by_slot": meter_sha256_by_slot,
        "frame_ack_sha256_by_slot": ack_sha256_by_slot,
        "preview_sha256": preview_row["sha256"],
        "transport_table_sha256": table_row["sha256"],
    }


def bind_capture_evidence_inventory(
    snapshot: CaptureEvidenceSnapshot,
    inventory: InventorySnapshot,
) -> dict[str, object]:
    """Bind file hashes to the lease's exact post-close directory inventory."""

    actual = dict(inventory.files)
    expected = dict(snapshot.identities)
    actual.pop(OUTPUT_LOCK_NAME, None)
    if actual != expected:
        raise InventoryConflict("capture evidence changed between hashing and lease inventory")
    return {
        "path": str(snapshot.root),
        "retained": True,
        "directory_count": len(inventory.directories),
        **snapshot.manifest,
    }
