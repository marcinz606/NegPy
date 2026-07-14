#!/usr/bin/env python3
"""Fail-closed PyUSB capture for the verified Nikon RGBI4x replay plan.

The default mode only validates the plan.  Hardware access requires the
explicit ``--live`` flag, a fresh scanner power cycle, and inserted film.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import secrets
import shutil
import struct
import sys
import time
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

from . import meter as meter_module
from .bundle import (
    CAPTURE_BUNDLE_COMPONENT_SHA256,
    CAPTURE_BUNDLE_SHA256,
    CAPTURE_WORKER_SHA256,
)
from .continuation_plan import (
    CANONICAL_CONTINUATION_PLAN_SHA256,
    derive_equivalent_continuation_blocks,
    verify_canonical_continuation_plan,
)
from .meter import (
    DEFAULT_EXPOSURES,
    MeterObservation,
    observe_meter_pass,
    propose_next_exposures,
    verify_final_convergence,
)
from .roll_index import (
    IndexGeometry,
    NativeFrameOrigin,
    RollDetection,
    TransportRecord,
    TransportMapping,
    decode_full_index_bytes,
    derive_transport_mapping,
    detect_roll_frames,
    parse_live_transport_records_bytes,
    transport_native_origin,
    validate_live_0x8e_bytes,
)
from .window import WindowBlock, decode_window_block


HERE = Path(__file__).resolve().parent
DATA_PACKAGE = "negpy.infrastructure.scanners.ls5000_single_pass.data"

EXPECTED_FINE_CDB = "280000000001032c0080"
EXPECTED_FINE_REQUEST = 207_872
EXPECTED_FINE_READS = 2_980
READY_POLL_SECONDS = 0.1
READY_POLL_DEADLINE_SECONDS = 120.0
RETRYABLE_BUSY_SENSES = {"020401"}
# TEST UNIT READY is an idempotent status query, so it is safe to keep
# polling through the cold-insert ``04/02`` state as well.  Do not add this
# sense to RETRYABLE_BUSY_SENSES: data-bearing commands must still refuse it
# instead of being reissued.
READY_POLL_TRANSIENT_SENSES = RETRYABLE_BUSY_SENSES | {"020402"}
CANONICAL_BUSY_STATUS = bytes.fromhex("0002040100000000")
STARTUP_UNIT_ATTENTION_SENSES = {"062800", "062900", "063f03"}
FINE_GET_WINDOW_SEQUENCES = (603, 604, 605, 606)
PREVIEW_SET_WINDOW_SEQUENCES = (88, 89, 90)
PREVIEW_READ_SEQUENCES = tuple(range(118, 166))
FRAME_TABLE_SEND_SEQUENCE = 174
FRAME_TABLE_SEND_RECORDS = 37
FRAME_TABLE_SEND_BYTES = 4 + FRAME_TABLE_SEND_RECORDS * 8
AUTOFOCUS_SEQUENCE = 231
DYNAMIC_WINDOW_GROUPS = (
    (503, 504, 505, 506),
    (530, 531, 532, 533),
    (556, 557, 558, 559),
    (581, 582, 583, 584),
)
DYNAMIC_WINDOW_SEQUENCES = tuple(
    sequence for group in DYNAMIC_WINDOW_GROUPS for sequence in group
)
METER_GET_WINDOW_GROUPS = (
    (518, 519, 520, 521),
    (544, 545, 546, 547),
    (570, 571, 572, 573),
)
METER_GET_WINDOW_SEQUENCES = tuple(
    sequence for group in METER_GET_WINDOW_GROUPS for sequence in group
)
METER_READ_GROUPS = (
    (522, 523, 524, 525, 526),
    (548, 549, 550, 551, 552),
    (574, 575, 576, 577, 578),
)
METER_READ_SEQUENCES = tuple(
    sequence for group in METER_READ_GROUPS for sequence in group
)
METER_GROUP_BYTES = 1_088_000
METER_CAPTURE_BYTES = len(METER_READ_GROUPS) * METER_GROUP_BYTES
METER_STOP_SEQUENCE = METER_READ_GROUPS[-1][-1]
WIRE_METER_COLORS = (9, 1, 2, 3)
WIRE_COLOR_TO_CONTROLLER_CHANNEL = {9: "IR", 1: "R", 2: "G", 3: "B"}
CONTROLLER_CHANNELS = ("R", "G", "B", "IR")
DRAINED_SCAN_READ_SEQUENCES = (
    PREVIEW_READ_SEQUENCES[-1],
    *(group[-1] for group in METER_READ_GROUPS),
)
FINE_NATIVE_WIDTH = 3_946
FINE_NATIVE_HEIGHT = 5_959
EXPECTED_PREVIEW_BYTES = 6_250_496
VARIABLE_FRAME_TABLE_SEQUENCE = 64
VARIABLE_FRAME_TABLE_CDB = "28008f00000300014a80"
VARIABLE_FRAME_TABLE_MAX_BYTES = 330


class ProtocolError(RuntimeError):
    pass


class SynchronizedProtocolError(ProtocolError):
    """The command status was fully consumed; another CDB is safe."""


class DesynchronizedProtocolError(ProtocolError):
    """The current USB/application phase is unknown; send no more CDBs."""


class CountedBulkReadError(OSError):
    def __init__(self, message: str, *, backend_error_code: int, transferred: int):
        error_number = errno.EPIPE if backend_error_code == -9 else None
        super().__init__(error_number, message)
        self.backend_error_code = backend_error_code
        self.transferred = transferred


class StartupFrameTable(TypedDict):
    """Validated summary of the bounded startup READ(0x8f) payload."""

    bytes: int
    count: int
    header: str
    sha256: str


@dataclass(frozen=True)
class LiveFrameSelection:
    """One frame selected solely from a same-traversal preview and 0x8e table."""

    frame: int
    frame_count: int
    geometry: IndexGeometry
    usable_rows: int
    detection: RollDetection
    mapping: TransportMapping
    base_selected: NativeFrameOrigin
    selected: NativeFrameOrigin
    requested_boundary_offset_rows: int
    applied_boundary_offset_rows: int
    preview_sha256: str
    table_sha256: str
    decode_report: dict[str, Any]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "frame_count": self.frame_count,
            "usable_rows": self.usable_rows,
            "preview_sha256": self.preview_sha256,
            "table_sha256": self.table_sha256,
            "geometry": {
                "requested_resolution": self.geometry.requested_resolution,
                "native_resolution": self.geometry.native_resolution,
                "pitch": self.geometry.pitch,
                "native_width": self.geometry.native_width,
                "native_height": self.geometry.native_height,
                "width": self.geometry.width,
                "height": self.geometry.height,
                "expected_stream_bytes": self.geometry.expected_stream_bytes,
            },
            "decode_report": self.decode_report,
            "boundary_offset": {
                "requested_rows": self.requested_boundary_offset_rows,
                "applied_rows": self.applied_boundary_offset_rows,
                "base_lookup_row": self.base_selected.lookup_row,
                "resolved_lookup_row": self.selected.lookup_row,
                "base_native_origin": self.base_selected.native_origin,
                "resolved_native_origin": self.selected.native_origin,
            },
            "detection": self.detection.diagnostics(),
            "transport_mapping": self.mapping.diagnostics(),
            "selected": {
                "frame": self.selected.frame,
                "lookup_row": self.selected.lookup_row,
                "code": self.selected.code,
                "selector": self.selected.selector,
                "native_origin": self.selected.native_origin,
                "automatic": self.selected.automatic,
                "manual_review": self.selected.manual_review,
                "method": self.selected.method,
            },
        }


@dataclass(frozen=True)
class ExecutableContinuationStep:
    """One pinned later-frame semantic step and its canonical USB template."""

    code: str
    entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BatchFrameSpec:
    """One frame and its parent-owned handshake paths inside a batch root."""

    slot: int
    boundary_offset_rows: int
    output: Path
    journal: Path
    ack: Path


@dataclass(frozen=True)
class LiveBatchJob:
    """Validated one-process/one-reservation batch contract."""

    session_id: str
    root: Path
    frames: tuple[BatchFrameSpec, ...]
    plan_sha256: str
    continuation_plan_sha256: str
    job_sha256: str

    @property
    def selected_slots(self) -> tuple[int, ...]:
        return tuple(frame.slot for frame in self.frames)


@dataclass
class SessionLifecycle:
    """Mutable scanner state shared with fail-closed batch cleanup."""

    at_transaction_boundary: bool = True
    scan_active: bool = False
    ready_required: bool = False


class CountedBulkInEndpoint:
    """libusb bulk-IN wrapper preserving the count returned on errors."""

    def __init__(self, endpoint: Any):
        self._endpoint = endpoint
        self.device = endpoint.device
        self.bEndpointAddress = endpoint.bEndpointAddress
        self._backend = self.device._ctx.backend
        self._handle = self.device._ctx.handle
        if (
            self._handle is None
            or not hasattr(self._handle, "handle")
            or not hasattr(self._backend, "lib")
            or not hasattr(self._backend.lib, "libusb_bulk_transfer")
        ):
            raise ProtocolError(
                "active PyUSB backend cannot report partial bulk-transfer counts"
            )
        self._buffers: dict[int, bytearray] = {}

    def read(self, size: int, timeout: int | None = None) -> bytes:
        buffer = self._buffers.setdefault(size, bytearray(size))
        raw_buffer = (ctypes.c_ubyte * size).from_buffer(buffer)
        transferred = ctypes.c_int()
        result = self._backend.lib.libusb_bulk_transfer(
            self._handle.handle,
            self.bEndpointAddress,
            ctypes.cast(raw_buffer, ctypes.POINTER(ctypes.c_ubyte)),
            size,
            ctypes.byref(transferred),
            0 if timeout is None else timeout,
        )
        if result == 0:
            return bytes(memoryview(buffer)[: transferred.value])
        raise CountedBulkReadError(
            f"libusb bulk read failed with code {result} after "
            f"{transferred.value} bytes",
            backend_error_code=result,
            transferred=transferred.value,
        )

    def clear_halt(self) -> None:
        self._endpoint.clear_halt()


@dataclass(frozen=True)
class TransactionResult:
    phase: int
    payload: bytes
    status: bytes
    sense: str
    stall_recoveries: int


def _write_exact(endpoint: Any, payload: bytes, timeout_ms: int) -> None:
    written = endpoint.write(payload, timeout=timeout_ms)
    if written != len(payload):
        raise ProtocolError(f"short USB write: {written} of {len(payload)} bytes")


def _is_pipe_error(error: Exception) -> bool:
    return (
        getattr(error, "errno", None) in (errno.EPIPE, 32)
        or "pipe" in str(error).lower()
    )


def _read_with_one_stall_recovery(
    endpoint: Any,
    size: int,
    timeout_ms: int,
) -> tuple[bytes, int]:
    try:
        return bytes(endpoint.read(size, timeout=timeout_ms)), 0
    except Exception as error:
        if not _is_pipe_error(error):
            raise DesynchronizedProtocolError(
                f"bulk read failed before command status: {error}"
            ) from error
        transferred = getattr(error, "transferred", None)
        if transferred != 0:
            detail = "unknown" if transferred is None else str(transferred)
            raise DesynchronizedProtocolError(
                f"PIPE after {detail} transferred bytes; refusing an ambiguous retry"
            ) from error
        endpoint.clear_halt()
        # Only a counted zero-byte PIPE proves the data phase was untouched.
        try:
            return bytes(endpoint.read(size, timeout=timeout_ms)), 1
        except Exception as retry_error:
            raise DesynchronizedProtocolError(
                f"bulk read failed after zero-byte PIPE recovery: {retry_error}"
            ) from retry_error


def perform_transaction(
    ep_out: Any,
    ep_in: Any,
    entry: dict,
    *,
    data_timeout_ms: int,
) -> TransactionResult:
    cdb = bytes.fromhex(entry["cdb"])

    def read_stage(size: int, stage: str) -> tuple[bytes, int]:
        try:
            return _read_with_one_stall_recovery(ep_in, size, data_timeout_ms)
        except DesynchronizedProtocolError as error:
            raise DesynchronizedProtocolError(
                f"command {entry['seq']} {entry.get('name')} CDB {entry['cdb']} "
                f"during {stage}: {error}"
            ) from error

    _write_exact(ep_out, cdb, 10_000)
    _write_exact(ep_out, b"\xd0", 10_000)

    try:
        phase_raw, phase_stalls = _read_with_one_stall_recovery(ep_in, 1, 30_000)
    except DesynchronizedProtocolError as error:
        raise DesynchronizedProtocolError(
            f"command {entry['seq']} {entry.get('name')} CDB {entry['cdb']} "
            f"during phase: {error}"
        ) from error
    if len(phase_raw) != 1:
        raise DesynchronizedProtocolError(
            f"command {entry['seq']}: phase length {len(phase_raw)} != 1"
        )
    phase = phase_raw[0]
    payload = b""
    data_stalls = 0
    if phase == 0x02:
        data_out = bytes.fromhex(entry.get("data_out", ""))
        if data_out:
            _write_exact(ep_out, data_out, 30_000)
    elif phase == 0x03:
        request_len = entry.get("request_len", 0)
        if request_len <= 0:
            raise DesynchronizedProtocolError(
                f"command {entry['seq']}: missing data-in request length"
            )
        request_parts = entry.get("request_parts") or [request_len]
        if sum(request_parts) != request_len or any(
            part <= 0 for part in request_parts
        ):
            raise DesynchronizedProtocolError(
                f"command {entry['seq']}: invalid data-in transfer parts {request_parts}"
            )
        payload_parts = []
        total_received = 0
        for part_index, part_len in enumerate(request_parts):
            remaining = part_len
            while remaining:
                part, part_stalls = read_stage(
                    remaining,
                    f"data part {part_index + 1}/{len(request_parts)} "
                    f"after {total_received} of {request_len} bytes",
                )
                data_stalls += part_stalls
                if not part:
                    raise DesynchronizedProtocolError(
                        f"command {entry['seq']}: zero-byte data transfer with "
                        f"{remaining} bytes still declared"
                    )
                payload_parts.append(part)
                remaining -= len(part)
                total_received += len(part)
                # A positive short packet completes one host USB transfer,
                # not necessarily Nikon's logical SCSI data phase.  Keep
                # reading until this command's *live-bound* allocation is
                # consumed.  For variable table 0x8e, command 171 supplies
                # that allocation: this roll declared 0x529a (21,146), not
                # the trace's stale 0x52c4.  Replaying 0x52c4 caused the real
                # ILI/ABORTED COMMAND/ASC 4B data-phase error in run 5; run 6
                # confirmed there were no additional 42 bytes.
        payload = b"".join(payload_parts)
    elif phase not in (0x00, 0x01, 0x04):
        raise DesynchronizedProtocolError(
            f"command {entry['seq']}: unknown phase 0x{phase:02x}"
        )

    # A complete data phase is followed by Nikon's explicit status trigger.
    _write_exact(ep_out, b"\x06", 10_000)
    try:
        status, status_stalls = _read_with_one_stall_recovery(ep_in, 8, 15_000)
    except DesynchronizedProtocolError as error:
        raise DesynchronizedProtocolError(
            f"command {entry['seq']} {entry.get('name')} CDB {entry['cdb']} "
            f"during status: {error}"
        ) from error
    if len(status) != 8:
        raise DesynchronizedProtocolError(
            f"command {entry['seq']}: status length {len(status)} != 8"
        )
    return TransactionResult(
        phase=phase,
        payload=payload,
        status=status,
        sense=status[1:4].hex(),
        stall_recoveries=phase_stalls + data_stalls + status_stalls,
    )


def validate_plan(plan: list[dict], manifest: dict | None = None) -> dict:
    if not plan:
        raise ProtocolError("replay plan is empty")
    if any(entry.get("resync_before", 0) for entry in plan):
        raise ProtocolError("replay plan contains parser resync loss")
    if len(plan) != 607 or [entry.get("seq") for entry in plan] != list(range(1, 608)):
        raise ProtocolError("replay plan is not the canonical 607-command prefix")
    allowed_opcodes = {
        0x00,
        0x12,
        0x15,
        0x16,
        0x1B,
        0x24,
        0x25,
        0x28,
        0x2A,
        0xC1,
        0xE0,
        0xE1,
    }
    for entry in plan:
        cdb = bytes.fromhex(entry.get("cdb", ""))
        if not cdb or cdb[0] not in allowed_opcodes:
            raise ProtocolError(
                f"command {entry.get('seq')}: disallowed CDB {entry.get('cdb')}"
            )
    target = plan[-1]
    if target.get("role") != "fine-rgbi4-template":
        raise ProtocolError("last replay entry is not the RGBI4x template")
    if target.get("cdb") != EXPECTED_FINE_CDB:
        raise ProtocolError("fine READ CDB is not the verified 207,872-byte command")
    if target.get("request_len") != EXPECTED_FINE_REQUEST:
        raise ProtocolError("fine request length is not 207,872 bytes")
    if target.get("repeat") != EXPECTED_FINE_READS:
        raise ProtocolError("fine repeat count is not the resync-free 2,980-read count")

    fine_scans = [
        entry
        for entry in plan
        if entry.get("name") == "SCAN"
        and entry.get("data_out") == "09010203"
        and entry.get("cdb") == "1b0000000400"
    ]
    # Earlier 285-dpi AE scans use the same four-color SCAN payload.  The
    # true 4000-dpi fine arm is the final four-command reissue chain before
    # the fine READ template.
    if [entry.get("expected_sense") for entry in fine_scans[-4:]] != [
        "098002",
        "098006",
        "098007",
        "000000",
    ]:
        raise ProtocolError("fine SCAN reissue chain is not 02 -> 06 -> 07 -> success")

    if manifest is not None:
        checks = {
            "fine_colors": [9, 1, 2, 3],
            "resolution": 4000,
            "samples_per_scan": 4,
            "fine_read_cdb": EXPECTED_FINE_CDB,
            "fine_request_bytes": EXPECTED_FINE_REQUEST,
            "fine_read_count": EXPECTED_FINE_READS,
            "expected_stream_bytes": EXPECTED_FINE_REQUEST * EXPECTED_FINE_READS,
            "pcap_snaplen": 65_535,
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise ProtocolError(
                    f"manifest {key}={manifest.get(key)!r}, expected {expected!r}"
                )
        for key in ("plan_sha256", "source_pcap_sha256"):
            value = manifest.get(key, "")
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ProtocolError(f"manifest {key} is not a SHA-256 digest")
    return target


def _plan_hash(plan_path: Path) -> str:
    digest = hashlib.sha256()
    with plan_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _meter_controller_sha256() -> str:
    """Return the meter identity without requiring loose source in a frozen app."""

    if getattr(sys, "frozen", False):
        return CAPTURE_BUNDLE_COMPONENT_SHA256["meter.py"]
    return _plan_hash(Path(meter_module.__file__).resolve())


def _load_validated_plan(
    plan_path: Path,
    manifest_path: Path,
) -> tuple[list[dict], dict, str]:
    plan_bytes = plan_path.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = [
        json.loads(line) for line in plan_bytes.decode("utf-8").splitlines() if line
    ]
    manifest = json.loads(manifest_path.read_text())
    validate_plan(plan, manifest)
    if plan_sha256 != manifest["plan_sha256"]:
        raise ProtocolError(
            f"plan SHA-256 {plan_sha256} != manifest {manifest['plan_sha256']}"
        )
    return plan, manifest, plan_sha256


def _entry(plan: list[dict], sequence: int) -> dict:
    try:
        entry = plan[sequence - 1]
    except IndexError as error:
        raise ProtocolError(f"plan is missing sequence {sequence}") from error
    if entry.get("seq") != sequence:
        raise ProtocolError(
            f"plan position {sequence} contains sequence {entry.get('seq')!r}"
        )
    return entry


def _continuation_template_groups(
    plan: list[dict],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Collapse the canonical frame segment into the observed 89 steps.

    Command 500 is deliberately absent: both later Nikon frames omit the
    first-frame-only READ(0x8c).  Consecutive TEST UNIT READY commands are one
    state-aware poll step; their captured repetition counts are timing noise.
    """

    groups: list[tuple[dict[str, Any], ...]] = []
    sequence = 225
    while sequence <= 606:
        if sequence == 500:
            sequence += 1
            continue
        current = _entry(plan, sequence)
        if current.get("name") != "TEST_UNIT_READY":
            groups.append((current,))
            sequence += 1
            continue
        ready: list[dict[str, Any]] = []
        while sequence <= 606 and sequence != 500:
            candidate = _entry(plan, sequence)
            if candidate.get("name") != "TEST_UNIT_READY":
                break
            ready.append(candidate)
            sequence += 1
        groups.append(tuple(ready))
    return tuple(groups)


def _window_semantic(
    code: str,
    entry: dict[str, Any],
) -> list[object]:
    raw = entry.get("data_out") if code == "S" else entry.get("expected_data_in")
    if not isinstance(raw, str):
        raise ProtocolError(
            f"continuation {code} command {entry.get('seq')} has no window payload"
        )
    window = decode_window_block(bytes.fromhex(raw))
    if window is None:
        raise ProtocolError(
            f"continuation {code} command {entry.get('seq')} has a malformed window"
        )
    return [
        code,
        window["color_id"],
        window["resx"],
        window["resy"],
        window["upper_left_x"],
        window["upper_left_y"],
        window["width"],
        window["height"],
        window["multiread_byte"],
        window["avg_negpos_byte"],
        window["scanning_kind_byte"],
        window["scanning_mode_byte"],
        window["color_interleaving_byte"],
        window["ae_byte"],
        window["exposure_raw_10ns"],
    ]


def _require_semantic_match(
    actual: Sequence[object],
    expected: Sequence[object],
    *,
    step_index: int,
) -> None:
    if len(actual) != len(expected):
        raise ProtocolError(
            f"continuation step {step_index} field count changed"
        )
    for field_index, (observed, required) in enumerate(zip(actual, expected)):
        if required in ("$Y", "$FOCUS", "$EXPOSURE"):
            if isinstance(observed, bool) or not isinstance(observed, int):
                raise ProtocolError(
                    f"continuation step {step_index} dynamic field "
                    f"{field_index} is not an integer"
                )
            continue
        if observed != required:
            raise ProtocolError(
                f"continuation step {step_index} field {field_index} is "
                f"{observed!r}, expected {required!r}"
            )


def compile_continuation_steps(
    plan: list[dict],
    continuation_plan: dict[str, Any],
) -> tuple[ExecutableContinuationStep, ...]:
    """Bind the pinned later-frame semantics to canonical command templates.

    The result contains no reservation, roll-index upload, release, or eject
    command.  Dynamic frame origin and exposure values remain whatever the
    already-bound canonical plan contains; every other byte is checked against
    the two Nikon trace blocks before any command can reach hardware.
    """

    validate_plan(plan)
    derive_equivalent_continuation_blocks(continuation_plan)
    raw_steps = continuation_plan["trace_equivalence"]["semantic_steps"]
    groups = _continuation_template_groups(plan)
    if len(groups) != 89 or len(raw_steps) != len(groups):
        raise ProtocolError(
            f"continuation compiler produced {len(groups)} steps, expected 89"
        )

    compiled: list[ExecutableContinuationStep] = []
    window_origins: set[int] = set()
    autofocus_y: int | None = None
    for step_index, (entries, raw_step) in enumerate(
        zip(groups, raw_steps),
        start=1,
    ):
        if not isinstance(raw_step, list) or not raw_step:
            raise ProtocolError(
                f"continuation semantic step {step_index} is malformed"
            )
        required_step = cast(list[object], raw_step)
        code = raw_step[0]
        if not isinstance(code, str):
            raise ProtocolError(
                f"continuation semantic step {step_index} has no code"
            )
        entry = entries[0]
        if code == "R":
            if any(item.get("name") != "TEST_UNIT_READY" for item in entries):
                raise ProtocolError(
                    f"continuation step {step_index} is not a ready group"
                )
            if entries[-1].get("expected_sense") != "000000":
                raise ProtocolError(
                    f"continuation ready step {step_index} does not end ready"
                )
            actual = ["R"]
        elif code == "A":
            if entry.get("name") != "VENDOR_E0:AUTOFOCUS_EXEC":
                raise ProtocolError(
                    f"continuation step {step_index} is not autofocus"
                )
            payload = bytes.fromhex(entry.get("data_out", ""))
            if len(payload) != 9 or payload[0] != 0:
                raise ProtocolError("continuation autofocus payload is malformed")
            autofocus_y = int.from_bytes(payload[5:9], "big")
            actual = [
                "A",
                int.from_bytes(payload[1:5], "big"),
                autofocus_y,
            ]
        elif code == "X":
            actual = ["X"] if entry.get("name") == "EXECUTE" else ["invalid"]
        elif code == "F":
            if entry.get("name") != "VENDOR_E1:READ_FOCUS":
                raise ProtocolError(
                    f"continuation step {step_index} is not focus readback"
                )
            payload = bytes.fromhex(entry.get("expected_data_in", ""))
            if len(payload) != 9:
                raise ProtocolError("continuation focus readback is malformed")
            actual = ["F", int.from_bytes(payload[4:6], "big")]
        elif code == "Q":
            if entry.get("cdb") != "28009300000100000c80":
                raise ProtocolError(
                    f"continuation step {step_index} is not vendor 0x93 READ"
                )
            actual = ["Q", entry.get("expected_data_in")]
        elif code in ("S", "G"):
            required_name = "SET_WINDOW" if code == "S" else "GET_WINDOW:"
            if not str(entry.get("name", "")).startswith(required_name):
                raise ProtocolError(
                    f"continuation step {step_index} is not a {code} window"
                )
            actual = _window_semantic(code, entry)
            window_origin = actual[5]
            if isinstance(window_origin, bool) or not isinstance(window_origin, int):
                raise ProtocolError(
                    f"continuation step {step_index} has no integer window origin"
                )
            window_origins.add(window_origin)
        elif code == "N":
            if entry.get("name") != "SCAN":
                raise ProtocolError(
                    f"continuation step {step_index} is not SCAN"
                )
            actual = ["N", entry.get("expected_sense"), entry.get("data_out")]
        elif code == "V":
            if not str(entry.get("cdb", "")).startswith("280087"):
                raise ProtocolError(
                    f"continuation step {step_index} is not vendor 0x87 READ"
                )
            actual = [
                "V",
                entry.get("request_len"),
                entry.get("expected_data_in"),
            ]
        elif code == "M":
            if entry.get("name") != "READ":
                raise ProtocolError(
                    f"continuation step {step_index} is not meter READ"
                )
            actual = ["M", entry.get("cdb")]
        else:
            raise ProtocolError(
                f"continuation step {step_index} has unsupported code {code!r}"
            )
        _require_semantic_match(actual, required_step, step_index=step_index)
        compiled.append(ExecutableContinuationStep(code, entries))

    if len(window_origins) != 1 or autofocus_y is None:
        raise ProtocolError("continuation dynamic frame origin is inconsistent")
    native_origin = next(iter(window_origins))
    if autofocus_y != native_origin + FINE_NATIVE_HEIGHT // 2:
        raise ProtocolError(
            "continuation autofocus is not centered on the bound frame"
        )
    forbidden = {"RESERVE_UNIT", "RELEASE_UNIT", "VENDOR_E0:EJECT"}
    for step in compiled:
        for entry in step.entries:
            if entry.get("name") in forbidden or (
                entry.get("name") == "SEND" and entry.get("cdb", "").startswith("2a008f")
            ):
                raise ProtocolError(
                    "continuation contains a forbidden session-level command"
                )
    return tuple(compiled)


def _derive_index_geometry(plan: list[dict]) -> IndexGeometry:
    """Derive the guarded preview raster geometry from the canonical plan."""

    windows = []
    all_resolutions = []
    for entry in plan:
        if entry.get("name") != "SET_WINDOW" or not entry.get("data_out"):
            continue
        decoded = decode_window_block(bytes.fromhex(entry["data_out"]))
        if decoded is None:
            raise ProtocolError(
                f"command {entry.get('seq')}: malformed SET_WINDOW payload"
            )
        all_resolutions.append(decoded["resy"])
    for sequence in PREVIEW_SET_WINDOW_SEQUENCES:
        decoded = decode_window_block(bytes.fromhex(_entry(plan, sequence)["data_out"]))
        if decoded is None:
            raise ProtocolError(
                f"command {sequence}: malformed preview SET_WINDOW payload"
            )
        windows.append(decoded)

    if [window["color_id"] for window in windows] != [1, 2, 3]:
        raise ProtocolError("preview SET_WINDOW order is not RGB")
    first = windows[0]
    shared = (
        "resx",
        "resy",
        "upper_left_x",
        "upper_left_y",
        "width",
        "height",
        "bit_depth",
    )
    if any(
        any(window[field] != first[field] for field in shared) for window in windows[1:]
    ):
        raise ProtocolError("preview SET_WINDOW geometry is inconsistent")
    if (
        first["resx"] != 97
        or first["resy"] != 97
        or first["upper_left_x"] != 0
        or first["upper_left_y"] != 0
        or first["width"] != FINE_NATIVE_WIDTH
        or first["height"] != 250_278
        or first["bit_depth"] != 16
    ):
        raise ProtocolError("preview SET_WINDOW geometry is not the proven roll index")

    native_resolution = max(all_resolutions, default=0)
    if native_resolution != 4_000:
        raise ProtocolError(
            f"plan native resolution {native_resolution}, expected 4000"
        )
    pitch = native_resolution // first["resy"]
    width = first["width"] // pitch
    height = first["height"] // pitch
    expected_stream_bytes = sum(
        _entry(plan, sequence).get("request_len", 0)
        for sequence in PREVIEW_READ_SEQUENCES
    )
    if (
        pitch != 41
        or width != 96
        or height != 6_104
        or expected_stream_bytes != EXPECTED_PREVIEW_BYTES
        or height % 2
        or (height // 2) * 2_048 != expected_stream_bytes
    ):
        raise ProtocolError(
            "preview stream geometry does not match its READ allocation"
        )
    return IndexGeometry(
        requested_resolution=first["resy"],
        native_resolution=native_resolution,
        pitch=pitch,
        native_width=first["width"],
        native_height=first["height"],
        width=width,
        height=height,
        block_bytes=2_048,
        expected_stream_bytes=expected_stream_bytes,
    )


def _derive_live_frame_selection(
    plan: list[dict],
    preview_data: bytes,
    table_data: bytes,
    *,
    frame: int,
    boundary_offset_rows: int = 0,
    expected_frame_count: int | None = None,
) -> LiveFrameSelection:
    """Resolve one automatic frame origin from same-traversal live data."""

    geometry = _derive_index_geometry(plan)
    validated_table, usable_rows = validate_live_0x8e_bytes(table_data, geometry.height)
    rgb16, known, decode_report = decode_full_index_bytes(
        preview_data, geometry, usable_rows=usable_rows
    )
    detection = detect_roll_frames(
        rgb16,
        known,
        nominal_frame_rows=FINE_NATIVE_HEIGHT // geometry.pitch,
        expected_frame_count=expected_frame_count,
    )
    if detection.confidence != "high":
        raise ProtocolError(
            f"roll boundary lattice confidence is {detection.confidence!r}; "
            "unattended frame binding requires 'high'"
        )
    records = parse_live_transport_records_bytes(
        validated_table, maximum_rows=geometry.height
    )
    mapping = derive_transport_mapping(
        detection.boundaries,
        len(detection.intervals),
        records,
    )
    frame_count = len(mapping.origins)
    if not 1 <= frame <= frame_count:
        raise ProtocolError(
            f"requested frame {frame} is outside detected roll 1..{frame_count}"
        )
    base_selected = mapping.origins[frame - 1]
    if base_selected.frame != frame:
        raise ProtocolError("transport mapping frame order is inconsistent")
    if not base_selected.automatic or base_selected.manual_review:
        raise ProtocolError(
            f"frame {frame} transport origin requires manual review; "
            "refusing an unattended fine scan"
        )
    mapping, selected = apply_boundary_offset(
        mapping,
        records,
        frame=frame,
        offset_rows=boundary_offset_rows,
    )
    return LiveFrameSelection(
        frame=frame,
        frame_count=frame_count,
        geometry=geometry,
        usable_rows=usable_rows,
        detection=detection,
        mapping=mapping,
        base_selected=base_selected,
        selected=selected,
        requested_boundary_offset_rows=boundary_offset_rows,
        applied_boundary_offset_rows=selected.lookup_row - base_selected.lookup_row,
        preview_sha256=hashlib.sha256(preview_data).hexdigest(),
        table_sha256=hashlib.sha256(validated_table).hexdigest(),
        decode_report=decode_report,
    )


def _derive_live_batch_selections(
    plan: list[dict],
    preview_data: bytes,
    table_data: bytes,
    frames: Sequence[BatchFrameSpec],
) -> tuple[LiveFrameSelection, ...]:
    """Pre-bind every batch frame to one same-traversal transport table."""

    if not frames:
        raise ProtocolError("live batch has no selected frames")
    # A zero-offset selection performs the expensive same-traversal decode and
    # gives us the unmodified detector mapping.  All requested offsets are then
    # applied together to that mapping before SEND(0x8f) can execute.
    context = _derive_live_frame_selection(
        plan,
        preview_data,
        table_data,
        frame=frames[0].slot,
        boundary_offset_rows=0,
        expected_frame_count=None,
    )
    validated_table, _usable_rows = validate_live_0x8e_bytes(
        table_data,
        context.geometry.height,
    )
    records = parse_live_transport_records_bytes(
        validated_table,
        maximum_rows=context.geometry.height,
    )
    combined, resolved = apply_batch_boundary_offsets(
        context.mapping,
        records,
        tuple((spec.slot, spec.boundary_offset_rows) for spec in frames),
    )
    for origin in combined.origins[:FRAME_TABLE_SEND_RECORDS]:
        if origin.native_origin + FINE_NATIVE_HEIGHT > context.geometry.native_height:
            raise ProtocolError(
                f"frame {origin.frame} fine window exceeds native transport height"
            )

    selections = tuple(
        LiveFrameSelection(
            frame=spec.slot,
            frame_count=len(combined.origins),
            geometry=context.geometry,
            usable_rows=context.usable_rows,
            detection=context.detection,
            mapping=combined,
            base_selected=base,
            selected=selected,
            requested_boundary_offset_rows=spec.boundary_offset_rows,
            applied_boundary_offset_rows=(selected.lookup_row - base.lookup_row),
            preview_sha256=context.preview_sha256,
            table_sha256=context.table_sha256,
            decode_report=context.decode_report,
        )
        for spec, (base, selected) in zip(frames, resolved, strict=True)
    )
    # Compile each binding before the first fine scan.  This proves that the
    # retained table and every later autofocus/window origin agree.
    for selection in selections:
        _bind_plan_to_live_selection(plan, selection)
    return selections


def _validate_boundary_offset(frame: int, offset_rows: int) -> None:
    if isinstance(offset_rows, bool) or not isinstance(offset_rows, int):
        raise ProtocolError("boundary offset must be an integer row count")
    minimum = 0 if frame == 1 else -144
    if not minimum <= offset_rows <= 144:
        raise ProtocolError(
            f"frame {frame} boundary offset must be in {minimum}..144 rows"
        )


def load_validated_batch_job(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_continuation_sha256: str,
) -> LiveBatchJob:
    """Load an exact path-confined parent/child batch handshake contract."""

    path = Path(path).expanduser().resolve()
    try:
        job_bytes = path.read_bytes()
        payload = json.loads(job_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"batch job could not be read: {error}") from error
    if not isinstance(payload, dict):
        raise ProtocolError("batch job must be a JSON object")
    expected_top_level = {
        "apply_all_boundary_offsets_before_first_frame": True,
        "capture_plan_sha256": expected_plan_sha256,
        "continuation_plan_sha256": expected_continuation_sha256,
        "parent_ack_required_after_every_frame": True,
        "release_once_after_last_frame": True,
        "schema_version": 1,
        "session_contract": "one-process-one-reservation",
    }
    expected_keys = set(expected_top_level) | {"frames", "session_id"}
    if set(payload) != expected_keys:
        raise ProtocolError(
            "batch job keys changed: "
            f"expected {sorted(expected_keys)}, got {sorted(payload)}"
        )
    for key, expected in expected_top_level.items():
        if payload.get(key) != expected:
            raise ProtocolError(
                f"batch job {key}={payload.get(key)!r}, expected {expected!r}"
            )
    session_id = payload.get("session_id")
    if (
        not isinstance(session_id, str)
        or not 1 <= len(session_id) <= 128
        or session_id[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in session_id
        )
    ):
        raise ProtocolError("batch job session_id is not filesystem-safe")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ProtocolError("batch job must contain at least one frame")
    root = path.parent
    frames: list[BatchFrameSpec] = []
    for index, raw in enumerate(raw_frames, start=1):
        if not isinstance(raw, dict):
            raise ProtocolError(f"batch frame {index} must be an object")
        raw = cast(dict[str, Any], raw)
        if set(raw) != {
            "ack",
            "boundary_offset_rows",
            "journal",
            "output",
            "slot",
        }:
            raise ProtocolError(f"batch frame {index} keys changed")
        slot = raw.get("slot")
        offset = raw.get("boundary_offset_rows")
        if isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot <= 40:
            raise ProtocolError(f"batch frame {index} has an invalid slot")
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ProtocolError(
                f"batch frame {index} has an invalid boundary offset"
            )
        _validate_boundary_offset(slot, offset)
        expected_directory = f"frame-{slot:03d}"
        expected_paths = {
            "output": f"{expected_directory}/capture.bin",
            "journal": f"{expected_directory}/journal.json",
            "ack": f"{expected_directory}/parent-ack.json",
        }
        for key, expected in expected_paths.items():
            if raw.get(key) != expected:
                raise ProtocolError(
                    f"batch frame {index} {key} must be {expected!r}"
                )
        frames.append(
            BatchFrameSpec(
                slot=slot,
                boundary_offset_rows=offset,
                output=root / expected_paths["output"],
                journal=root / expected_paths["journal"],
                ack=root / expected_paths["ack"],
            )
        )
    slots = tuple(frame.slot for frame in frames)
    if slots != tuple(sorted(set(slots))):
        raise ProtocolError("batch frame slots must be unique and increasing")
    return LiveBatchJob(
        session_id=session_id,
        root=root,
        frames=tuple(frames),
        plan_sha256=expected_plan_sha256,
        continuation_plan_sha256=expected_continuation_sha256,
        job_sha256=hashlib.sha256(job_bytes).hexdigest(),
    )


def wait_for_parent_ack(
    path: Path,
    *,
    session_id: str,
    frame_index: int,
    slot: int,
    nonce: str,
    timeout_seconds: float = 1_800.0,
    poll_seconds: float = 0.1,
) -> str:
    """Wait at a transaction boundary for one exact continue/stop decision."""

    if timeout_seconds < 0 or poll_seconds < 0:
        raise ValueError("parent ACK timing values cannot be negative")
    path = Path(path).expanduser().resolve()
    deadline = time.monotonic() + timeout_seconds
    while True:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SynchronizedProtocolError(
                    f"parent ACK for slot {slot} is unreadable: {error}"
                ) from error
            expected = {
                "ack_nonce": nonce,
                "frame_index": frame_index,
                "schema_version": 1,
                "session_id": session_id,
                "slot": slot,
            }
            if not isinstance(payload, dict):
                raise SynchronizedProtocolError(
                    f"parent ACK for slot {slot} must be an object"
                )
            if set(payload) != {*expected, "action"}:
                raise SynchronizedProtocolError(
                    f"parent ACK for slot {slot} has an unexpected schema"
                )
            for key, required in expected.items():
                if payload.get(key) != required:
                    raise SynchronizedProtocolError(
                        f"parent ACK for slot {slot} has {key}="
                        f"{payload.get(key)!r}, expected {required!r}"
                    )
            action = payload.get("action")
            if action not in ("continue", "stop"):
                raise SynchronizedProtocolError(
                    f"parent ACK for slot {slot} has invalid action {action!r}"
                )
            return action
        if time.monotonic() >= deadline:
            raise SynchronizedProtocolError(
                f"parent ACK for slot {slot} did not arrive before timeout"
            )
        if poll_seconds:
            time.sleep(poll_seconds)


def apply_boundary_offset(
    mapping: TransportMapping,
    records: Sequence[TransportRecord],
    *,
    frame: int,
    offset_rows: int,
) -> tuple[TransportMapping, NativeFrameOrigin]:
    """Resolve an operator offset through this traversal's raw 0x8e records.

    The offset is applied in preview rows, then snapped to the exact record at
    that row.  The returned mapping replaces the selected SEND(0x8f) entry, so
    autofocus and all RGBI SET_WINDOW commands remain bound to one raw
    transport identity instead of independently editing a native coordinate.
    """

    if not 1 <= frame <= len(mapping.origins):
        raise ProtocolError(
            f"requested frame {frame} is outside mapping 1..{len(mapping.origins)}"
        )
    _validate_boundary_offset(frame, offset_rows)
    base = mapping.origins[frame - 1]
    if base.frame != frame:
        raise ProtocolError("transport mapping frame order is inconsistent")
    resolved_row = base.lookup_row + offset_rows
    if not 0 <= resolved_row < len(records):
        raise ProtocolError(
            f"frame {frame} boundary offset resolves outside the live 0x8e table"
        )
    record = records[resolved_row]
    if record.row != resolved_row:
        raise ProtocolError("live 0x8e records are not indexed by preview row")
    if transport_native_origin(record.code, record.selector) != record.native_origin:
        raise ProtocolError("resolved 0x8e record does not reproduce its origin")
    selected = replace(
        base,
        lookup_row=resolved_row,
        code=record.code,
        selector=record.selector,
        native_origin=record.native_origin,
        method=(
            base.method
            if offset_rows == 0
            else f"{base.method}+operator-boundary-offset"
        ),
        automatic=base.automatic,
    )
    origins = list(mapping.origins)
    origins[frame - 1] = selected
    return replace(mapping, origins=tuple(origins)), selected


def apply_batch_boundary_offsets(
    mapping: TransportMapping,
    records: Sequence[TransportRecord],
    frames: Sequence[tuple[int, int]],
) -> tuple[
    TransportMapping,
    tuple[tuple[NativeFrameOrigin, NativeFrameOrigin], ...],
]:
    """Apply every selected-frame offset to one shared retained mapping.

    Nikon receives ``SEND(0x8f)`` only once per reserved roll session.  Every
    frame's operator offset therefore has to be encoded into that one table
    before the first fine scan; later autofocus and window commands can then
    bind to the same immutable mapping without resending the table.
    """

    if not frames:
        raise ProtocolError("batch boundary binding requires at least one frame")
    ordered_slots = tuple(frame for frame, _offset in frames)
    if tuple(sorted(set(ordered_slots))) != ordered_slots:
        raise ProtocolError(
            "batch boundary frames must be unique and strictly increasing"
        )
    original = mapping
    combined = mapping
    resolved: list[tuple[NativeFrameOrigin, NativeFrameOrigin]] = []
    for frame, offset_rows in frames:
        if not 1 <= frame <= len(original.origins):
            raise ProtocolError(
                f"requested frame {frame} is outside mapping "
                f"1..{len(original.origins)}"
            )
        base = original.origins[frame - 1]
        if base.frame != frame:
            raise ProtocolError("transport mapping frame order is inconsistent")
        if not base.automatic or base.manual_review:
            raise ProtocolError(
                f"frame {frame} transport origin requires manual review; "
                "refusing an unattended batch"
            )
        combined, selected = apply_boundary_offset(
            combined,
            records,
            frame=frame,
            offset_rows=offset_rows,
        )
        resolved.append((base, selected))

    # Validate the exact fixed-size payload now, before any selected frame is
    # scanned.  This also catches offsets that would make origins overlap.
    table = build_live_frame_table_payload(combined)
    table_count = table[2]
    for frame in ordered_slots:
        if frame > table_count:
            raise ProtocolError(
                f"requested frame {frame} is outside the scanner-addressable "
                f"table 1..{table_count}"
            )
    return combined, tuple(resolved)


def build_live_frame_table_payload(mapping: TransportMapping) -> bytes:
    """Encode Nikon SEND(0x8f) records from this traversal's raw 0x8e fields."""

    # The preview UI deliberately exposes every aligned candidate cell, including
    # a partial final cell when the index raster ends mid-frame.  That advisory
    # slot is not necessarily scanner-addressable: its local 0x8e lookup can land
    # in Nikon's end-of-roll records (observed as code 0x83xx), which SEND(0x8f)
    # rejects with ILLEGAL REQUEST 05/26/00.  Keep the UI's slot numbering intact,
    # but stop the hardware table before the first candidate that the detector
    # proved lies outside the raster or has invalid physical spacing.
    non_addressable_reasons = {"outside-index-raster", "spacing-outlier"}
    origins = []
    for origin in mapping.origins:
        if (
            non_addressable_reasons.intersection(origin.review_reasons)
            or abs(origin.affine_residual_rows) > 2.0
        ):
            break
        origins.append(origin)
    # This parameter page is fixed-size even though the preview exposes up to
    # 40 candidate cells. Every Nikon host SEND observed on this firmware uses
    # 37 records / 300 bytes. The device accepted that exact shape and rejected
    # otherwise well-formed 36-, 39-, and 40-record variants with 05/26/00.
    if len(origins) < FRAME_TABLE_SEND_RECORDS:
        raise ProtocolError(
            "live mapping has fewer than 37 scanner-addressable frame records"
        )
    origins = origins[:FRAME_TABLE_SEND_RECORDS]
    origins = tuple(origins)
    if len(origins) != FRAME_TABLE_SEND_RECORDS:
        raise ProtocolError("SEND(0x8f) frame table is not the proven 37 records")
    payload = bytearray((0x01, 0x2A, len(origins), 0x00))
    previous = -1
    for expected_frame, origin in enumerate(origins, start=1):
        if origin.frame != expected_frame:
            raise ProtocolError("dynamic frame table order is not consecutive")
        if origin.native_origin <= previous:
            raise ProtocolError("dynamic frame table origins are not increasing")
        if (
            transport_native_origin(origin.code, origin.selector)
            != origin.native_origin
        ):
            raise ProtocolError(
                f"frame {origin.frame} transport identity does not reproduce origin"
            )
        payload.extend(
            struct.pack(">IHH", origin.native_origin, origin.selector, origin.code)
        )
        previous = origin.native_origin
    return bytes(payload)


def _patch_window_origin(entry: dict, native_origin: int) -> None:
    payload = bytearray.fromhex(entry.get("data_out", ""))
    decoded = decode_window_block(payload)
    if decoded is None:
        raise ProtocolError(f"command {entry.get('seq')}: malformed SET_WINDOW payload")
    payload[18:22] = native_origin.to_bytes(4, "big")
    entry["data_out"] = payload.hex()


def _bind_plan_to_live_selection(
    plan: list[dict], selection: LiveFrameSelection
) -> list[dict]:
    """Return a plan whose frame-bearing fields share one proven live origin."""

    validate_plan(plan)
    if selection.frame_count != len(selection.mapping.origins):
        raise ProtocolError("selected frame count disagrees with transport mapping")
    if selection.selected != selection.mapping.origins[selection.frame - 1]:
        raise ProtocolError("selected origin is not owned by the supplied mapping")
    bound = [dict(entry) for entry in plan]
    native_origin = selection.selected.native_origin

    table_payload = build_live_frame_table_payload(selection.mapping)
    table_count = table_payload[2]
    if selection.frame > table_count:
        raise ProtocolError(
            f"requested frame {selection.frame} is outside the scanner-addressable "
            f"table 1..{table_count}"
        )
    for origin in selection.mapping.origins[:table_count]:
        if origin.native_origin + FINE_NATIVE_HEIGHT > selection.geometry.native_height:
            raise ProtocolError(
                f"frame {origin.frame} fine window exceeds native transport height"
            )
    table_entry = _entry(bound, FRAME_TABLE_SEND_SEQUENCE)
    table_cdb = bytearray.fromhex(table_entry.get("cdb", ""))
    if (
        len(table_cdb) != 10
        or table_cdb[:6] != bytes.fromhex("2a008f000003")
        or table_cdb[9] != 0
    ):
        raise ProtocolError("command 174 is not the canonical SEND(0x8f)")
    table_cdb[6:9] = len(table_payload).to_bytes(3, "big")
    table_entry["cdb"] = table_cdb.hex()
    table_entry["data_out"] = table_payload.hex()

    autofocus_entry = _entry(bound, AUTOFOCUS_SEQUENCE)
    autofocus = bytearray.fromhex(autofocus_entry.get("data_out", ""))
    if (
        len(autofocus) != 9
        or autofocus[0] != 0
        or int.from_bytes(autofocus[1:5], "big") != FINE_NATIVE_WIDTH // 2
    ):
        raise ProtocolError("command 231 autofocus payload is not 00 + X + Y")
    autofocus_y = native_origin + FINE_NATIVE_HEIGHT // 2
    autofocus[5:9] = autofocus_y.to_bytes(4, "big")
    autofocus_entry["data_out"] = autofocus.hex()

    expected_colors = [9, 1, 2, 3]
    for group in DYNAMIC_WINDOW_GROUPS:
        colors = []
        for sequence in group:
            entry = _entry(bound, sequence)
            decoded = decode_window_block(bytes.fromhex(entry.get("data_out", "")))
            if decoded is None:
                raise ProtocolError(f"command {sequence}: malformed SET_WINDOW")
            colors.append(decoded["color_id"])
            _patch_window_origin(entry, native_origin)
        if colors != expected_colors:
            raise ProtocolError(f"SET_WINDOW group {group} is not IR,R,G,B")

    for sequence in (*METER_GET_WINDOW_SEQUENCES, *FINE_GET_WINDOW_SEQUENCES):
        entry = _entry(bound, sequence)
        expected = bytearray.fromhex(entry.get("expected_data_in", ""))
        if len(expected) != 58:
            raise ProtocolError(
                f"command {sequence}: missing canonical GET_WINDOW response"
            )
        expected[18:22] = native_origin.to_bytes(4, "big")
        entry["expected_data_in"] = expected.hex()

    # Recheck the exact values that cross the unsafe hardware boundary.
    if (
        len(table_payload) != FRAME_TABLE_SEND_BYTES
        or int.from_bytes(table_cdb[6:9], "big") != FRAME_TABLE_SEND_BYTES
    ):
        raise ProtocolError("SEND(0x8f) transfer is not the proven 300 bytes")
    if int.from_bytes(autofocus[5:9], "big") != autofocus_y:
        raise ProtocolError("dynamic autofocus Y was not bound")
    for sequence in DYNAMIC_WINDOW_SEQUENCES:
        decoded = decode_window_block(
            bytes.fromhex(_entry(bound, sequence)["data_out"])
        )
        if decoded is None or decoded["upper_left_y"] != native_origin:
            raise ProtocolError(
                f"command {sequence}: dynamic SET_WINDOW origin was not bound"
            )
    return bound


def _write_journal(path: Path, journal: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(journal, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_parent_directory(path)


def _fsync_parent_directory(path: Path) -> None:
    """Make a newly created or replaced file name durable in its directory."""

    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Persist one provenance artifact without overwriting prior evidence."""

    with path.open("xb") as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise ProtocolError(
                f"short artifact write {written} of {len(payload)} bytes to {path}"
            )
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_parent_directory(path)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_exclusive(path, data)


def _live_index_artifact_paths(output_path: Path) -> dict[str, Path]:
    stem = output_path.stem
    return {
        "preview": output_path.with_name(f"{stem}-preview.bin"),
        "table": output_path.with_name(f"{stem}-008e.bin"),
        "mapping": output_path.with_name(f"{stem}-frame-map.json"),
    }


def _full_capture_meter_path(output_path: Path) -> Path:
    """Return the exclusive raw-meter sidecar owned by one full capture."""

    return output_path.with_name(f"{output_path.stem}-meter.bin")


def _validate_live_preview_windows(
    payloads: list[bytes], geometry: IndexGeometry
) -> list[WindowBlock]:
    if len(payloads) != 3:
        raise SynchronizedProtocolError("preview GET_WINDOW responses are incomplete")
    decoded: list[WindowBlock] = []
    for payload in payloads:
        window = decode_window_block(payload)
        if window is None:
            raise SynchronizedProtocolError(
                "preview GET_WINDOW responses are incomplete"
            )
        decoded.append(window)
    for color, window in zip((1, 2, 3), decoded, strict=True):
        checks = (
            ("color_id", window["color_id"], color),
            ("resx", window["resx"], geometry.requested_resolution),
            ("resy", window["resy"], geometry.requested_resolution),
            ("upper_left_x", window["upper_left_x"], 0),
            ("upper_left_y", window["upper_left_y"], 0),
            ("width", window["width"], geometry.native_width),
            ("height", window["height"], geometry.native_height),
            ("bit_depth", window["bit_depth"], 16),
        )
        for key, actual, expected in checks:
            if actual != expected:
                raise SynchronizedProtocolError(
                    f"preview GET_WINDOW color {color}: {key}={actual!r}, "
                    f"expected {expected!r}"
                )
    return decoded


def _window_exposures(plan: list[dict], sequences: tuple[int, ...]) -> dict[int, int]:
    exposures: dict[int, int] = {}
    for sequence in sequences:
        window = decode_window_block(
            bytes.fromhex(_entry(plan, sequence).get("data_out", ""))
        )
        if window is None:
            raise ProtocolError(f"command {sequence}: malformed SET_WINDOW")
        exposures[window["color_id"]] = window["exposure_raw_10ns"]
    if list(exposures) != [9, 1, 2, 3]:
        raise ProtocolError(f"SET_WINDOW group {sequences} is not IR,R,G,B")
    return exposures


def _controller_exposures_from_wire(
    exposures: dict[int, int],
) -> dict[str, int]:
    """Translate the scanner's IR,R,G,B identifiers to R,G,B,IR planes."""

    if tuple(exposures) != WIRE_METER_COLORS:
        raise ProtocolError(
            f"wire exposures must be ordered {WIRE_METER_COLORS}, got "
            f"{tuple(exposures)}"
        )
    return {
        channel: exposures[color]
        for channel in CONTROLLER_CHANNELS
        for color, mapped_channel in WIRE_COLOR_TO_CONTROLLER_CHANNEL.items()
        if mapped_channel == channel
    }


def _wire_exposures_from_controller(
    exposures: dict[str, int],
) -> dict[int, int]:
    """Translate R,G,B,IR controller values to scanner colors 9,1,2,3."""

    if tuple(exposures) != CONTROLLER_CHANNELS:
        raise ProtocolError(
            f"controller exposures must be ordered {CONTROLLER_CHANNELS}, got "
            f"{tuple(exposures)}"
        )
    wire = {
        color: exposures[channel]
        for color, channel in WIRE_COLOR_TO_CONTROLLER_CHANNEL.items()
    }
    for color, exposure in wire.items():
        if isinstance(exposure, bool) or not isinstance(exposure, int):
            raise ProtocolError(f"wire color {color} exposure is not an integer")
        if not 0 <= exposure <= 0xFFFFFFFF:
            raise ProtocolError(f"wire color {color} exposure is out of uint32 range")
    return wire


def _patched_window_exposure(
    payload: bytes,
    *,
    expected_color: int,
    exposure: int,
    sequence: int,
) -> bytes:
    decoded = decode_window_block(payload)
    if decoded is None or decoded["color_id"] != expected_color:
        raise ProtocolError(
            f"command {sequence}: expected window color {expected_color}"
        )
    mutable = bytearray(payload)
    mutable[54:58] = exposure.to_bytes(4, "big")
    patched = bytes(mutable)
    verified = decode_window_block(patched)
    if (
        verified is None
        or verified["color_id"] != expected_color
        or verified["exposure_raw_10ns"] != exposure
    ):
        raise ProtocolError(f"command {sequence}: exposure patch did not verify")
    return patched


def _patch_exposure_contract(
    plan: list[dict],
    set_sequences: tuple[int, ...],
    get_sequences: tuple[int, ...],
    controller_exposures: dict[str, int],
) -> dict[int, int]:
    """Atomically patch one SET group and its exact GET echo expectations."""

    if len(set_sequences) != len(WIRE_METER_COLORS) or len(get_sequences) != len(
        WIRE_METER_COLORS
    ):
        raise ProtocolError("exposure contract must contain four SET and GET windows")
    wire_exposures = _wire_exposures_from_controller(controller_exposures)
    patched_sets: list[tuple[dict, str]] = []
    patched_gets: list[tuple[dict, str]] = []
    for sequence, color in zip(set_sequences, WIRE_METER_COLORS, strict=True):
        entry = _entry(plan, sequence)
        patched = _patched_window_exposure(
            bytes.fromhex(entry.get("data_out", "")),
            expected_color=color,
            exposure=wire_exposures[color],
            sequence=sequence,
        )
        patched_sets.append((entry, patched.hex()))
    for sequence, color in zip(get_sequences, WIRE_METER_COLORS, strict=True):
        entry = _entry(plan, sequence)
        patched = _patched_window_exposure(
            bytes.fromhex(entry.get("expected_data_in", "")),
            expected_color=color,
            exposure=wire_exposures[color],
            sequence=sequence,
        )
        patched_gets.append((entry, patched.hex()))
    for entry, payload in patched_sets:
        entry["data_out"] = payload
    for entry, payload in patched_gets:
        entry["expected_data_in"] = payload
    return wire_exposures


def _validate_live_meter_windows(
    payloads: list[bytes],
    *,
    expected_origin: int,
    expected_exposures: dict[int, int],
) -> list[WindowBlock]:
    """Prove each live meter pass uses the requested frame and exposures."""

    if len(payloads) != 4:
        raise SynchronizedProtocolError("meter GET_WINDOW responses are incomplete")
    decoded: list[WindowBlock] = []
    for payload in payloads:
        window = decode_window_block(payload)
        if window is None:
            raise SynchronizedProtocolError(
                "meter GET_WINDOW responses are incomplete"
            )
        decoded.append(window)
    for color, window in zip((9, 1, 2, 3), decoded, strict=True):
        checks = (
            ("color_id", window["color_id"], color),
            ("resx", window["resx"], 285),
            ("resy", window["resy"], 285),
            ("upper_left_x", window["upper_left_x"], 0),
            ("upper_left_y", window["upper_left_y"], expected_origin),
            ("width", window["width"], FINE_NATIVE_WIDTH),
            ("height", window["height"], FINE_NATIVE_HEIGHT),
            ("multiread_byte", window["multiread_byte"], 0x00),
            ("avg_negpos_byte", window["avg_negpos_byte"], 0x80),
            (
                "samples_per_scan_minus1_nibble",
                window["samples_per_scan_minus1_nibble"],
                0,
            ),
            ("scanning_kind_byte", window["scanning_kind_byte"], 0x01),
            ("scanning_mode_byte", window["scanning_mode_byte"], 0x02),
            (
                "color_interleaving_byte",
                window["color_interleaving_byte"],
                0x02,
            ),
            ("ae_byte", window["ae_byte"], 0xFF),
            ("bit_depth", window["bit_depth"], 0x10),
            (
                "exposure_raw_10ns",
                window["exposure_raw_10ns"],
                expected_exposures[color],
            ),
        )
        for key, actual, expected in checks:
            if actual != expected:
                raise SynchronizedProtocolError(
                    f"meter GET_WINDOW color {color}: {key}={actual!r}, "
                    f"expected {expected!r}"
                )
    return decoded


def _validate_scanner_identity(payload: bytes) -> None:
    if len(payload) != 36:
        raise SynchronizedProtocolError(
            f"standard INQUIRY returned {len(payload)} bytes, expected 36"
        )
    vendor = payload[8:16].decode("ascii", errors="replace").strip()
    product = payload[16:32].decode("ascii", errors="replace").strip()
    revision = payload[32:36].decode("ascii", errors="replace").strip()
    if (vendor, product, revision) != ("Nikon", "LS-5000 ED", "1.03"):
        raise SynchronizedProtocolError(
            "unexpected scanner identity "
            f"vendor={vendor!r} product={product!r} revision={revision!r}"
        )


def _validate_live_fine_windows(
    payloads: list[bytes],
    *,
    expected_origin: int,
    expected_exposures: dict[int, int] | None = None,
) -> list[WindowBlock]:
    if len(payloads) != 4:
        raise SynchronizedProtocolError("fine GET_WINDOW responses are incomplete")
    decoded: list[WindowBlock] = []
    for payload in payloads:
        window = decode_window_block(payload)
        if window is None:
            raise SynchronizedProtocolError(
                "fine GET_WINDOW responses are incomplete"
            )
        decoded.append(window)
    expected_colors = [9, 1, 2, 3]
    for color, window in zip(expected_colors, decoded, strict=True):
        checks = (
            ("color_id", window["color_id"], color),
            ("resx", window["resx"], 4000),
            ("resy", window["resy"], 4000),
            ("upper_left_x", window["upper_left_x"], 0),
            ("upper_left_y", window["upper_left_y"], expected_origin),
            ("width", window["width"], 3946),
            ("height", window["height"], 5959),
            ("multiread_byte", window["multiread_byte"], 0x30),
            ("avg_negpos_byte", window["avg_negpos_byte"], 0x00),
            (
                "samples_per_scan_minus1_nibble",
                window["samples_per_scan_minus1_nibble"],
                3,
            ),
            ("scanning_kind_byte", window["scanning_kind_byte"], 0x01),
            ("scanning_mode_byte", window["scanning_mode_byte"], 0x10),
            (
                "color_interleaving_byte",
                window["color_interleaving_byte"],
                0x40,
            ),
            ("ae_byte", window["ae_byte"], 0xFF),
            ("bit_depth", window["bit_depth"], 0x10),
        )
        for key, actual, expected in checks:
            if actual != expected:
                raise SynchronizedProtocolError(
                    f"fine GET_WINDOW color {color}: {key}={actual!r}, "
                    f"expected {expected!r}"
                )
        if expected_exposures is not None:
            expected_exposure = expected_exposures.get(color)
            if expected_exposure is None:
                raise ProtocolError(f"fine exposure contract is missing color {color}")
            if window["exposure_raw_10ns"] != expected_exposure:
                raise SynchronizedProtocolError(
                    f"fine GET_WINDOW color {color}: exposure "
                    f"{window['exposure_raw_10ns']} != SET_WINDOW "
                    f"{expected_exposure}"
                )
    return decoded


def _connect_device():
    import usb.core
    import usb.util

    device = usb.core.find(idVendor=0x04B0, idProduct=0x4002)
    if device is None:
        raise ProtocolError("Nikon LS-5000 (04b0:4002) is not on the USB bus")
    try:
        configuration = device.get_active_configuration()
    except usb.core.USBError:
        device.set_configuration()
        configuration = device.get_active_configuration()
    interface = configuration[(0, 0)]
    usb.util.claim_interface(device, interface.bInterfaceNumber)
    try:
        ep_out = usb.util.find_descriptor(
            interface,
            custom_match=lambda endpoint: (
                usb.util.endpoint_direction(endpoint.bEndpointAddress)
                == usb.util.ENDPOINT_OUT
            ),
        )
        ep_in_raw = usb.util.find_descriptor(
            interface,
            custom_match=lambda endpoint: (
                usb.util.endpoint_direction(endpoint.bEndpointAddress)
                == usb.util.ENDPOINT_IN
            ),
        )
        if ep_out is None or ep_in_raw is None:
            raise ProtocolError("scanner bulk endpoints were not found")
        if ep_out.bEndpointAddress != 0x01 or ep_in_raw.bEndpointAddress != 0x82:
            raise ProtocolError(
                "unexpected LS-5000 endpoints: "
                f"OUT=0x{ep_out.bEndpointAddress:02x}, "
                f"IN=0x{ep_in_raw.bEndpointAddress:02x}"
            )
        if (
            usb.util.endpoint_type(ep_out.bmAttributes) != usb.util.ENDPOINT_TYPE_BULK
            or usb.util.endpoint_type(ep_in_raw.bmAttributes)
            != usb.util.ENDPOINT_TYPE_BULK
        ):
            raise ProtocolError("LS-5000 endpoints are not bulk endpoints")
        ep_in = CountedBulkInEndpoint(ep_in_raw)
        return device, interface, ep_out, ep_in, usb.util
    except BaseException:
        # `_connect_device` has not returned ownership to the caller yet, so
        # its outer `finally` cannot release a partially constructed endpoint.
        try:
            usb.util.release_interface(device, interface.bInterfaceNumber)
        except Exception:
            pass
        try:
            usb.util.dispose_resources(device)
        except Exception:
            pass
        raise


def _require_trace_result(entry: dict, result: TransactionResult) -> None:
    expected_phase = entry.get("expected_phase")
    if expected_phase is not None and result.phase != expected_phase:
        raise SynchronizedProtocolError(
            f"command {entry['seq']}: phase 0x{result.phase:02x} "
            f"!= expected 0x{expected_phase:02x}"
        )
    expected = entry.get("expected_sense", "")
    accepted_senses = set(entry.get("accepted_senses") or [expected])
    if result.sense not in accepted_senses:
        raise SynchronizedProtocolError(
            f"command {entry['seq']}: sense {result.sense} not in "
            f"accepted {sorted(accepted_senses)}"
        )
    expected_status = entry.get("expected_status")
    if (
        result.sense == expected
        and expected_status
        and result.status.hex() != expected_status
    ):
        raise SynchronizedProtocolError(
            f"command {entry['seq']}: full status {result.status.hex()} "
            f"!= expected {expected_status}"
        )
    if result.phase == 0x03:
        maximum = entry.get("request_len", 0)
        minimum = entry.get("minimum_data_in", maximum)
        if not minimum <= len(result.payload) <= maximum:
            raise SynchronizedProtocolError(
                f"command {entry['seq']}: data length {len(result.payload)} "
                f"outside accepted {minimum}..{maximum}"
            )


def _bind_live_sub_8e_read(entry: dict, header: bytes) -> dict:
    """Bind the variable 0x8e table READ to its preceding live header.

    Nikon first returns ``00 8e 00 00 LL LL`` and then expects the host to
    request exactly that many bytes from subcommand 0x8e.  The Windows trace
    used 0x52c4, while this live roll returned 0x529a.  Replaying the stale
    allocation made the scanner correctly report ILI/data-phase error.
    """
    if len(header) != 6 or header[:4] != b"\x00\x8e\x00\x00":
        raise SynchronizedProtocolError(
            f"command 171: malformed live 0x8e header {header.hex()}"
        )
    live_length = int.from_bytes(header[4:6], "big")
    if live_length == 0:
        raise SynchronizedProtocolError(
            "command 172: live 0x8e header declared a zero-length table"
        )

    cdb = bytearray.fromhex(entry["cdb"])
    if len(cdb) != 10 or cdb[:3] != b"\x28\x00\x8e":
        raise ProtocolError(f"command 172: unexpected 0x8e READ CDB {cdb.hex()}")
    cdb[7:9] = live_length.to_bytes(2, "big")

    # Preserve the scanner/host's large first-transfer boundary, then ask
    # for the entire live remainder in one transfer.  Do not retain the
    # stale trace's final 196-byte split when the live table is longer or
    # shorter (observed live remainders: 154 and 224 bytes).
    traced_parts = entry.get("request_parts") or [live_length]
    first_part = min(traced_parts[0], live_length)
    live_parts = [first_part]
    if live_length > first_part:
        live_parts.append(live_length - first_part)

    bound = dict(entry)
    bound["cdb"] = cdb.hex()
    bound["request_len"] = live_length
    bound["request_parts"] = live_parts
    bound["minimum_data_in"] = live_length
    bound["live_length_source"] = header.hex()
    return bound


def _validate_variable_frame_table_payload(payload: bytes) -> StartupFrameTable:
    """Validate the bounded, self-describing startup READ(0x8f) response."""

    payload = bytes(payload)
    length = len(payload)
    if not 10 <= length <= VARIABLE_FRAME_TABLE_MAX_BYTES:
        raise ProtocolError(f"0x8f payload length {length} is outside 10..330")
    if payload[:4] != b"\x8f\x00\x00\x00":
        raise ProtocolError("0x8f payload has invalid magic")
    outer = int.from_bytes(payload[4:6], "big")
    inner = int.from_bytes(payload[6:8], "big")
    count = payload[8]
    if (
        outer != length - 6
        or inner != length - 8
        or not 1 <= count <= 40
        or payload[9] != 0
        or length != 10 + count * 8
    ):
        raise ProtocolError("0x8f payload is not a complete self-declared table")
    return {
        "bytes": length,
        "count": count,
        "header": payload[:10].hex(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _perform_variable_frame_table_transaction(
    ep_out: Any,
    ep_in: Any,
    entry: dict,
    *,
    data_timeout_ms: int,
) -> TransactionResult:
    """Accept sequence 64's positive short response only if it is complete."""

    checks = {
        "seq": VARIABLE_FRAME_TABLE_SEQUENCE,
        "name": "READ",
        "cdb": VARIABLE_FRAME_TABLE_CDB,
        "request_len": VARIABLE_FRAME_TABLE_MAX_BYTES,
        "request_parts": [VARIABLE_FRAME_TABLE_MAX_BYTES],
        "expected_phase": 0x03,
        "expected_sense": "000000",
        "expected_status": "0000000000000000",
    }
    for key, expected in checks.items():
        if entry.get(key) != expected:
            raise ProtocolError(
                f"command 64 {key}={entry.get(key)!r}, expected {expected!r}"
            )

    _write_exact(ep_out, bytes.fromhex(entry["cdb"]), 10_000)
    _write_exact(ep_out, b"\xd0", 10_000)
    try:
        phase_raw, phase_stalls = _read_with_one_stall_recovery(ep_in, 1, 30_000)
    except DesynchronizedProtocolError as error:
        raise DesynchronizedProtocolError(
            f"command 64 during phase: {error}"
        ) from error
    if phase_raw != b"\x03":
        raise DesynchronizedProtocolError(
            f"command 64 expected data-IN phase 03, got {phase_raw.hex()}"
        )
    try:
        payload, data_stalls = _read_with_one_stall_recovery(
            ep_in, VARIABLE_FRAME_TABLE_MAX_BYTES, data_timeout_ms
        )
    except DesynchronizedProtocolError as error:
        raise DesynchronizedProtocolError(
            f"command 64 during bounded 0x8f read: {error}"
        ) from error
    try:
        _validate_variable_frame_table_payload(payload)
    except ProtocolError as error:
        # An invalid short response does not prove the data phase ended.
        raise DesynchronizedProtocolError(
            f"command 64 malformed bounded 0x8f response: {error}"
        ) from error

    _write_exact(ep_out, b"\x06", 10_000)
    try:
        status, status_stalls = _read_with_one_stall_recovery(ep_in, 8, 15_000)
    except DesynchronizedProtocolError as error:
        raise DesynchronizedProtocolError(
            f"command 64 during status: {error}"
        ) from error
    if len(status) != 8:
        raise DesynchronizedProtocolError(
            f"command 64 status length {len(status)} != 8"
        )
    if status.hex() != entry["expected_status"]:
        raise SynchronizedProtocolError(
            f"command 64 status {status.hex()} != {entry['expected_status']}"
        )
    return TransactionResult(
        phase=0x03,
        payload=payload,
        status=status,
        sense=status[1:4].hex(),
        stall_recoveries=phase_stalls + data_stalls + status_stalls,
    )


def _perform_with_busy_retry(
    ep_out: Any,
    ep_in: Any,
    entry: dict,
    *,
    data_timeout_ms: int,
    deadline_seconds: float = READY_POLL_DEADLINE_SECONDS,
    allow_busy_retry: bool = False,
) -> TransactionResult:
    """Complete one command, retrying only a fully consumed busy response."""
    deadline = time.monotonic() + deadline_seconds
    while True:
        result = perform_transaction(
            ep_out, ep_in, entry, data_timeout_ms=data_timeout_ms
        )
        if result.sense not in RETRYABLE_BUSY_SENSES:
            _require_trace_result(entry, result)
            return result
        if not allow_busy_retry:
            _require_trace_result(entry, result)
            return result
        if (
            result.phase != 0x04
            or result.payload
            or result.status != CANONICAL_BUSY_STATUS
        ):
            raise SynchronizedProtocolError(
                f"command {entry['seq']}: refusing non-canonical busy response "
                f"phase=0x{result.phase:02x}, payload={len(result.payload)} bytes, "
                f"status={result.status.hex()}"
            )
        if time.monotonic() >= deadline:
            raise SynchronizedProtocolError(
                f"command {entry['seq']}: still busy ({result.sense}) after "
                f"{deadline_seconds:.0f}s"
            )
        time.sleep(READY_POLL_SECONDS)


def _perform_ready_group(
    ep_out: Any,
    ep_in: Any,
    entries: list[dict],
    *,
    additional_terminal_senses: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    """Collapse one traced TEST UNIT READY run into a state-aware poll.

    The trace often contains hundreds of 100 ms polls and one 55-second UI
    pause.  Replay the state condition, not the workstation's wall-clock
    delay: accept only senses observed in the group and stop at its terminal
    sense.
    """
    if not entries or any(entry.get("name") != "TEST_UNIT_READY" for entry in entries):
        raise ProtocolError("invalid TEST UNIT READY group")
    template = entries[-1]
    terminal_sense = template.get("expected_sense", "")
    allowed_senses = {entry.get("expected_sense", "") for entry in entries}
    deadline = time.monotonic() + READY_POLL_DEADLINE_SECONDS
    polls = 0
    stalls = 0
    while True:
        result = perform_transaction(ep_out, ep_in, template, data_timeout_ms=30_000)
        polls += 1
        stalls += result.stall_recoveries
        if result.phase != template.get("expected_phase"):
            raise SynchronizedProtocolError(
                f"ready group {entries[0]['seq']}-{entries[-1]['seq']}: "
                f"phase 0x{result.phase:02x} != expected "
                f"0x{template.get('expected_phase'):02x}"
            )
        if result.sense == terminal_sense:
            _require_trace_result(template, result)
            return polls, stalls
        if result.sense in additional_terminal_senses:
            # Callers may name a semantically safe terminal state that differs
            # from the oracle trace.  This is deliberately opt-in so a no-media
            # result cannot make an ordinary scan readiness check succeed.
            return polls, stalls
        if terminal_sense == "023a00" and result.sense == "000000":
            # The oracle began with the feeder not yet presenting media.  A
            # scanner that already reports ready-with-media is a stronger
            # startup state and does not need to be forced back to no-media.
            return polls, stalls
        if (
            isinstance(entries[0].get("seq"), int)
            and entries[0]["seq"] <= 60
            and (
                result.sense in STARTUP_UNIT_ATTENTION_SENSES
                or result.sense.startswith("06")
            )
        ):
            # Fresh power/medium/configuration changes are reported once and
            # cleared by this completed TUR.  Keep polling toward the group's
            # semantic terminal state instead of treating them as success.
            if time.monotonic() >= deadline:
                raise SynchronizedProtocolError(
                    f"ready group {entries[0]['seq']}-{entries[-1]['seq']}: "
                    "startup unit attention did not clear before deadline"
                )
            time.sleep(READY_POLL_SECONDS)
            continue
        if result.sense in READY_POLL_TRANSIENT_SENSES:
            if time.monotonic() >= deadline:
                raise SynchronizedProtocolError(
                    f"ready group {entries[0]['seq']}-{entries[-1]['seq']}: "
                    f"scanner remained not ready ({result.sense}) past deadline"
                )
            time.sleep(READY_POLL_SECONDS)
            continue
        if result.sense not in allowed_senses:
            raise SynchronizedProtocolError(
                f"ready group {entries[0]['seq']}-{entries[-1]['seq']}: "
                f"untraced sense {result.sense}; terminal {terminal_sense}"
            )
        if time.monotonic() >= deadline:
            raise SynchronizedProtocolError(
                f"ready group {entries[0]['seq']}-{entries[-1]['seq']}: "
                f"terminal sense {terminal_sense} not reached after "
                f"{READY_POLL_DEADLINE_SECONDS:.0f}s"
            )
        time.sleep(READY_POLL_SECONDS)


def _release_unit(ep_out: Any, ep_in: Any) -> TransactionResult:
    entry = {
        "seq": "teardown",
        "name": "RELEASE_UNIT",
        "cdb": "170000000000",
        "expected_phase": 0x01,
        "expected_sense": "000000",
    }
    return _perform_with_busy_retry(
        ep_out, ep_in, entry, data_timeout_ms=30_000, deadline_seconds=30.0
    )


def _cancel_scan(ep_out: Any, ep_in: Any) -> TransactionResult:
    entry = {
        "seq": "cancel",
        "name": "CANCEL",
        "cdb": "c00000000000",
        "expected_phase": 0x01,
        "expected_sense": "000000",
    }
    return _perform_with_busy_retry(
        ep_out, ep_in, entry, data_timeout_ms=30_000, deadline_seconds=30.0
    )


def _wait_post_scan_ready(
    ep_out: Any,
    ep_in: Any,
    *,
    allow_medium_absent: bool = False,
) -> tuple[int, int]:
    base = {
        "name": "TEST_UNIT_READY",
        "cdb": "000000000000",
        "expected_phase": 0x01,
    }
    return _perform_ready_group(
        ep_out,
        ep_in,
        [
            {**base, "seq": "post-scan-busy", "expected_sense": "020401"},
            {**base, "seq": "post-scan-ready", "expected_sense": "000000"},
        ],
        # Only recovery after a successful CANCEL may treat no-medium as an
        # equally safe terminal state.  Normal successful teardown must still
        # flag unexpected media loss.
        additional_terminal_senses=(
            frozenset({"023a00"}) if allow_medium_absent else frozenset()
        ),
    )


def _cleanup_synchronized(
    ep_out: Any,
    ep_in: Any,
    *,
    scan_active: bool,
    ready_required: bool = False,
    reserved: bool,
) -> dict:
    cleanup: dict[str, Any] = {
        "attempted": True,
        "release_attempted": False,
        "release_succeeded": False,
    }
    cancelled = False
    try:
        if scan_active:
            result = _cancel_scan(ep_out, ep_in)
            cleanup["cancel_status"] = result.status.hex()
            cancelled = True
        if scan_active or ready_required:
            polls, stalls = _wait_post_scan_ready(
                ep_out,
                ep_in,
                allow_medium_absent=cancelled,
            )
            cleanup["ready_polls"] = polls
            cleanup["stall_recoveries"] = stalls
        if reserved:
            cleanup["release_attempted"] = True
            result = _release_unit(ep_out, ep_in)
            cleanup["release_status"] = result.status.hex()
            cleanup["release_succeeded"] = True
        cleanup["complete"] = True
    except BaseException as cleanup_error:
        cleanup["complete"] = False
        cleanup["error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
    return cleanup


def _scan_lifecycle_after_transaction(
    entry: dict,
    result: TransactionResult,
    *,
    scan_active: bool,
    ready_required: bool,
) -> tuple[bool, bool]:
    """Track whether cleanup must CANCEL or merely wait for READY."""

    if entry.get("name") == "SCAN":
        # Reissue senses mean the arm was rejected and no data scan began.
        if result.sense == "000000":
            return True, False
        return False, ready_required
    if entry.get("seq") in DRAINED_SCAN_READ_SEQUENCES:
        # The scan data phase is fully drained.  Cleanup must wait for READY
        # and RELEASE, never send an idle CANCEL that could prevent release.
        return False, True
    return scan_active, ready_required


def _run_live_continuation_frame(
    ep_out: Any,
    ep_in: Any,
    plan: list[dict],
    plan_path: Path,
    plan_sha256: str,
    continuation_plan: dict[str, Any],
    continuation_plan_sha256: str,
    frame_spec: BatchFrameSpec,
    selection: LiveFrameSelection,
    *,
    batch_job: LiveBatchJob,
    frame_index: int,
    lifecycle: SessionLifecycle,
) -> dict[str, Any]:
    """Capture one later frame without reconnecting, reserving, or releasing."""

    target = validate_plan(plan)
    if continuation_plan_sha256 != CANONICAL_CONTINUATION_PLAN_SHA256:
        raise ProtocolError("continuation plan digest is not canonical")
    expected_bytes = EXPECTED_FINE_READS * target["request_len"]
    output_path = frame_spec.output
    journal_path = frame_spec.journal
    meter_path = _full_capture_meter_path(output_path)
    output_path.parent.mkdir(parents=False, exist_ok=False)
    for candidate in (output_path, journal_path, frame_spec.ack, meter_path):
        if candidate.exists():
            raise ProtocolError(f"refusing to overwrite {candidate}")
    free_bytes = shutil.disk_usage(output_path.parent).free
    required_free = expected_bytes + max(1_073_741_824, expected_bytes // 10)
    if free_bytes < required_free:
        raise ProtocolError(
            f"only {free_bytes} free bytes; continuation requires {required_free}"
        )

    if (
        selection.frame != frame_spec.slot
        or selection.requested_boundary_offset_rows
        != frame_spec.boundary_offset_rows
        or selection.applied_boundary_offset_rows
        != frame_spec.boundary_offset_rows
    ):
        raise ProtocolError(
            "continuation frame does not match its prevalidated batch selection"
        )
    active_plan = _bind_plan_to_live_selection(plan, selection)
    initial_wire_exposures = _patch_exposure_contract(
        active_plan,
        DYNAMIC_WINDOW_GROUPS[0],
        METER_GET_WINDOW_GROUPS[0],
        dict(DEFAULT_EXPOSURES),
    )
    steps = compile_continuation_steps(active_plan, continuation_plan)
    batch_identity = {
        "frame_index": frame_index,
        "frame_total": len(batch_job.frames),
        "selected_slots": list(batch_job.selected_slots),
        "session_id": batch_job.session_id,
    }
    journal: dict[str, Any] = {
        "status": "starting",
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "continuation_plan_sha256": continuation_plan_sha256,
        "capture_engine_sha256": CAPTURE_WORKER_SHA256,
        "capture_bundle_sha256": CAPTURE_BUNDLE_SHA256,
        "meter_controller_sha256": _meter_controller_sha256(),
        "output": str(output_path.resolve()),
        "capture_mode": "full",
        "requested_frame": frame_spec.slot,
        "expected_frame_count": None,
        "requested_boundary_offset_rows": frame_spec.boundary_offset_rows,
        "applied_boundary_offset_rows": selection.applied_boundary_offset_rows,
        "resolved_lookup_row": selection.selected.lookup_row,
        "resolved_native_origin": selection.selected.native_origin,
        "expected_reads": EXPECTED_FINE_READS,
        "expected_bytes": expected_bytes,
        "completed_reads": 0,
        "completed_bytes": 0,
        "stall_recoveries": 0,
        "started_unix": time.time(),
        "scanner_identity": "Nikon LS-5000 ED 1.03",
        "preview_geometry_validated_before_reads": True,
        "live_frame_selection": selection.diagnostics(),
        "batch_session": batch_identity,
        "session_reservation_retained": True,
        "unit_released": False,
        "meter_evidence_path": str(meter_path.resolve()),
        "meter_observed_exposures_raw_10ns": [],
        "meter_layout": {
            "wire_colors": list(WIRE_METER_COLORS),
            "controller_channels": list(CONTROLLER_CHANNELS),
            "pass_count": len(METER_READ_GROUPS),
            "group_bytes": METER_GROUP_BYTES,
        },
        "meter_completed_reads": 0,
        "meter_completed_bytes": 0,
        "meter_pass_exposures_raw_10ns": [],
        "meter_pass_commanded_exposures": [],
        "meter_controller_proposals": [],
        "meter_controller_seed": {
            "controller_channels_raw_10ns": dict(DEFAULT_EXPOSURES),
            "wire_colors_raw_10ns": {
                str(color): exposure
                for color, exposure in initial_wire_exposures.items()
            },
        },
    }
    _write_journal(journal_path, journal)

    meter_window_payloads: list[list[bytes]] = [
        [] for _group in METER_GET_WINDOW_GROUPS
    ]
    fine_window_payloads: list[bytes] = []
    meter_group_bytes = [0] * len(METER_READ_GROUPS)
    meter_group_payloads = [bytearray() for _group in METER_READ_GROUPS]
    meter_commanded_wire: list[dict[int, int] | None] = [
        None for _group in METER_READ_GROUPS
    ]
    meter_observations: list[MeterObservation] = []
    meter_evidence_sha256 = hashlib.sha256()
    output_sha256 = hashlib.sha256()
    final_controller_accepted = False
    final_wire_exposures: dict[int, int] | None = None
    meter_evidence_persisted = False

    try:
        with output_path.open("xb") as output, meter_path.open("xb") as meter_output:
            for step in steps:
                if step.code == "R":
                    journal["current_command"] = {
                        "seq": (
                            f"{step.entries[0]['seq']}..{step.entries[-1]['seq']}"
                        ),
                        "name": "TEST_UNIT_READY group",
                        "cdb": step.entries[0]["cdb"],
                    }
                    lifecycle.at_transaction_boundary = False
                    polls, stalls = _perform_ready_group(
                        ep_out,
                        ep_in,
                        list(step.entries),
                    )
                    lifecycle.at_transaction_boundary = True
                    lifecycle.ready_required = False
                    journal["ready_polls"] = journal.get("ready_polls", 0) + polls
                    journal["stall_recoveries"] += stalls
                    continue

                entry = step.entries[0]
                sequence = entry["seq"]
                if sequence == DYNAMIC_WINDOW_GROUPS[-1][0]:
                    if (
                        not final_controller_accepted
                        or not meter_evidence_persisted
                        or final_wire_exposures is None
                    ):
                        raise SynchronizedProtocolError(
                            "continuation fine SET_WINDOW reached without accepted "
                            "meter evidence"
                        )
                    preflight = _validate_live_fine_windows(
                        [
                            bytes.fromhex(
                                _entry(active_plan, item).get("data_out", "")
                            )
                            for item in DYNAMIC_WINDOW_GROUPS[-1]
                        ],
                        expected_origin=selection.selected.native_origin,
                        expected_exposures=final_wire_exposures,
                    )
                    journal["fine_set_windows_preflight"] = [
                        {
                            "color_id": window["color_id"],
                            "origin": [
                                window["upper_left_x"],
                                window["upper_left_y"],
                            ],
                            "resolution": [window["resx"], window["resy"]],
                            "size": [window["width"], window["height"]],
                            "samples": (
                                window["samples_per_scan_minus1_nibble"] + 1
                            ),
                            "exposure_raw_10ns": window["exposure_raw_10ns"],
                        }
                        for window in preflight
                    ]
                    journal["fine_set_windows_preflight_before_sequence"] = sequence
                    _write_journal(journal_path, journal)

                request = entry.get("request_len", 0)
                timeout = 120_000 if request > 60_000 else 30_000
                journal["current_command"] = {
                    "seq": sequence,
                    "name": entry.get("name"),
                    "cdb": entry["cdb"],
                    "request_len": request,
                    "request_parts": entry.get("request_parts"),
                }
                lifecycle.at_transaction_boundary = False
                result = _perform_with_busy_retry(
                    ep_out,
                    ep_in,
                    entry,
                    data_timeout_ms=timeout,
                )
                lifecycle.at_transaction_boundary = True
                journal["stall_recoveries"] += result.stall_recoveries
                (
                    lifecycle.scan_active,
                    lifecycle.ready_required,
                ) = _scan_lifecycle_after_transaction(
                    entry,
                    result,
                    scan_active=lifecycle.scan_active,
                    ready_required=lifecycle.ready_required,
                )

                if sequence in METER_GET_WINDOW_SEQUENCES:
                    group_index = next(
                        index
                        for index, group in enumerate(METER_GET_WINDOW_GROUPS)
                        if sequence in group
                    )
                    meter_window_payloads[group_index].append(result.payload)
                    if sequence == METER_GET_WINDOW_GROUPS[group_index][-1]:
                        expected_exposures = {
                            window["color_id"]: window["exposure_raw_10ns"]
                            for window in (
                                decode_window_block(
                                    bytes.fromhex(
                                        _entry(active_plan, item).get(
                                            "data_out", ""
                                        )
                                    )
                                )
                                for item in DYNAMIC_WINDOW_GROUPS[group_index]
                            )
                            if window is not None
                        }
                        observed = _validate_live_meter_windows(
                            meter_window_payloads[group_index],
                            expected_origin=selection.selected.native_origin,
                            expected_exposures=expected_exposures,
                        )
                        observed_wire = {
                            window["color_id"]: window["exposure_raw_10ns"]
                            for window in observed
                        }
                        meter_commanded_wire[group_index] = observed_wire
                        observed_named = _controller_exposures_from_wire(
                            observed_wire
                        )
                        observed_wire_json = {
                            str(color): exposure
                            for color, exposure in observed_wire.items()
                        }
                        journal["meter_observed_exposures_raw_10ns"].append(
                            observed_wire_json
                        )
                        journal["meter_pass_exposures_raw_10ns"].append(
                            observed_wire_json
                        )
                        journal["meter_pass_commanded_exposures"].append(
                            {
                                "pass": group_index + 1,
                                "controller_channels_raw_10ns": observed_named,
                                "wire_colors_raw_10ns": observed_wire_json,
                            }
                        )
                        _write_journal(journal_path, journal)

                if sequence in FINE_GET_WINDOW_SEQUENCES:
                    fine_window_payloads.append(result.payload)

                if sequence in METER_READ_SEQUENCES:
                    written = meter_output.write(result.payload)
                    if written != len(result.payload):
                        raise SynchronizedProtocolError(
                            f"short meter file write {written} of "
                            f"{len(result.payload)} bytes"
                        )
                    meter_evidence_sha256.update(result.payload)
                    group_index = next(
                        index
                        for index, group in enumerate(METER_READ_GROUPS)
                        if sequence in group
                    )
                    meter_group_bytes[group_index] += len(result.payload)
                    meter_group_payloads[group_index].extend(result.payload)
                    journal["meter_completed_reads"] += 1
                    journal["meter_completed_bytes"] += len(result.payload)
                    if sequence == METER_READ_GROUPS[group_index][-1]:
                        if meter_group_bytes[group_index] != METER_GROUP_BYTES:
                            raise SynchronizedProtocolError(
                                f"meter pass {group_index + 1} has "
                                f"{meter_group_bytes[group_index]} bytes; expected "
                                f"{METER_GROUP_BYTES}"
                            )
                        meter_output.flush()
                        os.fsync(meter_output.fileno())
                        _fsync_parent_directory(meter_path)
                        journal["meter_evidence"] = {
                            "path": str(meter_path.resolve()),
                            "bytes": journal["meter_completed_bytes"],
                            "sha256": meter_evidence_sha256.hexdigest(),
                            "complete": False,
                            "durable_completed_passes": group_index + 1,
                        }
                        _write_journal(journal_path, journal)
                        observed_wire = meter_commanded_wire[group_index]
                        if observed_wire is None:
                            raise SynchronizedProtocolError(
                                f"meter pass {group_index + 1} has no validated "
                                "GET_WINDOW exposure echo"
                            )
                        observation = observe_meter_pass(
                            bytes(meter_group_payloads[group_index]),
                            _controller_exposures_from_wire(observed_wire),
                        )
                        meter_observations.append(observation)
                        if group_index < len(METER_READ_GROUPS) - 1:
                            previous = (
                                meter_observations[-2]
                                if len(meter_observations) > 1
                                else None
                            )
                            proposal = propose_next_exposures(
                                observation,
                                previous=previous,
                            )
                            proposal_record: dict[str, object] = {
                                "pass": group_index + 1,
                                **proposal.to_dict(),
                            }
                            journal["meter_controller_proposals"].append(
                                proposal_record
                            )
                            if not proposal.accepted:
                                codes = ", ".join(
                                    refusal.code for refusal in proposal.refusals
                                )
                                raise SynchronizedProtocolError(
                                    f"meter pass {group_index + 1} controller "
                                    f"refused: {codes}"
                                )
                            next_group = group_index + 1
                            patched_wire = _patch_exposure_contract(
                                active_plan,
                                DYNAMIC_WINDOW_GROUPS[next_group],
                                METER_GET_WINDOW_GROUPS[next_group],
                                proposal.proposed_exposures,
                            )
                            proposal_record[
                                "applied_to_next_pass_wire_colors_raw_10ns"
                            ] = {
                                str(color): exposure
                                for color, exposure in patched_wire.items()
                            }
                            _write_journal(journal_path, journal)
                        else:
                            final_result = verify_final_convergence(
                                observation,
                                previous=meter_observations[-2],
                            )
                            journal["meter_controller_final_result"] = (
                                final_result.to_dict()
                            )
                            if (
                                not final_result.accepted
                                or final_result.final_exposures is None
                            ):
                                codes = ", ".join(
                                    refusal.code
                                    for refusal in final_result.refusals
                                )
                                raise SynchronizedProtocolError(
                                    "meter pass 3 final controller refused: "
                                    f"{codes}"
                                )
                            final_wire = _patch_exposure_contract(
                                active_plan,
                                DYNAMIC_WINDOW_GROUPS[-1],
                                FINE_GET_WINDOW_SEQUENCES,
                                final_result.final_exposures,
                            )
                            final_wire_exposures = dict(final_wire)
                            journal["meter_final_exposures"] = {
                                "controller_channels_raw_10ns": dict(
                                    final_result.final_exposures
                                ),
                                "wire_colors_raw_10ns": {
                                    str(color): exposure
                                    for color, exposure in final_wire.items()
                                },
                            }
                            final_controller_accepted = True
                            _write_journal(journal_path, journal)

                    if sequence == METER_STOP_SEQUENCE:
                        if any(
                            size != METER_GROUP_BYTES
                            for size in meter_group_bytes
                        ):
                            raise SynchronizedProtocolError(
                                f"meter groups have sizes {meter_group_bytes}"
                            )
                        meter_evidence = b"".join(meter_group_payloads)
                        if len(meter_evidence) != METER_CAPTURE_BYTES:
                            raise SynchronizedProtocolError(
                                "raw meter evidence has the wrong size"
                            )
                        meter_sha256 = hashlib.sha256(meter_evidence).hexdigest()
                        if meter_sha256 != meter_evidence_sha256.hexdigest():
                            raise SynchronizedProtocolError(
                                "meter evidence digests disagree"
                            )
                        journal["meter_group_bytes"] = meter_group_bytes
                        journal["meter_group_offsets"] = [
                            index * METER_GROUP_BYTES
                            for index in range(len(METER_READ_GROUPS))
                        ]
                        journal["meter_evidence"] = {
                            "path": str(meter_path.resolve()),
                            "bytes": len(meter_evidence),
                            "sha256": meter_sha256,
                            "complete": True,
                            "durable_completed_passes": len(METER_READ_GROUPS),
                        }
                        meter_output.flush()
                        os.fsync(meter_output.fileno())
                        journal["meter_evidence_persisted_before_fine_arm"] = True
                        meter_evidence_persisted = True
                        _write_journal(journal_path, journal)

            if (
                not final_controller_accepted
                or not meter_evidence_persisted
                or final_wire_exposures is None
            ):
                raise SynchronizedProtocolError(
                    "continuation fine capture lacks accepted metering evidence"
                )
            final_windows = [
                decode_window_block(
                    bytes.fromhex(_entry(active_plan, sequence)["data_out"])
                )
                for sequence in DYNAMIC_WINDOW_GROUPS[-1]
            ]
            if any(window is None for window in final_windows):
                raise ProtocolError("continuation final SET_WINDOW is malformed")
            expected_exposures = {
                window["color_id"]: window["exposure_raw_10ns"]
                for window in final_windows
                if window is not None
            }
            fine_windows = _validate_live_fine_windows(
                fine_window_payloads,
                expected_origin=selection.selected.native_origin,
                expected_exposures=expected_exposures,
            )
            journal["fine_windows"] = [
                {
                    "color_id": window["color_id"],
                    "resolution": [window["resx"], window["resy"]],
                    "origin": [window["upper_left_x"], window["upper_left_y"]],
                    "size": [window["width"], window["height"]],
                    "samples": window["samples_per_scan_minus1_nibble"] + 1,
                    "interleave": window["color_interleaving_byte"],
                    "exposure_raw_10ns": window["exposure_raw_10ns"],
                }
                for window in fine_windows
            ]
            journal["status"] = "fine-capture"
            _write_journal(journal_path, journal)
            for read_index in range(EXPECTED_FINE_READS):
                timeout = 180_000 if read_index == 0 else 60_000
                journal["current_command"] = {
                    "seq": target["seq"],
                    "name": "fine READ",
                    "cdb": target["cdb"],
                    "read_index": read_index,
                    "request_len": target["request_len"],
                    "request_parts": target.get("request_parts"),
                }
                lifecycle.at_transaction_boundary = False
                result = _perform_with_busy_retry(
                    ep_out,
                    ep_in,
                    target,
                    data_timeout_ms=timeout,
                    allow_busy_retry=True,
                )
                lifecycle.at_transaction_boundary = True
                if read_index + 1 == EXPECTED_FINE_READS:
                    lifecycle.scan_active = False
                    lifecycle.ready_required = True
                written = output.write(result.payload)
                if written != len(result.payload):
                    raise SynchronizedProtocolError(
                        f"short file write {written} of {len(result.payload)} bytes"
                    )
                output_sha256.update(result.payload)
                journal["completed_reads"] = read_index + 1
                journal["completed_bytes"] += len(result.payload)
                journal["stall_recoveries"] += result.stall_recoveries
                if (read_index + 1) % 25 == 0:
                    _write_journal(journal_path, journal)
            output.flush()
            os.fsync(output.fileno())
            _fsync_parent_directory(output_path)

        journal["output_sha256"] = output_sha256.hexdigest()
        journal["disk_bytes"] = output_path.stat().st_size
        if (
            journal["completed_bytes"] != expected_bytes
            or journal["disk_bytes"] != expected_bytes
        ):
            raise SynchronizedProtocolError(
                "continuation capture has the wrong final byte count"
            )
        lifecycle.at_transaction_boundary = False
        polls, stalls = _wait_post_scan_ready(ep_out, ep_in)
        lifecycle.at_transaction_boundary = True
        lifecycle.scan_active = False
        lifecycle.ready_required = False
        journal["post_scan_ready_polls"] = polls
        journal["stall_recoveries"] += stalls
        journal["ack_nonce"] = secrets.token_hex(16)
        journal["frame_complete"] = True
        journal["recovery_required"] = None
        journal["status"] = "frame-complete"
        journal["finished_unix"] = time.time()
        _write_journal(journal_path, journal)
        return journal
    except BaseException as error:
        journal["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        journal["error"] = f"{type(error).__name__}: {error}"
        journal["finished_unix"] = time.time()
        try:
            _write_journal(journal_path, journal)
        except Exception:
            pass
        raise


def run_live_capture(
    plan: list[dict],
    plan_path: Path,
    plan_sha256: str,
    output_path: Path,
    journal_path: Path,
    read_count: int,
    *,
    frame: int | None = None,
    boundary_offset_rows: int = 0,
    meter_only: bool = False,
    preview_only: bool = False,
    expected_frame_count: int | None = None,
    batch_job: LiveBatchJob | None = None,
    continuation_plan: dict[str, Any] | None = None,
    continuation_plan_sha256: str | None = None,
    session_journal_path: Path | None = None,
) -> None:
    target = validate_plan(plan)
    if meter_only and preview_only:
        raise ProtocolError("live capture cannot be both meter-only and preview-only")
    batch_values = (
        batch_job,
        continuation_plan,
        continuation_plan_sha256,
        session_journal_path,
    )
    batch_mode = any(value is not None for value in batch_values)
    if batch_mode and any(value is None for value in batch_values):
        raise ProtocolError("live batch capture requires every batch component")
    if batch_mode and (meter_only or preview_only):
        raise ProtocolError("live batch capture supports full frames only")
    if batch_mode:
        assert batch_job is not None
        assert continuation_plan is not None
        assert continuation_plan_sha256 is not None
        assert session_journal_path is not None
        if continuation_plan_sha256 != CANONICAL_CONTINUATION_PLAN_SHA256:
            raise ProtocolError("live batch continuation digest is not canonical")
        if batch_job.plan_sha256 != plan_sha256:
            raise ProtocolError("live batch plan digest does not match its job")
        if batch_job.continuation_plan_sha256 != continuation_plan_sha256:
            raise ProtocolError(
                "live batch continuation digest does not match its job"
            )
        derive_equivalent_continuation_blocks(continuation_plan)
        first_spec = batch_job.frames[0]
        if (
            frame != first_spec.slot
            or boundary_offset_rows != first_spec.boundary_offset_rows
            or output_path.resolve() != first_spec.output.resolve()
            or journal_path.resolve() != first_spec.journal.resolve()
        ):
            raise ProtocolError("first live batch frame does not match its job")
        if session_journal_path.exists():
            raise ProtocolError(
                f"refusing to overwrite batch session journal {session_journal_path}"
            )
        if (
            session_journal_path.resolve()
            != (batch_job.root / "session-journal.json").resolve()
        ):
            raise ProtocolError("live batch session journal is outside its job root")
    if preview_only and frame is not None:
        raise ProtocolError("preview-only capture does not accept --frame")
    if preview_only and boundary_offset_rows != 0:
        raise ProtocolError("preview-only capture does not accept a boundary offset")
    if not preview_only and frame is None:
        raise ProtocolError("live capture requires an explicit same-traversal --frame")
    if frame is not None:
        _validate_boundary_offset(frame, boundary_offset_rows)
    if preview_only and expected_frame_count is not None:
        raise ProtocolError(
            "preview-only capture does not accept an expected frame count"
        )
    if expected_frame_count is not None and (
        isinstance(expected_frame_count, bool) or not 2 <= expected_frame_count <= 40
    ):
        raise ProtocolError("expected frame count must be an integer in 2..40")
    if (
        not meter_only
        and not preview_only
        and (read_count != EXPECTED_FINE_READS or read_count != target["repeat"])
    ):
        raise ProtocolError(
            "live fine capture requires the complete 2,980-read stream; "
            "a one-read probe is unsafe"
        )
    if output_path.exists():
        raise ProtocolError(f"refusing to overwrite {output_path}")
    if journal_path.exists():
        raise ProtocolError(f"refusing to overwrite {journal_path}")

    artifact_paths = _live_index_artifact_paths(output_path)
    meter_sidecar_path = (
        None if meter_only or preview_only else _full_capture_meter_path(output_path)
    )
    for artifact in artifact_paths.values():
        if artifact.exists():
            raise ProtocolError(f"refusing to overwrite {artifact}")
    if meter_sidecar_path is not None and meter_sidecar_path.exists():
        raise ProtocolError(f"refusing to overwrite {meter_sidecar_path}")

    expected_bytes = (
        0
        if preview_only
        else (METER_CAPTURE_BYTES if meter_only else read_count * target["request_len"])
    )
    free_bytes = shutil.disk_usage(output_path.parent).free
    required_free = expected_bytes + max(1_073_741_824, expected_bytes // 10)
    if free_bytes < required_free:
        raise ProtocolError(
            f"only {free_bytes} free bytes; capture requires {required_free}"
        )
    journal: dict[str, Any] = {
        "status": "starting",
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "capture_engine_sha256": CAPTURE_WORKER_SHA256,
        "capture_bundle_sha256": CAPTURE_BUNDLE_SHA256,
        "meter_controller_sha256": _meter_controller_sha256(),
        "output": str(output_path.resolve()),
        "capture_mode": (
            "preview-only" if preview_only else ("meter-only" if meter_only else "full")
        ),
        "requested_frame": frame,
        "expected_frame_count": expected_frame_count,
        "requested_boundary_offset_rows": boundary_offset_rows,
        "applied_boundary_offset_rows": None,
        "resolved_lookup_row": None,
        "resolved_native_origin": None,
        "expected_reads": (
            0
            if preview_only
            else (len(METER_READ_SEQUENCES) if meter_only else read_count)
        ),
        "expected_bytes": expected_bytes,
        "completed_reads": 0,
        "completed_bytes": 0,
        "stall_recoveries": 0,
        "started_unix": time.time(),
    }
    session_journal: dict[str, Any] | None = None
    frame_journal_finalized = False
    batch_stopped = False
    batch_lifecycle = SessionLifecycle()
    if batch_mode:
        assert batch_job is not None
        assert continuation_plan_sha256 is not None
        assert session_journal_path is not None
        journal.update(
            {
                "ack_nonce": None,
                "batch_session": {
                    "frame_index": 1,
                    "frame_total": len(batch_job.frames),
                    "selected_slots": list(batch_job.selected_slots),
                    "session_id": batch_job.session_id,
                },
                "continuation_plan_sha256": continuation_plan_sha256,
                "frame_complete": False,
                "session_reservation_retained": False,
                "unit_released": False,
            }
        )
        session_journal = {
            "status": "capturing",
            "session_id": batch_job.session_id,
            "selected_slots": list(batch_job.selected_slots),
            "completed_slots": [],
            "active_frame_index": 1,
            "active_slot": batch_job.frames[0].slot,
            "batch_job_sha256": batch_job.job_sha256,
            "capture_engine_sha256": CAPTURE_WORKER_SHA256,
            "capture_bundle_sha256": CAPTURE_BUNDLE_SHA256,
            "plan_sha256": plan_sha256,
            "continuation_plan_sha256": continuation_plan_sha256,
            "reservation_acquired": False,
            "unit_release_attempts": 0,
            "unit_released": False,
            "recovery_required": None,
            "started_unix": time.time(),
        }
        _write_journal(session_journal_path, session_journal)
    if artifact_paths:
        journal["live_index_artifacts"] = {
            key: str(path.resolve()) for key, path in artifact_paths.items()
        }
    if meter_sidecar_path is not None:
        journal["meter_evidence_path"] = str(meter_sidecar_path.resolve())
    _write_journal(journal_path, journal)

    device = interface = ep_out = ep_in = usb_util = None
    at_transaction_boundary = True
    reserved = False
    scan_active = False
    ready_required = False
    meter_output = None
    try:
        device, interface, ep_out, ep_in, usb_util = _connect_device()
        journal["status"] = "preamble"
        journal["endpoint_out"] = f"0x{ep_out.bEndpointAddress:02x}"
        journal["endpoint_in"] = f"0x{ep_in.bEndpointAddress:02x}"
        _write_journal(journal_path, journal)

        with output_path.open("xb") as output:
            if meter_sidecar_path is not None:
                meter_output = meter_sidecar_path.open("xb")
            active_plan = [dict(entry) for entry in plan]
            preamble = active_plan[:-1]
            geometry = _derive_index_geometry(active_plan)
            entry_index = 0
            fine_window_payloads: list[bytes] = []
            preview_window_payloads: list[bytes] = []
            meter_window_payloads: list[list[bytes]] = [
                [] for _group in METER_GET_WINDOW_GROUPS
            ]
            preview_data = bytearray()
            live_sub_8e_header: bytes | None = None
            live_sub_8e_table: bytes | None = None
            live_selection: LiveFrameSelection | None = None
            batch_selections: tuple[LiveFrameSelection, ...] | None = None
            startup_table: StartupFrameTable | None = None
            meter_group_bytes = [0] * len(METER_READ_GROUPS)
            meter_group_payloads = [bytearray() for _group in METER_READ_GROUPS]
            meter_commanded_wire: list[dict[int, int] | None] = [
                None for _group in METER_READ_GROUPS
            ]
            meter_observations: list[MeterObservation] = []
            meter_controller_proposals: list[dict[str, object]] = []
            meter_evidence_persisted = False
            meter_evidence_sha256 = hashlib.sha256()
            final_controller_accepted = False
            final_wire_exposures: dict[int, int] | None = None
            output_sha256 = hashlib.sha256()
            while entry_index < len(preamble):
                entry = preamble[entry_index]
                if entry["seq"] == DYNAMIC_WINDOW_GROUPS[-1][0]:
                    if not final_controller_accepted:
                        raise SynchronizedProtocolError(
                            "fine SET_WINDOW reached without an accepted final "
                            "meter-controller result"
                        )
                    if not meter_only and not meter_evidence_persisted:
                        raise SynchronizedProtocolError(
                            "fine SET_WINDOW reached before raw meter evidence "
                            "was durably persisted"
                        )
                    if live_selection is None or final_wire_exposures is None:
                        raise SynchronizedProtocolError(
                            "fine SET_WINDOW reached without a live origin and "
                            "final exposure contract"
                        )
                    preflight_windows = _validate_live_fine_windows(
                        [
                            bytes.fromhex(
                                _entry(active_plan, sequence).get("data_out", "")
                            )
                            for sequence in DYNAMIC_WINDOW_GROUPS[-1]
                        ],
                        expected_origin=live_selection.selected.native_origin,
                        expected_exposures=final_wire_exposures,
                    )
                    journal["fine_set_windows_preflight"] = [
                        {
                            "color_id": window["color_id"],
                            "origin": [
                                window["upper_left_x"],
                                window["upper_left_y"],
                            ],
                            "resolution": [window["resx"], window["resy"]],
                            "size": [window["width"], window["height"]],
                            "samples": window["samples_per_scan_minus1_nibble"] + 1,
                            "exposure_raw_10ns": window["exposure_raw_10ns"],
                        }
                        for window in preflight_windows
                    ]
                    journal["fine_set_windows_preflight_before_sequence"] = entry["seq"]
                    _write_journal(journal_path, journal)
                if entry["seq"] == 172:
                    if live_sub_8e_header is None:
                        raise SynchronizedProtocolError(
                            "command 172: missing live 0x8e length header"
                        )
                    entry = _bind_live_sub_8e_read(entry, live_sub_8e_header)
                    journal["live_sub_8e_length"] = entry["request_len"]
                    journal["live_sub_8e_cdb"] = entry["cdb"]
                    _write_journal(journal_path, journal)
                if entry.get("name") == "TEST_UNIT_READY":
                    group_end = entry_index + 1
                    while (
                        group_end < len(preamble)
                        and preamble[group_end].get("name") == "TEST_UNIT_READY"
                    ):
                        group_end += 1
                    journal["current_command"] = {
                        "seq": f"{entry['seq']}..{preamble[group_end - 1]['seq']}",
                        "name": "TEST_UNIT_READY group",
                        "cdb": entry["cdb"],
                    }
                    at_transaction_boundary = False
                    polls, stalls = _perform_ready_group(
                        ep_out, ep_in, preamble[entry_index:group_end]
                    )
                    at_transaction_boundary = True
                    ready_required = False
                    journal["ready_polls"] = journal.get("ready_polls", 0) + polls
                    journal["stall_recoveries"] += stalls
                    entry_index = group_end
                    continue
                request = entry.get("request_len", 0)
                timeout = 120_000 if request > 60_000 else 30_000
                journal["current_command"] = {
                    "seq": entry["seq"],
                    "name": entry.get("name"),
                    "cdb": entry["cdb"],
                    "request_len": request,
                    "request_parts": entry.get("request_parts"),
                }
                at_transaction_boundary = False
                if entry["seq"] == VARIABLE_FRAME_TABLE_SEQUENCE:
                    result = _perform_variable_frame_table_transaction(
                        ep_out, ep_in, entry, data_timeout_ms=timeout
                    )
                else:
                    result = _perform_with_busy_retry(
                        ep_out, ep_in, entry, data_timeout_ms=timeout
                    )
                at_transaction_boundary = True
                journal["stall_recoveries"] += result.stall_recoveries
                scan_active, ready_required = _scan_lifecycle_after_transaction(
                    entry,
                    result,
                    scan_active=scan_active,
                    ready_required=ready_required,
                )
                if entry["seq"] == 1:
                    _validate_scanner_identity(result.payload)
                    journal["scanner_identity"] = "Nikon LS-5000 ED 1.03"
                if entry["seq"] == 17:
                    reserved = True
                    if session_journal is not None and session_journal_path is not None:
                        session_journal["reservation_acquired"] = True
                        _write_journal(session_journal_path, session_journal)
                if entry["seq"] == VARIABLE_FRAME_TABLE_SEQUENCE:
                    startup_table = _validate_variable_frame_table_payload(
                        result.payload
                    )
                    journal["live_startup_0x8f"] = startup_table
                    _write_journal(journal_path, journal)
                if entry["seq"] in (115, 116, 117):
                    preview_window_payloads.append(result.payload)
                    if entry["seq"] == 117:
                        preview_windows = _validate_live_preview_windows(
                            preview_window_payloads, geometry
                        )
                        journal["preview_windows"] = [
                            {
                                "color_id": window["color_id"],
                                "resolution": [window["resx"], window["resy"]],
                                "origin": [
                                    window["upper_left_x"],
                                    window["upper_left_y"],
                                ],
                                "size": [window["width"], window["height"]],
                                "bit_depth": window["bit_depth"],
                            }
                            for window in preview_windows
                        ]
                        journal["preview_geometry_validated_before_reads"] = True
                        _write_journal(journal_path, journal)
                if entry["seq"] in PREVIEW_READ_SEQUENCES:
                    if len(result.payload) != request:
                        raise SynchronizedProtocolError(
                            f"preview READ {entry['seq']} delivered "
                            f"{len(result.payload)} bytes, expected {request}"
                        )
                    preview_data.extend(result.payload)
                if entry["seq"] == 171:
                    live_sub_8e_header = result.payload
                    journal["live_sub_8e_header"] = result.payload.hex()
                    _write_journal(journal_path, journal)
                if entry["seq"] == 172:
                    if (
                        live_sub_8e_header is None
                        or result.payload[:6] != live_sub_8e_header
                    ):
                        raise SynchronizedProtocolError(
                            "command 172 table does not repeat command 171 header"
                        )
                    live_sub_8e_table = result.payload
                if entry["seq"] == 172:
                    if live_sub_8e_table is None:
                        raise SynchronizedProtocolError(
                            "command 172 completed without a live 0x8e table"
                        )
                    if len(preview_data) != EXPECTED_PREVIEW_BYTES:
                        raise SynchronizedProtocolError(
                            f"live preview has {len(preview_data)} bytes, expected "
                            f"{EXPECTED_PREVIEW_BYTES}"
                        )
                    preview_bytes = bytes(preview_data)
                    preview_sha256 = hashlib.sha256(preview_bytes).hexdigest()
                    table_sha256 = hashlib.sha256(live_sub_8e_table).hexdigest()
                    _write_bytes_exclusive(artifact_paths["preview"], preview_bytes)
                    _write_bytes_exclusive(artifact_paths["table"], live_sub_8e_table)
                    journal["live_index_evidence"] = {
                        "status": "persisted-before-frame-detection",
                        "preview_bytes": len(preview_bytes),
                        "preview_sha256": preview_sha256,
                        "table_bytes": len(live_sub_8e_table),
                        "table_sha256": table_sha256,
                    }
                    _write_journal(journal_path, journal)
                    if preview_only:
                        if startup_table is None:
                            raise SynchronizedProtocolError(
                                "preview completed without a validated startup frame table"
                            )
                        preview_receipt = {
                            "status": "preview-only-complete",
                            "slot_capacity_hint": startup_table["count"],
                            "slot_capacity_semantics": (
                                "scanner-addressable preview slots; not an exposure count"
                            ),
                            "preview_bytes": len(preview_bytes),
                            "preview_sha256": preview_sha256,
                            "table_bytes": len(live_sub_8e_table),
                            "table_sha256": table_sha256,
                            "frame_detection": "deferred-offline",
                        }
                        _write_json_exclusive(
                            artifact_paths["mapping"], preview_receipt
                        )
                        journal["preview_only_receipt"] = preview_receipt
                        journal["status"] = "preview-captured"
                        _write_journal(journal_path, journal)
                        break
                    try:
                        if frame is None:
                            raise ProtocolError(
                                "full capture lost its explicit frame binding"
                            )
                        if batch_mode:
                            assert batch_job is not None
                            batch_selections = _derive_live_batch_selections(
                                active_plan,
                                preview_bytes,
                                live_sub_8e_table,
                                batch_job.frames,
                            )
                            live_selection = batch_selections[0]
                            journal["batch_prevalidated_frame_selections"] = [
                                selection.diagnostics()
                                for selection in batch_selections
                            ]
                        else:
                            live_selection = _derive_live_frame_selection(
                                active_plan,
                                preview_bytes,
                                live_sub_8e_table,
                                frame=frame,
                                boundary_offset_rows=boundary_offset_rows,
                                expected_frame_count=expected_frame_count,
                            )
                    except Exception as selection_error:
                        refusal = {
                            "status": "refused-before-frame-binding",
                            "requested_frame": frame,
                            "requested_boundary_offset_rows": boundary_offset_rows,
                            "expected_frame_count": expected_frame_count,
                            "error_type": type(selection_error).__name__,
                            "error": str(selection_error),
                            "preview_bytes": len(preview_bytes),
                            "preview_sha256": preview_sha256,
                            "table_bytes": len(live_sub_8e_table),
                            "table_sha256": table_sha256,
                        }
                        try:
                            _write_json_exclusive(artifact_paths["mapping"], refusal)
                            refusal["diagnostic_artifact_persisted"] = True
                        except Exception as artifact_error:
                            refusal["diagnostic_artifact_persisted"] = False
                            refusal["diagnostic_artifact_error"] = (
                                f"{type(artifact_error).__name__}: {artifact_error}"
                            )
                        journal["live_frame_selection_refusal"] = refusal
                        _write_journal(journal_path, journal)
                        raise
                    bound_plan = _bind_plan_to_live_selection(
                        active_plan, live_selection
                    )
                    journal["applied_boundary_offset_rows"] = (
                        live_selection.applied_boundary_offset_rows
                    )
                    journal["resolved_lookup_row"] = live_selection.selected.lookup_row
                    journal["resolved_native_origin"] = (
                        live_selection.selected.native_origin
                    )
                    initial_wire_exposures = _patch_exposure_contract(
                        bound_plan,
                        DYNAMIC_WINDOW_GROUPS[0],
                        METER_GET_WINDOW_GROUPS[0],
                        dict(DEFAULT_EXPOSURES),
                    )
                    # `preamble` holds these dictionaries by reference.  Update
                    # them in place so the very next command (SEND 0x8f) is the
                    # live-bound version; no stale frame-bearing command can run.
                    for sequence in range(FRAME_TABLE_SEND_SEQUENCE, 608):
                        active_plan[sequence - 1].clear()
                        active_plan[sequence - 1].update(bound_plan[sequence - 1])
                    _write_json_exclusive(
                        artifact_paths["mapping"], live_selection.diagnostics()
                    )
                    journal["live_frame_selection"] = live_selection.diagnostics()
                    journal["meter_observed_exposures_raw_10ns"] = []
                    journal["meter_layout"] = {
                        "passes": 3,
                        "rows_per_pass": 425,
                        "columns": 281,
                        "decoded_raster_channel_order": ["R", "G", "B", "IR"],
                        "wire_window_color_order": list(WIRE_METER_COLORS),
                        "wire_color_to_controller_channel": {
                            str(color): channel
                            for color, channel in (
                                WIRE_COLOR_TO_CONTROLLER_CHANNEL.items()
                            )
                        },
                        "sample_byte_order": "big-endian-u16",
                        "row_core_bytes": 2_248,
                        "row_stride_bytes": 2_560,
                        "row_tail_bytes": 312,
                    }
                    journal["meter_completed_reads"] = 0
                    journal["meter_completed_bytes"] = 0
                    journal["meter_pass_exposures_raw_10ns"] = []
                    journal["meter_pass_commanded_exposures"] = []
                    meter_controller_proposals.clear()
                    journal["meter_controller_proposals"] = (
                        meter_controller_proposals
                    )
                    journal["meter_controller_seed"] = {
                        "controller_channels_raw_10ns": dict(DEFAULT_EXPOSURES),
                        "wire_colors_raw_10ns": {
                            str(color): exposure
                            for color, exposure in initial_wire_exposures.items()
                        },
                    }
                    journal["status"] = "metering"
                    _write_journal(journal_path, journal)
                if entry["seq"] in METER_GET_WINDOW_SEQUENCES:
                    group_index = next(
                        index
                        for index, group in enumerate(METER_GET_WINDOW_GROUPS)
                        if entry["seq"] in group
                    )
                    meter_window_payloads[group_index].append(result.payload)
                    if entry["seq"] == METER_GET_WINDOW_GROUPS[group_index][-1]:
                        if live_selection is None:
                            raise SynchronizedProtocolError(
                                "meter GET_WINDOW reached without live frame binding"
                            )
                        expected_exposures = _window_exposures(
                            active_plan, DYNAMIC_WINDOW_GROUPS[group_index]
                        )
                        observed = _validate_live_meter_windows(
                            meter_window_payloads[group_index],
                            expected_origin=live_selection.selected.native_origin,
                            expected_exposures=expected_exposures,
                        )
                        observed_wire = {
                            window["color_id"]: window["exposure_raw_10ns"]
                            for window in observed
                        }
                        meter_commanded_wire[group_index] = observed_wire
                        observed_named = _controller_exposures_from_wire(observed_wire)
                        observed_wire_json = {
                            str(color): exposure
                            for color, exposure in observed_wire.items()
                        }
                        journal["meter_observed_exposures_raw_10ns"].append(
                            observed_wire_json
                        )
                        journal["meter_pass_exposures_raw_10ns"].append(
                            observed_wire_json
                        )
                        journal["meter_pass_commanded_exposures"].append(
                            {
                                "pass": group_index + 1,
                                "controller_channels_raw_10ns": observed_named,
                                "wire_colors_raw_10ns": observed_wire_json,
                            }
                        )
                        _write_journal(journal_path, journal)
                if entry["seq"] in FINE_GET_WINDOW_SEQUENCES:
                    fine_window_payloads.append(result.payload)
                if entry["seq"] in METER_READ_SEQUENCES:
                    meter_destination = output if meter_only else meter_output
                    if meter_destination is None:
                        raise ProtocolError("meter evidence destination is not open")
                    written = meter_destination.write(result.payload)
                    if written != len(result.payload):
                        raise SynchronizedProtocolError(
                            f"short meter file write {written} of "
                            f"{len(result.payload)} bytes"
                        )
                    meter_evidence_sha256.update(result.payload)
                    if meter_only:
                        output_sha256.update(result.payload)
                    group_index = next(
                        index
                        for index, group in enumerate(METER_READ_GROUPS)
                        if entry["seq"] in group
                    )
                    meter_group_bytes[group_index] += len(result.payload)
                    meter_group_payloads[group_index].extend(result.payload)
                    journal["meter_completed_reads"] += 1
                    journal["meter_completed_bytes"] += len(result.payload)
                    if meter_only:
                        journal["completed_reads"] += 1
                        journal["completed_bytes"] += len(result.payload)
                    if entry["seq"] == METER_READ_GROUPS[group_index][-1]:
                        if meter_group_bytes[group_index] != METER_GROUP_BYTES:
                            raise SynchronizedProtocolError(
                                f"meter pass {group_index + 1} has "
                                f"{meter_group_bytes[group_index]} bytes; expected "
                                f"{METER_GROUP_BYTES}"
                            )
                        meter_destination.flush()
                        os.fsync(meter_destination.fileno())
                        meter_evidence_path = output_path
                        if not meter_only:
                            if meter_sidecar_path is None:
                                raise ProtocolError(
                                    "full capture has no meter sidecar path"
                                )
                            meter_evidence_path = meter_sidecar_path
                        _fsync_parent_directory(meter_evidence_path)
                        journal["meter_evidence"] = {
                            "path": str(meter_evidence_path.resolve()),
                            "bytes": journal["meter_completed_bytes"],
                            "sha256": meter_evidence_sha256.hexdigest(),
                            "complete": False,
                            "durable_completed_passes": group_index + 1,
                        }
                        _write_journal(journal_path, journal)
                        observed_wire = meter_commanded_wire[group_index]
                        if observed_wire is None:
                            raise SynchronizedProtocolError(
                                f"meter pass {group_index + 1} has no validated "
                                "GET_WINDOW exposure echo"
                            )
                        observation = observe_meter_pass(
                            bytes(meter_group_payloads[group_index]),
                            _controller_exposures_from_wire(observed_wire),
                        )
                        meter_observations.append(observation)
                        if group_index < len(METER_READ_GROUPS) - 1:
                            previous = (
                                meter_observations[-2]
                                if len(meter_observations) > 1
                                else None
                            )
                            proposal = propose_next_exposures(
                                observation, previous=previous
                            )
                            proposal_record: dict[str, object] = {
                                "pass": group_index + 1,
                                **proposal.to_dict(),
                            }
                            meter_controller_proposals.append(proposal_record)
                            _write_journal(journal_path, journal)
                            if not proposal.accepted:
                                codes = ", ".join(
                                    refusal.code for refusal in proposal.refusals
                                )
                                raise SynchronizedProtocolError(
                                    f"meter pass {group_index + 1} controller "
                                    f"refused: {codes}"
                                )
                            next_group = group_index + 1
                            patched_wire = _patch_exposure_contract(
                                active_plan,
                                DYNAMIC_WINDOW_GROUPS[next_group],
                                METER_GET_WINDOW_GROUPS[next_group],
                                proposal.proposed_exposures,
                            )
                            proposal_record[
                                "applied_to_next_pass_wire_colors_raw_10ns"
                            ] = {
                                str(color): exposure
                                for color, exposure in patched_wire.items()
                            }
                            _write_journal(journal_path, journal)
                        else:
                            final_result = verify_final_convergence(
                                observation,
                                previous=meter_observations[-2],
                            )
                            journal["meter_controller_final_result"] = (
                                final_result.to_dict()
                            )
                            _write_journal(journal_path, journal)
                            if (
                                not final_result.accepted
                                or final_result.final_exposures is None
                            ):
                                codes = ", ".join(
                                    refusal.code for refusal in final_result.refusals
                                )
                                raise SynchronizedProtocolError(
                                    f"meter pass 3 final controller refused: {codes}"
                                )
                            final_wire = _patch_exposure_contract(
                                active_plan,
                                DYNAMIC_WINDOW_GROUPS[-1],
                                FINE_GET_WINDOW_SEQUENCES,
                                final_result.final_exposures,
                            )
                            final_wire_exposures = dict(final_wire)
                            journal["meter_final_exposures"] = {
                                "controller_channels_raw_10ns": dict(
                                    final_result.final_exposures
                                ),
                                "wire_colors_raw_10ns": {
                                    str(color): exposure
                                    for color, exposure in final_wire.items()
                                },
                            }
                            final_controller_accepted = True
                            _write_journal(journal_path, journal)
                entry_index += 1

                if entry["seq"] == METER_STOP_SEQUENCE:
                    if live_selection is None:
                        raise SynchronizedProtocolError(
                            "meter capture reached its stop without live frame binding"
                        )
                    if any(size != METER_GROUP_BYTES for size in meter_group_bytes):
                        raise SynchronizedProtocolError(
                            f"meter groups have sizes {meter_group_bytes}, expected "
                            f"{METER_GROUP_BYTES} each"
                        )
                    journal["meter_group_bytes"] = meter_group_bytes
                    journal["meter_group_offsets"] = [
                        index * METER_GROUP_BYTES
                        for index in range(len(METER_READ_GROUPS))
                    ]
                    meter_evidence = b"".join(meter_group_payloads)
                    if len(meter_evidence) != METER_CAPTURE_BYTES:
                        raise SynchronizedProtocolError(
                            f"raw meter evidence has {len(meter_evidence)} bytes, "
                            f"expected {METER_CAPTURE_BYTES}"
                        )
                    meter_sha256 = hashlib.sha256(meter_evidence).hexdigest()
                    if meter_sha256 != meter_evidence_sha256.hexdigest():
                        raise SynchronizedProtocolError(
                            "streamed meter evidence digest disagrees with "
                            "in-memory pass assembly"
                        )
                    meter_evidence_path = output_path
                    if not meter_only:
                        if meter_sidecar_path is None:
                            raise ProtocolError(
                                "full capture has no meter sidecar path"
                            )
                        meter_evidence_path = meter_sidecar_path
                    journal["meter_evidence"] = {
                        "path": str(meter_evidence_path.resolve()),
                        "bytes": len(meter_evidence),
                        "sha256": meter_sha256,
                        "complete": True,
                        "durable_completed_passes": len(METER_READ_GROUPS),
                    }
                    if meter_only:
                        meter_evidence_persisted = True
                        _write_journal(journal_path, journal)
                        break
                    if meter_sidecar_path is None or meter_output is None:
                        raise ProtocolError("full capture has no meter sidecar")
                    meter_output.flush()
                    os.fsync(meter_output.fileno())
                    if os.fstat(meter_output.fileno()).st_size != METER_CAPTURE_BYTES:
                        raise SynchronizedProtocolError(
                            "meter sidecar size disagrees with completed evidence"
                        )
                    journal["meter_evidence_persisted_before_fine_arm"] = True
                    meter_evidence_persisted = True
                    _write_journal(journal_path, journal)

            if meter_only or preview_only:
                fine_windows = []
            else:
                if live_selection is None:
                    raise SynchronizedProtocolError(
                        "fine capture reached without live frame binding"
                    )
                if not final_controller_accepted or not meter_evidence_persisted:
                    raise SynchronizedProtocolError(
                        "fine capture reached without accepted metering evidence"
                    )
                fine_origin = live_selection.selected.native_origin
                final_windows: list[WindowBlock] = []
                for sequence in DYNAMIC_WINDOW_GROUPS[-1]:
                    window = decode_window_block(
                        bytes.fromhex(_entry(active_plan, sequence)["data_out"])
                    )
                    if window is None:
                        raise ProtocolError(
                            "final SET_WINDOW exposure contract is malformed"
                        )
                    final_windows.append(window)
                expected_exposures = {
                    window["color_id"]: window["exposure_raw_10ns"]
                    for window in final_windows
                }
                fine_windows = _validate_live_fine_windows(
                    fine_window_payloads,
                    expected_origin=fine_origin,
                    expected_exposures=expected_exposures,
                )
            journal["fine_windows"] = [
                {
                    "color_id": window["color_id"],
                    "resolution": [window["resx"], window["resy"]],
                    "origin": [window["upper_left_x"], window["upper_left_y"]],
                    "size": [window["width"], window["height"]],
                    "samples": window["samples_per_scan_minus1_nibble"] + 1,
                    "interleave": window["color_interleaving_byte"],
                    "exposure_raw_10ns": window["exposure_raw_10ns"],
                }
                for window in fine_windows
            ]
            if not meter_only and not preview_only:
                journal["status"] = "fine-capture"
                _write_journal(journal_path, journal)
                for read_index in range(read_count):
                    timeout = 180_000 if read_index == 0 else 60_000
                    journal["current_command"] = {
                        "seq": target["seq"],
                        "name": "fine READ",
                        "cdb": target["cdb"],
                        "read_index": read_index,
                        "request_len": target["request_len"],
                        "request_parts": target.get("request_parts"),
                    }
                    at_transaction_boundary = False
                    result = _perform_with_busy_retry(
                        ep_out,
                        ep_in,
                        target,
                        data_timeout_ms=timeout,
                        allow_busy_retry=True,
                    )
                    at_transaction_boundary = True
                    if read_count == target["repeat"] and read_index + 1 == read_count:
                        scan_active = False
                        ready_required = True
                    written = output.write(result.payload)
                    if written != len(result.payload):
                        raise SynchronizedProtocolError(
                            f"short file write {written} of {len(result.payload)} bytes"
                        )
                    output_sha256.update(result.payload)
                    journal["completed_reads"] = read_index + 1
                    journal["completed_bytes"] += len(result.payload)
                    journal["stall_recoveries"] += result.stall_recoveries
                    if (read_index + 1) % 25 == 0:
                        _write_journal(journal_path, journal)

            output.flush()
            os.fsync(output.fileno())
            _fsync_parent_directory(output_path)
            if meter_output is not None:
                meter_output.close()
                meter_output = None

        journal["output_sha256"] = output_sha256.hexdigest()

        if journal["completed_bytes"] != expected_bytes:
            raise SynchronizedProtocolError(
                f"final size {journal['completed_bytes']} != expected {expected_bytes}"
            )
        disk_bytes = output_path.stat().st_size
        journal["disk_bytes"] = disk_bytes
        if disk_bytes != expected_bytes:
            raise SynchronizedProtocolError(
                f"file size {disk_bytes} != expected {expected_bytes}"
            )
        journal["status"] = "teardown"
        _write_journal(journal_path, journal)
        at_transaction_boundary = False
        polls, stalls = _wait_post_scan_ready(ep_out, ep_in)
        at_transaction_boundary = True
        scan_active = False
        ready_required = False
        journal["post_scan_ready_polls"] = polls
        journal["stall_recoveries"] += stalls
        if batch_mode:
            assert batch_job is not None
            assert continuation_plan is not None
            assert continuation_plan_sha256 is not None
            assert session_journal is not None
            assert session_journal_path is not None
            if (
                live_sub_8e_table is None
                or len(preview_data) != EXPECTED_PREVIEW_BYTES
                or batch_selections is None
            ):
                raise SynchronizedProtocolError(
                    "batch first frame lost its retained roll-index evidence"
                )

            journal["ack_nonce"] = secrets.token_hex(16)
            journal["frame_complete"] = True
            journal["recovery_required"] = None
            journal["session_reservation_retained"] = True
            journal["unit_released"] = False
            journal["status"] = "frame-complete"
            journal["finished_unix"] = time.time()
            _write_journal(journal_path, journal)
            frame_journal_finalized = True

            completed_slots = session_journal["completed_slots"]
            if not isinstance(completed_slots, list):
                raise ProtocolError("batch session completed_slots is not a list")
            completed_slots.append(batch_job.frames[0].slot)
            session_journal.update(
                {
                    "active_frame_index": 1,
                    "active_slot": batch_job.frames[0].slot,
                    "recovery_required": None,
                    "status": "awaiting-parent-ack",
                }
            )
            _write_journal(session_journal_path, session_journal)
            action = wait_for_parent_ack(
                batch_job.frames[0].ack,
                session_id=batch_job.session_id,
                frame_index=1,
                slot=batch_job.frames[0].slot,
                nonce=journal["ack_nonce"],
            )
            batch_stopped = action == "stop"

            if not batch_stopped:
                for frame_index, (frame_spec, selection) in enumerate(
                    zip(
                        batch_job.frames[1:],
                        batch_selections[1:],
                        strict=True,
                    ),
                    start=2,
                ):
                    session_journal.update(
                        {
                            "active_frame_index": frame_index,
                            "active_slot": frame_spec.slot,
                            "status": "capturing",
                        }
                    )
                    _write_journal(session_journal_path, session_journal)
                    frame_journal = _run_live_continuation_frame(
                        ep_out,
                        ep_in,
                        plan,
                        plan_path,
                        plan_sha256,
                        continuation_plan,
                        continuation_plan_sha256,
                        frame_spec,
                        selection,
                        batch_job=batch_job,
                        frame_index=frame_index,
                        lifecycle=batch_lifecycle,
                    )
                    completed_slots.append(frame_spec.slot)
                    session_journal.update(
                        {
                            "active_frame_index": frame_index,
                            "active_slot": frame_spec.slot,
                            "status": "awaiting-parent-ack",
                        }
                    )
                    _write_journal(session_journal_path, session_journal)
                    action = wait_for_parent_ack(
                        frame_spec.ack,
                        session_id=batch_job.session_id,
                        frame_index=frame_index,
                        slot=frame_spec.slot,
                        nonce=frame_journal["ack_nonce"],
                    )
                    if action == "stop":
                        batch_stopped = True
                        break

            # Persist the attempt count before crossing the release boundary.
            # If release itself fails, the exception path must never retry it.
            session_journal.update(
                {
                    "status": "releasing",
                    "unit_release_attempts": 1,
                }
            )
            _write_journal(session_journal_path, session_journal)
            at_transaction_boundary = False
            batch_lifecycle.at_transaction_boundary = False
            teardown = _release_unit(ep_out, ep_in)
            at_transaction_boundary = True
            batch_lifecycle.at_transaction_boundary = True
            reserved = False
            session_journal.update(
                {
                    "active_frame_index": None,
                    "active_slot": None,
                    "finished_unix": time.time(),
                    "recovery_required": "none",
                    "release_stall_recoveries": teardown.stall_recoveries,
                    "release_status": teardown.status.hex(),
                    "status": "stopped" if batch_stopped else "complete",
                    "unit_released": True,
                }
            )
            _write_journal(session_journal_path, session_journal)
        else:
            at_transaction_boundary = False
            teardown = _release_unit(ep_out, ep_in)
            at_transaction_boundary = True
            reserved = False
            journal["stall_recoveries"] += teardown.stall_recoveries
            journal["unit_released"] = True
            journal["status"] = "complete"
            journal["finished_unix"] = time.time()
            _write_journal(journal_path, journal)
    except BaseException as error:
        cleanup_scan_active = scan_active
        cleanup_ready_required = ready_required
        cleanup_boundary = at_transaction_boundary
        if batch_mode and frame_journal_finalized:
            cleanup_scan_active = batch_lifecycle.scan_active
            cleanup_ready_required = batch_lifecycle.ready_required
            cleanup_boundary = batch_lifecycle.at_transaction_boundary
        synchronized = (
            cleanup_boundary or isinstance(error, SynchronizedProtocolError)
        ) and not isinstance(error, DesynchronizedProtocolError)
        release_already_attempted = bool(
            session_journal is not None
            and session_journal.get("unit_release_attempts") == 1
        )
        if (
            synchronized
            and ep_out is not None
            and ep_in is not None
            and not release_already_attempted
            and (reserved or cleanup_scan_active or cleanup_ready_required)
        ):
            journal["cleanup"] = _cleanup_synchronized(
                ep_out,
                ep_in,
                scan_active=cleanup_scan_active,
                ready_required=cleanup_ready_required,
                reserved=reserved,
            )
        cleanup_complete = journal.get("cleanup", {}).get("complete", False)
        no_cleanup_needed = (
            synchronized
            and not reserved
            and not cleanup_scan_active
            and not cleanup_ready_required
        )
        recovery_required = (
            "none"
            if synchronized and (cleanup_complete or no_cleanup_needed)
            else "power-cycle scanner before another attempt"
        )
        if release_already_attempted and reserved:
            recovery_required = "power-cycle scanner before another attempt"

        if session_journal is not None and session_journal_path is not None:
            cleanup = journal.get("cleanup", {})
            if cleanup:
                session_journal["cleanup"] = cleanup
            if cleanup.get("release_attempted") is True:
                session_journal["unit_release_attempts"] = 1
                session_journal["unit_released"] = (
                    cleanup.get("release_succeeded") is True
                )
            session_journal.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "finished_unix": time.time(),
                    "recovery_required": recovery_required,
                    "status": (
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                }
            )
            try:
                _write_journal(session_journal_path, session_journal)
            except Exception:
                pass

        # A frame-complete journal is an immutable parent/child handoff.  The
        # parent may already have hashed, promoted, and deleted its scratch
        # stream, so a later ACK/continuation/release failure belongs only in
        # the batch session receipt.
        if not frame_journal_finalized:
            journal["status"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            journal["error"] = f"{type(error).__name__}: {error}"
            journal["recovery_required"] = recovery_required
            journal["finished_unix"] = time.time()
            try:
                _write_journal(journal_path, journal)
            except Exception:
                pass
        raise
    finally:
        if meter_output is not None:
            try:
                meter_output.close()
            except Exception:
                pass
        if device is not None and interface is not None and usb_util is not None:
            try:
                usb_util.release_interface(device, interface.bInterfaceNumber)
            finally:
                usb_util.dispose_resources(device)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            str(files(DATA_PACKAGE).joinpath("replay-first-rgbi4-plan.jsonl"))
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            str(files(DATA_PACKAGE).joinpath("replay-first-rgbi4-manifest.json"))
        ),
    )
    parser.add_argument(
        "--batch-job",
        type=Path,
        help="one-process/one-reservation ordered frame job",
    )
    parser.add_argument(
        "--continuation-plan",
        type=Path,
        help="pinned later-frame continuation recipe required by --batch-job",
    )
    parser.add_argument(
        "--session-journal",
        type=Path,
        help="durable batch-level release receipt required by --batch-job",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument(
        "--reads",
        type=int,
        choices=(EXPECTED_FINE_READS,),
        default=EXPECTED_FINE_READS,
        help=(
            "fine mode always drains the complete 2,980-READ stream; "
            "partial live probes are disabled"
        ),
    )
    parser.add_argument(
        "--confirm-full-capture",
        action="store_true",
        help="required with --reads 2980",
    )
    parser.add_argument(
        "--frame",
        type=int,
        choices=range(1, 41),
        help="physical frame selected from this traversal's live roll index",
    )
    parser.add_argument(
        "--boundary-offset-rows",
        type=int,
        default=0,
        help=(
            "operator boundary adjustment in 97-dpi preview rows; frame 1 "
            "accepts 0..144 and later frames accept -144..144"
        ),
    )
    parser.add_argument(
        "--expected-frame-count",
        type=int,
        choices=range(2, 41),
        help=(
            "optional operator label for diagnostics; never changes aligned "
            "candidate-slot geometry"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--meter-only",
        action="store_true",
        help=(
            "capture three guarded 285-dpi RGB+IR metering rasters and stop "
            "before the 4000-dpi fine arm; requires --frame"
        ),
    )
    mode.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "capture and persist the whole-roll preview plus transport table, "
            "then release before frame binding or metering; does not accept --frame"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="access the scanner; without this flag only validate the plan",
    )
    args = parser.parse_args(argv)

    plan, manifest, plan_sha256 = _load_validated_plan(args.plan, args.manifest)
    target = validate_plan(plan, manifest)
    batch_job: LiveBatchJob | None = None
    continuation_plan: dict[str, Any] | None = None
    continuation_plan_sha256: str | None = None
    batch_requested = any(
        value is not None
        for value in (
            args.batch_job,
            args.continuation_plan,
            args.session_journal,
        )
    )
    if batch_requested:
        if any(
            value is None
            for value in (
                args.batch_job,
                args.continuation_plan,
                args.session_journal,
            )
        ):
            raise ProtocolError(
                "--batch-job, --continuation-plan, and --session-journal are inseparable"
            )
        if (
            args.output is not None
            or args.journal is not None
            or args.frame is not None
            or args.boundary_offset_rows != 0
            or args.expected_frame_count is not None
            or args.meter_only
            or args.preview_only
            or args.confirm_full_capture
        ):
            raise ProtocolError(
                "batch capture owns frame, output, journal, and mode arguments"
            )
        continuation_payload = args.continuation_plan.read_bytes()
        continuation_plan_sha256 = hashlib.sha256(
            continuation_payload
        ).hexdigest()
        continuation_plan = verify_canonical_continuation_plan(
            continuation_payload
        )
        batch_job = load_validated_batch_job(
            args.batch_job,
            expected_plan_sha256=plan_sha256,
            expected_continuation_sha256=continuation_plan_sha256,
        )
        expected_session_journal = batch_job.root / "session-journal.json"
        if args.session_journal.resolve() != expected_session_journal.resolve():
            raise ProtocolError(
                "batch session journal must be session-journal.json in the job root"
            )
        first_frame = batch_job.frames[0]
        output = first_frame.output
        journal = first_frame.journal
        print(
            "validated RGBI4x batch: slots "
            + ", ".join(str(slot) for slot in batch_job.selected_slots)
            + "; one child, one reservation, one final release"
        )
        if not args.live:
            print("dry run only; scanner was not accessed")
            return
        output.parent.mkdir(parents=False, exist_ok=False)
        run_live_capture(
            plan,
            args.plan,
            plan_sha256,
            output,
            journal,
            args.reads,
            frame=first_frame.slot,
            boundary_offset_rows=first_frame.boundary_offset_rows,
            batch_job=batch_job,
            continuation_plan=continuation_plan,
            continuation_plan_sha256=continuation_plan_sha256,
            session_journal_path=args.session_journal,
        )
        return
    if args.preview_only and args.frame is not None:
        raise ProtocolError("--preview-only does not accept --frame")
    if args.preview_only and args.boundary_offset_rows != 0:
        raise ProtocolError("--preview-only does not accept --boundary-offset-rows")
    if args.preview_only and args.expected_frame_count is not None:
        raise ProtocolError("--preview-only does not accept --expected-frame-count")
    if args.meter_only and args.frame is None:
        raise ProtocolError("--meter-only requires --frame")
    if not args.meter_only and not args.preview_only and args.frame is None:
        raise ProtocolError("fine capture requires --frame")
    if not args.meter_only and not args.preview_only and not args.confirm_full_capture:
        raise ProtocolError("fine capture requires --confirm-full-capture")
    if args.frame is not None:
        _validate_boundary_offset(args.frame, args.boundary_offset_rows)
    if args.output is not None:
        output = args.output
    elif args.preview_only:
        output = HERE / "rgbi4-roll-preview.bin"
    elif args.meter_only:
        output = HERE / f"rgbi4-meter-frame{args.frame:02d}.bin"
    else:
        output = HERE / "rgbi4-full-frame.bin"
    journal = args.journal or output.with_suffix(".json")
    if args.preview_only:
        print(
            "validated preview-only plan: persist whole-roll preview and "
            "transport table; hard stop before frame binding and metering"
        )
    elif args.meter_only:
        print(
            "validated guarded meter plan: "
            f"frame {args.frame}, 3 x {METER_GROUP_BYTES} = "
            f"{METER_CAPTURE_BYTES} bytes; hard stop before fine SET_WINDOW"
        )
    else:
        print(
            "validated RGBI4x plan: "
            f"selected {args.reads} x {target['request_len']} = "
            f"{args.reads * target['request_len']} bytes"
        )
    if not args.live:
        print("dry run only; scanner was not accessed")
        return
    run_live_capture(
        plan,
        args.plan,
        plan_sha256,
        output,
        journal,
        args.reads,
        frame=args.frame,
        boundary_offset_rows=args.boundary_offset_rows,
        meter_only=args.meter_only,
        preview_only=args.preview_only,
        expected_frame_count=args.expected_frame_count,
    )


if __name__ == "__main__":
    main()
