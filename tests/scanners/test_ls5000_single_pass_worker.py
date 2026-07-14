"""Hardware-free contracts for the packaged LS-5000 capture worker."""

from __future__ import annotations

import hashlib
import struct

import pytest

from negpy.infrastructure.scanners.ls5000_single_pass.roll_index import (
    NativeFrameOrigin,
    TransportMapping,
)
from negpy.infrastructure.scanners.ls5000_single_pass.plan import load_canonical_plan
from negpy.infrastructure.scanners.ls5000_single_pass.worker import (
    FRAME_TABLE_SEND_BYTES,
    FRAME_TABLE_SEND_RECORDS,
    ProtocolError,
    apply_boundary_offset,
    build_live_frame_table_payload,
)
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
