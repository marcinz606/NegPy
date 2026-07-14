"""Hardware-free proof for the Nikon later-frame continuation recipe."""

from __future__ import annotations

import hashlib
import json

from negpy.infrastructure.scanners.ls5000_single_pass.continuation_plan import (
    CANONICAL_CONTINUATION_PLAN_SHA256,
    CANONICAL_CONTINUATION_SEMANTIC_SHA256,
    canonical_continuation_plan_bytes,
    derive_equivalent_continuation_blocks,
    load_canonical_continuation_plan,
)


def test_canonical_continuation_plan_is_pinned_to_two_later_nikon_frames() -> None:
    payload = canonical_continuation_plan_bytes()
    plan = load_canonical_continuation_plan()

    assert hashlib.sha256(payload).hexdigest() == CANONICAL_CONTINUATION_PLAN_SHA256
    assert plan["source"] == {
        "capture": "oracle-ice-on-1.pcapng",
        "capture_sha256": "5e3a890fa7c61f4f6c9cf597f55c0e7ea71628818d449aa9c53760fb017d07be",
        "usb_bus": 1,
        "usb_device": 2,
        "command_blocks": [[3581, 3719], [6700, 6849]],
    }
    assert plan["session_contract"] == {
        "requires_existing_reservation": True,
        "requires_existing_frame_table": True,
        "forbidden_commands": ["RESERVE_UNIT", "SEND:sub_008f", "RELEASE_UNIT", "EJECT"],
    }
    assert plan["autofocus"]["x"] == 1973
    assert plan["autofocus"]["y"] == "resolved_native_origin+2979"
    assert plan["metering"]["passes"] == 3
    assert plan["metering"]["colors"] == [9, 1, 2, 3]
    assert plan["fine"]["colors"] == [9, 1, 2, 3]
    assert plan["fine"]["resolution"] == 4000
    assert plan["fine"]["samples_per_scan"] == 4
    assert plan["fine"]["read_cdb"] == "280000000001032c0080"
    assert plan["fine"]["read_count"] == 2980


def test_two_nikon_blocks_reduce_to_the_same_89_step_continuation() -> None:
    plan = load_canonical_continuation_plan()
    blocks = derive_equivalent_continuation_blocks(plan)
    semantic_steps = plan["trace_equivalence"]["semantic_steps"]
    encoded_steps = json.dumps(
        semantic_steps,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert [block.command_range for block in blocks] == [
        (3581, 3719),
        (6700, 6849),
    ]
    assert [block.raw_command_count for block in blocks] == [139, 150]
    assert [block.ready_poll_count for block in blocks] == [65, 76]
    assert {block.semantic_step_count for block in blocks} == {89}
    assert len(semantic_steps) == 89
    assert hashlib.sha256(encoded_steps).hexdigest() == (
        CANONICAL_CONTINUATION_SEMANTIC_SHA256
    )
    assert {block.autofocus_window_delta for block in blocks} == {2979}
    assert len({block.semantic_sha256 for block in blocks}) == 1
    # The traces genuinely differ: Nikon happened to poll busy one extra time
    # in several groups.  Equivalence therefore cannot depend on poll counts.
    assert blocks[0].ready_groups != blocks[1].ready_groups
