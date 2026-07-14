"""Offline contracts for an LS-5000 whole-roll preview session."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from negpy.infrastructure.scanners.ls5000_single_pass import roll_index
from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    AttemptPaths,
    CaptureAttemptResult,
    CaptureMode,
    CaptureOutcome,
    CaptureRequest,
)
from negpy.infrastructure.scanners.ls5000_single_pass.plan import CANONICAL_PLAN_SHA256
from negpy.services.scanning.ls5000_roll_session import (
    CaptureRoute,
    RollSessionIntegrityError,
    build_roll_preview_session,
    reload_thumbnail,
)
from negpy.services.scanning.roll_preview_controls import ScanMaterial


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encode_index(rgb16: np.ndarray) -> bytes:
    """Encode the compact RGB96 index rows used by the scanner."""

    blocks = np.zeros(
        (rgb16.shape[0] // 2, roll_index.INDEX_BLOCK_WORDS),
        dtype=np.uint16,
    )
    blocks[:, 0:96] = rgb16[0::2, :, 0]
    blocks[:, 96:192] = rgb16[0::2, :, 1]
    blocks[:, 192:288] = rgb16[0::2, :, 2]
    blocks[:, 512:608] = rgb16[1::2, :, 0]
    blocks[:, 608:704] = rgb16[1::2, :, 1]
    blocks[:, 704:800] = rgb16[1::2, :, 2]
    blocks[:, 800::2] = roll_index.INDEX_TRAILER_MARK
    blocks[:, 801::2] = np.arange(
        roll_index.INDEX_TRAILER_COUNTER0,
        roll_index.INDEX_TRAILER_COUNTER0 + roll_index.INDEX_TRAILER_WORDS // 2,
        dtype=np.uint16,
    )
    return blocks.astype(">u2", copy=False).tobytes()


def _synthetic_index(
    *,
    height: int = 6_104,
    content_frames: int = 40,
) -> np.ndarray:
    """Make 40 textured cells separated by physical clear-film gaps."""

    pitch = 143
    leader = 128
    boundaries = [leader + index * pitch for index in range(41)]
    y = np.arange(height, dtype=np.int64)[:, None]
    x = np.arange(90, dtype=np.int64)[None, :]
    texture = (x * 173 + y * 71 + (x * y) % 997) % 7_000
    aperture = np.empty((height, 90, 3), dtype=np.int64)
    for channel, base in enumerate((7_000, 5_500, 4_000)):
        aperture[:, :, channel] = base + texture * (3 - channel) // 2
    clear_base = np.asarray((34_200, 25_500, 17_800), dtype=np.int64)
    clear_noise = ((x * 19 + y * 13) % 301 - 150)[:, :, None]
    aperture[: boundaries[0]] = clear_base + clear_noise[: boundaries[0]]
    aperture[boundaries[-1] :] = clear_base + clear_noise[boundaries[-1] :]
    for boundary in boundaries:
        aperture[boundary - 3 : boundary + 3] = clear_base + clear_noise[boundary - 3 : boundary + 3]
    if content_frames < 40:
        clear_start = boundaries[content_frames]
        aperture[clear_start:] = clear_base + clear_noise[clear_start:]
    rgb = np.empty((height, 96, 3), dtype=np.int64)
    rgb[:, 2:92] = aperture
    rgb[:, :2] = np.asarray((1_300, 1_000, 700))
    rgb[:, 92:] = np.asarray((1_100, 850, 600))
    return rgb.clip(0, 65_535).astype(np.uint16)


def _transport_table(rows: int) -> bytes:
    records = bytearray()
    for row in range(rows):
        records.extend(struct.pack(">HH", 6 * (row % 18), row // 18))
    total = 8 + len(records)
    return b"\x00\x8e\x00\x00" + total.to_bytes(2, "big") + b"\x00\x00" + bytes(records)


@dataclass(frozen=True)
class PreviewFixture:
    result: CaptureAttemptResult
    rgb: np.ndarray


def _preview_fixture(
    tmp_path: Path,
    *,
    content_frames: int = 40,
    slot_capacity_hint: int = 40,
) -> PreviewFixture:
    attempt = tmp_path / "preview-attempt"
    attempt.mkdir()
    output = attempt / "capture.bin"
    output.write_bytes(b"")
    preview_path = attempt / "capture-preview.bin"
    table_path = attempt / "capture-008e.bin"
    mapping_path = attempt / "capture-frame-map.json"
    rgb = _synthetic_index(content_frames=content_frames)
    preview = _encode_index(rgb)
    table = _transport_table(len(rgb))
    preview_path.write_bytes(preview)
    table_path.write_bytes(table)
    receipt = {
        "status": "preview-only-complete",
        "slot_capacity_hint": slot_capacity_hint,
        "slot_capacity_semantics": ("scanner-addressable preview slots; not an exposure count"),
        "preview_bytes": len(preview),
        "preview_sha256": _sha256(preview),
        "table_bytes": len(table),
        "table_sha256": _sha256(table),
        "frame_detection": "deferred-offline",
    }
    mapping_path.write_text(json.dumps(receipt), encoding="utf-8")
    journal_path = attempt / "journal.json"
    engine_sha256 = "b" * 64
    journal = {
        "status": "complete",
        "capture_mode": "preview-only",
        "requested_frame": None,
        "requested_boundary_offset_rows": 0,
        "expected_frame_count": None,
        "expected_reads": 0,
        "completed_reads": 0,
        "expected_bytes": 0,
        "completed_bytes": 0,
        "disk_bytes": 0,
        "unit_released": True,
        "output": str(output.resolve()),
        "output_sha256": _sha256(b""),
        "plan_sha256": CANONICAL_PLAN_SHA256,
        "capture_engine_sha256": engine_sha256,
        "scanner_identity": "Nikon LS-5000 ED 1.03",
        "preview_geometry_validated_before_reads": True,
        "preview_windows": [
            {
                "color_id": color,
                "resolution": [97, 97],
                "origin": [0, 0],
                "size": [3_946, 250_278],
                "bit_depth": 16,
            }
            for color in (1, 2, 3)
        ],
        "live_startup_0x8f": {"count": slot_capacity_hint},
        "live_index_artifacts": {
            "mapping": str(mapping_path.resolve()),
            "preview": str(preview_path.resolve()),
            "table": str(table_path.resolve()),
        },
        "live_index_evidence": {
            "status": "persisted-before-frame-detection",
            "preview_bytes": len(preview),
            "preview_sha256": _sha256(preview),
            "table_bytes": len(table),
            "table_sha256": _sha256(table),
        },
        "preview_only_receipt": receipt,
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    paths = AttemptPaths(
        directory=attempt,
        output=output,
        journal=journal_path,
        plan=attempt / "plan.jsonl",
        manifest=attempt / "manifest.json",
        stdout=attempt / "stdout.txt",
        stderr=attempt / "stderr.txt",
    )
    result = CaptureAttemptResult(
        outcome=CaptureOutcome.COMPLETE,
        request=CaptureRequest(mode=CaptureMode.PREVIEW),
        paths=paths,
        argv=("worker", "--preview-only"),
        returncode=0,
        stdout="",
        stderr="",
        journal=journal,
    )
    return PreviewFixture(result=result, rgb=rgb)


def test_complete_preview_builds_fixed_order_session_with_exact_transport_origins(
    tmp_path: Path,
) -> None:
    fixture = _preview_fixture(tmp_path)

    session = build_roll_preview_session(
        fixture.result,
        material=ScanMaterial.COLOR_NEGATIVE,
    )

    assert session.geometry == roll_index.IndexGeometry(
        requested_resolution=97,
        native_resolution=4_000,
        pitch=41,
        native_width=3_946,
        native_height=250_278,
        width=96,
        height=6_104,
        block_bytes=2_048,
        expected_stream_bytes=6_250_496,
    )
    assert [slot.slot_id for slot in session.slots] == list(range(1, 41))
    assert all(slot.boundary_offset_rows == 0 for slot in session.slots)
    assert all(slot.thumbnail.shape[1:] == (96, 3) for slot in session.slots)
    assert all(slot.thumbnail.dtype == np.uint16 for slot in session.slots)
    assert session.selected_slots == ()
    origin = session.resolve_origin(18, 0)
    assert origin.native_origin == 42 * origin.lookup_row
    assert origin is session.slots[17].base_origin


def test_preview_refuses_a_live_startup_table_that_is_not_40_slots(
    tmp_path: Path,
) -> None:
    fixture = _preview_fixture(tmp_path, slot_capacity_hint=6)

    with pytest.raises(
        RollSessionIntegrityError,
        match="40-slot|SA-30",
    ):
        build_roll_preview_session(fixture.result)


def test_reload_thumbnail_recrops_the_saved_full_width_preview_at_exact_offset(
    tmp_path: Path,
) -> None:
    fixture = _preview_fixture(tmp_path)
    session = build_roll_preview_session(fixture.result)
    slot = session.slots[17]

    reloaded = reload_thumbnail(session.preview, slot, -21)

    np.testing.assert_array_equal(
        reloaded,
        fixture.rgb[
            slot.start_boundary_row - 21 : slot.end_boundary_row - 21,
            :,
            :,
        ],
    )
    assert reloaded.shape[1:] == (96, 3)
    assert reloaded.flags.writeable is False


def test_material_change_uses_explicit_rgb4_routes_without_claiming_bw_ir(
    tmp_path: Path,
) -> None:
    color = build_roll_preview_session(_preview_fixture(tmp_path).result)

    bw = color.with_material(ScanMaterial.BLACK_AND_WHITE_NEGATIVE)

    assert color.recipe.capture_route is CaptureRoute.SINGLE_PASS_RGBI4
    assert color.recipe.capture_ir is True
    assert color.recipe.repair_with_ir_after_import is True
    assert bw.recipe.capture_route is CaptureRoute.SANE_RGB4
    assert bw.recipe.dpi == 4_000
    assert bw.recipe.bit_depth == 16
    assert bw.recipe.rgb_samples == 4
    assert bw.recipe.capture_ir is False
    assert bw.recipe.repair_with_ir_after_import is False


def test_selected_slots_are_an_immutable_ordered_operator_choice(tmp_path: Path) -> None:
    session = build_roll_preview_session(_preview_fixture(tmp_path).result)

    selected = session.with_selected_slots((3, 7, 18))

    assert session.selected_slots == ()
    assert selected.selected_slots == (3, 7, 18)


def test_boundary_offset_reloads_only_that_slot_and_resolves_exact_table_record(
    tmp_path: Path,
) -> None:
    session = build_roll_preview_session(_preview_fixture(tmp_path).result)
    base = session.slots[17]

    adjusted = session.with_boundary_offset(18, -21)

    assert session.slots[17].boundary_offset_rows == 0
    assert adjusted.slots[17].boundary_offset_rows == -21
    assert adjusted.slots[16] is session.slots[16]
    resolved = adjusted.resolve_origin(18, -21)
    assert resolved.lookup_row == base.base_origin.lookup_row - 21
    assert resolved.native_origin == 42 * resolved.lookup_row
    np.testing.assert_array_equal(
        adjusted.slots[17].thumbnail,
        reload_thumbnail(session.preview, base, -21),
    )


def test_session_json_round_trip_revalidates_sources_and_restores_operator_state(
    tmp_path: Path,
) -> None:
    session = (
        build_roll_preview_session(_preview_fixture(tmp_path).result)
        .with_material(ScanMaterial.BLACK_AND_WHITE_NEGATIVE)
        .with_selected_slots((3, 7, 18))
        .with_boundary_offset(18, -21)
    )

    restored = type(session).from_json(session.to_json())

    assert restored.material is ScanMaterial.BLACK_AND_WHITE_NEGATIVE
    assert restored.recipe.capture_route is CaptureRoute.SANE_RGB4
    assert restored.selected_slots == (3, 7, 18)
    assert restored.slots[17].boundary_offset_rows == -21
    assert restored.preview.preview_artifact == session.preview.preview_artifact
    np.testing.assert_array_equal(
        restored.slots[17].thumbnail,
        session.slots[17].thumbnail,
    )


def test_preview_journal_alias_is_rejected_before_decoding(tmp_path: Path) -> None:
    fixture = _preview_fixture(tmp_path)
    original = fixture.result.paths.journal
    durable = original.with_name("durable-journal.json")
    original.rename(durable)
    original.symlink_to(durable.name)
    aliased = replace(
        fixture.result,
        paths=replace(fixture.result.paths, journal=original),
    )

    with np.testing.assert_raises_regex(
        RollSessionIntegrityError,
        "journal.*alias|regular file",
    ):
        build_roll_preview_session(aliased)


def test_blank_tail_remains_selectable_fixed_slots_with_manual_review_warnings(
    tmp_path: Path,
) -> None:
    session = build_roll_preview_session(
        _preview_fixture(tmp_path, content_frames=36).result,
        expected_frame_count=36,
    )

    assert len(session.slots) == 40
    assert session.detection.expected_frame_count_matches is True
    for slot in session.slots[36:]:
        assert slot.manual_review is True
        assert {
            "ambiguous-content-tail-boundary",
            "beyond-advisory-content-end",
        }.intersection(slot.warnings)


def test_manual_origin_approval_is_bound_to_exact_reviewed_thumbnail_and_roll(
    tmp_path: Path,
) -> None:
    session = build_roll_preview_session(
        _preview_fixture(tmp_path, content_frames=36).result,
        expected_frame_count=36,
    )

    fingerprint = session.reviewed_fingerprint()
    approval = session.approve_manual_origin(37, 0)

    assert fingerprint.source_preview_sha256 == session.preview.preview_artifact.sha256
    assert fingerprint.source_table_sha256 == session.preview.table_artifact.sha256
    assert len(fingerprint.frame_visual_hashes) == len(session.slots) == 40
    assert approval.reviewed_fingerprint_sha256 == fingerprint.binding_sha256
    assert approval.slot == 37
    assert approval.boundary_offset_rows == 0
    assert approval.review_reasons
    assert session.validate_manual_approval(approval, slot_id=37, boundary_offset_rows=0)

    with pytest.raises(ValueError, match="does not require manual review"):
        session.approve_manual_origin(2, 0)


def test_session_pixel_arrays_cannot_be_made_writeable(tmp_path: Path) -> None:
    session = build_roll_preview_session(_preview_fixture(tmp_path).result)

    with pytest.raises(ValueError):
        session.preview.rgb.setflags(write=True)
    with pytest.raises(ValueError):
        session.slots[0].thumbnail.setflags(write=True)
