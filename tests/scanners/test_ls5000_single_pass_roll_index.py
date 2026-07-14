"""Regression contracts for whole-roll index decoding and dynamic frame origins."""

import struct
from pathlib import Path

import numpy as np
import pytest

from negpy.infrastructure.scanners.ls5000_single_pass import roll_index as roll


CAMPAIGN_WIRE = Path(__file__).resolve().parents[3] / "single-pass-wire"
GOLD36_PREVIEW = CAMPAIGN_WIRE / "rgbi4-gold36-frame18-meter2-preview.bin"
GOLD36_TABLE = CAMPAIGN_WIRE / "rgbi4-gold36-frame18-meter2-008e.bin"


def _encode_index(rgb16: np.ndarray) -> bytes:
    assert rgb16.shape[1:] == (96, 3)
    assert rgb16.shape[0] % 2 == 0
    blocks = np.zeros((rgb16.shape[0] // 2, roll.INDEX_BLOCK_WORDS), dtype=np.uint16)
    blocks[:, 0:96] = rgb16[0::2, :, 0]
    blocks[:, 96:192] = rgb16[0::2, :, 1]
    blocks[:, 192:288] = rgb16[0::2, :, 2]
    blocks[:, 512:608] = rgb16[1::2, :, 0]
    blocks[:, 608:704] = rgb16[1::2, :, 1]
    blocks[:, 704:800] = rgb16[1::2, :, 2]
    blocks[:, 800::2] = roll.INDEX_TRAILER_MARK
    blocks[:, 801::2] = np.arange(
        roll.INDEX_TRAILER_COUNTER0,
        roll.INDEX_TRAILER_COUNTER0 + roll.INDEX_TRAILER_WORDS // 2,
        dtype=np.uint16,
    )
    return blocks.astype(">u2", copy=False).tobytes()


def _live_extent(usable_rows: int) -> bytes:
    records = bytearray()
    for row in range(usable_rows):
        records.extend(struct.pack(">HH", 6 * (row % 18), row // 18))
    total = 8 + len(records)
    return b"\x00\x8e\x00\x00" + total.to_bytes(2, "big") + b"\x00\x00" + bytes(records)


def _synthetic_roll(
    frame_count: int,
    *,
    pitch: int = 143,
    leader: int = 24,
    tail: int = 24,
) -> tuple[np.ndarray, list[int]]:
    boundaries = [leader + index * pitch for index in range(frame_count + 1)]
    height = boundaries[-1] + tail
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
        aperture[max(0, boundary - 3) : min(height, boundary + 3)] = (
            clear_base + clear_noise[max(0, boundary - 3) : min(height, boundary + 3)]
        )
    rgb = np.empty((height, 96, 3), dtype=np.int64)
    rgb[:, 2:92] = aperture
    rgb[:, :2] = np.asarray((1_300, 1_000, 700))
    rgb[:, 92:] = np.asarray((1_100, 850, 600))
    return rgb.clip(0, 65_535).astype(np.uint16), boundaries


def _detect(
    rgb: np.ndarray,
    *,
    expected_frame_count: int | None = None,
) -> roll.RollDetection:
    return roll.detect_roll_frames(
        rgb,
        np.ones_like(rgb, dtype=bool),
        nominal_frame_rows=145,
        expected_frame_count=expected_frame_count,
    )


def test_complete_index_bytes_decode_every_rgb_row_and_validate_sync() -> None:
    rgb = np.arange(20 * 96 * 3, dtype=np.uint16).reshape(20, 96, 3)
    stream = _encode_index(rgb)
    geometry = roll.IndexGeometry(97, 4000, 41, 3946, 20, 96, 20, 2048, len(stream))

    decoded, known, report = roll.decode_full_index_bytes(stream, geometry)

    np.testing.assert_array_equal(decoded, rgb)
    assert known.all()
    assert report["valid_trailers"] == 10
    assert report["odd_housekeeping"] == "canonical-aa55-counter"
    with pytest.raises(roll.IncompleteIndexError, match="exactly"):
        roll.decode_full_index_bytes(stream[:-1], geometry)


def test_complete_index_accepts_the_stable_observed_odd_housekeeping_variant() -> None:
    rgb = np.arange(20 * 96 * 3, dtype=np.uint16).reshape(20, 96, 3)
    rows = np.frombuffer(_encode_index(rgb), dtype=">u2").copy().reshape(20, -1)
    variant = (
        np.arange(roll.INDEX_TRAILER_WORDS, dtype=np.uint16) * 6 + 8
    ).astype(np.uint16)
    rows[1::2, roll.INDEX_RGB_WORDS_PER_ROW :] = variant
    stream = rows.astype(">u2", copy=False).tobytes()
    geometry = roll.IndexGeometry(97, 4000, 41, 3946, 20, 96, 20, 2048, len(stream))

    decoded, known, report = roll.decode_full_index_bytes(stream, geometry)

    np.testing.assert_array_equal(decoded, rgb)
    assert known.all()
    assert report["odd_housekeeping"] == "stable-observed"


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (roll.IndexGeometry(97, 4000, 41, 3946, 20, 95, 20, 2048, 20 * 1024), "width must be 96"),
        (roll.IndexGeometry(97, 4000, 41, 3946, 20, 96.0, 20, 2048, 20 * 1024), "width must be 96"),
        (roll.IndexGeometry(97, 4000, 41, 3946, 20, 96, 20, 1024, 20 * 1024), "block size must be 2048"),
        (roll.IndexGeometry(97, 4000, 41, 3946, 19, 96, 19, 2048, 19 * 1024), "positive even row count"),
        (roll.IndexGeometry(97, 4000, 41, 3946, 20, 96, 20, 2048, 18 * 1024), "allocation mismatch"),
        (roll.IndexGeometry(97, 4000, 41, 3946, 20, 96, 20, 2048, float(20 * 1024)), "integer byte count"),
    ],
)
def test_exported_index_decoder_refuses_malformed_geometry(
    geometry: roll.IndexGeometry,
    message: str,
) -> None:
    valid_rgb = np.zeros((20, 96, 3), dtype=np.uint16)
    stream = _encode_index(valid_rgb)
    with pytest.raises(roll.IndexDecodeError, match=message):
        roll.decode_full_index_bytes(stream, geometry)


@pytest.mark.parametrize(
    ("leader", "tail", "candidate_slots", "content_ends"),
    [
        (17, 21, 37, (36,)),
        (300, 260, 40, (38, 39)),
        (83, 47, 37, (36,)),
    ],
)
def test_variable_leader_and_trailer_are_visible_without_guessing_roll_count(
    leader: int,
    tail: int,
    candidate_slots: int,
    content_ends: tuple[int, ...],
) -> None:
    rgb, boundaries = _synthetic_roll(36, leader=leader, tail=tail)

    detection = _detect(rgb)

    assert len(detection.intervals) == candidate_slots
    assert len(detection.boundaries) == candidate_slots + 1
    assert detection.content_end_candidates == content_ends
    assert detection.count_confidence == "user-selection-required"
    assert detection.candidate_slot_count == candidate_slots
    assert detection.confidence == "high"
    assert detection.pitch_rows == pytest.approx(143.0, abs=0.25)
    assert all(
        min(abs(item.output_row - expected) for item in detection.boundaries) <= 1
        for expected in boundaries
    )


def test_channel_gain_changes_do_not_move_detected_boundaries() -> None:
    rgb, _boundaries = _synthetic_roll(24, leader=41, tail=67)
    baseline = _detect(rgb)
    gained = np.clip(rgb.astype(np.float64) * np.asarray((0.70, 1.15, 1.30)), 0, 65_535).astype(np.uint16)
    adjusted = _detect(gained)
    assert [item.output_row for item in adjusted.boundaries] == [item.output_row for item in baseline.boundaries]


def test_one_true_interior_blank_cell_is_flagged_without_renumbering() -> None:
    rgb, boundaries = _synthetic_roll(24, leader=17, tail=21)
    baseline = _detect(rgb)
    blank_frame = 12
    rgb[boundaries[blank_frame - 1] : boundaries[blank_frame], 2:92] = np.asarray((34_200, 25_500, 17_800), dtype=np.uint16)

    detection = _detect(rgb)

    assert len(detection.intervals) == 25
    assert detection.frame_starts == baseline.frame_starts
    assert detection.bridged_cell_count == 0
    interval = detection.intervals[blank_frame - 1]
    assert not interval.count_supported
    assert not interval.count_bridged
    assert interval.manual_review
    assert "low-content-support" in interval.review_reasons


def test_blank_tail_cells_do_not_bridge_to_a_single_dense_artifact() -> None:
    pitch = 143
    rgb, boundaries = _synthetic_roll(
        36,
        pitch=pitch,
        leader=128,
        tail=3 * pitch + 21,
    )
    terminal = boundaries[-1]
    scene_cell = rgb[boundaries[4] : boundaries[5], 2:92].copy()
    rgb[terminal + 2 * pitch : terminal + 3 * pitch, 2:92] = scene_cell

    detection = _detect(rgb, expected_frame_count=36)

    assert len(detection.intervals) == 40
    assert detection.content_end_candidates == (39,)
    assert detection.bridged_cell_count == 0
    assert detection.confidence == "high"
    assert not detection.expected_frame_count_matches
    matching = _detect(rgb, expected_frame_count=39)
    assert matching.frame_starts == detection.frame_starts
    assert matching.expected_frame_count_matches


def test_missing_terminal_physical_gap_keeps_geometry_for_user_selection() -> None:
    rgb, boundaries = _synthetic_roll(24, leader=17, tail=21)
    terminal = boundaries[-1]
    observed_tail = len(rgb) - (terminal - 3)
    replacement = rgb[terminal - observed_tail - 6 : terminal - 6, 2:92].copy()
    rgb[terminal - 3 :, 2:92] = replacement
    detection = _detect(rgb)
    assert len(detection.intervals) == 25
    assert detection.content_end_candidates == ()
    assert detection.count_confidence == "user-selection-required"
    assert detection.intervals[-1].manual_review


def _erase_terminal_gap(rgb: np.ndarray, boundaries: list[int]) -> np.ndarray:
    altered = rgb.copy()
    terminal = boundaries[-1]
    observed_tail = len(altered) - (terminal - 3)
    replacement = altered[terminal - observed_tail - 6 : terminal - 6, 2:92].copy()
    altered[terminal - 3 :, 2:92] = replacement
    return altered


def test_expected_count_is_informational_when_terminal_is_missing() -> None:
    rgb, boundaries = _synthetic_roll(24, leader=17, tail=21)
    altered = _erase_terminal_gap(rgb, boundaries)

    detection = _detect(altered, expected_frame_count=24)

    assert len(detection.intervals) == 25
    assert detection.confidence == "high"
    assert detection.expected_frame_count == 24
    assert detection.count_confirmation == "candidate-slots-user-selection"
    assert detection.expected_frame_count_matches is False
    assert detection.boundaries[-1].manual_review
    assert 24 in detection.manual_review_frames
    assert detection.diagnostics()["count_confirmation"] == "candidate-slots-user-selection"
    for wrong_count in (23, 25):
        warned = _detect(altered, expected_frame_count=wrong_count)
        assert warned.frame_starts == detection.frame_starts
        assert warned.expected_frame_count_matches is False
        assert any("informational" in warning for warning in warned.warnings)


@pytest.mark.parametrize("frame_count,leader,tail", [(8, 91, 47), (36, 300, 21)])
def test_expected_count_never_changes_candidate_geometry(
    frame_count: int,
    leader: int,
    tail: int,
) -> None:
    rgb, boundaries = _synthetic_roll(frame_count, leader=leader, tail=tail)

    detection = _detect(
        _erase_terminal_gap(rgb, boundaries),
        expected_frame_count=frame_count,
    )

    baseline = _detect(_erase_terminal_gap(rgb, boundaries))
    assert detection.frame_starts == baseline.frame_starts
    assert detection.confidence == "high"
    assert detection.count_confirmation == "candidate-slots-user-selection"
    assert detection.expected_frame_count_matches is False


def test_missing_internal_boundary_is_flagged_without_renumbering() -> None:
    rgb, boundaries = _synthetic_roll(24, leader=17, tail=21)
    altered = _erase_terminal_gap(rgb, boundaries)
    internal = boundaries[8]
    altered[internal - 3 : internal + 3, 2:92] = altered[internal + 8 : internal + 14, 2:92]

    detection = _detect(altered, expected_frame_count=24)
    assert len(detection.intervals) == 25
    assert 8 in detection.manual_review_frames
    assert 9 in detection.manual_review_frames
    assert detection.expected_frame_count_matches is False


@pytest.mark.parametrize("expected", [True, 1, 41])
def test_expected_count_must_be_an_integer_in_supported_range(expected: int) -> None:
    rgb, _boundaries = _synthetic_roll(8, leader=91, tail=47)
    with pytest.raises(roll.IndexDecodeError, match="integer in 2..40"):
        _detect(rgb, expected_frame_count=expected)


def _boundary(
    index: int,
    row: int,
    *,
    support: str = "direct",
    evidence_run: tuple[int, int] | None = None,
) -> roll.GapBoundary:
    evidence_run = evidence_run or (row - 3, row + 4)
    reasons = () if support == "direct" else ("broad-clear-region",)
    return roll.GapBoundary(
        index=index,
        output_row=row,
        fitted_row=float(row),
        evidence=0.9,
        transmission=0.95,
        nonuniformity=0.08,
        support=support,
        evidence_run=evidence_run,
        manual_review=bool(reasons),
        review_reasons=reasons,
    )


def test_same_traversal_transport_table_maps_dynamic_frame_origins() -> None:
    records = roll.parse_live_transport_records_bytes(_live_extent(900), maximum_rows=900)
    rows = [20, 163, 306, 449, 592]
    boundaries = [_boundary(index, row) for index, row in enumerate(rows)]
    boundaries[2] = _boundary(2, rows[2], support="cadence-broad", evidence_run=(286, 326))

    mapping = roll.derive_transport_mapping(boundaries, len(rows), records)

    assert mapping.native_units_per_preview_row == pytest.approx(42.0)
    assert mapping.origins[1].lookup_row == rows[1] + 4
    assert mapping.origins[2].manual_review
    assert "transport-origin-inferred" in mapping.origins[2].review_reasons
    assert mapping.origins[2].native_origin == 42 * (rows[2] + 4)


def test_transport_envelope_and_anchor_residuals_fail_closed() -> None:
    with pytest.raises(roll.IndexDecodeError, match="0x8e"):
        roll.parse_live_transport_records_bytes(_live_extent(50)[:-1], maximum_rows=50)
    records = roll.parse_live_transport_records_bytes(_live_extent(1_200), maximum_rows=1_200)
    rows = [20, 163, 306, 449, 592, 735]
    boundaries = [_boundary(index, row) for index, row in enumerate(rows)]
    boundaries[3] = _boundary(3, rows[3], evidence_run=(rows[3] - 3, rows[3] + 24))
    with pytest.raises(roll.IndexDecodeError, match="transport anchor residual"):
        roll.derive_transport_mapping(boundaries, len(rows), records)


@pytest.mark.skipif(
    not GOLD36_PREVIEW.is_file() or not GOLD36_TABLE.is_file(),
    reason="persisted Gold 36 index capture is unavailable",
)
def test_persisted_gold36_expected_count_and_frame18_origin() -> None:
    geometry = roll.IndexGeometry(
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
    table, usable_rows = roll.validate_live_0x8e_bytes(GOLD36_TABLE.read_bytes(), geometry.height)
    rgb, known, _report = roll.decode_full_index_bytes(
        GOLD36_PREVIEW.read_bytes(),
        geometry,
        usable_rows=usable_rows,
    )

    detection = roll.detect_roll_frames(
        rgb,
        known,
        nominal_frame_rows=5_959 // geometry.pitch,
        expected_frame_count=36,
    )
    mapping = roll.derive_transport_mapping(
        detection.boundaries,
        len(detection.intervals),
        roll.parse_live_transport_records_bytes(
            table,
            maximum_rows=geometry.height,
        ),
    )

    assert len(detection.intervals) == 40
    assert len(detection.boundaries) == 41
    assert detection.confidence == "high"
    assert detection.count_confidence == "user-selection-required"
    assert detection.content_end_candidates == (36, 37, 38)
    assert detection.expected_frame_count_matches
    assert (
        detection.boundaries[0].output_row,
        detection.intervals[-1].start_row,
        detection.intervals[-1].end_row,
    ) == (128, 5_687, 5_750)
    assert mapping.origins[17].native_origin == 109_060
    assert mapping.origins[17].automatic
    mismatched = roll.detect_roll_frames(
        rgb,
        known,
        nominal_frame_rows=5_959 // geometry.pitch,
        expected_frame_count=24,
    )
    assert mismatched.frame_starts == detection.frame_starts
    assert not mismatched.expected_frame_count_matches
