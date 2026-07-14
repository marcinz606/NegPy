"""Pure LS-5000 whole-roll index decoding and frame-position logic.

This is the scanner-free subset of the proven roll-preview implementation.
It accepts complete persisted byte responses and NumPy rasters; USB transport,
pcap discovery, rendering, and scanner lifecycle remain outside this module.
"""

from __future__ import annotations

import math
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

INDEX_BLOCK_BYTES = 2_048
INDEX_BLOCK_WORDS = INDEX_BLOCK_BYTES // 2
INDEX_CHANNELS = 3
INDEX_ROWS_PER_BLOCK = 2
INDEX_ROW_WORDS = INDEX_BLOCK_WORDS // INDEX_ROWS_PER_BLOCK
INDEX_RGB_WORDS_PER_ROW = 96 * INDEX_CHANNELS
INDEX_TRAILER_WORDS = 224
INDEX_TRAILER_MARK = 0xAA55
INDEX_TRAILER_COUNTER0 = 0xAAE5


class IndexDecodeError(ValueError):
    """Input is not a self-consistent LS-5000 roll-index capture."""


class IncompleteIndexError(IndexDecodeError):
    """Frame detection was requested on a raster with missing rows."""


@dataclass(frozen=True)
class Window:
    sequence: int
    color_id: int
    resolution: int
    native_width: int
    native_height: int
    native_y: int
    bit_depth: int


@dataclass(frozen=True)
class ReadSegment:
    sequence: int
    stream_offset: int
    request_bytes: int
    captured: bytes
    captured_original_bytes: int


@dataclass(frozen=True)
class IndexGeometry:
    requested_resolution: int
    native_resolution: int
    pitch: int
    native_width: int
    native_height: int
    width: int
    height: int
    block_bytes: int
    expected_stream_bytes: int


@dataclass(frozen=True)
class OracleBoundary:
    frame: int
    native_origin: int
    selector_code: int
    edge_flags: int


@dataclass(frozen=True)
class GapBoundary:
    """One physical clear-film boundary in the decoded index raster."""

    index: int
    output_row: int
    fitted_row: float
    evidence: float
    transmission: float
    nonuniformity: float
    support: str
    evidence_run: tuple[int, int] | None
    manual_review: bool
    review_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FrameInterval:
    """One complete frame cell bounded by two clear-film gap centers."""

    frame: int
    start_row: int
    end_row: int
    height_rows: int
    start_boundary: int
    end_boundary: int
    content_fraction: float
    coverage_fraction: float
    count_supported: bool
    count_bridged: bool
    manual_review: bool
    review_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TransportRecord:
    """One row-indexed transport lookup record from a live READ(0x8e)."""

    row: int
    code: int
    selector: int
    native_origin: int


@dataclass(frozen=True)
class NativeFrameOrigin:
    """One preview boundary mapped into the scanner's native Y units."""

    frame: int
    boundary_index: int
    boundary_output_row: int
    lookup_row: int
    code: int
    selector: int
    native_origin: int
    method: str
    automatic: bool
    manual_review: bool
    review_reasons: tuple[str, ...]
    affine_residual_rows: float


@dataclass(frozen=True)
class TransportMapping:
    """Per-capture preview-to-native mapping derived from READ(0x8e)."""

    record_count: int
    native_intercept: float
    native_units_per_preview_row: float
    anchor_mae_rows: float
    anchor_max_error_rows: float
    origins: tuple[NativeFrameOrigin, ...]

    def diagnostics(self) -> dict:
        manual = [item.frame for item in self.origins if item.manual_review]
        return {
            "method": "same-capture READ(0x8e) row transport lookup",
            "formula": (
                "native_origin = 756*selector + 7*((code & 0xff) + 22*(code >> 8))"
            ),
            "record_count": self.record_count,
            "native_intercept": self.native_intercept,
            "native_units_per_preview_row": (self.native_units_per_preview_row),
            "anchor_mae_rows": self.anchor_mae_rows,
            "anchor_max_error_rows": self.anchor_max_error_rows,
            "frame_count": len(self.origins),
            "manual_review_frames": manual,
            "status": ("manual-review-required" if manual else "automatic"),
            "origins": [asdict(item) for item in self.origins],
        }


@dataclass(frozen=True)
class RollDetection:
    """Aligned candidate slots from one complete whole-roll index."""

    aperture_columns: tuple[int, int]
    nominal_frame_rows: int
    autocorrelation_lag: int
    autocorrelation_peak: float
    autocorrelation_best_non_neighbor: float
    pitch_rows: float
    phase_rows: float
    lattice_score: float
    alternative_lattice_score: float
    lattice_margin_fraction: float
    mean_boundary_evidence: float
    minimum_boundary_evidence: float
    content_level_threshold: float
    content_range_threshold: float
    candidate_cell_count: int
    bridged_cell_count: int
    expected_frame_count: int | None
    expected_frame_count_matches: bool | None
    count_confirmation: str
    count_confidence: str
    content_end_candidates: tuple[int, ...]
    confidence: str
    warnings: tuple[str, ...]
    boundaries: tuple[GapBoundary, ...]
    intervals: tuple[FrameInterval, ...]
    manual_review_frames: tuple[int, ...]

    @property
    def frame_starts(self) -> list[int]:
        return [interval.start_row for interval in self.intervals]

    @property
    def candidate_slot_count(self) -> int:
        return len(self.intervals)

    @property
    def alignment_confidence(self) -> str:
        """Compatibility-friendly explicit name for physical lattice confidence."""

        return self.confidence

    def diagnostics(self) -> dict:
        return {
            "method": (
                "physical clear-film transmission/uniformity comb with "
                "content-independent pitch fitting"
            ),
            "assumed_capacity": None,
            "frame_count": len(self.intervals),
            "candidate_slot_count": self.candidate_slot_count,
            "boundary_count": len(self.boundaries),
            "aperture_columns": list(self.aperture_columns),
            "nominal_frame_rows": self.nominal_frame_rows,
            "autocorrelation_lag": self.autocorrelation_lag,
            "autocorrelation_peak": self.autocorrelation_peak,
            "autocorrelation_best_non_neighbor": (
                self.autocorrelation_best_non_neighbor
            ),
            "pitch_rows": self.pitch_rows,
            "phase_rows": self.phase_rows,
            "lattice_score": self.lattice_score,
            "alternative_lattice_score": self.alternative_lattice_score,
            "lattice_margin_fraction": self.lattice_margin_fraction,
            "mean_boundary_evidence": self.mean_boundary_evidence,
            "minimum_boundary_evidence": self.minimum_boundary_evidence,
            "count_method": (
                "all physically aligned, scanner-addressable lattice slots; "
                "scene content only supplies advisory flags"
            ),
            "content_level_threshold": self.content_level_threshold,
            "content_range_threshold": self.content_range_threshold,
            "candidate_cell_count": self.candidate_cell_count,
            "bridged_cell_count": self.bridged_cell_count,
            "expected_frame_count": self.expected_frame_count,
            "expected_frame_count_matches": self.expected_frame_count_matches,
            "count_confirmation": self.count_confirmation,
            "count_confidence": self.count_confidence,
            "content_end_candidates": list(self.content_end_candidates),
            "confidence": self.confidence,
            "alignment_confidence": self.alignment_confidence,
            "warnings": list(self.warnings),
            "manual_review_frames": list(self.manual_review_frames),
            "origins_output_rows": self.frame_starts,
            "boundaries": [asdict(item) for item in self.boundaries],
            "intervals": [asdict(item) for item in self.intervals],
        }
def parse_oracle_boundaries(payload: bytes) -> list[OracleBoundary]:
    """Parse Nikon Scan's same-roll SEND(0x8f) oracle table.

    This payload was produced by Nikon Scan's own detection for this physical
    roll.  It is validation data, not a protocol constant or a reusable list.
    Only the count and first u32 origin are assigned semantic names; the two
    u16 fields are retained losslessly but remain undocumented.
    """

    if len(payload) < 4 or payload[:2] != b"\x01\x2a" or payload[3] != 0:
        raise IndexDecodeError("not a recognized Nikon SEND(0x8f) boundary table")
    count = payload[2]
    expected = 4 + count * 8
    if len(payload) != expected:
        raise IndexDecodeError(
            f"boundary header says {count} records ({expected} B), got {len(payload)} B"
        )
    result = []
    for index in range(count):
        origin, selector, flags = struct.unpack_from(">IHH", payload, 4 + index * 8)
        result.append(OracleBoundary(index + 1, origin, selector, flags))
    if any(a.native_origin >= b.native_origin for a, b in zip(result, result[1:])):
        raise IndexDecodeError("boundary origins are not strictly increasing")
    return result


def validate_index_block(block: np.ndarray) -> bool:
    """Return whether a complete block has the fixed 112-pair sync trailer."""

    if block.shape != (INDEX_BLOCK_WORDS,):
        return False
    trailer = block[800:]
    return bool(
        np.all(trailer[0::2] == INDEX_TRAILER_MARK)
        and np.array_equal(
            trailer[1::2],
            np.arange(
                INDEX_TRAILER_COUNTER0,
                INDEX_TRAILER_COUNTER0 + INDEX_TRAILER_WORDS // 2,
                dtype=np.uint16,
            ),
        )
    )


def _sync_housekeeping() -> np.ndarray:
    trailer = np.empty(INDEX_TRAILER_WORDS, dtype=np.uint16)
    trailer[0::2] = INDEX_TRAILER_MARK
    trailer[1::2] = np.arange(
        INDEX_TRAILER_COUNTER0,
        INDEX_TRAILER_COUNTER0 + INDEX_TRAILER_WORDS // 2,
        dtype=np.uint16,
    )
    return trailer


def validate_live_0x8e_bytes(data: bytes, maximum_rows: int) -> tuple[bytes, int]:
    """Validate one complete live READ(0x8e) response already in memory."""

    data = bytes(data)
    if len(data) < 12 or data[:4] != b"\x00\x8e\x00\x00":
        raise IndexDecodeError("live 0x8e response has an invalid envelope")
    declared = int.from_bytes(data[4:6], "big")
    if declared != len(data):
        raise IndexDecodeError(
            f"live 0x8e declared {declared:,} bytes but persisted {len(data):,}"
        )
    if (declared - 8) % 4:
        raise IndexDecodeError("live 0x8e length is not 8 + 4 bytes per row")
    usable_rows = (declared - 8) // 4
    if not 0 < usable_rows <= maximum_rows:
        raise IndexDecodeError(
            f"live 0x8e usable height {usable_rows:,} is outside 1..{maximum_rows:,}"
        )
    return data, usable_rows


def _read_live_0x8e(frame_table: Path | str, maximum_rows: int) -> tuple[bytes, int]:
    """Read and validate the common live READ(0x8e) envelope."""

    return validate_live_0x8e_bytes(Path(frame_table).read_bytes(), maximum_rows)


def parse_live_usable_rows(frame_table: Path | str, geometry: IndexGeometry) -> int:
    """Derive the usable index height from a complete live READ(0x8e).

    Two independent captures show an eight-byte envelope followed by one
    four-byte record per usable 97 dpi row.  The declared response length and
    the persisted byte count must agree before this value can constrain image
    decoding.
    """

    _data, usable_rows = _read_live_0x8e(frame_table, geometry.height)
    return usable_rows


def transport_native_origin(code: int, selector: int) -> int:
    """Decode Nikon's exact 0x8e/0x8f transport-coordinate identity.

    The identity reproduces all 37 origins in the canonical Nikon SEND(0x8f)
    table and independent frame 3/7/20 fine SET_WINDOW captures.  ``code`` is
    deliberately retained as an opaque two-byte field; only this lossless
    arithmetic interpretation is assigned here.
    """

    if not 0 <= code <= 0xFFFF or not 0 <= selector <= 0xFFFF:
        raise IndexDecodeError("transport code and selector must be u16 values")
    subposition = (code & 0xFF) + 22 * (code >> 8)
    return 756 * selector + 7 * subposition


def parse_live_transport_records(
    frame_table: Path | str, *, maximum_rows: int
) -> tuple[TransportRecord, ...]:
    """Parse one row-indexed transport record for every usable preview row."""

    return parse_live_transport_records_bytes(
        Path(frame_table).read_bytes(), maximum_rows=maximum_rows
    )


def parse_live_transport_records_bytes(
    data: bytes, *, maximum_rows: int
) -> tuple[TransportRecord, ...]:
    """Parse live row transport records without a filesystem round trip."""

    data, usable_rows = validate_live_0x8e_bytes(data, maximum_rows)
    records = []
    for row, (code, selector) in enumerate(struct.iter_unpack(">HH", data[8:])):
        records.append(
            TransportRecord(
                row=row,
                code=code,
                selector=selector,
                native_origin=transport_native_origin(code, selector),
            )
        )
    if len(records) != usable_rows:
        raise IndexDecodeError(
            f"live 0x8e yielded {len(records):,} transport records for "
            f"{usable_rows:,} usable rows"
        )
    return tuple(records)


def _validate_full_index_rows(rows: np.ndarray, usable_rows: int) -> dict:
    """Validate stable parity framing plus a strictly repeated EOF suffix.

    The scanner has emitted two valid odd-row housekeeping records: the
    Nikon-trace ``AA55,counter`` form and a coordinate-like form observed on
    a consecutive live preview.  Their semantic payload is opaque; block
    alignment is established by requiring each parity's complete 224-word
    record to repeat exactly across every usable row.
    """

    if rows.ndim != 2 or rows.shape[1] != INDEX_ROW_WORDS:
        raise IndexDecodeError("complete index does not contain 1,024-byte rows")
    if not 0 < usable_rows <= len(rows):
        raise IndexDecodeError(
            f"usable row count {usable_rows:,} is outside 1..{len(rows):,}"
        )

    prefix = rows[:usable_rows]
    canonical_sync = _sync_housekeeping()
    parity_housekeeping = {
        parity: rows[parity, INDEX_RGB_WORDS_PER_ROW:].copy()
        for parity in range(min(2, len(rows)))
    }
    for parity, expected in parity_housekeeping.items():
        actual = prefix[parity::2, INDEX_RGB_WORDS_PER_ROW:]
        if len(actual) == 0:
            continue
        matches = np.all(actual == expected, axis=1)
        if not bool(matches.all()):
            local = int(np.flatnonzero(~matches)[0])
            row = parity + 2 * local
            raise IndexDecodeError(f"index row framing mismatch at usable row {row}")

    padding = rows[usable_rows:]
    padding_record_type = None
    if len(padding):
        expected_housekeeping = parity_housekeeping[usable_rows % 2]
        first = padding[0]
        if bool(np.any(first[:INDEX_RGB_WORDS_PER_ROW])) or not np.array_equal(
            first[INDEX_RGB_WORDS_PER_ROW:], expected_housekeeping
        ):
            raise IndexDecodeError(
                "terminal padding does not begin with the expected blank row record"
            )
        if not bool(np.all(padding == first)):
            raise IndexDecodeError(
                "terminal padding is not one byte-identical blank-row suffix"
            )
        padding_record_type = (
            "even-housekeeping" if usable_rows % 2 == 0 else "sync-trailer"
        )

    return {
        "allocated_rows": int(len(rows)),
        "usable_rows": int(usable_rows),
        "padding_rows": int(len(rows) - usable_rows),
        "allocated_blocks": int(len(rows) // INDEX_ROWS_PER_BLOCK),
        "usable_blocks": int(math.ceil(usable_rows / INDEX_ROWS_PER_BLOCK)),
        "padding_record_type": padding_record_type,
        "odd_housekeeping": (
            "canonical-aa55-counter"
            if np.array_equal(parity_housekeeping[1], canonical_sync)
            else "stable-observed"
        ),
    }


def _validate_exported_geometry(geometry: IndexGeometry) -> None:
    """Refuse geometries that cannot describe the observed RGB96 block layout."""

    if not isinstance(geometry, IndexGeometry):
        raise TypeError("geometry must be an IndexGeometry")
    if (
        isinstance(geometry.width, bool)
        or not isinstance(geometry.width, (int, np.integer))
        or geometry.width != INDEX_RGB_WORDS_PER_ROW // INDEX_CHANNELS
    ):
        raise IndexDecodeError(f"index geometry width must be 96, got {geometry.width}")
    if (
        isinstance(geometry.block_bytes, bool)
        or not isinstance(geometry.block_bytes, (int, np.integer))
        or geometry.block_bytes != INDEX_BLOCK_BYTES
    ):
        raise IndexDecodeError(
            f"index geometry block size must be {INDEX_BLOCK_BYTES}, "
            f"got {geometry.block_bytes}"
        )
    if (
        isinstance(geometry.height, bool)
        or not isinstance(geometry.height, (int, np.integer))
        or geometry.height <= 0
        or geometry.height % INDEX_ROWS_PER_BLOCK
    ):
        raise IndexDecodeError(
            "index geometry allocated height must be a positive even row count"
        )
    if isinstance(geometry.expected_stream_bytes, bool) or not isinstance(
        geometry.expected_stream_bytes, (int, np.integer)
    ):
        raise IndexDecodeError(
            "index geometry allocation must be an integer byte count"
        )
    expected_stream_bytes = geometry.height * INDEX_ROW_WORDS * 2
    if geometry.expected_stream_bytes != expected_stream_bytes:
        raise IndexDecodeError(
            "index geometry allocation mismatch: "
            f"{geometry.height} rows require {expected_stream_bytes:,} bytes, "
            f"not {geometry.expected_stream_bytes:,}"
        )


def decode_full_index_bytes(
    data: bytes,
    geometry: IndexGeometry,
    *,
    usable_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode one complete index-only stream already held in memory."""

    _validate_exported_geometry(geometry)
    data = bytes(data)
    if len(data) != geometry.expected_stream_bytes:
        raise IncompleteIndexError(
            f"complete index must be exactly {geometry.expected_stream_bytes:,} B; "
            f"got {len(data):,} B"
        )
    words = np.frombuffer(data, dtype=">u2")
    rows = words.reshape(geometry.height, INDEX_ROW_WORDS)
    if usable_rows is None:
        usable_rows = geometry.height
    report = _validate_full_index_rows(rows, usable_rows)

    pixels = rows[:usable_rows, :INDEX_RGB_WORDS_PER_ROW]
    rgb16 = np.stack(
        [
            pixels[:, channel * geometry.width : (channel + 1) * geometry.width]
            for channel in range(INDEX_CHANNELS)
        ],
        axis=2,
    ).copy()
    known = np.ones(rgb16.shape, dtype=bool)
    report.update(
        {
            "decoded_rows": int(usable_rows),
            "complete_blocks": int(len(rows) // INDEX_ROWS_PER_BLOCK),
            "valid_trailers": int(usable_rows // INDEX_ROWS_PER_BLOCK),
            "invalid_trailers": 0,
        }
    )
    return rgb16, known, report


def decode_full_index_stream(
    stream: Path | str,
    geometry: IndexGeometry,
    *,
    usable_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode one complete index-only stream persisted by a live capture."""

    return decode_full_index_bytes(
        Path(stream).read_bytes(), geometry, usable_rows=usable_rows
    )
def _largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    runs = _true_runs(mask)
    return max(runs, key=lambda item: item[1] - item[0], default=None)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs from a one-dimensional boolean mask."""

    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _active_aperture(rgb16: np.ndarray) -> tuple[int, int]:
    """Find the illuminated film aperture while excluding fixed dark rails."""

    if rgb16.ndim != 3 or rgb16.shape[2] != 3 or rgb16.shape[1] < 8:
        raise IndexDecodeError("roll detector requires an HxWx3 RGB raster")
    levels = np.median(rgb16.astype(np.float32).mean(axis=2), axis=0)
    middle = levels[len(levels) // 4 : 3 * len(levels) // 4]
    reference = float(np.median(middle))
    if reference <= 0:
        raise IndexDecodeError("roll detector found no illuminated film aperture")
    run = _largest_true_run(levels > 0.25 * reference)
    if run is None or run[1] - run[0] < rgb16.shape[1] // 2:
        raise IndexDecodeError("roll detector could not isolate the film aperture")
    return run


def _physical_gap_evidence(
    rgb16: np.ndarray, aperture: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure clear-base coverage and spatial uniformity for every row."""

    lo, hi = aperture
    pixels = rgb16[:, lo:hi].astype(np.float32)
    row_median = np.median(pixels, axis=1)
    row_q10 = np.percentile(pixels, 10, axis=1)
    row_q90 = np.percentile(pixels, 90, axis=1)

    # A few clipped dust pixels must not discard a row.  The all-channel
    # saturated EOF record, however, must never become a film boundary.
    safe = (pixels >= 65_000).mean(axis=(1, 2)) < 0.5
    if int(safe.sum()) < 3:
        raise IndexDecodeError("roll index has too few unsaturated rows for CV")
    clear_base = np.percentile(row_median[safe], 99.0, axis=0)
    if bool(np.any(clear_base <= 0)):
        raise IndexDecodeError("roll index clear-film reference is degenerate")

    transmission = np.min(row_q10 / clear_base, axis=1)
    nonuniformity = np.max((row_q90 - row_q10) / np.maximum(row_median, 1.0), axis=1)
    evidence = (
        np.clip((transmission - 0.72) / 0.20, 0.0, 1.0)
        * np.clip((0.28 - nonuniformity) / 0.18, 0.0, 1.0)
        * safe
    )
    smoothed = np.convolve(evidence, np.ones(5) / 5.0, mode="same")
    direct = (transmission >= 0.87) & (nonuniformity <= 0.18) & safe
    return smoothed, transmission, nonuniformity, direct


def _lattice_positions(height: int, pitch: float, phase: float) -> np.ndarray:
    positions = np.floor(np.arange(phase, height, pitch) + 0.5 + 1e-9).astype(np.int64)
    return positions[(positions >= 2) & (positions < height - 2)]


def _local_lattice_evidence(
    evidence: np.ndarray, positions: np.ndarray, *, radius: int = 3
) -> np.ndarray:
    return np.asarray(
        [
            evidence[max(0, row - radius) : min(len(evidence), row + radius + 1)].max()
            for row in positions
        ],
        dtype=np.float64,
    )


def _lattice_objective(evidence: np.ndarray, positions: np.ndarray) -> float:
    if len(positions) < 3:
        return 0.0
    return float(_local_lattice_evidence(evidence, positions).mean())


def _extended_lattice(
    height: int, pitch: float, phase: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical and rounded boundaries beyond both raster edges."""

    first = math.floor((0.0 - phase) / pitch) - 1
    last = math.ceil((height - phase) / pitch) + 1
    indices = np.arange(first, last + 1, dtype=np.int64)
    theoretical = phase + indices.astype(np.float64) * pitch
    rounded = np.floor(theoretical + 0.5 + 1e-9).astype(np.int64)
    unique = np.concatenate(([True], np.diff(rounded) != 0))
    return theoretical[unique], rounded[unique]


def _content_support(
    rgb16: np.ndarray,
    aperture: tuple[int, int],
    boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Measure scene content and coverage for every candidate lattice cell."""

    lo, hi = aperture
    gray = rgb16[:, lo:hi].astype(np.float32).mean(axis=2)
    row_level = np.median(gray, axis=1)
    row_range = np.percentile(gray, 90, axis=1) - np.percentile(gray, 10, axis=1)
    level_threshold = float(np.percentile(row_level, 80))
    range_threshold = float(np.percentile(row_range, 20))
    # A fully unilluminated aperture row is scanner padding/opaque leader, not
    # scene content.  Without this guard, a long run of zero rows satisfies
    # the low-level branch below and can masquerade as one or more frames.
    has_signal = np.any(rgb16[:, lo:hi] != 0, axis=(1, 2))
    content_rows = has_signal & (
        (row_level < level_threshold) | (row_range > range_threshold)
    )

    coverage = []
    content = []
    supported = []
    for start, end in zip(boundaries, boundaries[1:]):
        clipped_start = max(0, int(start))
        clipped_end = min(len(rgb16), int(end))
        cell_rows = max(1, int(end - start))
        observed_rows = max(0, clipped_end - clipped_start)
        cell_coverage = observed_rows / cell_rows
        content_fraction = (
            float(content_rows[clipped_start:clipped_end].mean())
            if observed_rows
            else 0.0
        )
        coverage.append(cell_coverage)
        content.append(content_fraction)
        supported.append(cell_coverage >= 0.50 and content_fraction >= 0.25)
    return (
        np.asarray(coverage, dtype=np.float64),
        np.asarray(content, dtype=np.float64),
        np.asarray(supported, dtype=bool),
        level_threshold,
        range_threshold,
    )


def _nearest_evidence_run(
    row: int, runs: Sequence[tuple[int, int]], *, max_distance: int = 3
) -> tuple[tuple[int, int] | None, int]:
    nearest = None
    nearest_distance = sys.maxsize
    for start, end in runs:
        if start <= row < end:
            return (start, end), 0
        distance = min(abs(row - start), abs(row - (end - 1)))
        if distance < nearest_distance:
            nearest = (start, end)
            nearest_distance = distance
    if nearest_distance > max_distance:
        return None, nearest_distance
    return nearest, nearest_distance


def detect_roll_frames(
    rgb16: np.ndarray,
    known: np.ndarray,
    *,
    nominal_frame_rows: int,
    expected_frame_count: int | None = None,
) -> RollDetection:
    """Detect arbitrary roll frames from physical clear-film gap evidence.

    The result contains N+1 lattice boundaries and N candidate slots.  The
    in-raster lattice and the 40-slot feeder protocol limit determine geometry;
    scene content and an expected roll count are advisory metadata only.
    """

    if expected_frame_count is not None and (
        isinstance(expected_frame_count, bool) or not 2 <= expected_frame_count <= 40
    ):
        raise IndexDecodeError("expected frame count must be an integer in 2..40")
    if rgb16.shape != known.shape:
        raise IndexDecodeError("RGB raster and completeness mask shapes differ")
    complete_rows = known.all(axis=(1, 2))
    if complete_rows.mean() < 0.995:
        raise IncompleteIndexError(
            f"CV boundary detection requires a persisted complete index; "
            f"row coverage is {complete_rows.mean():.1%}"
        )
    if nominal_frame_rows < 16 or len(rgb16) < 3 * nominal_frame_rows:
        raise IndexDecodeError("roll index is too short for pitch detection")

    aperture = _active_aperture(rgb16)
    evidence, transmission, nonuniformity, direct = _physical_gap_evidence(
        rgb16, aperture
    )
    evidence_runs = _true_runs(direct)
    narrow_runs = [
        (start, end) for start, end in evidence_runs if 3 <= end - start <= 12
    ]
    if len(narrow_runs) < 3:
        raise IndexDecodeError("roll detector found too few distinct physical gaps")
    anchor_signal = np.zeros_like(evidence)
    anchor_centers = []
    for start, end in narrow_runs:
        anchor_signal[start:end] = evidence[start:end]
        anchor_centers.append((start + end - 1) / 2.0)

    centered = anchor_signal - float(anchor_signal.mean())
    lag_min = max(3, int(math.floor(nominal_frame_rows * 0.90)))
    lag_max = min(len(evidence) - 3, int(math.ceil(nominal_frame_rows * 1.10)))
    correlations: list[tuple[float, int]] = []
    for lag in range(lag_min, lag_max + 1):
        value = float(np.dot(centered[:-lag], centered[lag:]) / (len(centered) - lag))
        correlations.append((value, lag))
    autocorrelation_peak, peak_lag = max(correlations)
    if autocorrelation_peak <= 0:
        raise IndexDecodeError("roll index has no physical gap periodicity")
    non_neighbors = [value for value, lag in correlations if abs(lag - peak_lag) >= 4]
    best_non_neighbor = max(non_neighbors, default=0.0)

    candidates: list[tuple[float, float, float]] = []
    for pitch in np.arange(peak_lag - 1.5, peak_lag + 1.501, 0.05):
        for phase in np.arange(0.0, pitch, 0.25):
            positions = _lattice_positions(
                len(anchor_signal), float(pitch), float(phase)
            )
            if len(positions) < 3:
                continue
            candidates.append(
                (
                    _lattice_objective(anchor_signal, positions),
                    float(pitch),
                    float(phase),
                )
            )
    if not candidates:
        raise IndexDecodeError("roll index produced no physical-gap lattice")
    _coarse_score, coarse_pitch, coarse_phase = max(candidates)

    # Refine the coarse comb against narrow physical-gap centers.  Anchors
    # more than eight rows from the comb are scene artifacts; a second robust
    # fit removes assignments whose residual still exceeds three rows.
    assignments: dict[int, tuple[float, float]] = {}
    for center in anchor_centers:
        lattice_index = int(np.rint((center - coarse_phase) / coarse_pitch))
        residual = abs(center - (coarse_phase + lattice_index * coarse_pitch))
        if residual > 8.0:
            continue
        previous = assignments.get(lattice_index)
        if previous is None or residual < previous[0]:
            assignments[lattice_index] = (residual, center)
    if len(assignments) < 3:
        raise IndexDecodeError(
            "roll detector could not anchor the physical-gap lattice"
        )
    anchor_indices = np.asarray(sorted(assignments), dtype=np.float64)
    anchored_centers = np.asarray(
        [assignments[int(index)][1] for index in anchor_indices],
        dtype=np.float64,
    )
    design = np.column_stack((np.ones(len(anchor_indices)), anchor_indices))
    intercept, pitch = np.linalg.lstsq(design, anchored_centers, rcond=None)[0]
    residuals = design @ np.asarray((intercept, pitch)) - anchored_centers
    keep = np.abs(residuals) <= 3.0
    if int(keep.sum()) < 3:
        raise IndexDecodeError("roll detector physical-gap anchors do not agree")
    intercept, pitch = np.linalg.lstsq(
        design[keep], anchored_centers[keep], rcond=None
    )[0]
    if not 0.85 * nominal_frame_rows <= pitch <= 1.15 * nominal_frame_rows:
        raise IndexDecodeError("roll detector refined pitch is outside film geometry")
    phase = float(intercept % pitch)

    theoretical, all_positions = _extended_lattice(len(evidence), float(pitch), phase)
    (
        cell_coverage,
        cell_content,
        cell_supported,
        content_level_threshold,
        content_range_threshold,
    ) = _content_support(rgb16, aperture, all_positions)
    content_runs = _true_runs(cell_supported)
    if not content_runs:
        raise IndexDecodeError("roll detector found no content-supported frame cells")

    def make_boundary(global_index: int, output_index: int) -> GapBoundary:
        fitted_float = float(theoretical[global_index])
        fitted = int(all_positions[global_index])
        if not 0 <= fitted < len(evidence):
            output_row = min(max(fitted, 0), len(evidence))
            return GapBoundary(
                index=output_index,
                output_row=output_row,
                fitted_row=fitted_float,
                evidence=0.0,
                transmission=0.0,
                nonuniformity=1.0,
                support="outside-raster",
                evidence_run=None,
                manual_review=True,
                review_reasons=("outside-index-raster",),
            )

        run, _distance = _nearest_evidence_run(fitted, evidence_runs)
        reasons: list[str] = []
        support = "cadence-weak"
        output_row = fitted
        if run is not None:
            width = run[1] - run[0]
            if 3 <= width <= 12:
                support = "direct"
                output_row = int(math.floor((run[0] + run[1] - 1) / 2.0 + 0.5))
            elif width > 12:
                support = "cadence-broad"
                reasons.append("broad-clear-region")
            else:
                reasons.append("narrow-gap-evidence")
        else:
            reasons.append("no-local-gap-run")
        if float(evidence[output_row]) < 0.45:
            reasons.append("low-gap-evidence")
        return GapBoundary(
            index=output_index,
            output_row=output_row,
            fitted_row=fitted_float,
            evidence=float(evidence[output_row]),
            transmission=float(transmission[output_row]),
            nonuniformity=float(nonuniformity[output_row]),
            support=support,
            evidence_run=run,
            manual_review=bool(reasons),
            review_reasons=tuple(reasons),
        )

    def boundary_is_supported(boundary: GapBoundary) -> bool:
        return bool(
            boundary.support in {"direct", "cadence-broad"}
            and boundary.evidence >= 0.40
            and boundary.transmission >= 0.82
            and boundary.nonuniformity <= 0.22
        )

    # Candidate numbering comes from the fitted physical lattice, never scene
    # brightness or local boundary visibility.  Start at its first in-raster
    # position and expose every addressable start through the captured 0x8e
    # extent, capped by the feeder's 40-slot protocol limit.  A weak/blank
    # leading slot remains slot 1 with warnings instead of silently renumbering
    # every later frame.  The final slot may likewise be partial.
    lattice_starts = [
        index
        for index, row in enumerate(all_positions[:-1])
        if 2 <= row < len(evidence)
    ]
    if not lattice_starts:
        raise IndexDecodeError("roll detector found no in-raster lattice origin")
    cell_start = lattice_starts[0]
    in_raster_starts = np.flatnonzero(
        (all_positions[:-1] >= all_positions[cell_start])
        & (all_positions[:-1] < len(evidence))
    )
    candidate_slot_count = min(40, int(len(in_raster_starts)))
    if candidate_slot_count < 2:
        raise IndexDecodeError("roll detector found fewer than two candidate slots")
    cell_end = cell_start + candidate_slot_count
    boundaries = [
        make_boundary(global_index, output_index)
        for output_index, global_index in enumerate(range(cell_start, cell_end + 1))
    ]

    # Content can suggest where photographed frames stop, but it can never
    # define slot geometry.  A content island is eligible only when its end has
    # physical terminal support.  This excludes opaque/padded tail artifacts
    # regardless of how many lattice cells they occupy.
    terminal_runs: list[tuple[tuple[int, int], GapBoundary]] = []
    for run in content_runs:
        terminal_index = run[1]
        if not cell_start < terminal_index <= cell_end:
            continue
        terminal = make_boundary(terminal_index, terminal_index - cell_start)
        if boundary_is_supported(terminal):
            terminal_runs.append((run, terminal))

    content_end_candidates: tuple[int, ...] = ()
    if terminal_runs:
        _terminal_content_run, terminal_boundary = max(
            terminal_runs,
            key=lambda item: item[0][1],
        )
        terminal_index = _terminal_content_run[1]
        terminal_evidence_run = terminal_boundary.evidence_run
        candidate_ends = [terminal_index]
        if (
            terminal_evidence_run is not None
            and terminal_evidence_run[1] - terminal_evidence_run[0] > 12
        ):
            candidate_ends = [
                index
                for index in range(terminal_index, cell_end + 1)
                if terminal_evidence_run[0]
                <= int(all_positions[index])
                < terminal_evidence_run[1]
            ]
        content_end_candidates = tuple(
            index - cell_start
            for index in candidate_ends
            if 1 <= index - cell_start <= candidate_slot_count
        )
        alignment_end = terminal_index
    else:
        alignment_end = min(
            cell_end,
            max(content_runs, key=lambda item: (item[1] - item[0], item[1]))[1],
        )
    alignment_end = max(cell_start + 2, alignment_end)
    fitted_theoretical = theoretical[cell_start : alignment_end + 1]
    fitted_positions = all_positions[cell_start : alignment_end + 1]
    fitted_in_raster = fitted_positions[
        (fitted_positions >= 0) & (fitted_positions < len(evidence))
    ]
    if len(fitted_in_raster) < 3:
        raise IndexDecodeError("roll detector has too few aligned in-raster boundaries")

    # Score physical alignment independently from how many candidate slots the
    # operator ultimately keeps.  An alternative must differ by at least four
    # RMS rows, outside the local gap-refinement radius.
    lattice_score = float(evidence[fitted_in_raster].mean())
    alternative_score = 0.0
    for candidate_pitch in np.arange(pitch - 1.5, pitch + 1.501, 0.05):
        for offset in np.arange(-10.0, 10.001, 0.25):
            candidate = np.floor(
                fitted_theoretical[0]
                + offset
                + np.arange(len(fitted_theoretical)) * candidate_pitch
                + 0.5
            ).astype(np.int64)
            if candidate[0] < 0 or candidate[-1] >= len(evidence):
                continue
            rms = float(np.sqrt(np.mean(np.square(candidate - fitted_positions))))
            if rms >= 4.0:
                alternative_score = max(
                    alternative_score, float(evidence[candidate].mean())
                )
    lattice_margin = (
        (lattice_score - alternative_score) / lattice_score
        if lattice_score > 0
        else 0.0
    )

    if len(boundaries) < 3:
        raise IndexDecodeError("roll detector found fewer than two frame intervals")
    alignment_boundary_count = alignment_end - cell_start + 1
    alignment_boundaries = boundaries[:alignment_boundary_count]
    if sum(item.support == "direct" for item in alignment_boundaries) < 3:
        raise IndexDecodeError(
            "roll detector found insufficient distinct physical gaps"
        )

    intervals: list[FrameInterval] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        reasons = []
        reasons.extend(f"start-{item}" for item in start.review_reasons)
        reasons.extend(f"end-{item}" for item in end.review_reasons)
        candidate_cell = cell_start + index
        if not cell_supported[candidate_cell]:
            reasons.append("low-content-support")
        if cell_coverage[candidate_cell] < 0.999:
            reasons.append("partial-index-coverage")
        height = end.output_row - start.output_row
        if not 0.85 * pitch <= height <= 1.15 * pitch:
            reasons.append("spacing-outlier")
        intervals.append(
            FrameInterval(
                frame=index + 1,
                start_row=start.output_row,
                end_row=end.output_row,
                height_rows=height,
                start_boundary=start.index,
                end_boundary=end.index,
                content_fraction=float(cell_content[candidate_cell]),
                coverage_fraction=float(cell_coverage[candidate_cell]),
                count_supported=bool(cell_supported[candidate_cell]),
                count_bridged=False,
                manual_review=bool(reasons),
                review_reasons=tuple(reasons),
            )
        )

    supported_alignment_boundaries = [
        item for item in alignment_boundaries if boundary_is_supported(item)
    ]
    boundary_evidence = np.asarray(
        [item.evidence for item in supported_alignment_boundaries], dtype=np.float64
    )
    direct_fraction = sum(
        item.support == "direct" for item in alignment_boundaries[:-1]
    ) / max(1, len(alignment_boundaries) - 1)
    if (
        lattice_score >= 0.65
        and lattice_margin >= 0.08
        and direct_fraction >= 0.80
        and float(boundary_evidence.min()) >= 0.40
    ):
        confidence = "high"
    elif lattice_score >= 0.45 and direct_fraction >= 0.60:
        confidence = "medium"
    else:
        confidence = "low"

    expected_frame_count_matches: bool | None = None
    warnings: list[str] = []
    if expected_frame_count is not None:
        expected_frame_count_matches = expected_frame_count in content_end_candidates
        if expected_frame_count_matches:
            warnings.append(
                f"expected frame count {expected_frame_count} is informational; "
                f"it did not alter the {candidate_slot_count}-slot geometry"
            )
        else:
            advisory = (
                ", ".join(str(item) for item in content_end_candidates)
                if content_end_candidates
                else "none"
            )
            warnings.append(
                f"expected frame count {expected_frame_count} is informational and "
                f"does not match advisory content ends [{advisory}]; geometry "
                "was not altered"
            )
    manual_review_frames = tuple(item.frame for item in intervals if item.manual_review)
    return RollDetection(
        aperture_columns=aperture,
        nominal_frame_rows=nominal_frame_rows,
        autocorrelation_lag=peak_lag,
        autocorrelation_peak=autocorrelation_peak,
        autocorrelation_best_non_neighbor=best_non_neighbor,
        pitch_rows=pitch,
        phase_rows=phase,
        lattice_score=lattice_score,
        alternative_lattice_score=alternative_score,
        lattice_margin_fraction=lattice_margin,
        mean_boundary_evidence=float(boundary_evidence.mean()),
        minimum_boundary_evidence=float(boundary_evidence.min()),
        content_level_threshold=content_level_threshold,
        content_range_threshold=content_range_threshold,
        candidate_cell_count=len(cell_supported),
        bridged_cell_count=0,
        expected_frame_count=expected_frame_count,
        expected_frame_count_matches=expected_frame_count_matches,
        count_confirmation="candidate-slots-user-selection",
        count_confidence="user-selection-required",
        content_end_candidates=content_end_candidates,
        confidence=confidence,
        warnings=tuple(warnings),
        boundaries=tuple(boundaries),
        intervals=tuple(intervals),
        manual_review_frames=manual_review_frames,
    )


def detect_live_boundaries(
    rgb16: np.ndarray,
    known: np.ndarray,
    *,
    nominal_frame_rows: int,
) -> list[int]:
    """Compatibility wrapper returning the N frame starts, not N+1 gaps."""

    return detect_roll_frames(
        rgb16, known, nominal_frame_rows=nominal_frame_rows
    ).frame_starts


def derive_transport_mapping(
    boundaries: Sequence[GapBoundary],
    frame_count: int,
    records: Sequence[TransportRecord],
    *,
    minimum_scale: float = 40.0,
    maximum_scale: float = 45.0,
    maximum_anchor_mae_rows: float = 1.0,
    maximum_anchor_error_rows: float = 2.0,
) -> TransportMapping:
    """Map physical preview gaps through the same capture's 0x8e lookup.

    A narrow physical clear-film run supplies a direct row anchor: Nikon's
    canonical table chooses the first row at or just beyond its trailing side.
    Broad/cut gaps are never called automatic.  They receive a local candidate
    chosen against the robust direct-anchor line so a UI can present it for
    review without silently pretending that the damaged boundary was exact.
    """

    if frame_count < 1 or len(boundaries) < frame_count:
        raise IndexDecodeError("transport mapping has fewer boundaries than frames")
    if len(records) < 1:
        raise IndexDecodeError("transport mapping has no live 0x8e records")
    frame_boundaries = list(boundaries[:frame_count])

    direct: list[tuple[GapBoundary, TransportRecord]] = []
    for boundary in frame_boundaries:
        if (
            boundary.support != "direct"
            or boundary.evidence_run is None
            or boundary.manual_review
        ):
            continue
        lookup_row = boundary.evidence_run[1]
        if not 0 <= lookup_row < len(records):
            raise IndexDecodeError(
                f"transport lookup row {lookup_row} for frame "
                f"{boundary.index + 1} is outside the live 0x8e table"
            )
        direct.append((boundary, records[lookup_row]))
    if len(direct) < 3:
        raise IndexDecodeError(
            "transport mapping requires at least three direct physical gaps"
        )

    x = np.asarray([item.output_row for item, _record in direct], dtype=np.float64)
    y = np.asarray([record.native_origin for _item, record in direct], dtype=np.float64)
    design = np.column_stack((np.ones(len(x)), x))
    intercept, scale = np.linalg.lstsq(design, y, rcond=None)[0]
    if not minimum_scale <= scale <= maximum_scale:
        raise IndexDecodeError(
            f"transport mapping scale {scale:.6f} is outside "
            f"{minimum_scale:.1f}..{maximum_scale:.1f} native units per preview row"
        )
    residual_rows = (design @ np.asarray((intercept, scale)) - y) / scale
    anchor_mae = float(np.mean(np.abs(residual_rows)))
    anchor_max = float(np.max(np.abs(residual_rows)))
    if anchor_mae > maximum_anchor_mae_rows or anchor_max > maximum_anchor_error_rows:
        raise IndexDecodeError(
            "transport anchor residual is inconsistent with one affine "
            f"preview traversal (MAE {anchor_mae:.3f} rows, max "
            f"{anchor_max:.3f} rows)"
        )

    direct_by_index = {item.index: record for item, record in direct}
    origins: list[NativeFrameOrigin] = []
    for frame, boundary in enumerate(frame_boundaries, start=1):
        prediction = float(intercept + scale * boundary.output_row)
        record = direct_by_index.get(boundary.index)
        if record is not None:
            method = "direct-gap-trailing-row"
            reasons = list(boundary.review_reasons)
            automatic = not boundary.manual_review
        else:
            # The canonical Nikon detector selects roughly 2..5 rows after a
            # physical gap centre.  Restrict the search to that local trailing
            # neighborhood, then use the direct-anchor line only to choose a
            # record already supplied by this capture's transport table.
            centre = int(math.floor(boundary.fitted_row + 0.5))
            start = max(0, centre + 1)
            end = min(len(records), centre + 8)
            if start >= end:
                raise IndexDecodeError(
                    f"transport candidate range for frame {frame} is outside "
                    "the live 0x8e table"
                )
            record = min(
                records[start:end],
                key=lambda item: abs(item.native_origin - prediction),
            )
            method = "affine-guided-local-lookup"
            reasons = list(boundary.review_reasons)
            reasons.append("transport-origin-inferred")
            automatic = False
        residual = (prediction - record.native_origin) / scale
        origins.append(
            NativeFrameOrigin(
                frame=frame,
                boundary_index=boundary.index,
                boundary_output_row=boundary.output_row,
                lookup_row=record.row,
                code=record.code,
                selector=record.selector,
                native_origin=record.native_origin,
                method=method,
                automatic=automatic,
                manual_review=not automatic,
                review_reasons=tuple(dict.fromkeys(reasons)),
                affine_residual_rows=float(residual),
            )
        )

    if any(
        first.native_origin >= second.native_origin
        for first, second in zip(origins, origins[1:])
    ):
        raise IndexDecodeError("transport-derived frame origins are not increasing")
    return TransportMapping(
        record_count=len(records),
        native_intercept=float(intercept),
        native_units_per_preview_row=float(scale),
        anchor_mae_rows=anchor_mae,
        anchor_max_error_rows=anchor_max,
        origins=tuple(origins),
    )


def detection_diagnostics(starts: Sequence[int], nominal_frame_rows: int) -> dict:
    """Describe fixed-spacing consistency without changing detector results."""

    values = np.asarray(starts, dtype=np.float64)
    if len(values) < 2:
        return {
            "frame_count": int(len(values)),
            "origins_output_rows": [int(value) for value in starts],
            "median_spacing_rows": None,
            "spacing_mad_rows": None,
            "fixed_spacing_residuals_rows": [],
            "confidence": "low",
        }
    spacings = np.diff(values)
    pitch = float(np.median(spacings))
    fitted = values[0] + np.arange(len(values)) * pitch
    residuals = values - fitted
    mad = float(np.median(np.abs(spacings - pitch)))
    confidence = (
        "provisional-medium" if mad <= max(2.0, nominal_frame_rows * 0.04) else "low"
    )
    return {
        "frame_count": int(len(values)),
        "origins_output_rows": [int(value) for value in starts],
        "median_spacing_rows": pitch,
        "spacing_mad_rows": mad,
        "fixed_spacing_residuals_rows": [round(float(value), 3) for value in residuals],
        "confidence": confidence,
    }
