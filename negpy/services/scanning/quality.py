"""Pure scanner-output quality measurements shared by capture workflows."""

from __future__ import annotations

import math
from hashlib import sha256
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import SplitIrAlignment
from negpy.infrastructure.scanners.split_alignment_policy import (
    SPLIT_MAX_CHANNEL_SPREAD_PX,
    SPLIT_MAX_HORIZONTAL_SHIFT_PX,
    SPLIT_MAX_VERTICAL_SHIFT_PX,
    SPLIT_MIN_ECC,
    SPLIT_MIN_PHASE_RESPONSE,
    SPLIT_MULTISCALE_HIGH_MIN_RESPONSE,
    SPLIT_MULTISCALE_LOW_MIN_RESPONSE,
    SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX,
    SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX,
    SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX,
    SPLIT_MULTISCALE_PERIODIC_ALIAS_DOMINANCE,
    SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_CHANNELS,
    SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_HORIZONTAL_PX,
    SPLIT_PHASE_ONLY_MAX_CHANNEL_SPREAD_PX,
    SPLIT_PHASE_ONLY_MAX_SHIFT_PX,
    SPLIT_PHASE_ONLY_MIN_RESPONSE,
    SPLIT_TILED_PHASE_MAX_CHANNEL_SPREAD_PX,
    SPLIT_TILED_PHASE_MAX_SHIFT_PX,
    SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX,
    SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE,
    SPLIT_TILED_PHASE_MIN_SUPPORT,
)


_STOPPED_TRANSPORT_MAX_RMS = 150.0
_STOPPED_TRANSPORT_MIN_CORRELATION = 0.9999
_STOPPED_TRANSPORT_MIN_TEXTURE_SPAN = 2048.0
_STOPPED_TRANSPORT_REFERENCE_RATIO = 4.0
_STOPPED_TRANSPORT_MAX_SAMPLED_COLUMNS = 512
_CLIPPING_MAX_CHUNK_ELEMENTS = 1 << 20


@dataclass(frozen=True)
class TiffPayloadInspection:
    """File-backed TIFF metadata used to validate persisted scanner masters."""

    sha256: str
    byte_length: int
    shape: tuple[int, ...]
    dtype: str
    page_count: int
    x_resolution: tuple[int, int] | None
    y_resolution: tuple[int, int] | None
    resolution_unit: str | None
    payload_within_file: bool


@dataclass(frozen=True)
class ScanClippingTelemetry:
    """Warning-only per-channel scanner-white clipping telemetry."""

    fractions: tuple[float, float, float]
    clip_level: float
    warning_fraction: float
    warning: bool


@dataclass(frozen=True)
class FocusDetailTelemetry:
    """Scene-dependent detail proxy; informational, never an autofocus gate."""

    method: str
    verdict: Literal["measured", "indeterminate"]
    score: float | None
    texture_span: float


def file_sha256(path: str | PathLike[str]) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rational_pair(value: object) -> tuple[int, int] | None:
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        if isinstance(numerator, int) and isinstance(denominator, int):
            return numerator, denominator
    if isinstance(value, int):
        return value, 1
    return None


def inspect_tiff_payload(
    path: str | PathLike[str],
) -> TiffPayloadInspection:
    """Inspect one TIFF without decoding its complete image payload."""

    source = Path(path)
    byte_length = source.stat().st_size
    with tifffile.TiffFile(source) as document:
        page_count = len(document.pages)
        if page_count == 0:
            raise ValueError(f"TIFF has no image pages: {source}")
        page = cast(tifffile.TiffPage, document.pages[0])
        x_tag = page.tags.get("XResolution")
        y_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        unit_value = unit_tag.value if unit_tag is not None else None
        unit_name = str(getattr(unit_value, "name", unit_value)) if unit_value is not None else None
        payload_within_file = all(
            int(offset) >= 0 and int(count) > 0 and int(offset) + int(count) <= byte_length
            for candidate in document.pages
            for offset, count in zip(
                candidate.dataoffsets,
                candidate.databytecounts,
                strict=True,
            )
        )
        return TiffPayloadInspection(
            sha256=file_sha256(source),
            byte_length=byte_length,
            shape=tuple(int(value) for value in page.shape),
            dtype=np.dtype(page.dtype).name,
            page_count=page_count,
            x_resolution=(_rational_pair(x_tag.value) if x_tag is not None else None),
            y_resolution=(_rational_pair(y_tag.value) if y_tag is not None else None),
            resolution_unit=unit_name,
            payload_within_file=payload_within_file,
        )


def measure_scan_clipping(
    rgb: np.ndarray,
    *,
    clip_level: float = 0.99,
    warning_fraction: float = 0.01,
) -> ScanClippingTelemetry:
    """Measure every RGB pixel at scanner white in bounded row chunks."""

    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must have shape (rows, columns, 3)")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("RGB must have non-zero area")
    if not 0.0 < clip_level <= 1.0:
        raise ValueError("clip_level must be in (0, 1]")
    if not 0.0 <= warning_fraction <= 1.0:
        raise ValueError("warning_fraction must be in [0, 1]")
    is_float = np.issubdtype(rgb.dtype, np.floating)
    if np.issubdtype(rgb.dtype, np.integer):
        white = float(np.iinfo(rgb.dtype).max)
    elif is_float:
        white = 1.0
    else:
        raise ValueError(f"unsupported RGB dtype {rgb.dtype}")

    height, width, _ = rgb.shape
    rows_per_chunk = max(1, _CLIPPING_MAX_CHUNK_ELEMENTS // (width * 3))
    clipped_counts = np.zeros(3, dtype=np.int64)
    for start_row in range(0, height, rows_per_chunk):
        chunk = rgb[start_row : start_row + rows_per_chunk]
        if is_float and not np.isfinite(chunk).all():
            raise ValueError("float RGB must contain only finite values")
        clipped_counts += np.count_nonzero(chunk >= clip_level * white, axis=(0, 1))

    total_pixels = height * width
    fractions = (
        float(clipped_counts[0] / total_pixels),
        float(clipped_counts[1] / total_pixels),
        float(clipped_counts[2] / total_pixels),
    )
    return ScanClippingTelemetry(
        fractions=fractions,
        clip_level=clip_level,
        warning_fraction=warning_fraction,
        warning=max(fractions) > warning_fraction,
    )


def measure_focus_detail(rgb: np.ndarray) -> FocusDetailTelemetry:
    """Measure an informational, exposure-normalized detail proxy."""

    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must have shape (rows, columns, 3)")
    sampled = rgb[::4, ::4].astype(np.float64, copy=False)
    gray = sampled[:, :, 0] * 0.2126 + sampled[:, :, 1] * 0.7152 + sampled[:, :, 2] * 0.0722
    low, high = np.percentile(gray, (5, 95))
    texture_span = float(high - low)
    if not math.isfinite(texture_span) or texture_span <= 0.0:
        return FocusDetailTelemetry(
            method="normalized-gradient-v1",
            verdict="indeterminate",
            score=None,
            texture_span=texture_span,
        )
    normalized = np.clip((gray - low) / texture_span, 0.0, 1.0)
    horizontal = np.diff(normalized, axis=1)
    vertical = np.diff(normalized, axis=0)
    score = float(math.sqrt(float(np.mean(horizontal * horizontal)) + float(np.mean(vertical * vertical))))
    return FocusDetailTelemetry(
        method="normalized-gradient-v1",
        verdict="measured",
        score=score,
        texture_span=texture_span,
    )


def _multiscale_alignment_metrics_confident(alignment: SplitIrAlignment, *, tiled: bool) -> bool:
    dimensions = alignment.multiscale_max_dimensions
    shifts_by_scale = alignment.multiscale_channel_shifts_px
    responses_by_scale = alignment.multiscale_responses
    scale_count = len(dimensions)
    if (
        alignment.estimator_version != 2
        or scale_count < 2
        or tuple(sorted(set(dimensions))) != dimensions
        or len(shifts_by_scale) != scale_count
        or len(responses_by_scale) != scale_count
    ):
        return False

    aggregates: list[tuple[float, float]] = []
    spreads: list[float] = []
    for index, (channel_shifts, responses) in enumerate(zip(shifts_by_scale, responses_by_scale, strict=True)):
        minimum_response = (
            SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE
            if tiled
            else (SPLIT_MULTISCALE_LOW_MIN_RESPONSE if index == 0 else SPLIT_MULTISCALE_HIGH_MIN_RESPONSE)
        )
        if (
            len(channel_shifts) != 3
            or len(responses) != 3
            or not all(math.isfinite(value) for shift in channel_shifts for value in shift)
            or not all(math.isfinite(response) and response >= minimum_response for response in responses)
            or any(max(abs(shift[0]), abs(shift[1])) > SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX for shift in channel_shifts)
        ):
            return False
        spread = max(
            max(shift[0] for shift in channel_shifts) - min(shift[0] for shift in channel_shifts),
            max(shift[1] for shift in channel_shifts) - min(shift[1] for shift in channel_shifts),
        )
        if spread > SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX:
            return False
        spreads.append(spread)
        aggregates.append(
            (
                sum(shift[0] for shift in channel_shifts) / 3.0,
                sum(shift[1] for shift in channel_shifts) / 3.0,
            )
        )

    measured_dx, measured_dy = aggregates[-1]
    applied_dx = float(round(measured_dx)) if abs(measured_dx - round(measured_dx)) < 0.05 else measured_dx
    applied_dy = float(round(measured_dy)) if abs(measured_dy - round(measured_dy)) < 0.05 else measured_dy
    if (
        max(shift[0] for shift in aggregates) - min(shift[0] for shift in aggregates) > SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX
        or max(shift[1] for shift in aggregates) - min(shift[1] for shift in aggregates) > SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX
        or not math.isclose(alignment.dx_px, applied_dx, abs_tol=1e-6)
        or not math.isclose(alignment.dy_px, applied_dy, abs_tol=1e-6)
        or tuple(alignment.phase_responses) != tuple(responses_by_scale[-1])
        or alignment.channel_spread_px is None
        or not math.isclose(alignment.channel_spread_px, spreads[-1], abs_tol=1e-6)
        or alignment.ecc_coefficient is not None
    ):
        return False
    if tiled:
        support_by_scale = alignment.multiscale_tile_support_counts
        tile_spreads_by_scale = alignment.multiscale_tile_shift_spreads_px
        alias_by_scale = alignment.multiscale_global_alias_shifts_px
        if len(support_by_scale) != scale_count or len(tile_spreads_by_scale) != scale_count or len(alias_by_scale) != scale_count:
            return False
        alias_signs: list[int] = []
        for support, tile_spreads, alias_shifts in zip(
            support_by_scale,
            tile_spreads_by_scale,
            alias_by_scale,
            strict=True,
        ):
            if (
                len(support) != 3
                or not all(count >= SPLIT_TILED_PHASE_MIN_SUPPORT for count in support)
                or len(tile_spreads) != 3
                or not all(math.isfinite(spread) and spread <= SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX for spread in tile_spreads)
                or len(alias_shifts) != 3
                or not all(math.isfinite(value) for shift in alias_shifts for value in shift)
            ):
                return False
            qualifying_aliases = [
                dx
                for dx, dy in alias_shifts
                if abs(dx) >= SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_HORIZONTAL_PX
                and abs(dx) >= SPLIT_MULTISCALE_PERIODIC_ALIAS_DOMINANCE * max(abs(dy), 1.0)
            ]
            if len(qualifying_aliases) < SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_CHANNELS:
                return False
            signs = {1 if dx > 0.0 else -1 for dx in qualifying_aliases}
            if len(signs) != 1:
                return False
            alias_signs.append(signs.pop())
        return (
            len(set(alias_signs)) == 1
            and tuple(alignment.tile_support_counts) == tuple(support_by_scale[-1])
            and alignment.tile_shift_spread_px is not None
            and math.isclose(
                alignment.tile_shift_spread_px,
                max(tile_spreads_by_scale[-1]),
                abs_tol=1e-6,
            )
        )
    return (
        not alignment.tile_support_counts
        and alignment.tile_shift_spread_px is None
        and not alignment.multiscale_tile_support_counts
        and not alignment.multiscale_tile_shift_spreads_px
        and not alignment.multiscale_global_alias_shifts_px
    )


def split_alignment_metrics_confident(
    alignment: SplitIrAlignment | None,
    *,
    image_width: int | None = None,
) -> bool:
    """Validate the alignment metrics persisted with a split capture.

    Production already rejected insufficient texture and overlap, plus ECC
    divergence from the phase estimate, before producing this telemetry. Those
    inputs are not persisted and therefore cannot be revalidated here.
    Same-reservation displacement caps are unconditional; ``image_width`` is
    retained for call-site compatibility with older acceptance code.
    """

    if alignment is None or not all(math.isfinite(value) for value in (alignment.dx_px, alignment.dy_px)):
        return False
    if abs(alignment.dx_px) > SPLIT_MAX_HORIZONTAL_SHIFT_PX or abs(alignment.dy_px) > SPLIT_MAX_VERTICAL_SHIFT_PX:
        return False
    if alignment.mode == "identity":
        return alignment.dx_px == 0.0 and alignment.dy_px == 0.0
    phase_responses = tuple(alignment.phase_responses)
    channel_spread = alignment.channel_spread_px
    ecc = alignment.ecc_coefficient
    if alignment.mode == "multiscale-global":
        return _multiscale_alignment_metrics_confident(alignment, tiled=False)
    if alignment.mode == "multiscale-tiled":
        return _multiscale_alignment_metrics_confident(alignment, tiled=True)
    if alignment.mode == "phase-only":
        return (
            len(phase_responses) == 3
            and all(math.isfinite(response) and response >= SPLIT_PHASE_ONLY_MIN_RESPONSE for response in phase_responses)
            and channel_spread is not None
            and math.isfinite(channel_spread)
            and channel_spread <= SPLIT_PHASE_ONLY_MAX_CHANNEL_SPREAD_PX
            and ecc is not None
            and math.isfinite(ecc)
            and ecc >= SPLIT_MIN_ECC
            and max(abs(alignment.dx_px), abs(alignment.dy_px)) <= SPLIT_PHASE_ONLY_MAX_SHIFT_PX
        )
    if alignment.mode == "tiled-phase":
        return (
            len(phase_responses) == 3
            and all(math.isfinite(response) and response >= SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE for response in phase_responses)
            and channel_spread is not None
            and math.isfinite(channel_spread)
            and channel_spread <= SPLIT_TILED_PHASE_MAX_CHANNEL_SPREAD_PX
            and len(alignment.tile_support_counts) == 3
            and all(count >= SPLIT_TILED_PHASE_MIN_SUPPORT for count in alignment.tile_support_counts)
            and alignment.tile_shift_spread_px is not None
            and math.isfinite(alignment.tile_shift_spread_px)
            and alignment.tile_shift_spread_px <= SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX
            and max(abs(alignment.dx_px), abs(alignment.dy_px)) <= SPLIT_TILED_PHASE_MAX_SHIFT_PX
        )
    if alignment.mode != "phase-ecc":
        return False
    return (
        len(phase_responses) == 3
        and all(math.isfinite(response) and response >= SPLIT_MIN_PHASE_RESPONSE for response in phase_responses)
        and channel_spread is not None
        and math.isfinite(channel_spread)
        and channel_spread <= SPLIT_MAX_CHANNEL_SPREAD_PX
        and ecc is not None
        and math.isfinite(ecc)
        and ecc >= SPLIT_MIN_ECC
    )


@dataclass(frozen=True)
class StoppedTransportSmearAssessment:
    """Fail-closed assessment of an abnormally repeated RGB tail."""

    verdict: Literal["clean", "smear", "indeterminate"]
    start_row: int | None
    suffix_rows: int
    minimum_matches: int
    tail_median_rms: float | None
    tail_min_corr: float | None
    pre_tail_median_rms: float | None
    texture_span: float | None
    reason: str


def assess_stopped_transport_smear(
    rgb: np.ndarray,
    *,
    dpi: int,
) -> StoppedTransportSmearAssessment:
    """Classify a contiguous, near-identical RGB suffix without copying the image.

    Only the central 90 percent of each inspected row is used, with the column
    stride capped to roughly 512 samples. Row-sized float buffers are created
    as needed; the full uint16 image is never converted to floating point.
    """

    minimum_matches = math.ceil(dpi * 0.016) if type(dpi) is int and dpi > 0 else 0

    def _assessment(
        verdict: Literal["clean", "smear", "indeterminate"],
        reason: str,
        *,
        start_row: int | None = None,
        suffix_rows: int = 0,
        tail_median_rms: float | None = None,
        tail_min_corr: float | None = None,
        pre_tail_median_rms: float | None = None,
        texture_span: float | None = None,
    ) -> StoppedTransportSmearAssessment:
        return StoppedTransportSmearAssessment(
            verdict=verdict,
            start_row=start_row,
            suffix_rows=suffix_rows,
            minimum_matches=minimum_matches,
            tail_median_rms=tail_median_rms,
            tail_min_corr=tail_min_corr,
            pre_tail_median_rms=pre_tail_median_rms,
            texture_span=texture_span,
            reason=reason,
        )

    if minimum_matches == 0:
        return _assessment("indeterminate", "dpi must be a positive integer")
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint16:
        return _assessment("indeterminate", "RGB must be a uint16 numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return _assessment("indeterminate", "RGB must have shape (rows, columns, 3)")
    height, width, _ = rgb.shape
    if height < 2 or width < 2:
        return _assessment(
            "indeterminate",
            "RGB is too small to measure row transitions",
        )

    left = math.floor(width * 0.05)
    right = math.ceil(width * 0.95)
    sampled_width = right - left
    stride = max(
        1,
        math.ceil(sampled_width / _STOPPED_TRANSPORT_MAX_SAMPLED_COLUMNS),
    )

    def _row(index: int) -> np.ndarray:
        return rgb[index, left:right:stride, :].reshape(-1)

    def _transition(end_row: int) -> tuple[float, float] | None:
        before = _row(end_row - 1).astype(np.float64, copy=False)
        after = _row(end_row).astype(np.float64, copy=False)
        difference = after - before
        rms = float(np.sqrt(np.mean(difference * difference)))
        before_centered = before - float(np.mean(before))
        after_centered = after - float(np.mean(after))
        denominator = float(np.sqrt(np.dot(before_centered, before_centered) * np.dot(after_centered, after_centered)))
        if denominator == 0.0 or not math.isfinite(denominator) or not math.isfinite(rms):
            return None
        correlation = float(np.dot(before_centered, after_centered) / denominator)
        if not math.isfinite(correlation):
            return None
        return rms, correlation

    # Measure spatial variation inside each channel. Flattening RGB together
    # makes a perfectly uniform colored band look textured merely because its
    # red, green, and blue levels differ.
    anchor_rgb = rgb[height - 1, left:right:stride, :]
    channel_texture_spans = np.percentile(anchor_rgb, 95, axis=0) - np.percentile(
        anchor_rgb,
        5,
        axis=0,
    )
    texture_span = float(np.max(channel_texture_spans))
    tail_rms: list[float] = []
    tail_correlations: list[float] = []
    end_row = height - 1
    while end_row >= 1:
        metrics = _transition(end_row)
        if metrics is None:
            return _assessment(
                "indeterminate",
                "an EOF row transition could not be measured",
                start_row=height - len(tail_rms) if tail_rms else None,
                suffix_rows=len(tail_rms),
                tail_median_rms=(float(np.median(tail_rms)) if tail_rms else None),
                tail_min_corr=(min(tail_correlations) if tail_correlations else None),
                texture_span=texture_span,
            )
        rms, correlation = metrics
        if not (rms < _STOPPED_TRANSPORT_MAX_RMS and correlation > _STOPPED_TRANSPORT_MIN_CORRELATION):
            break
        tail_rms.append(rms)
        tail_correlations.append(correlation)
        end_row -= 1

    matching_transitions = len(tail_rms)
    if matching_transitions == 0:
        return _assessment(
            "clean",
            "the final row transition does not match a stopped-transport suffix",
            texture_span=texture_span,
        )

    start_row = height - matching_transitions
    tail_median_rms = float(np.median(tail_rms))
    tail_min_corr = min(tail_correlations)
    if matching_transitions < minimum_matches:
        return _assessment(
            "indeterminate",
            "the EOF has a partial repeated-row signature",
            start_row=start_row,
            suffix_rows=matching_transitions,
            tail_median_rms=tail_median_rms,
            tail_min_corr=tail_min_corr,
            texture_span=texture_span,
        )
    if texture_span < _STOPPED_TRANSPORT_MIN_TEXTURE_SPAN:
        return _assessment(
            "indeterminate",
            "the repeated anchor row lacks enough texture to distinguish smear",
            start_row=start_row,
            suffix_rows=matching_transitions,
            tail_median_rms=tail_median_rms,
            tail_min_corr=tail_min_corr,
            texture_span=texture_span,
        )
    if start_row < minimum_matches + 1:
        return _assessment(
            "indeterminate",
            "the repeated suffix has no complete preceding reference band",
            start_row=start_row,
            suffix_rows=matching_transitions,
            tail_median_rms=tail_median_rms,
            tail_min_corr=tail_min_corr,
            texture_span=texture_span,
        )

    reference_rms: list[float] = []
    for reference_end_row in range(
        start_row - 1,
        start_row - minimum_matches - 1,
        -1,
    ):
        metrics = _transition(reference_end_row)
        if metrics is None:
            return _assessment(
                "indeterminate",
                "a preceding reference transition could not be measured",
                start_row=start_row,
                suffix_rows=matching_transitions,
                tail_median_rms=tail_median_rms,
                tail_min_corr=tail_min_corr,
                texture_span=texture_span,
            )
        reference_rms.append(metrics[0])
    pre_tail_median_rms = float(np.median(reference_rms))
    if pre_tail_median_rms < _STOPPED_TRANSPORT_REFERENCE_RATIO * tail_median_rms:
        return _assessment(
            "indeterminate",
            "the repeated suffix is not separated by an abrupt row-change boundary",
            start_row=start_row,
            suffix_rows=matching_transitions,
            tail_median_rms=tail_median_rms,
            tail_min_corr=tail_min_corr,
            pre_tail_median_rms=pre_tail_median_rms,
            texture_span=texture_span,
        )
    return _assessment(
        "smear",
        "a textured repeated-row suffix follows an abrupt transport boundary",
        start_row=start_row,
        suffix_rows=matching_transitions,
        tail_median_rms=tail_median_rms,
        tail_min_corr=tail_min_corr,
        pre_tail_median_rms=pre_tail_median_rms,
        texture_span=texture_span,
    )


__all__ = [
    "FocusDetailTelemetry",
    "ScanClippingTelemetry",
    "StoppedTransportSmearAssessment",
    "TiffPayloadInspection",
    "assess_stopped_transport_smear",
    "file_sha256",
    "inspect_tiff_payload",
    "measure_focus_detail",
    "measure_scan_clipping",
    "split_alignment_metrics_confident",
]
