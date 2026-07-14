#!/usr/bin/env python3
"""Scanner-free, fail-closed exposure metering for LS-5000 C-41 scans.

The wire capture layer owns USB sequencing.  This module only decodes the
three 285-dpi RGB+IR meter rasters and decides whether an exposure update is
safe.  It deliberately has no scanner or filesystem dependencies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


CHANNELS = ("R", "G", "B", "IR")
METER_ROWS = 425
METER_WIDTH = 281
METER_CORE_SAMPLES = len(CHANNELS) * METER_WIDTH
METER_ROW_SAMPLES = 1280
METER_TAIL_SAMPLES = METER_ROW_SAMPLES - METER_CORE_SAMPLES
METER_PASS_BYTES = METER_ROWS * METER_ROW_SAMPLES * 2
FULL_SCALE = 65_535.0
DEFAULT_EXPOSURES = {"R": 90_000, "G": 180_000, "B": 165_000, "IR": 280_000}
TARGET_FRACTIONS = {"R": 0.85, "G": 0.85, "B": 0.85, "IR": 0.84}
CEILING_FRACTION = 0.95
EXPOSURE_MIN = 50_000
EXPOSURE_MAX = 400_000
UPDATE_RATIO_MIN = 0.5
UPDATE_RATIO_MAX = 2.0
FINAL_CHANGE_LIMIT = 0.05
CENTRAL_INSET_FRACTION = 0.10
LINEARITY_GAIN_ERROR_LIMIT = 0.10
LINEARITY_CORRELATION_MIN = 0.98
LINEARITY_MIN_SAMPLES = 256
LINEARITY_CORRELATION_BIN_WIDTH = 9
LINEARITY_MIN_AGGREGATES = 256


@dataclass(frozen=True)
class DecodedMeterPass:
    """One decoded pass plus the opaque per-row suffix from the scanner."""

    image: np.ndarray
    row_tail: np.ndarray
    channels: tuple[str, ...] = CHANNELS

    def to_bytes(self) -> bytes:
        """Re-encode without altering the unknown row-tail samples."""

        rows = np.empty((METER_ROWS, METER_ROW_SAMPLES), dtype=">u2")
        rows[:, :METER_CORE_SAMPLES] = self.image.transpose(0, 2, 1).reshape(METER_ROWS, METER_CORE_SAMPLES)
        rows[:, METER_CORE_SAMPLES:] = self.row_tail
        return rows.tobytes()


@dataclass(frozen=True)
class ChannelStatistics:
    black: float
    central_low: float
    central_high: float
    full_high: float
    central_signal: float
    texture_span: float
    ceiling_value: float

    def to_dict(self) -> dict[str, float]:
        return {
            "black": float(self.black),
            "central_low_q1": float(self.central_low),
            "central_high_q99_9": float(self.central_high),
            "full_high_q99_99": float(self.full_high),
            "central_signal": float(self.central_signal),
            "texture_span": float(self.texture_span),
            "ceiling_value": float(self.ceiling_value),
        }


@dataclass(frozen=True)
class MeterObservation:
    decoded: DecodedMeterPass
    exposures: dict[str, int]
    channel_statistics: dict[str, ChannelStatistics]

    def to_dict(self) -> dict[str, Any]:
        tail_bytes = self.decoded.row_tail.astype(">u2", copy=False).tobytes()
        return {
            "geometry": {
                "rows": METER_ROWS,
                "columns": METER_WIDTH,
                "channels": list(CHANNELS),
                "sample_format": "big-endian-u16",
            },
            "exposures_raw_10ns": dict(self.exposures),
            "channels": {channel: self.channel_statistics[channel].to_dict() for channel in CHANNELS},
            "opaque_row_tail": {
                "shape": [METER_ROWS, METER_TAIL_SAMPLES],
                "sha256": hashlib.sha256(tail_bytes).hexdigest(),
            },
        }


@dataclass(frozen=True)
class SafetyRefusal:
    code: str
    message: str
    channel: str | None = None

    def to_dict(self) -> dict[str, str]:
        result = {"code": self.code, "message": self.message}
        if self.channel is not None:
            result["channel"] = self.channel
        return result


@dataclass(frozen=True)
class MeterProposal:
    observation: MeterObservation
    proposed_exposures: dict[str, int]
    channel_diagnostics: dict[str, dict[str, Any]]
    refusals: tuple[SafetyRefusal, ...]

    @property
    def accepted(self) -> bool:
        return not self.refusals

    @property
    def max_change_fraction(self) -> float:
        return max(
            abs(self.proposed_exposures[channel] - self.observation.exposures[channel]) / max(abs(self.observation.exposures[channel]), 1)
            for channel in CHANNELS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "current_exposures_raw_10ns": dict(self.observation.exposures),
            "proposed_exposures_raw_10ns": dict(self.proposed_exposures),
            "max_change_fraction": float(self.max_change_fraction),
            "refusals": [refusal.to_dict() for refusal in self.refusals],
            "channels": {channel: dict(self.channel_diagnostics[channel]) for channel in CHANNELS},
            "observation": self.observation.to_dict(),
        }


@dataclass(frozen=True)
class MeterResult:
    accepted: bool
    final_exposures: dict[str, int] | None
    steps: tuple[MeterProposal, ...]
    refusals: tuple[SafetyRefusal, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "final_exposures_raw_10ns": (dict(self.final_exposures) if self.final_exposures is not None else None),
            "final_change_limit": FINAL_CHANGE_LIMIT,
            "refusals": [refusal.to_dict() for refusal in self.refusals],
            "steps": [step.to_dict() for step in self.steps],
        }


def decode_meter_pass(payload: bytes) -> DecodedMeterPass:
    """Decode exactly one 1,088,000-byte, big-endian 285-dpi meter pass."""

    if len(payload) != METER_PASS_BYTES:
        raise ValueError(f"meter pass is {len(payload)} bytes; expected {METER_PASS_BYTES}")
    rows = np.frombuffer(payload, dtype=">u2").reshape(METER_ROWS, METER_ROW_SAMPLES)
    image = rows[:, :METER_CORE_SAMPLES].reshape(METER_ROWS, len(CHANNELS), METER_WIDTH).transpose(0, 2, 1).astype(np.uint16, copy=True)
    row_tail = rows[:, METER_CORE_SAMPLES:].copy()
    return DecodedMeterPass(image=image, row_tail=row_tail)


def _normalise_exposures(
    exposures: Mapping[str, int] | Sequence[int],
) -> dict[str, int]:
    if isinstance(exposures, Mapping):
        missing = [channel for channel in CHANNELS if channel not in exposures]
        extra = [str(key) for key in exposures if key not in CHANNELS]
        if missing or extra:
            raise ValueError(f"exposures must contain exactly {CHANNELS}; missing={missing}, extra={extra}")
        values = [exposures[channel] for channel in CHANNELS]
    else:
        values = list(exposures)
        if len(values) != len(CHANNELS):
            raise ValueError(f"expected four exposures in {CHANNELS} order")
    result: dict[str, int] = {}
    for channel, value in zip(CHANNELS, values, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{channel} exposure must be an integer")
        result[channel] = int(value)
    return result


def _channel_statistics(channel_image: np.ndarray) -> ChannelStatistics:
    row_inset = max(1, round(METER_ROWS * CENTRAL_INSET_FRACTION))
    column_inset = max(1, round(METER_WIDTH * CENTRAL_INSET_FRACTION))
    central = channel_image[
        row_inset : METER_ROWS - row_inset,
        column_inset : METER_WIDTH - column_inset,
    ].astype(np.float64, copy=False)
    full = channel_image.astype(np.float64, copy=False)
    black = float(np.quantile(full, 0.001))
    central_low = float(np.quantile(central, 0.01))
    central_high = float(np.quantile(central, 0.999))
    full_high = float(np.quantile(full, 0.9999))
    ceiling_value = black + CEILING_FRACTION * (FULL_SCALE - black)
    return ChannelStatistics(
        black=black,
        central_low=central_low,
        central_high=central_high,
        full_high=full_high,
        central_signal=max(0.0, central_high - black),
        texture_span=max(0.0, central_high - central_low),
        ceiling_value=ceiling_value,
    )


def observe_meter_pass(
    meter_pass: bytes | DecodedMeterPass,
    exposures: Mapping[str, int] | Sequence[int] = (90_000, 180_000, 165_000, 280_000),
) -> MeterObservation:
    """Decode and meter one pass without making a control decision."""

    decoded = decode_meter_pass(meter_pass) if isinstance(meter_pass, (bytes, bytearray, memoryview)) else meter_pass
    if not isinstance(decoded, DecodedMeterPass):
        raise TypeError("meter_pass must be bytes or DecodedMeterPass")
    normalised = _normalise_exposures(exposures)
    statistics = {channel: _channel_statistics(decoded.image[:, :, index]) for index, channel in enumerate(CHANNELS)}
    return MeterObservation(
        decoded=decoded,
        exposures=normalised,
        channel_statistics=statistics,
    )


def _linearity_diagnostic(
    previous: MeterObservation,
    current: MeterObservation,
    channel: str,
    channel_index: int,
) -> dict[str, Any]:
    row_inset = max(1, round(METER_ROWS * CENTRAL_INSET_FRACTION))
    column_inset = max(1, round(METER_WIDTH * CENTRAL_INSET_FRACTION))
    rows = slice(row_inset, METER_ROWS - row_inset)
    columns = slice(column_inset, METER_WIDTH - column_inset)
    previous_stats = previous.channel_statistics[channel]
    current_stats = current.channel_statistics[channel]
    x_grid = previous.decoded.image[rows, columns, channel_index].astype(np.float64) - previous_stats.black
    y_grid = current.decoded.image[rows, columns, channel_index].astype(np.float64) - current_stats.black
    previous_range = max(1.0, FULL_SCALE - previous_stats.black)
    current_range = max(1.0, FULL_SCALE - current_stats.black)
    valid_grid = (
        (x_grid > 0.02 * previous_range)
        & (y_grid > 0.02 * current_range)
        & (x_grid < 0.90 * previous_range)
        & (y_grid < 0.90 * current_range)
    )
    x = x_grid[valid_grid]
    y = y_grid[valid_grid]

    # Nikon's IR meter plane has real pixel-scale read noise.  Average only
    # across width so every transport-axis row remains independently visible;
    # one shifted row must not be hidden by a 2-D blur.  A block participates
    # only when all nine raw pairs meet the same gain-regression validity mask.
    aggregate_width = (x_grid.shape[1] // LINEARITY_CORRELATION_BIN_WIDTH) * LINEARITY_CORRELATION_BIN_WIDTH
    x_blocks = x_grid[:, :aggregate_width].reshape(x_grid.shape[0], -1, LINEARITY_CORRELATION_BIN_WIDTH)
    y_blocks = y_grid[:, :aggregate_width].reshape(y_grid.shape[0], -1, LINEARITY_CORRELATION_BIN_WIDTH)
    valid_blocks = valid_grid[:, :aggregate_width].reshape(valid_grid.shape[0], -1, LINEARITY_CORRELATION_BIN_WIDTH)
    all_raw_valid = np.all(valid_blocks, axis=2)
    x_aggregated = np.mean(x_blocks, axis=2)[all_raw_valid]
    y_aggregated = np.mean(y_blocks, axis=2)[all_raw_valid]

    previous_exposure = previous.exposures[channel]
    expected_gain = current.exposures[channel] / previous_exposure if previous_exposure > 0 else None
    aggregation = {
        "axis": "width",
        "size": LINEARITY_CORRELATION_BIN_WIDTH,
        "requires_all_raw_valid": True,
    }
    if x.size < LINEARITY_MIN_SAMPLES or x_aggregated.size < LINEARITY_MIN_AGGREGATES:
        return {
            "valid_samples": int(x.size),
            "correlation_samples": int(x_aggregated.size),
            "expected_gain": (float(expected_gain) if expected_gain is not None else None),
            "measured_gain": None,
            "gain_error_fraction": None,
            "correlation": None,
            "raw_pixel_correlation": None,
            "correlation_aggregation": aggregation,
            "accepted": False,
            "reason": "insufficient_unclipped_samples",
        }
    denominator = float(np.dot(x, x))
    measured_gain = float(np.dot(x, y) / denominator) if denominator else 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        raw_pixel_correlation = 0.0
    else:
        raw_pixel_correlation = float(np.corrcoef(x, y)[0, 1])
    if np.std(x_aggregated) == 0 or np.std(y_aggregated) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(x_aggregated, y_aggregated)[0, 1])
    gain_error = abs(measured_gain / expected_gain - 1.0) if expected_gain is not None and expected_gain > 0 else None
    if not np.isfinite(raw_pixel_correlation):
        raw_pixel_correlation = 0.0
    if not np.isfinite(correlation):
        correlation = 0.0
    accepted = gain_error is not None and gain_error <= LINEARITY_GAIN_ERROR_LIMIT and correlation >= LINEARITY_CORRELATION_MIN
    return {
        "valid_samples": int(x.size),
        "correlation_samples": int(x_aggregated.size),
        "expected_gain": (float(expected_gain) if expected_gain is not None else None),
        "measured_gain": float(measured_gain),
        "gain_error_fraction": (float(gain_error) if gain_error is not None else None),
        "correlation": float(correlation),
        "raw_pixel_correlation": float(raw_pixel_correlation),
        "correlation_aggregation": aggregation,
        "accepted": bool(accepted),
    }


def propose_next_exposures(
    current: MeterObservation,
    previous: MeterObservation | MeterProposal | None = None,
) -> MeterProposal:
    """Propose the next bounded exposure set, refusing unsafe observations."""

    if isinstance(previous, MeterProposal):
        previous = previous.observation
    if previous is not None and not isinstance(previous, MeterObservation):
        raise TypeError("previous must be MeterObservation, MeterProposal, or None")

    refusals: list[SafetyRefusal] = []
    proposed: dict[str, int] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        exposure = current.exposures[channel]
        stats = current.channel_statistics[channel]
        target_fraction = TARGET_FRACTIONS[channel]
        target_signal = target_fraction * (FULL_SCALE - stats.black)
        if stats.central_signal <= 0:
            raw_ratio = UPDATE_RATIO_MAX
        else:
            raw_ratio = target_signal / stats.central_signal
        bounded_ratio = float(np.clip(raw_ratio, UPDATE_RATIO_MIN, UPDATE_RATIO_MAX))
        ratio_bounded_exposure = int(round(exposure * bounded_ratio))
        absolute_bounded_exposure = int(np.clip(ratio_bounded_exposure, EXPOSURE_MIN, EXPOSURE_MAX))
        requested_increase = exposure > 0 and absolute_bounded_exposure > exposure
        full_signal = max(0.0, stats.full_high - stats.black)
        predictive_ceiling_ratio = (stats.ceiling_value - stats.black) / full_signal if full_signal > 0 else None
        predictive_ceiling_cap = (
            int(np.floor(exposure * predictive_ceiling_ratio)) if exposure > 0 and predictive_ceiling_ratio is not None else None
        )
        if predictive_ceiling_cap is not None:
            predictive_ceiling_cap = max(0, predictive_ceiling_cap)
            while predictive_ceiling_cap > 0 and stats.black + full_signal * predictive_ceiling_cap / exposure > stats.ceiling_value:
                predictive_ceiling_cap -= 1
        bounded_exposure = absolute_bounded_exposure
        if requested_increase and predictive_ceiling_cap is not None:
            bounded_exposure = min(bounded_exposure, max(exposure, predictive_ceiling_cap))
        effective_ratio = bounded_exposure / exposure if exposure > 0 else None
        predicted_full_high = stats.black + full_signal * effective_ratio if effective_ratio is not None else None
        proposed[channel] = bounded_exposure
        channel_diagnostic = stats.to_dict()
        channel_diagnostic.update(
            {
                "target_fraction": float(target_fraction),
                "target_signal": float(target_signal),
                "raw_update_ratio": float(raw_ratio),
                "bounded_update_ratio": float(bounded_ratio),
                "effective_update_ratio": (float(effective_ratio) if effective_ratio is not None else None),
                "bounded_by_ratio": not np.isclose(raw_ratio, bounded_ratio),
                "bounded_by_absolute_exposure": (ratio_bounded_exposure != absolute_bounded_exposure),
                "predictive_ceiling_ratio": (float(predictive_ceiling_ratio) if predictive_ceiling_ratio is not None else None),
                "predictive_ceiling_exposure_cap": predictive_ceiling_cap,
                "bounded_by_predictive_ceiling": (bounded_exposure < absolute_bounded_exposure),
                "predicted_full_high": (float(predicted_full_high) if predicted_full_high is not None else None),
            }
        )
        diagnostics[channel] = channel_diagnostic

        if not EXPOSURE_MIN <= exposure <= EXPOSURE_MAX:
            refusals.append(
                SafetyRefusal(
                    "exposure_out_of_bounds",
                    f"current exposure {exposure} is outside [{EXPOSURE_MIN}, {EXPOSURE_MAX}]",
                    channel,
                )
            )
        usable_range = max(1.0, FULL_SCALE - stats.black)
        if stats.central_signal < 0.02 * usable_range:
            refusals.append(
                SafetyRefusal(
                    "dense_or_no_signal",
                    "robust highlight signal is too small to meter safely",
                    channel,
                )
            )
        if stats.texture_span < 0.005 * usable_range:
            refusals.append(
                SafetyRefusal(
                    "blank_or_uniform",
                    "meter raster has too little robust tonal variation",
                    channel,
                )
            )
        if stats.full_high >= stats.ceiling_value and requested_increase:
            refusals.append(
                SafetyRefusal(
                    "clipped_increase",
                    "full-frame high quantile is above the 95% ceiling; an exposure increase is unsafe",
                    channel,
                )
            )

        if previous is not None:
            linearity = _linearity_diagnostic(previous, current, channel, CHANNELS.index(channel))
            channel_diagnostic["linearity"] = linearity
            if linearity["valid_samples"] < LINEARITY_MIN_SAMPLES or linearity["correlation_samples"] < LINEARITY_MIN_AGGREGATES:
                refusals.append(
                    SafetyRefusal(
                        "linearity_insufficient",
                        "too few unclipped pixels remain for pass-linearity proof",
                        channel,
                    )
                )
            else:
                gain_error = linearity["gain_error_fraction"]
                correlation = linearity["correlation"]
                if gain_error is None or gain_error > LINEARITY_GAIN_ERROR_LIMIT:
                    refusals.append(
                        SafetyRefusal(
                            "nonlinear_gain",
                            "observed pass gain differs from exposure gain by more than 10%",
                            channel,
                        )
                    )
                if correlation is None or correlation < LINEARITY_CORRELATION_MIN:
                    refusals.append(
                        SafetyRefusal(
                            "low_correlation",
                            "consecutive meter passes do not preserve image structure",
                            channel,
                        )
                    )

    return MeterProposal(
        observation=current,
        proposed_exposures=proposed,
        channel_diagnostics=diagnostics,
        refusals=tuple(refusals),
    )


def _convergence_refusal(proposal: MeterProposal) -> SafetyRefusal | None:
    if proposal.max_change_fraction <= FINAL_CHANGE_LIMIT:
        return None
    return SafetyRefusal(
        "nonconverged",
        f"final exposure update is {proposal.max_change_fraction:.6f}; limit is {FINAL_CHANGE_LIMIT:.6f}",
    )


def verify_final_convergence(
    current: MeterObservation,
    previous: MeterObservation | None = None,
) -> MeterResult:
    """Require fresh prior-pass linearity and a final update of at most 5%."""

    if not isinstance(current, MeterObservation):
        raise TypeError("current must be a MeterObservation, not a cached proposal")
    if previous is not None and not isinstance(previous, MeterObservation):
        raise TypeError("previous must be a MeterObservation or None")
    proposal = propose_next_exposures(current, previous=previous)
    refusals = list(proposal.refusals)
    if previous is None:
        refusals.append(
            SafetyRefusal(
                "missing_previous_linearity",
                "final acceptance requires a fresh previous-pass linearity check",
            )
        )
    convergence_refusal = _convergence_refusal(proposal)
    if convergence_refusal is not None:
        refusals.append(convergence_refusal)
    accepted = not refusals
    return MeterResult(
        accepted=accepted,
        final_exposures=(dict(proposal.proposed_exposures) if accepted else None),
        steps=(proposal,),
        refusals=tuple(refusals),
    )


def evaluate_meter_sequence(
    meter_passes: Sequence[bytes | DecodedMeterPass],
    exposure_history: Sequence[Mapping[str, int] | Sequence[int]],
) -> MeterResult:
    """Evaluate the three guarded meter passes as one fail-closed decision."""

    if len(meter_passes) != 3 or len(exposure_history) != 3:
        raise ValueError("exactly three meter passes and exposure sets are required")
    observations = [observe_meter_pass(meter_pass, exposures) for meter_pass, exposures in zip(meter_passes, exposure_history, strict=True)]
    proposals: list[MeterProposal] = []
    previous: MeterObservation | None = None
    for observation in observations:
        proposal = propose_next_exposures(observation, previous=previous)
        proposals.append(proposal)
        previous = observation

    refusals = [refusal for proposal in proposals for refusal in proposal.refusals]
    for proposal_index in range(len(proposals) - 1):
        expected = proposals[proposal_index].proposed_exposures
        observed = observations[proposal_index + 1].exposures
        for channel in CHANNELS:
            if observed[channel] != expected[channel]:
                refusals.append(
                    SafetyRefusal(
                        "exposure_history_mismatch",
                        f"pass {proposal_index + 2} exposure {observed[channel]} "
                        f"does not match expected prior proposal "
                        f"{expected[channel]}",
                        channel,
                    )
                )
    convergence_refusal = _convergence_refusal(proposals[-1])
    if convergence_refusal is not None:
        refusals.append(convergence_refusal)
    accepted = not refusals
    return MeterResult(
        accepted=accepted,
        final_exposures=(dict(proposals[-1].proposed_exposures) if accepted else None),
        steps=tuple(proposals),
        refusals=tuple(refusals),
    )
