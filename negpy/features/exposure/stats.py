"""
Negative statistics for the Analysis panel.

Pure presentation logic: turns the metrics already measured every render
(density range, metered anchor, effective slope, histogram clipping) into
human-readable rows. No pipeline math here — just labelling.

Densities are *relative* (normalized decode), meaningful for contrast and across
a roll, not absolute scanner D.
"""

from dataclasses import dataclass
from typing import List, Optional

from negpy.features.exposure.models import EXPOSURE_CONSTANTS

# Qualitative band thresholds (display-only, tune to taste).
_DENSITY_LOW = 0.95  # luminance ΔD below this reads flat
_DENSITY_HIGH = 1.75  # above this reads contrasty
_KEY_DEV = 0.04  # metered-anchor deviation from neutral for low/high key
_SLOPE_SOFT = 3.5  # effective slope below this prints soft
_SLOPE_HARD = 6.0  # above this prints hard
_CLIP_WARN = 0.01  # >1% of a channel clipped flags a warning

_EMPTY = "—"


@dataclass(frozen=True)
class StatRow:
    """One labelled read-out row. `tag` is an optional qualitative chip."""

    name: str
    value: str
    tag: str = ""
    warn: bool = False


def _density_row(norm_density_range: Optional[float]) -> StatRow:
    if norm_density_range is None:
        return StatRow("Density range", _EMPTY)
    dr = float(norm_density_range)
    if dr < _DENSITY_LOW:
        tag = "Low contrast"
    elif dr > _DENSITY_HIGH:
        tag = "High contrast"
    else:
        tag = "Normal"
    return StatRow("Density range", f"{dr:.2f}", tag)


_LOG10_2 = 0.30103  # one stop in log10-density


def _exposure_row(metered_anchor: Optional[float], norm_density_range: Optional[float]) -> StatRow:
    if metered_anchor is None:
        return StatRow("Exposure", _EMPTY)
    dev = float(metered_anchor) - float(EXPOSURE_CONSTANTS["assumed_anchor"])
    if dev < -_KEY_DEV:
        tag = "Low-key"
    elif dev > _KEY_DEV:
        tag = "High-key"
    else:
        tag = "Balanced"
    # Express the midtone offset as stops, scaling the normalized deviation by the
    # negative's density range. Positive = brighter (high-key). Approximate.
    if norm_density_range:
        ev = dev * float(norm_density_range) / _LOG10_2
        return StatRow("Exposure", f"{ev:+.1f} EV", tag)
    return StatRow("Exposure", "", tag)


def _contrast_row(slope: Optional[float], effective_range: Optional[float]) -> StatRow:
    if slope is None:
        return StatRow("Contrast", _EMPTY)
    from negpy.features.exposure.logic import slope_to_grade

    k = float(slope)
    if k < _SLOPE_SOFT:
        tag = "Soft"
    elif k > _SLOPE_HARD:
        tag = "Hard"
    else:
        tag = "Normal"
    # Effective slope expressed on the ISO R paper scale (same as the Grade
    # slider), rounded to a tidy R step.
    r = slope_to_grade(k, effective_range)
    return StatRow("Contrast", f"R{int(round(r / 5.0) * 5)}", tag)


def _clipping_row(clip_low: Optional[float], clip_high: Optional[float]) -> StatRow:
    if clip_low is None or clip_high is None:
        return StatRow("Clipping", _EMPTY)
    lo, hi = float(clip_low), float(clip_high)
    warn = lo > _CLIP_WARN or hi > _CLIP_WARN
    return StatRow("Clipping", f"Sh {lo * 100:.1f}% · Hi {hi * 100:.1f}%", warn=warn)


def negative_statistics(
    norm_density_range: Optional[float],
    metered_anchor: Optional[float],
    slope: Optional[float],
    clip_low: Optional[float],
    clip_high: Optional[float],
    effective_range: Optional[float] = None,
) -> List[StatRow]:
    """
    Core-4 negative read-out, in display order. `effective_range` is the density
    range that produced `slope` (after Auto Grade blending) — used to express the
    contrast on the ISO R scale; falls back to `norm_density_range`.
    """
    return [
        _density_row(norm_density_range),
        _exposure_row(metered_anchor, norm_density_range),
        _contrast_row(slope, effective_range if effective_range is not None else norm_density_range),
        _clipping_row(clip_low, clip_high),
    ]
