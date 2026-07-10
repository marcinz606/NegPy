"""
Negative statistics for the Analysis panel.

Pure presentation logic: turns the metrics already measured every render
(density range, metered anchor, effective slope, histogram clipping) into
read-out rows. No pipeline math here — just formatting.

Densities are *relative* (normalized decode), meaningful for contrast and across
a roll, not absolute scanner D.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from negpy.features.exposure.models import EXPOSURE_CONSTANTS

_CLIP_WARN = 0.01  # >1% of a channel clipped flags a warning

_EMPTY = "—"


@dataclass(frozen=True)
class StatRow:
    """One labelled read-out row."""

    name: str
    value: str
    warn: bool = False


def _density_row(norm_density_range: Optional[float]) -> StatRow:
    if norm_density_range is None:
        return StatRow("Density range", _EMPTY)
    return StatRow("Density range", f"{float(norm_density_range):.2f}")


_LOG10_2 = 0.30103  # one stop in log10-density


def _exposure_row(metered_anchor: Optional[float], norm_density_range: Optional[float]) -> StatRow:
    if metered_anchor is None or not norm_density_range:
        return StatRow("Exposure", _EMPTY)
    dev = float(metered_anchor) - float(EXPOSURE_CONSTANTS["assumed_anchor"])
    # Express the midtone offset as stops, scaling the normalized deviation by the
    # negative's density range. Positive = brighter (high-key). Approximate.
    ev = dev * float(norm_density_range) / _LOG10_2
    return StatRow("Exposure", f"{ev:+.1f} EV")


def _contrast_row(slope: Optional[float], effective_range: Optional[float]) -> StatRow:
    if slope is None:
        return StatRow("Contrast", _EMPTY)
    from negpy.features.exposure.logic import slope_to_grade

    # Effective slope expressed on the ISO R paper scale (same as the Grade
    # slider), rounded to a tidy R step.
    r = slope_to_grade(float(slope), effective_range)
    return StatRow("Contrast", f"R{int(round(r / 5.0) * 5)}")


# Dichroic-head complements for negative CC values: less cyan = red, etc.
_CC_LETTERS = (("C", "R"), ("M", "G"), ("Y", "B"))


def format_cc(wb_cmy: Tuple[float, float, float]) -> str:
    """WB sliders as Kodak CC filtration (1 CC = 0.01 density; slider 1.0 = 20cc)."""
    cc_per_unit = float(EXPOSURE_CONSTANTS["cmy_max_density"]) * 100.0
    parts = []
    for value, (pos, neg) in zip(wb_cmy, _CC_LETTERS):
        cc = int(round(float(value) * cc_per_unit))
        if cc:
            parts.append(f"{abs(cc)}{pos if cc > 0 else neg}")
    return " ".join(parts)


def print_exposure_stops(density: float, norm_density_range: float) -> float:
    """Density slider offset expressed as print-exposure stops (+ = darker print)."""
    delta_log_e = (float(density) - 1.0) * float(EXPOSURE_CONSTANTS["density_multiplier"]) * float(norm_density_range)
    return delta_log_e / _LOG10_2


def _print_row(
    density: Optional[float],
    wb_cmy: Optional[Tuple[float, float, float]],
    norm_density_range: Optional[float],
) -> StatRow:
    if density is None or wb_cmy is None:
        return StatRow("Print", _EMPTY)
    parts = []
    if norm_density_range:
        parts.append(f"{print_exposure_stops(density, norm_density_range):+.2f} stop")
    cc = format_cc(wb_cmy)
    if cc:
        parts.append(cc)
    if not parts:
        return StatRow("Print", _EMPTY)
    return StatRow("Print", " · ".join(parts))


def _clipping_row(clip_low: Optional[float], clip_high: Optional[float]) -> StatRow:
    if clip_low is None or clip_high is None:
        return StatRow("Clipping", _EMPTY)
    lo, hi = float(clip_low), float(clip_high)
    warn = lo > _CLIP_WARN or hi > _CLIP_WARN
    return StatRow("Clipping", f"Sh {lo * 100:.1f}% · Hi {hi * 100:.1f}%", warn=warn)


def _scan_clip_row(scan_clip: Optional[Tuple[float, float, float]]) -> StatRow:
    if scan_clip is None:
        return StatRow("Scan clip", _EMPTY)
    r, g, b = (float(v) for v in scan_clip)
    warn = max(r, g, b) > float(EXPOSURE_CONSTANTS["scan_clip_warn"])
    return StatRow("Scan clip", f"R {r * 100:.1f}% · G {g * 100:.1f}% · B {b * 100:.1f}%", warn=warn)


def negative_statistics(
    norm_density_range: Optional[float],
    metered_anchor: Optional[float],
    slope: Optional[float],
    clip_low: Optional[float],
    clip_high: Optional[float],
    effective_range: Optional[float] = None,
    density: Optional[float] = None,
    wb_cmy: Optional[Tuple[float, float, float]] = None,
    scan_clip: Optional[Tuple[float, float, float]] = None,
) -> List[StatRow]:
    """
    Negative read-out, in display order. `effective_range` is the density
    range that produced `slope` (after Auto Grade blending) — used to express the
    contrast on the ISO R scale; falls back to `norm_density_range`.
    `density`/`wb_cmy` are the print-exposure settings shown in darkroom units;
    `scan_clip` is the per-channel sensor-white clipped fraction of the source scan.
    """
    return [
        _density_row(norm_density_range),
        _exposure_row(metered_anchor, norm_density_range),
        _contrast_row(slope, effective_range if effective_range is not None else norm_density_range),
        _print_row(density, wb_cmy, norm_density_range),
        _clipping_row(clip_low, clip_high),
        _scan_clip_row(scan_clip),
    ]
