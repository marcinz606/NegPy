"""
Negative statistics for the Analysis panel.

Pure presentation logic: turns the metrics already measured every render
(density range, metered anchor, histogram clipping) into read-out rows.
No pipeline math here — just formatting.

Densities are *relative* (normalized decode), meaningful for contrast and across
a roll, not absolute scanner D.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from negpy.features.exposure.models import EXPOSURE_CONSTANTS

_CLIP_WARN = 0.01  # >1% of a channel clipped flags a warning

# Negative character: measured range vs default_grade_range().
_DIAG_FLAT = 0.80
_DIAG_CONTRASTY = 1.25

_EMPTY = "—"


@dataclass(frozen=True)
class StatRow:
    """One labelled read-out row."""

    name: str
    value: str
    warn: bool = False


_LOG10_2 = 0.30103  # one stop in log10-density


def _negative_row(norm_density_range: Optional[float]) -> StatRow:
    if not norm_density_range:
        return StatRow("Negative", _EMPTY)
    from negpy.features.exposure.logic import default_grade_range

    ratio = float(norm_density_range) / default_grade_range()
    if ratio < _DIAG_FLAT:
        character = "flat (≈N−1)"
    elif ratio > _DIAG_CONTRASTY:
        character = "contrasty (≈N+1)"
    else:
        character = "normal"
    return StatRow("Negative", f"{float(norm_density_range):.2f} · {character}")


def _exposure_row(metered_anchor: Optional[float], norm_density_range: Optional[float]) -> StatRow:
    if metered_anchor is None or not norm_density_range:
        return StatRow("Exposure", _EMPTY)
    dev = float(metered_anchor) - float(EXPOSURE_CONSTANTS["assumed_anchor"])
    # Express the midtone offset as stops, scaling the normalized deviation by the
    # negative's density range. Positive = brighter (high-key). Approximate.
    ev = dev * float(norm_density_range) / _LOG10_2
    return StatRow("Exposure", f"{ev:+.1f} EV")


def _clipping_row(clip_low: Optional[float], clip_high: Optional[float]) -> StatRow:
    if clip_low is None or clip_high is None:
        return StatRow("Clipping", _EMPTY)
    lo, hi = float(clip_low), float(clip_high)
    warn = lo > _CLIP_WARN or hi > _CLIP_WARN
    return StatRow("Clipping", f"Sh {lo * 100:.1f}% · Hi {hi * 100:.1f}%", warn=warn)


def _scan_clip_row(scan_clip: Optional[Tuple[float, float, float]]) -> Optional[StatRow]:
    """Returns None when clean — the row only appears when it warns."""
    if scan_clip is None:
        return None
    r, g, b = (float(v) for v in scan_clip)
    if max(r, g, b) <= float(EXPOSURE_CONSTANTS["scan_clip_warn"]):
        return None
    return StatRow("Scan clip", f"R {r * 100:.1f}% · G {g * 100:.1f}% · B {b * 100:.1f}%", warn=True)


def negative_statistics(
    norm_density_range: Optional[float],
    metered_anchor: Optional[float],
    clip_low: Optional[float],
    clip_high: Optional[float],
    scan_clip: Optional[Tuple[float, float, float]] = None,
) -> List[StatRow]:
    """
    Negative read-out, in display order. `scan_clip` is the per-channel
    sensor-white clipped fraction of the source scan; its row appears only
    when it warns.
    """
    rows = [
        _negative_row(norm_density_range),
        _exposure_row(metered_anchor, norm_density_range),
        _clipping_row(clip_low, clip_high),
    ]
    scan_row = _scan_clip_row(scan_clip)
    if scan_row is not None:
        rows.append(scan_row)
    return rows
