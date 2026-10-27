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
_REPAIR_WARN = 0.05  # a route rewriting >5% of the scan is repairing the picture, not the dust
_GAMUT_WARN = 0.02  # >2% unprintable is worth a look before you print

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
    detail: str = ""  # lines the row's hover adds under its fixed explanation
    per_channel: bool = False  # the value is R · G · B, which the read-out colors per channel


_LOG10_2 = 0.30103  # one stop in log10-density


_NO_BASE = (
    "Dmax, Dmed and Dmin need a film base. Click the clear rebate with the Film Base tool on the "
    "Metering card; without one, the base is read from the rebate when the scan shows a film edge."
)
_SOURCE = {
    "picked": "Above the film base set with the Film Base tool",
    "read": "Above the clear base read from the rebate",
}
_READ_CAVEAT = " The rebate reader can take bare light or a neighbor frame for the base; the Film Base tool pins it."


def _density_detail(density_above_base: Optional[Tuple[float, float, float]], base_source: str) -> str:
    if density_above_base is None:
        return _NO_BASE
    dmin, dmed, dmax = (float(v) for v in density_above_base)
    return (
        f"{_SOURCE.get(base_source, _SOURCE['read'])}: Dmax {dmax:.2f} · Dmed {dmed:.2f} · Dmin {dmin:.2f}. "
        "Dmin and Dmax are the bounds the frame is normalized with, before the White and Black Point trims "
        "(the roll's under Roll Analysis); Dmed is the metered median. A Dmin below zero means the metered area "
        "holds something thinner than the base: bare light through a perforation or past the film's edge. "
        "An estimate from the scan: flare reads dense areas low." + (_READ_CAVEAT if base_source != "picked" else "")
    )


def _negative_row(
    norm_density_range: Optional[float], density_above_base: Optional[Tuple[float, float, float]] = None, base_source: str = ""
) -> StatRow:
    detail = _density_detail(density_above_base, base_source)
    if not norm_density_range:
        return StatRow("Negative", _EMPTY, detail=detail)
    from negpy.features.exposure.logic import default_grade_range

    ratio = float(norm_density_range) / default_grade_range()
    word = "Flat" if ratio < _DIAG_FLAT else "Contrasty" if ratio > _DIAG_CONTRASTY else "Normal"
    # The value is only the range, so the read-out fits a narrow panel; the hover names it.
    return StatRow("Negative", f"Δ{float(norm_density_range):.2f}", detail=f"{word} negative. {detail}")


def _exposure_row(metered_anchor: Optional[float], norm_density_range: Optional[float]) -> StatRow:
    if metered_anchor is None or not norm_density_range:
        return StatRow("Exposure", _EMPTY)
    dev = float(metered_anchor) - float(EXPOSURE_CONSTANTS["assumed_anchor"])
    # Express the midtone offset as stops, scaling the normalized deviation by the negative's
    # density range. Positive is brighter (high-key). Approximate.
    ev = dev * float(norm_density_range) / _LOG10_2
    return StatRow("Exposure", f"{ev:+.1f} EV")


def clipping_row(clip_low: Optional[float], clip_high: Optional[float], compact: bool = False) -> StatRow:
    """The Clipping row: worst-channel shares at the black and white ends, warning above 1%.
    `compact` drops the Sh and Hi names into the hover, for the Analysis read-out."""
    if clip_low is None or clip_high is None:
        return StatRow("Clipping", _EMPTY)
    lo, hi = float(clip_low), float(clip_high)
    warn = lo > _CLIP_WARN or hi > _CLIP_WARN
    if compact:
        return StatRow(
            "Clipping", f"{lo * 100:.1f} · {hi * 100:.1f}%", warn=warn, detail=f"Shadows {lo * 100:.1f}% · highlights {hi * 100:.1f}%"
        )
    return StatRow("Clipping", f"Sh {lo * 100:.1f}% · Hi {hi * 100:.1f}%", warn=warn)


def _scan_clip_row(scan_clip: Optional[Tuple[float, float, float]]) -> StatRow:
    if scan_clip is None:
        return StatRow("Scan clip", _EMPTY)
    r, g, b = (float(v) for v in scan_clip)
    warn = max(r, g, b) > float(EXPOSURE_CONSTANTS["scan_clip_warn"])
    return StatRow(
        "Scan clip",
        f"{r * 100:.1f} · {g * 100:.1f} · {b * 100:.1f}%",
        warn=warn,
        detail=f"Red {r * 100:.1f}% · green {g * 100:.1f}% · blue {b * 100:.1f}%",
        per_channel=True,
    )


def _repair_row(repair: Optional[Tuple[float, float, float]]) -> StatRow:
    if repair is None:
        return StatRow("Repair", _EMPTY)
    parts = [f"{label} {frac * 100:.2f}%" for label, frac in zip(("IR", "dust", "painted"), repair) if frac > 0.0]
    if not parts:
        return StatRow("Repair", "none")
    return StatRow("Repair", " · ".join(parts), warn=max(repair) > _REPAIR_WARN)


def _gamut_row(gamut: Optional[float]) -> StatRow:
    """Blank when nothing is being proofed: with no output profile there is no gamut to be
    outside of."""
    if gamut is None:
        return StatRow("Gamut", _EMPTY)
    return StatRow("Gamut", f"{gamut * 100:.1f}%", warn=gamut > _GAMUT_WARN)


def negative_statistics(
    norm_density_range: Optional[float],
    metered_anchor: Optional[float],
    clip_low: Optional[float],
    clip_high: Optional[float],
    scan_clip: Optional[Tuple[float, float, float]] = None,
    repair: Optional[Tuple[float, float, float]] = None,
    gamut: Optional[float] = None,
    density_above_base: Optional[Tuple[float, float, float]] = None,
    base_source: str = "",
) -> List[StatRow]:
    """
    Negative read-out, in display order. Every row is always present, so the
    panel height stays fixed. `scan_clip` is the per-channel sensor-white clipped
    fraction of the source scan. `repair` is the (IR, dust, painted) share each
    repair route rewrote. `gamut` is the share the proofed output profile cannot print.
    `density_above_base` is (Dmin, Dmed, Dmax) above the clear base, for the Negative row's hover;
    `base_source` says whether that base was "picked" with the Film Base tool or "read" from the rebate.
    """
    return [
        _negative_row(norm_density_range, density_above_base, base_source),
        _exposure_row(metered_anchor, norm_density_range),
        clipping_row(clip_low, clip_high, compact=True),
        _scan_clip_row(scan_clip),
        _repair_row(repair),
        _gamut_row(gamut),
    ]
