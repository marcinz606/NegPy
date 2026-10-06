"""Auto Density and Auto Grade on their sliders.

The stored value stays an offset on the meter; the slider shows meter + offset in the
slider's own units, and an edit converts back. A toggle flips only the flag, so the
slider gains or drops the meter's share.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

from negpy.features.exposure.logic import (
    auto_base_slope,
    auto_highlight_from_metrics,
    effective_grade_range,
    grade_to_slope,
    slope_to_grade,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.features.exposure.papers import effective_constants, effective_paper_profile

# The offsets at which the meter alone decides.
NEUTRAL: Dict[str, float] = {f: float(getattr(ExposureConfig(), f)) for f in ("density", "grade", "highlight_density")}


def _paper(exposure: ExposureConfig, process_mode: Optional[str]) -> Tuple[Any, float]:
    profile = effective_paper_profile(exposure.paper_profile, process_mode)
    return profile, (profile.d_min if exposure.paper_dmin else 0.0)


def _density_shift(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[float]:
    """Print Density the metered anchor is worth: compute_pivot's pivot is linear in both."""
    anchor = metrics.get("metered_anchor")
    if not exposure.auto_exposure or anchor is None:
        return None
    c = effective_constants(_paper(exposure, process_mode)[0])
    return (float(anchor) - float(c["assumed_anchor"])) / float(c["density_multiplier"])


def _auto_slope(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[float]:
    lum_range, textural = metrics.get("norm_density_range"), metrics.get("textural_range")
    if not exposure.auto_normalize_contrast or lum_range is None or textural is None:
        return None
    profile, d_min = _paper(exposure, process_mode)
    anchor = metrics.get("metered_anchor") if exposure.auto_exposure else None
    return auto_base_slope(exposure.grade, True, lum_range, textural, metrics.get("shadow_point"), anchor, d_min, profile)


def print_shown_values(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Manual-equivalent value of every slider an auto drives, for the autos that are on
    and metered. Grade is the ISO R that prints the auto's slope on the frame's own range."""
    shown: Dict[str, float] = {}
    shift = _density_shift(exposure, process_mode, metrics)
    if shift is not None:
        shown["density"] = exposure.density - shift
    slope = _auto_slope(exposure, process_mode, metrics)
    if slope is not None:
        shown["grade"] = slope_to_grade(slope, metrics["norm_density_range"])
    if exposure.auto_normalize_contrast and metrics.get("highlight_point") is not None:
        shown["highlight_density"] = exposure.highlight_density + auto_highlight_from_metrics(exposure, process_mode, metrics)
    return shown


def print_stored_value(
    exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any], field: str, shown: float
) -> float:
    """Inverse of print_shown_values for one field. Where Shadow Reach floors the slope,
    a softer grade is stored but does not print."""
    if field == "density":
        shift = _density_shift(exposure, process_mode, metrics)
        return shown if shift is None else shown + shift
    if field == "grade":
        if _auto_slope(exposure, process_mode, metrics) is None:
            return shown
        lum_range = metrics["norm_density_range"]
        r_eff = effective_grade_range(True, lum_range, metrics["textural_range"])
        return slope_to_grade(grade_to_slope(shown, lum_range), r_eff)
    if field == "highlight_density" and exposure.auto_normalize_contrast and metrics.get("highlight_point") is not None:
        return shown - auto_highlight_from_metrics(exposure, process_mode, metrics)
    return shown
