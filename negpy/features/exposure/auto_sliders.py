"""Auto Density and Auto Grade on their sliders.

The stored value is an offset on the meter; the slider shows meter + offset, and an edit
converts back. A toggle flips only the flag.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

from negpy.features.exposure.logic import (
    auto_base_slope,
    auto_highlight_from_metrics,
    auto_shadow_from_metrics,
    compute_pivot,
    curve_params_from_metrics,
    effective_grade_range,
    effective_midtone_gamma,
    grade_to_slope,
    slope_to_grade,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.features.exposure.papers import effective_constants, effective_paper_profile, slope_gain

# The offsets at which the meter alone decides.
NEUTRAL: Dict[str, float] = {f: float(getattr(ExposureConfig(), f)) for f in ("density", "grade", "shadow_density", "highlight_density")}


def _paper(exposure: ExposureConfig, process_mode: Optional[str]) -> Tuple[Any, float]:
    profile = effective_paper_profile(exposure.paper_profile, process_mode)
    return profile, (profile.d_min if exposure.paper_dmin else 0.0)


def _density_shift(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[float]:
    """Print Density the metered anchor is worth: the gap between the metered and the
    reference pivot at green's slope, as compute_pivot is linear in density."""
    anchor = metrics.get("metered_anchor")
    if not exposure.auto_exposure or anchor is None:
        return None
    profile, d_min = _paper(exposure, process_mode)
    slope = curve_params_from_metrics(exposure, process_mode, metrics)[0][1]
    mg = effective_midtone_gamma(profile, exposure.midtone_gamma)
    gap = compute_pivot(slope, 1.0, d_min, float(anchor), profile, mg) - compute_pivot(slope, 1.0, d_min, paper=profile)
    return gap / float(effective_constants(profile)["density_multiplier"])


def _auto_slope(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[float]:
    lum_range, textural = metrics.get("norm_density_range"), metrics.get("textural_range")
    if not exposure.auto_normalize_contrast or lum_range is None or textural is None:
        return None
    profile, d_min = _paper(exposure, process_mode)
    anchor = metrics.get("metered_anchor") if exposure.auto_exposure else None
    mg = effective_midtone_gamma(profile, exposure.midtone_gamma)
    return auto_base_slope(exposure.grade, True, lum_range, textural, metrics.get("shadow_point"), anchor, d_min, profile, mg)


def print_shown_values(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Manual-equivalent value of every slider an auto drives, for the autos that are on
    and metered. Grade is the ISO R that prints the auto's slope on the frame's own range."""
    shown: Dict[str, float] = {}
    shift = _density_shift(exposure, process_mode, metrics)
    if shift is not None:
        shown["density"] = exposure.density - shift
    slope = _auto_slope(exposure, process_mode, metrics)
    if slope is not None:
        shown["grade"] = slope_to_grade(slope, metrics["norm_density_range"], slope_gain(_paper(exposure, process_mode)[0]))
    if exposure.auto_normalize_contrast and metrics.get("highlight_point") is not None:
        shown["highlight_density"] = exposure.highlight_density + auto_highlight_from_metrics(exposure, process_mode, metrics)
    if exposure.auto_normalize_contrast and metrics.get("shadow_point") is not None:
        shown["shadow_density"] = exposure.shadow_density + auto_shadow_from_metrics(exposure, process_mode, metrics)
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
        gain = slope_gain(_paper(exposure, process_mode)[0])
        return slope_to_grade(grade_to_slope(shown, lum_range, gain), r_eff, gain)
    if field == "highlight_density" and exposure.auto_normalize_contrast and metrics.get("highlight_point") is not None:
        return shown - auto_highlight_from_metrics(exposure, process_mode, metrics)
    if field == "shadow_density" and exposure.auto_normalize_contrast and metrics.get("shadow_point") is not None:
        return shown - auto_shadow_from_metrics(exposure, process_mode, metrics)
    return shown
