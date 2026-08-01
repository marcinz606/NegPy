"""Zone placement: solve Print Density (and ISO-R Grade) so pinned tones print on
their target zones. Inverts the chart's own forward model (curve_params_from_metrics
-> print_curve -> print_curve_output -> zone_of_encoded) by bisection — the composite
is monotone in density and grade but not analytically invertible."""

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from negpy.features.exposure.analysis import zone_of_encoded
from negpy.features.exposure.logic import curve_params_from_metrics, print_curve, print_curve_output

DENSITY_RANGE = (0.0, 2.0)  # mirrors the Print Density slider

# Half the sliders' precision: bisecting finer than the 0.01 / 1 R rounding is wasted work.
_DENSITY_RESOLUTION = 0.005
_GRADE_RESOLUTION = 0.5
_CLAMP_TOL = 1e-6
_DEGENERATE_VAL_GAP = 1e-3


@dataclass(frozen=True)
class ZonePin:
    """One probed spot: content-normalized position, frozen normalized-log sample,
    and the zone the user wants it to print on."""

    nx: float
    ny: float
    val_rgb: Tuple[float, float, float]
    val_luma: float
    target_zone: float
    label: str = ""
    # Set once a zone is asked for: a dragged pin keeps it instead of re-reading.
    retargeted: bool = False


@dataclass(frozen=True)
class PlacementSolution:
    fields: Dict[str, Any]  # ExposureConfig replacements
    achieved: Tuple[float, ...]  # zone per pin at the rounded fields, pin order
    clamped: bool


def predicted_zone(exposure: Any, process_mode: Optional[str], metrics: Any, val_luma: float) -> float:
    """Zone the achromatic print curve puts `val_luma` on under `exposure`."""
    slopes, pivots, curvs = curve_params_from_metrics(exposure, process_mode, metrics)
    curve = print_curve(exposure, slopes[1], pivots[1], process_mode, curvature=curvs[1])
    enc = float(print_curve_output(curve, [val_luma])[0])
    return float(zone_of_encoded(enc))


def solve_placement(
    exposure: Any,
    process_mode: Optional[str],
    metrics: Any,
    pins: Sequence[ZonePin],
) -> Optional[PlacementSolution]:
    """1 pin: Print Density. 2 pins: Print Density + Grade. Solved against the
    post-Apply config (autos off), clamped to the slider ranges, rounded to the
    sliders' precision with achieved zones recomputed at the rounded values."""
    if len(pins) == 1:
        candidate = replace(exposure, auto_exposure=False)
        density, clamped = _solve_density(candidate, process_mode, metrics, pins[0])
        fields: Dict[str, Any] = {"density": round(density, 2), "auto_exposure": False}
    elif len(pins) == 2:
        dark, light = sorted(pins, key=lambda p: p.val_luma, reverse=True)
        if abs(dark.val_luma - light.val_luma) < _DEGENERATE_VAL_GAP:
            return None
        candidate = replace(exposure, auto_exposure=False, auto_normalize_contrast=False)
        grade, density, clamped = _solve_grade_and_density(candidate, process_mode, metrics, dark, light)
        fields = {
            "density": round(density, 2),
            "grade": float(round(grade)),
            "auto_exposure": False,
            "auto_normalize_contrast": False,
        }
    else:
        return None
    applied = replace(exposure, **fields)
    achieved = tuple(predicted_zone(applied, process_mode, metrics, p.val_luma) for p in pins)
    return PlacementSolution(fields, achieved, clamped)


def _solve_density(candidate: Any, process_mode: Optional[str], metrics: Any, pin: ZonePin) -> Tuple[float, bool]:
    def residual(density: float) -> float:
        placed = replace(candidate, density=density)
        return predicted_zone(placed, process_mode, metrics, pin.val_luma) - pin.target_zone

    return _bisect_decreasing(residual, DENSITY_RANGE[0], DENSITY_RANGE[1], _DENSITY_RESOLUTION)


def _solve_grade_and_density(
    candidate: Any,
    process_mode: Optional[str],
    metrics: Any,
    dark: ZonePin,
    light: ZonePin,
) -> Tuple[float, float, bool]:
    """Outer bisection on grade against the light pin, inner density solve pinning
    the dark pin at every candidate grade. Softer (higher R) lowers the light
    pin's zone with the dark pin held, so the residual is decreasing in R."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    r_lo, r_hi = float(EXPOSURE_CONSTANTS["iso_r_min"]), float(EXPOSURE_CONSTANTS["iso_r_max"])

    def light_residual(grade: float) -> float:
        graded = replace(candidate, grade=grade)
        density, _ = _solve_density(graded, process_mode, metrics, dark)
        placed = replace(graded, density=density)
        return predicted_zone(placed, process_mode, metrics, light.val_luma) - light.target_zone

    grade, grade_clamped = _bisect_decreasing(light_residual, r_lo, r_hi, _GRADE_RESOLUTION)
    density, density_clamped = _solve_density(replace(candidate, grade=grade), process_mode, metrics, dark)
    return grade, density, grade_clamped or density_clamped


def _bisect_decreasing(f: Callable[[float], float], lo: float, hi: float, resolution: float) -> Tuple[float, bool]:
    """Root of a decreasing f on [lo, hi] to within `resolution`; (closest
    endpoint, True) when the target sits outside the bracket."""
    f_lo = f(lo)
    if f_lo <= 0.0:
        return lo, f_lo < -_CLAMP_TOL
    f_hi = f(hi)
    if f_hi >= 0.0:
        return hi, f_hi > _CLAMP_TOL
    while hi - lo > resolution:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False
