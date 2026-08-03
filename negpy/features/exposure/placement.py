"""Zone placement: solve Print Density (and ISO-R Grade, and one knee control) so pinned
tones print on their target zones. Inverts the chart's own forward model
(curve_params_from_metrics -> print_curve -> print_curve_output -> zone_of_encoded) by
bisection — the composite is monotone in each solved control but not analytically
invertible."""

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
class KneeCandidate:
    """One control a third pin can be solved on: its field, slider bounds, bisection
    resolution and commit rounding. Bounds mirror sidebar/tone.py."""

    field: str
    lo: float
    hi: float
    resolution: float
    ndigits: int


KNEE_CANDIDATES = (
    KneeCandidate("shadow_grade", -50.0, 50.0, _GRADE_RESOLUTION, 1),
    KneeCandidate("highlight_grade", -50.0, 50.0, _GRADE_RESOLUTION, 1),
    KneeCandidate("midtone_gamma", -0.5, 0.5, _DENSITY_RESOLUTION, 2),
)

MAX_PINS = 3  # density, grade, one knee control — a fourth pin has nothing left to solve

_KNEE_PROBE = 0.2  # fraction of a candidate's range perturbed to measure its purchase
_KNEE_SENSITIVITY_FLOOR = 0.05  # zones; below this the control cannot place the pin
_KNEE_PASSES = 3
_KNEE_SETTLED = 0.02  # zone movement between passes that ends the iteration
_ON_TARGET_TOL = 1.0 / 6.0  # a third of a stepper step: on target, and off-target honestly


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
    knee: str = ""  # the knee field a third pin was solved on; "" when none was needed


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
    """1 pin: Print Density. 2 pins: Print Density + Grade. 3 pins: those two from the
    outer tones plus one knee control for the middle one. Solved against the post-Apply
    config (autos off), clamped to the slider ranges, rounded to the sliders' precision
    with achieved zones recomputed at the rounded values."""
    knee = ""
    if len(pins) == 1:
        candidate = replace(exposure, auto_exposure=False)
        density, clamped = _solve_density(candidate, process_mode, metrics, pins[0])
        fields: Dict[str, Any] = {"density": round(density, 2), "auto_exposure": False}
    elif len(pins) in (2, 3):
        ordered = sorted(pins, key=lambda p: p.val_luma, reverse=True)
        dark, light = ordered[0], ordered[-1]
        if abs(dark.val_luma - light.val_luma) < _DEGENERATE_VAL_GAP:
            return None
        candidate = replace(exposure, auto_exposure=False, auto_normalize_contrast=False)
        if len(pins) == 3:
            candidate, grade, density, knee, clamped = _solve_with_knee(candidate, process_mode, metrics, dark, light, ordered[1])
        else:
            grade, density, clamped = _solve_grade_and_density(candidate, process_mode, metrics, dark, light)
        fields = {
            "density": round(density, 2),
            "grade": float(round(grade)),
            "auto_exposure": False,
            "auto_normalize_contrast": False,
        }
        if knee:
            digits = next(c.ndigits for c in KNEE_CANDIDATES if c.field == knee)
            fields[knee] = round(float(getattr(candidate, knee)), digits)
    else:
        return None
    applied = replace(exposure, **fields)
    achieved = tuple(predicted_zone(applied, process_mode, metrics, p.val_luma) for p in pins)
    # The knee iteration can settle short of every ask: say so rather than report the ask.
    if knee and any(abs(a - p.target_zone) > _ON_TARGET_TOL for a, p in zip(achieved, pins)):
        clamped = True
    return PlacementSolution(fields, achieved, clamped, knee)


def _solve_with_knee(
    candidate: Any,
    process_mode: Optional[str],
    metrics: Any,
    dark: ZonePin,
    light: ZonePin,
    mid: ZonePin,
) -> Tuple[Any, float, float, str, bool]:
    """Place the outer tones, then the middle one on the knee control it can actually
    move, alternating until it settles. Iterated rather than nested: a third bisection
    inside the 2-pin solve would multiply its cost by its own iteration count."""
    grade, density, clamped = _solve_grade_and_density(candidate, process_mode, metrics, dark, light)
    placed = replace(candidate, grade=grade, density=density)
    landed = predicted_zone(placed, process_mode, metrics, mid.val_luma)
    if abs(landed - mid.target_zone) <= _ON_TARGET_TOL:
        return candidate, grade, density, "", clamped

    knee = _pick_knee(placed, process_mode, metrics, mid)
    if knee is None:
        return candidate, grade, density, "", clamped

    previous = None
    for _ in range(_KNEE_PASSES):
        value, knee_clamped = _solve_knee_field(replace(candidate, grade=grade, density=density), process_mode, metrics, mid, knee)
        candidate = replace(candidate, **{knee.field: value})
        grade, density, clamped = _solve_grade_and_density(candidate, process_mode, metrics, dark, light)
        landed = predicted_zone(replace(candidate, grade=grade, density=density), process_mode, metrics, mid.val_luma)
        if previous is not None and abs(landed - previous) < _KNEE_SETTLED:
            break
        previous = landed
    return candidate, grade, density, knee.field, clamped or knee_clamped


def _pick_knee(placed: Any, process_mode: Optional[str], metrics: Any, pin: ZonePin) -> Optional[KneeCandidate]:
    """The control that moves this pin most, or None when none of them can.

    Measured, not inferred from the pin's zone: each control's weight is an expit window
    on *print density*, and both zone grades also lose purchase at their own centre —
    a zone-number rule would have to mirror those constants and the working OETF.
    """
    here = predicted_zone(placed, process_mode, metrics, pin.val_luma)
    best: Optional[KneeCandidate] = None
    best_delta = _KNEE_SENSITIVITY_FLOOR
    for cand in KNEE_CANDIDATES:
        current = float(getattr(placed, cand.field))
        step = _KNEE_PROBE * (cand.hi - cand.lo)
        probe = current + step if current + step <= cand.hi else current - step
        moved = predicted_zone(replace(placed, **{cand.field: probe}), process_mode, metrics, pin.val_luma)
        delta = abs(moved - here)
        if delta > best_delta:
            best, best_delta = cand, delta
    return best


def _solve_knee_field(
    base: Any,
    process_mode: Optional[str],
    metrics: Any,
    pin: ZonePin,
    knee: KneeCandidate,
) -> Tuple[float, bool]:
    def residual(value: float) -> float:
        placed = replace(base, **{knee.field: value})
        return predicted_zone(placed, process_mode, metrics, pin.val_luma) - pin.target_zone

    return _bisect_monotone(residual, knee.lo, knee.hi, knee.resolution)


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


def _bisect_monotone(f: Callable[[float], float], lo: float, hi: float, resolution: float) -> Tuple[float, bool]:
    """Root of a monotone f, direction read off the endpoints. The knee controls carry a
    (v - centre) factor that flips sign either side of their zone centre, so neither
    direction can be assumed the way density and grade can."""
    if f(lo) >= f(hi):
        return _bisect_decreasing(f, lo, hi, resolution)
    return _bisect_decreasing(lambda v: -f(v), lo, hi, resolution)


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
