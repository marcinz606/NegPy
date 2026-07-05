from typing import Any, Optional, Tuple

import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.papers import PaperProfile, effective_constants, resolve_dye_matrix
from negpy.kernel.image.validation import ensure_image


def _expit(x: Any) -> Any:
    """Numpy implementation of the logistic sigmoid function (scipy.special.expit fallback).

    expit(x) = exp(-logaddexp(0, -x)) — exact and overflow-free for any x.
    """
    return np.exp(-np.logaddexp(0.0, -x))


@njit(inline="always")
def _fast_sigmoid(x: float) -> float:
    """
    Fast implementation of the logistic sigmoid function.
    expit(x) = 1 / (1 + exp(-x))
    """
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    else:
        z = np.exp(x)
        return float(z / (1.0 + z))


@njit(inline="always")
def _softplus(x: float) -> float:
    """
    Numerically stable softplus: log(1 + exp(x)). Antiderivative of the sigmoid.
    """
    if x > 0:
        return float(x + np.log1p(np.exp(-x)))
    return float(np.log1p(np.exp(x)))


def _inv_softplus_np(y: Any) -> Any:
    """Inverse of softplus: log(exp(y) - 1), stable for y > 0 (pivot solve)."""
    return np.where(y > 20.0, y, np.log(np.expm1(np.maximum(y, 1e-12))))


@njit(cache=True, fastmath=True)
def _apply_print_curve_kernel(
    img: np.ndarray,
    pivots: np.ndarray,
    slopes: np.ndarray,
    curvatures: np.ndarray,
    toe: float,
    shoulder: float,
    toe_width: float,
    shoulder_width: float,
    cmy_offsets: np.ndarray,
    shadow_cmy: np.ndarray,
    highlight_cmy: np.ndarray,
    d_min_rgb: np.ndarray,
    d_max: float,
    a_toe_base: float,
    a_sh_base: float,
    width_ref: float,
    toe_height: float,
    sh_height: float,
    zone_center: float,
    v_star: float,
    midtone_gamma: float,
    gamma_width: float,
    dye_mix: np.ndarray,
    use_dye_mix: bool,
    ev_map: np.ndarray,
    ev_scale: np.ndarray,
    use_ev: bool,
    flare: float = 0.0,
    surround_gamma: float = 1.0,
) -> np.ndarray:
    """
    Asymmetric H&D print curve: a straight line of slope `slope` through the
    exposure pivot, smoothly bounded above by the toe (shadows -> paper black
    d_max) and below by the shoulder (highlights -> paper white d_min). Toe and
    shoulder are independent softplus bounds, so the `toe` slider shapes only
    shadows and `shoulder` only highlights (film/print convention). `toe`/`shoulder`
    arrive pre-scaled by toe_shoulder_strength.

    d_min_rgb: per-channel paper-white floor (base+fog incl. tint). dye_mix:
    dye coupling above that floor (D_rgb = M · D_dye) when use_dye_mix is set.
    ev_map/ev_scale: per-pixel dodge/burn print-exposure offset (EV stops ×
    normalized-space stop size) when use_ev is set; same domain as cmy_offsets.

    Output is linear reflectance (transmittance = 10^-D); the working-space OETF is
    applied at the engine output, not here.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6

    # Roll-off sharpness from width (larger width = gentler); slider sets height.
    # toe -> shadow (upper / paper-black) bound; shoulder -> highlight (lower /
    # paper-white) bound. a_toe_base/a_sh_base carry the shadow/highlight sharpness.
    a_hl = a_sh_base * width_ref / max(shoulder_width, eps)
    a_sh = a_toe_base * width_ref / max(toe_width, eps)
    if toe >= 0.0:
        d_max_base = d_max - toe * toe_height
    else:
        # Negative toe: tighten the shadow roll-off (sharper knee) rather than
        # extending d_max_eff beyond paper black (perceptually near-zero effect).
        d_max_base = d_max
        a_sh = a_sh * (1.0 - toe * 4.0)

    d_min_eff = np.empty(3, dtype=np.float64)
    d_max_eff = np.empty(3, dtype=np.float64)
    flare_white = np.empty(3, dtype=np.float64)
    for ch in range(3):
        dmn = d_min_rgb[ch] + shoulder * sh_height
        if dmn < 0.0:
            dmn = 0.0
        dmx = d_max_base
        if dmx < dmn + 0.1:
            dmx = dmn + 0.1
        d_min_eff[ch] = dmn
        d_max_eff[ch] = dmx
        flare_white[ch] = 10.0 ** (-d_min_rgb[ch])

    dens = np.empty(3, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                val = img[y, x, ch] + cmy_offsets[ch]
                if use_ev:
                    val = val + ev_map[y, x] * ev_scale[ch]
                # Quadratic per-channel core (curvature 0 -> the original straight line).
                v = slopes[ch] * (val - pivots[ch]) + curvatures[ch] * val * val

                # Variable-gamma paper S-curve: extra local gamma at the midtone
                # centre (v_star), easing to zero toward toe/shoulder. Centred on
                # v_star so the reference tone is preserved.
                if midtone_gamma != 0.0:
                    v = v + midtone_gamma * gamma_width * np.tanh((v - v_star) / gamma_width)

                # Regional CMY: shadow weight rises with density, highlight falls.
                w_sh = _fast_sigmoid(3.0 * (v - zone_center))
                w_hi = 1.0 - w_sh
                v = v + shadow_cmy[ch] * w_sh + highlight_cmy[ch] * w_hi

                # Shoulder: smooth lower bound at paper white (highlights).
                v1 = d_min_eff[ch] + _softplus(a_hl * (v - d_min_eff[ch])) / a_hl
                # Toe: smooth upper bound at paper black (shadows).
                dens[ch] = d_max_eff[ch] - _softplus(a_sh * (d_max_eff[ch] - v1)) / a_sh

            if use_dye_mix:
                # Dye unwanted absorptions: mix the densities above paper base.
                e0 = dens[0] - d_min_rgb[0]
                e1 = dens[1] - d_min_rgb[1]
                e2 = dens[2] - d_min_rgb[2]
                dens[0] = d_min_rgb[0] + dye_mix[0, 0] * e0 + dye_mix[0, 1] * e1 + dye_mix[0, 2] * e2
                dens[1] = d_min_rgb[1] + dye_mix[1, 0] * e0 + dye_mix[1, 1] * e1 + dye_mix[1, 2] * e2
                dens[2] = d_min_rgb[2] + dye_mix[2, 0] * e0 + dye_mix[2, 1] * e1 + dye_mix[2, 2] * e2

            for ch in range(3):
                density = dens[ch]
                if surround_gamma != 1.0:
                    density = d_min_rgb[ch] + surround_gamma * (density - d_min_rgb[ch])

                transmittance = 10.0 ** (-density)
                if flare != 0.0:
                    transmittance = (transmittance + flare * flare_white[ch]) / (1.0 + flare)

                final_val = transmittance
                if final_val < 0.0:
                    final_val = 0.0
                elif final_val > 1.0:
                    final_val = 1.0
                res[y, x, ch] = final_val
    return res


class CharacteristicCurve:
    """
    Asymmetric H&D print curve (toe-linear-shoulder) in density space — the NumPy
    mirror of _apply_print_curve_kernel, used by the curve chart so the displayed
    curve matches the render. Returns density (pre-transmittance/encode). Neutral
    (no regional CMY), since the chart shows the achromatic transfer.
    """

    def __init__(
        self,
        contrast: float,
        pivot: float,
        d_min: float = 0.0,
        toe: float = 0.0,
        toe_width: float = 2.5,
        shoulder: float = 0.0,
        shoulder_width: float = 2.5,
        flare: float = 0.0,
        surround_gamma: float = 1.0,
        paper: Optional[PaperProfile] = None,
    ):
        c = effective_constants(paper)
        ts = float(c["toe_shoulder_strength"])
        self.k = float(contrast)
        self.x0 = float(pivot)
        self.d_min = float(d_min)
        self.v_star = _reference_linear_value(d_min, paper)
        self.midtone_gamma = float(c["paper_midtone_gamma"])
        self.gamma_width = float(c["paper_gamma_width"])
        self.d_max = float(c["d_max"])
        self.flare = float(flare)
        self.surround_gamma = float(surround_gamma)
        wr = float(c["toeshoulder_width_ref"])
        # toe -> shadow (upper) bound; shoulder -> highlight (lower) bound.
        self.a_hl = float(c["shoulder_sharpness_base"]) * wr / max(shoulder_width, 1e-6)
        a_sh_base = float(c["toe_sharpness_base"]) * wr / max(toe_width, 1e-6)
        self.d_min_eff = max(0.0, self.d_min + shoulder * ts * float(c["shoulder_height"]))
        toe_eff = toe * ts
        if toe_eff >= 0.0:
            self.d_max_eff = self.d_max - toe_eff * float(c["toe_height"])
            self.a_sh = a_sh_base
        else:
            self.d_max_eff = self.d_max
            self.a_sh = a_sh_base * (1.0 - toe_eff * 4.0)
        if self.d_max_eff < self.d_min_eff + 0.1:
            self.d_max_eff = self.d_min_eff + 0.1

    def __call__(self, x: ImageBuffer) -> ImageBuffer:
        v = self.k * (np.asarray(x, dtype=np.float64) - self.x0)
        if self.midtone_gamma != 0.0:
            v = v + self.midtone_gamma * self.gamma_width * np.tanh((v - self.v_star) / self.gamma_width)
        v1 = self.d_min_eff + np.logaddexp(0.0, self.a_hl * (v - self.d_min_eff)) / self.a_hl
        res = self.d_max_eff - np.logaddexp(0.0, self.a_sh * (self.d_max_eff - v1)) / self.a_sh

        if self.surround_gamma != 1.0:
            res = self.d_min + self.surround_gamma * (res - self.d_min)

        if self.flare != 0.0:
            white = 10.0 ** (-self.d_min)
            t = 10.0 ** (-res)
            t = (t + self.flare * white) / (1.0 + self.flare)
            res = -np.log10(np.maximum(t, 1e-12))

        return ensure_image(res)


def paper_dmin_rgb(d_min: float, paper: Optional[PaperProfile]) -> Tuple[float, float, float]:
    """
    Per-channel paper-white floor: d_min plus the paper's base tint (a minimum
    dye density — tints highlights, fades toward d_max). All-zero when d_min is 0.
    """
    if d_min <= 0.0 or paper is None:
        base = max(d_min, 0.0)
        return (base, base, base)
    t = paper.base_tint_cmy
    return (max(d_min + t[0], 0.0), max(d_min + t[1], 0.0), max(d_min + t[2], 0.0))


def apply_characteristic_curve(
    img: ImageBuffer,
    params_r: Tuple[float, float],
    params_g: Tuple[float, float],
    params_b: Tuple[float, float],
    toe: float = 0.0,
    toe_width: float = 2.5,
    shoulder: float = 0.0,
    shoulder_width: float = 2.5,
    shadow_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    d_min: float = 0.0,
    flare: float = 0.0,
    surround_gamma: float = 1.0,
    midtone_gamma: Optional[float] = None,
    curvatures: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    paper: Optional[PaperProfile] = None,
    ev_map: Optional[np.ndarray] = None,
    ev_scale: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ImageBuffer:
    """Applies the asymmetric H&D print curve per channel in log-density space.

    ev_map (H×W, EV stops; positive = dodge) with ev_scale (see local_ev_scale)
    applies per-pixel dodge/burn as print-exposure offsets ahead of the curve."""
    c = effective_constants(paper)
    ts = c["toe_shoulder_strength"]
    if midtone_gamma is None:
        midtone_gamma = float(c["paper_midtone_gamma"])
    v_star = _reference_linear_value(d_min, paper)
    pivots = np.ascontiguousarray(np.array([params_r[0], params_g[0], params_b[0]], dtype=np.float32))
    slopes = np.ascontiguousarray(np.array([params_r[1], params_g[1], params_b[1]], dtype=np.float32))
    curvs = np.ascontiguousarray(np.array(curvatures, dtype=np.float32))
    offsets = np.ascontiguousarray(np.array(cmy_offsets, dtype=np.float32))
    s_cmy = np.ascontiguousarray(np.array(shadow_cmy, dtype=np.float32))
    h_cmy = np.ascontiguousarray(np.array(highlight_cmy, dtype=np.float32))
    dye = resolve_dye_matrix(paper)
    dye_mix = np.ascontiguousarray(np.eye(3) if dye is None else dye)
    use_ev = ev_map is not None
    ev_arr = np.ascontiguousarray(ev_map.astype(np.float32)) if ev_map is not None else np.zeros((1, 1), dtype=np.float32)

    res = _apply_print_curve_kernel(
        np.ascontiguousarray(img.astype(np.float32)),
        pivots,
        slopes,
        curvs,
        float(toe * ts),
        float(shoulder * ts),
        float(toe_width),
        float(shoulder_width),
        offsets,
        s_cmy,
        h_cmy,
        d_min_rgb=np.array(paper_dmin_rgb(d_min, paper), dtype=np.float64),
        d_max=float(c["d_max"]),
        a_toe_base=float(c["toe_sharpness_base"]),
        a_sh_base=float(c["shoulder_sharpness_base"]),
        width_ref=float(c["toeshoulder_width_ref"]),
        toe_height=float(c["toe_height"]),
        sh_height=float(c["shoulder_height"]),
        zone_center=float(c["anchor_target_density"]),
        v_star=float(v_star),
        midtone_gamma=float(midtone_gamma),
        gamma_width=float(c["paper_gamma_width"]),
        dye_mix=dye_mix,
        use_dye_mix=dye is not None,
        ev_map=ev_arr,
        ev_scale=np.ascontiguousarray(np.array(ev_scale, dtype=np.float32)),
        use_ev=use_ev,
        flare=float(flare),
        surround_gamma=float(surround_gamma),
    )
    return ensure_image(res)


def flat_curve_params() -> Tuple[float, float]:
    """
    Fixed (gain, lift) for the flat log master — scene-independent (no per-frame
    metering) so an evenly-exposed roll renders identically.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    return float(c["flat_log_gain"]), float(c["flat_log_lift"])


def apply_flat_curve(
    image: ImageBuffer,
    gain: float,
    lift: float,
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ImageBuffer:
    """
    True log master: emit the normalized log signal directly as the code value
    (positive-oriented 1 - val), linearly remapped with headroom. No 10^-D decode
    and no sRGB OETF — the flat, log-video look, fully invertible for downstream
    editing. WB rides as an additive per-channel shift in log space.
    """
    arr = np.asarray(image, dtype=np.float32)
    off = np.asarray(cmy_offsets, dtype=np.float32)
    code = lift + gain * (1.0 - (arr + off))
    return ensure_image(np.clip(code, 0.0, 1.0).astype(np.float32))


def default_grade_range() -> float:
    """Fallback density range when none is measured: auto_grade_target * nominal ratio."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    return float(c["auto_grade_target"]) * float(c["auto_grade_nominal_ratio"])


def grade_to_slope(grade: float, density_range: Optional[float]) -> float:
    """
    Straight-line slope k from the grade given as an ISO R paper exposure range
    (R180 very soft ... R50 very hard; R110 ~ classic grade 2 paper). k is the
    literal H&D gamma: contrast = negative density range / paper exposure range,
    like real graded paper — k = grade_contrast_scale * range / (R/100).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    rng_in = default_grade_range() if density_range is None else density_range
    er = min(max(grade, c["iso_r_min"]), c["iso_r_max"]) / 100.0
    rng = min(max(abs(float(rng_in)), 0.3), 3.5)
    k = float(c["grade_contrast_scale"]) * rng / er
    return float(min(max(k, c["slope_min"]), c["slope_max"]))


def slope_to_grade(slope: float, density_range: Optional[float]) -> float:
    """
    Inverse of grade_to_slope: the ISO R paper grade equivalent to an effective
    slope, given the density range that produced it. Used to display the contrast
    the conversion is actually applying (including Auto Grade), on the same ISO R
    scale as the Grade slider. Clamped to the slider's R range.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    rng_in = default_grade_range() if density_range is None else density_range
    rng = min(max(abs(float(rng_in)), 0.3), 3.5)
    if slope <= 0:
        return float(c["iso_r_max"])
    er = float(c["grade_contrast_scale"]) * rng / float(slope)
    return float(min(max(er * 100.0, c["iso_r_min"]), c["iso_r_max"]))


def grade_coupled_shape(slope_g: float, toe: float, shoulder: float) -> Tuple[float, float]:
    """
    Grade-coupled baseline toe/shoulder: hard grades (VC paper) physically have
    snappier toes and compressed shoulders. slope_norm = 0 at the softest grade,
    1 at the hardest. Single source of truth for the CPU engine, the GPU uniform
    packing and the curve chart — they must all draw the same knees.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    slope_norm = (float(slope_g) - float(c["slope_min"])) / (float(c["slope_max"]) - float(c["slope_min"]))
    slope_norm = min(max(slope_norm, 0.0), 1.0)
    toe_eff = float(toe) + float(c["toe_grade_strength"]) * slope_norm
    shoulder_eff = float(shoulder) + float(c["shoulder_grade_strength"]) * slope_norm
    return toe_eff, shoulder_eff


def effective_grade_range(
    auto_normalize_contrast: bool,
    floor_ceil_range: Optional[float],
    textural_range: Optional[float],
) -> Optional[float]:
    """
    Range fed to grade_to_slope. Auto Grade off: the measured floor-to-ceil range.
    Auto Grade on: hold printed midtone contrast partially constant, damping the
    floor_ceil/textural ratio toward the nominal frame:
    effective = target * (nominal + strength * (ratio - nominal)).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    if not auto_normalize_contrast:
        return floor_ceil_range
    if textural_range is None or floor_ceil_range is None:
        return default_grade_range()
    measured = abs(float(textural_range))
    if measured < 1e-6:
        # Degenerate (near-flat) frame: let grade_to_slope's clamp cap the boost.
        return 3.5
    k = float(c["auto_grade_target"])
    nominal = float(c["auto_grade_nominal_ratio"])
    strength = float(c["auto_grade_strength"])
    ratio = abs(float(floor_ceil_range)) / measured
    return k * (nominal + strength * (ratio - nominal))


def _reference_linear_value(d_min: float = 0.0, paper: Optional[PaperProfile] = None) -> float:
    """
    Straight-line density value v* that the base shoulder+toe bounds map onto the
    target density (anchor_target_density). The reference tone is placed here so it
    prints at target, and the paper S-curve is centred here so the anchor is
    preserved. Closed form via inverse softplus at the base toe/shoulder sharpness.
    """
    c = effective_constants(paper)
    t = float(c["anchor_target_density"])
    d_max = float(c["d_max"])
    a_hl = float(c["shoulder_sharpness_base"])  # highlight (lower) bound
    a_sh = float(c["toe_sharpness_base"])  # shadow (upper) bound
    v1 = d_max - _inv_softplus_np(a_sh * (d_max - t)) / a_sh
    return float(d_min + _inv_softplus_np(a_hl * (v1 - d_min)) / a_hl)


def compute_pivot(
    slope: float, density: float, d_min: float = 0.0, anchor: Optional[float] = None, paper: Optional[PaperProfile] = None
) -> float:
    """
    Fixed calibrated exposure: solve the curve pivot so the reference tone
    prints at anchor_target_density for the current effective slope — grade
    changes rotate around that reference tone instead of shifting brightness.
    The density slider offsets exposure around it. The reference tone defaults
    to assumed_anchor (a typical negative's normalized median); pass `anchor`
    to use a per-frame metered median (auto-exposure) instead.
    """
    c = effective_constants(paper)
    ref = c["assumed_anchor"] if anchor is None else anchor
    v_star = _reference_linear_value(d_min, paper)
    base = ref - v_star / slope
    return base + (1.0 - density) * c["density_multiplier"]


def normalize_refs(
    refs: Tuple[float, float, float],
    floors: Tuple[float, float, float],
    ceils: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Per-channel reference densities -> normalized [0, 1] position in the same
    floor->ceil stretch the image is normalized with. Shared by the CPU/GPU/chart
    call sites (Cast Removal shadow refs) so they can't drift.
    """
    epsilon = 1e-6
    out = []
    for ch in range(3):
        denom = ceils[ch] - floors[ch]
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        out.append((refs[ch] - floors[ch]) / denom)
    return (out[0], out[1], out[2])


def normalized_shadow_refs(bounds: Any, refs: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    """Shadow refs normalized against `bounds`, or None if either is missing."""
    if bounds is None or refs is None:
        return None
    return normalize_refs(refs, bounds.floors, bounds.ceils)


def normalized_neutral_axis(bounds: Any, refs: Any) -> Any:
    """(midtone, shadow, highlight) neutral refs normalized against `bounds`; highlight may be
    None (2-point), or the whole thing None if either core ref is missing."""
    if bounds is None or refs is None:
        return None
    mid, shadow, highlight = refs[0], refs[1], refs[2]  # refs may carry a trailing confidence
    norm = lambda r: normalize_refs(r, bounds.floors, bounds.ceils) if r is not None else None  # noqa: E731
    return (norm(mid), norm(shadow), norm(highlight))


def effective_cast_strength(strength: float, auto: bool, confidence: Optional[float]) -> float:
    """Applied cast-removal strength. Auto biases the slider by the neutral-reference
    confidence (clean greys → full, ambiguous → gentler); the slider trims on top."""
    if auto and confidence is not None:
        return confidence * strength
    return strength


def per_channel_curve_params(
    grade: float,
    density: float,
    auto_normalize_contrast: bool,
    strength: float,
    lum_range: Optional[float],
    shadow_refs_norm: Optional[Tuple[float, float, float]],
    textural_range: Optional[float],
    d_min: float = 0.0,
    anchor: Optional[float] = None,
    paper: Optional[PaperProfile] = None,
    neutral_axis_norm: Any = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-channel (slopes, pivots, curvatures); single source of truth for CPU/GPU/chart.
    Core: v = slope*(u - pivot) + curv*u² (curv 0 = a straight line).

    Cast Removal fits R/B to green's neutral axis (green is the reference; its pivot rides the
    luma anchor, so exposure is unchanged). neutral_axis_norm (midtone, shadow, highlight) ->
    quadratic through all three; (midtone, shadow) -> line; shadow_refs_norm only -> one-point
    shadow tie. Off / no refs (E6/B&W): one shared linear curve.
    """
    c = effective_constants(paper)
    # Per-channel slope multipliers (paper dye-layer contrast crossover). The
    # pivot is re-solved per channel so neutrals stay neutral and colour diverges
    # only away from the midtone.
    cg = paper.channel_gamma if paper is not None else (1.0, 1.0, 1.0)
    slope_min = float(c["slope_min"])
    slope_max = float(c["slope_max"])
    r_eff = effective_grade_range(auto_normalize_contrast, lum_range, textural_range)
    base_slope = grade_to_slope(grade, r_eff)

    epsilon = 1e-6

    if strength > 0.0 and neutral_axis_norm is not None:
        # Line through the green-matched midtone+shadow, plus a highlight-driven curvature
        # (when present) so highlights don't extrapolate past neutral. Clamped monotonic.
        mid_norm, sh_norm, hl_norm = neutral_axis_norm
        limit = float(c["midtone_cast_max_offset"])
        curv_lim = float(c["neutral_axis_curv_max_ratio"])
        m_g, s_g = float(mid_norm[1]), float(sh_norm[1])
        slope_g = min(max(base_slope * cg[1], slope_min), slope_max)
        pivot_g = compute_pivot(slope_g, density, d_min=d_min, anchor=anchor, paper=paper)
        target = lambda g: slope_g * (g - pivot_g)  # noqa: E731  green's core at a green ref
        t_m, t_s = target(m_g), target(s_g)
        h_g = float(hl_norm[1]) if hl_norm is not None else None
        clamp_dev = lambda g, v: g + min(max(strength * (v - g), -limit), limit)  # noqa: E731

        slopes, pivots, curvs = [], [], []
        for ch in range(3):
            if ch == 1:
                slopes.append(slope_g)
                pivots.append(pivot_g)
                curvs.append(0.0)
                continue
            u_m = clamp_dev(m_g, float(mid_norm[ch]))
            u_s = clamp_dev(s_g, float(sh_norm[ch]))

            curv = 0.0
            if h_g is not None and hl_norm is not None:
                u_h = clamp_dev(h_g, float(hl_norm[ch]))
                # Leading coeff of the quadratic through the three green-matched points.
                m = np.array([[1.0, u_h, u_h * u_h], [1.0, u_m, u_m * u_m], [1.0, u_s, u_s * u_s]])
                try:
                    curv = float(np.linalg.solve(m, np.array([target(h_g), t_m, t_s]))[2])
                except np.linalg.LinAlgError:
                    curv = 0.0
                curv = min(max(curv, -curv_lim * slope_g), curv_lim * slope_g)

            # Re-pin midtone+shadow with this curvature (mid exact via pivot, like the 2-pt solve).
            du = u_m - u_s
            slope_ch = slope_g if abs(du) < epsilon else ((t_m - t_s) - curv * (u_m * u_m - u_s * u_s)) / du
            slope_ch = min(max(slope_ch * cg[ch], slope_min), slope_max)
            curv_ch = curv * cg[ch]
            pivot_ch = u_m - (t_m - curv_ch * u_m * u_m) / slope_ch if abs(slope_ch) > epsilon else pivot_g
            slopes.append(slope_ch)
            pivots.append(pivot_ch)
            curvs.append(curv_ch)
        return (slopes[0], slopes[1], slopes[2]), (pivots[0], pivots[1], pivots[2]), (curvs[0], curvs[1], curvs[2])

    if strength > 0.0 and shadow_refs_norm is not None:
        # Fallback one-point tie: slope-tilt each channel so its shadow ref lands on
        # green's, with the luma anchor pinning the midtone (used when no neutral axis).
        anchor_val = float(c["assumed_anchor"]) if anchor is None else float(anchor)
        limit = float(c["cast_removal_max_offset"])
        r_green = float(shadow_refs_norm[1])
        numer = anchor_val - r_green

        slopes = []
        pivots = []
        for ch in range(3):
            # Clamp the shadow cast before solving, bounding the correction.
            cast = min(max(strength * (r_green - float(shadow_refs_norm[ch])), -limit), limit)
            denom = anchor_val - (r_green - cast)
            if ch == 1 or abs(denom) < epsilon:
                slope_ch = base_slope
            else:
                slope_ch = base_slope * numer / denom
                slope_ch = min(max(slope_ch, slope_min), slope_max)
            slope_ch = min(max(slope_ch * cg[ch], slope_min), slope_max)
            slopes.append(slope_ch)
            pivots.append(compute_pivot(slope_ch, density, d_min=d_min, anchor=anchor, paper=paper))
        return (slopes[0], slopes[1], slopes[2]), (pivots[0], pivots[1], pivots[2]), (0.0, 0.0, 0.0)

    # Base curve: Cast Removal off, or on with no measured refs (E6/B&W/no neutrals).
    s0 = min(max(base_slope * cg[0], slope_min), slope_max)
    s1 = min(max(base_slope * cg[1], slope_min), slope_max)
    s2 = min(max(base_slope * cg[2], slope_min), slope_max)
    p0 = compute_pivot(s0, density, d_min=d_min, anchor=anchor, paper=paper)
    p1 = compute_pivot(s1, density, d_min=d_min, anchor=anchor, paper=paper)
    p2 = compute_pivot(s2, density, d_min=d_min, anchor=anchor, paper=paper)
    return (s0, s1, s2), (p0, p1, p2), (0.0, 0.0, 0.0)


def filtration_offsets(wb_cmy: Tuple[float, float, float], bounds: Any) -> Tuple[float, float, float]:
    """
    WB sliders as normalized-space offsets: slider · cmy_max_density is an
    absolute density (1.0 = 20cc), divided by each channel's stretch range so
    the same slider prints the same filtration on every frame. abs() keeps the
    slider direction uniform across C-41/E-6. Range 1 when bounds are None.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    cmy_max = float(EXPOSURE_CONSTANTS["cmy_max_density"])
    out = []
    for ch in range(3):
        d = float(wb_cmy[ch]) * cmy_max
        if bounds is not None:
            d = d / max(abs(bounds.ceils[ch] - bounds.floors[ch]), 1e-6)
        out.append(d)
    return (out[0], out[1], out[2])


def local_ev_scale(bounds: Any) -> Tuple[float, float, float]:
    """
    Normalized-space size of one dodge/burn EV stop per channel: -log10(2) over
    the channel's stretch range (like filtration_offsets); negative so positive
    EV (dodge) lowers print exposure. Range 1 when bounds are None.
    """
    step = -float(np.log10(2.0))
    if bounds is None:
        return (step, step, step)
    out = []
    for ch in range(3):
        out.append(step / max(abs(bounds.ceils[ch] - bounds.floors[ch]), 1e-6))
    return (out[0], out[1], out[2])


def cmy_to_density(val: float, log_range: float = 1.0) -> float:
    """
    Converts a CMY slider value (-1.0..1.0) to a physical density shift (D).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    absolute_density = val * EXPOSURE_CONSTANTS["cmy_max_density"]
    return float(absolute_density / max(log_range, 1e-6))


def density_to_cmy(density: float, log_range: float = 1.0) -> float:
    """
    Converts a physical density shift (D) back to a normalized CMY slider value.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    absolute_density = density * log_range
    return float(absolute_density / EXPOSURE_CONSTANTS["cmy_max_density"])


TEMP_REF_KELVIN = 5500.0
TEMP_MIN_KELVIN, TEMP_MAX_KELVIN = 3000.0, 12000.0
# Slider units per mired, red-anchored (wb_cyan stays 0): Wien slopes
# c2*log10(e)*(1/lambda - 1/lambda_R) at ~460/550/690 nm, over a nominal
# effective print gamma (~4) and cmy_max_density (0.2) — so the Kelvin
# readout is nominal, not colorimetric. Calibration knobs: retune by eye.
_TEMP_K_MAGENTA = 0.0029
_TEMP_K_YELLOW = 0.0057


def wb_to_kelvin(magenta: float, yellow: float) -> float:
    """
    Measured print temperature: least-squares projection of the global (M, Y)
    pair onto the Planckian (mired) direction; 5500K at neutral.
    """
    km, ky = _TEMP_K_MAGENTA, _TEMP_K_YELLOW
    dmu = (km * magenta + ky * yellow) / (km * km + ky * ky)
    # Clamp in the mired domain: a Kelvin-domain clamp breaks when mu goes negative.
    mu = min(max(1e6 / TEMP_REF_KELVIN + dmu, 1e6 / TEMP_MAX_KELVIN), 1e6 / TEMP_MIN_KELVIN)
    return float(1e6 / mu)


def kelvin_to_wb(kelvin: float, magenta: float, yellow: float) -> Tuple[float, float]:
    """
    Moves (M, Y) along the Planckian direction to land on `kelvin`, keeping the
    off-locus green-magenta tint component untouched. Clips to slider range.
    """
    km, ky = _TEMP_K_MAGENTA, _TEMP_K_YELLOW
    kelvin = min(max(kelvin, TEMP_MIN_KELVIN), TEMP_MAX_KELVIN)
    dmu_cur = (km * magenta + ky * yellow) / (km * km + ky * ky)
    d = (1e6 / kelvin - 1e6 / TEMP_REF_KELVIN) - dmu_cur
    m2 = min(max(magenta + km * d, -1.0), 1.0)
    y2 = min(max(yellow + ky * d, -1.0), 1.0)
    return float(m2), float(y2)


def calculate_wb_shifts(sampled_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Magenta and Yellow shifts to neutralize sampled color in positive space.
    """
    r, g, b = np.clip(sampled_rgb, 1e-6, 1.0)
    d_m = np.log10(g) - np.log10(r)
    d_y = np.log10(b) - np.log10(r)

    shift_m = density_to_cmy(d_m)
    shift_y = density_to_cmy(d_y)

    return float(shift_m), float(shift_y)


def calculate_wb_shifts_from_log(sampled_log_rgb: np.ndarray, bounds: Any = None) -> Tuple[float, float]:
    """
    Calculates Magenta and Yellow shifts from data in Negative Log-Density space.
    `bounds` converts the deviation to absolute density (see filtration_offsets).
    """
    r, g, b = sampled_log_rgb[:3]
    d_m = r - g
    d_y = r - b

    rng = lambda ch: abs(bounds.ceils[ch] - bounds.floors[ch]) if bounds is not None else 1.0  # noqa: E731
    shift_m = density_to_cmy(d_m, rng(1))
    shift_y = density_to_cmy(d_y, rng(2))

    return float(shift_m), float(shift_y)
