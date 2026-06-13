from typing import Any, Optional, Tuple

import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import ImageBuffer
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


@njit(inline="always")
def _srgb_oetf(t: float) -> float:
    """
    sRGB opto-electronic transfer function (linear -> display encoding).
    Matches the sRGB decode used by the downstream Lab stage.
    """
    if t <= 0.0031308:
        return float(12.92 * t)
    return float(1.055 * t ** (1.0 / 2.4) - 0.055)


@njit(cache=True, fastmath=True)
def _apply_photometric_fused_kernel(
    img: np.ndarray,
    pivots: np.ndarray,
    slopes: np.ndarray,
    toe: float,
    toe_width: float,
    shoulder: float,
    shoulder_width: float,
    cmy_offsets: np.ndarray,
    shadow_cmy: np.ndarray,
    highlight_cmy: np.ndarray,
    d_max: float = 2.3,
    d_min: float = 0.0,
    d_onset: float = 1.2,
    asymptote: float = 3.2,
    shoulder_beta: float = 8.0,
    nu: float = 1.0,
    mode: int = 0,
) -> np.ndarray:
    """
    Fused JIT kernel for H&D curve application with integrated toe/shoulder.

    Shoulder modulates local contrast on the input axis: gamma(d) = 1 - shoulder*M_s(d),
    with the closed-form integral (softplus) as curve argument, anchored at the
    pivot (x(0) = 0). Toe works in the DENSITY domain as a shadow lever —
    raising or crushing print tones darker than the onset — because the shadow
    zone above the pivot is too narrow for an input-axis toe to have useful
    strength. Both are
    monotone and smooth by construction; shoulder leaves the pivot tone
    invariant, toe (anchored at D = 0) leaves highlights invariant.
    `toe`/`shoulder` arrive pre-scaled by EXPOSURE_CONSTANTS["toe_shoulder_strength"].
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    epsilon = 1e-6

    # Density-domain toe geometry: onset where print shadows begin
    # (d_onset = 1.2 D, ~0.28 sRGB), so the slider works as a shadow
    # lever — raise or crush everything darker than that.
    # Anchored at D = 0 with its tangent removed (zero value AND zero slope),
    # so highlights are invariant even at very soft widths.
    b_t = toe_width * 2.0
    sp_toe0 = _softplus(b_t * (0.0 - d_onset)) / b_t
    sig_toe0 = _fast_sigmoid(b_t * (0.0 - d_onset))

    # Per-channel mask geometry (same centers/widths as the legacy masks).
    a_t = np.empty(3, dtype=np.float64)
    c_t = np.empty(3, dtype=np.float64)
    a_s = np.empty(3, dtype=np.float64)
    c_s = np.empty(3, dtype=np.float64)
    sig_s0 = np.empty(3, dtype=np.float64)
    for ch in range(3):
        p = float(pivots[ch])
        a_t[ch] = toe_width / max(1.0 - p, epsilon)
        c_t[ch] = 0.5 * (1.0 - p)
        a_s[ch] = shoulder_width / max(p, epsilon)
        c_s[ch] = -0.5 * p
        sig_s0[ch] = -_softplus(-a_s[ch] * (0.0 - c_s[ch])) / a_s[ch]

    for y in range(h):
        for x in range(w):
            for ch in range(3):
                val = img[y, x, ch] + cmy_offsets[ch]
                diff = val - pivots[ch]

                zt = a_t[ch] * (diff - c_t[ch])
                zs = -a_s[ch] * (diff - c_s[ch])
                toe_mask = _fast_sigmoid(zt)
                shoulder_mask = _fast_sigmoid(zs)

                sig_s = -_softplus(zs) / a_s[ch]

                x_adj = diff - shoulder * (sig_s - sig_s0[ch])
                arg = x_adj + shadow_cmy[ch] * toe_mask + highlight_cmy[ch] * shoulder_mask

                # Richards curve toward the projected (virtual) asymptote: the
                # nu exponent shortens the toe (whites snap to paper white) and
                # lengthens the top approach, like real paper. Physical paper
                # black is enforced by the soft clamp below.
                density = d_min + (asymptote - d_min) * _fast_sigmoid(float(slopes[ch]) * arg) ** nu

                if toe != 0.0:
                    sp_d = _softplus(b_t * (density - d_onset)) / b_t
                    density = density - toe * (sp_d - sp_toe0 - sig_toe0 * density)

                # Abrupt smooth saturation shoulder at paper Dmax.
                density = density - _softplus(shoulder_beta * (density - d_max)) / shoulder_beta

                transmittance = 10.0 ** (-density)
                final_val = _srgb_oetf(transmittance)

                if final_val < 0.0:
                    final_val = 0.0
                elif final_val > 1.0:
                    final_val = 1.0

                res[y, x, ch] = final_val
    return res


class LogisticSigmoid:
    """
    H&D curve with integrated toe/shoulder — same math as the fused kernel
    (used for the curve display, so chart and render stay identical).
    Returns density (pre-transmittance/encode).
    """

    def __init__(
        self,
        contrast: float,
        pivot: float,
        d_max: Optional[float] = None,
        d_min: float = 0.0,
        toe: float = 0.0,
        toe_width: float = 3.0,
        shoulder: float = 0.0,
        shoulder_width: float = 3.0,
        shadow_cmy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        highlight_cmy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS

        ts = EXPOSURE_CONSTANTS["toe_shoulder_strength"]
        self.k = contrast
        self.x0 = pivot
        # L is the projected (virtual) asymptote; d_max is the physical paper
        # black enforced by the soft saturation clamp in __call__.
        self.L = EXPOSURE_CONSTANTS["curve_asymptote"] if d_max is None else d_max
        self.d_max = EXPOSURE_CONSTANTS["d_max"]
        self.shoulder_beta = EXPOSURE_CONSTANTS["dmax_shoulder"]
        self.d_min = d_min
        self.nu = float(EXPOSURE_CONSTANTS["paper_toe_nu"])
        self.d_onset = EXPOSURE_CONSTANTS["toe_onset_density"]
        self.toe = toe * ts
        self.toe_width = toe_width
        self.shoulder = shoulder * ts
        self.shoulder_width = shoulder_width
        self.shadow_cmy = shadow_cmy
        self.highlight_cmy = highlight_cmy

    def __call__(self, x: ImageBuffer) -> ImageBuffer:
        diff = x - self.x0
        epsilon = 1e-6

        a_s = self.shoulder_width / max(self.x0, epsilon)
        c_s = -0.5 * self.x0

        zs = -a_s * (diff - c_s)

        # np.logaddexp(0, z) is a numerically stable softplus.
        sig_s = -np.logaddexp(0.0, zs) / a_s
        sig_s0 = -np.logaddexp(0.0, -a_s * (0.0 - c_s)) / a_s

        x_adj = diff - self.shoulder * (sig_s - sig_s0)

        res = self.d_min + (self.L - self.d_min) * _expit(self.k * x_adj) ** self.nu

        if self.toe != 0.0:
            # Density-domain toe (shadow lever), anchored at D = 0 with
            # its tangent removed so highlights are invariant at any width.
            b_t = self.toe_width * 2.0
            d_onset = self.d_onset
            sp_d = np.logaddexp(0.0, b_t * (res - d_onset)) / b_t
            sp_0 = np.logaddexp(0.0, b_t * (0.0 - d_onset)) / b_t
            sig_0 = _expit(b_t * (0.0 - d_onset))
            res = res - self.toe * (sp_d - sp_0 - sig_0 * res)

        # Abrupt smooth saturation shoulder at paper Dmax.
        res = res - np.logaddexp(0.0, self.shoulder_beta * (res - self.d_max)) / self.shoulder_beta

        return ensure_image(res)


def apply_characteristic_curve(
    img: ImageBuffer,
    params_r: Tuple[float, float],
    params_g: Tuple[float, float],
    params_b: Tuple[float, float],
    toe: float = 0.0,
    toe_width: float = 3.0,
    shoulder: float = 0.0,
    shoulder_width: float = 3.0,
    shadow_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    d_min: float = 0.0,
    mode: int = 0,
) -> ImageBuffer:
    """
    Applies a film/paper characteristic curve (Sigmoid) per channel in Log-Density space.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    ts = EXPOSURE_CONSTANTS["toe_shoulder_strength"]
    pivots = np.ascontiguousarray(np.array([params_r[0], params_g[0], params_b[0]], dtype=np.float32))
    slopes = np.ascontiguousarray(np.array([params_r[1], params_g[1], params_b[1]], dtype=np.float32))
    offsets = np.ascontiguousarray(np.array(cmy_offsets, dtype=np.float32))
    s_cmy = np.ascontiguousarray(np.array(shadow_cmy, dtype=np.float32))
    h_cmy = np.ascontiguousarray(np.array(highlight_cmy, dtype=np.float32))

    res = _apply_photometric_fused_kernel(
        np.ascontiguousarray(img.astype(np.float32)),
        pivots,
        slopes,
        float(toe * ts),
        float(toe_width),
        float(shoulder * ts),
        float(shoulder_width),
        offsets,
        s_cmy,
        h_cmy,
        d_max=float(EXPOSURE_CONSTANTS["d_max"]),
        d_min=float(d_min),
        d_onset=float(EXPOSURE_CONSTANTS["toe_onset_density"]),
        asymptote=float(EXPOSURE_CONSTANTS["curve_asymptote"]),
        shoulder_beta=float(EXPOSURE_CONSTANTS["dmax_shoulder"]),
        nu=float(EXPOSURE_CONSTANTS["paper_toe_nu"]),
        mode=mode,
    )

    return ensure_image(res)


def sigmoid_span(nu: float) -> float:
    """
    Curve-argument span between 10% and 90% of the Richards asymptote —
    generalizes ln 81 (the nu = 1 value). Used to map ISO R to slope so the
    grade keeps its ISO meaning for any paper-toe sharpness.
    """

    def _logit(s: float) -> float:
        return float(np.log(s / (1.0 - s)))

    return _logit(0.9 ** (1.0 / nu)) - _logit(0.1 ** (1.0 / nu))


def grade_to_slope(grade: float, density_range: Optional[float]) -> float:
    """
    Slope from the grade given as an ISO R paper exposure range
    (R180 very soft ... R50 very hard; R110 ~ classic grade 2 paper).
    The curve's 10-90% span covers the paper's exposure range expressed in
    normalized negative-density units, so contrast = negative density range /
    paper exposure range — like real graded paper.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    rng_in = c["auto_grade_ref_range"] if density_range is None else density_range
    er = min(max(grade, c["iso_r_min"]), c["iso_r_max"]) / 100.0
    rng = min(max(abs(float(rng_in)), 0.3), 3.5)
    k = sigmoid_span(float(c["paper_toe_nu"])) * rng / er
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
    rng_in = c["auto_grade_ref_range"] if density_range is None else density_range
    rng = min(max(abs(float(rng_in)), 0.3), 3.5)
    if slope <= 0:
        return float(c["iso_r_max"])
    er = sigmoid_span(float(c["paper_toe_nu"])) * rng / float(slope)
    return float(min(max(er * 100.0, c["iso_r_min"]), c["iso_r_max"]))


def effective_grade_range(
    auto_normalize_contrast: bool,
    floor_ceil_range: Optional[float],
    textural_range: Optional[float],
) -> Optional[float]:
    """
    Range fed to grade_to_slope for the contrast (grade) decision.

    - Auto Grade off (physical): the floor-to-ceil range — contrast fully tracks
      the negative's measured density range.
    - Auto Grade on: a damped blend between a pleasing reference and the per-frame
      textural range, so contrast leans gently with the scene without printing to
      a hard extreme. strength = auto_grade_adapt (0 = constant, 1 = full track).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    if not auto_normalize_contrast:
        return floor_ceil_range
    ref = float(c["auto_grade_ref_range"])
    if textural_range is None:
        return ref
    s = float(c["auto_grade_adapt"])
    return (1.0 - s) * ref + s * abs(float(textural_range))


def compute_pivot(slope: float, density: float, d_min: float = 0.0, anchor: Optional[float] = None) -> float:
    """
    Fixed calibrated exposure: solve the curve pivot so the reference tone
    prints at anchor_target_density for the current effective slope — grade
    changes rotate around that reference tone instead of shifting brightness.
    The density slider offsets exposure around it. The reference tone defaults
    to assumed_anchor (a typical negative's normalized median); pass `anchor`
    to use a per-frame metered median (auto-exposure) instead.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    t = c["anchor_target_density"]
    if anchor is not None:
        # Auto Density prints a touch bright; nudge the metered tone darker.
        t = t + c["auto_density_target_offset"]
    nu = float(c["paper_toe_nu"])
    ref = c["assumed_anchor"] if anchor is None else anchor
    # Solve against the projected asymptote (the target sits well below the
    # Dmax saturation shoulder, so the soft clamp doesn't shift it):
    # sigmoid(slope*(anchor - pivot))^nu == s  =>  pivot = anchor - logit(s^(1/nu))/slope.
    s = (t - d_min) / (c["curve_asymptote"] - d_min)
    root = s ** (1.0 / nu)
    base = ref - float(np.log(root / (1.0 - root))) / slope
    return base + (1.0 - density) * c["density_multiplier"]


def shadow_neutral_offsets(
    refs: Tuple[float, float, float],
    floors: Tuple[float, float, float],
    ceils: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Normalized-space shadow CMY offsets that print the measured shadow
    reference tone neutral — like filtration printing film base+fog to a
    neutral black. Green-referenced (consistent with the green-referenced
    density range), clamped to shadow_neutral_max_offset.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    epsilon = 1e-6
    limit = float(EXPOSURE_CONSTANTS["shadow_neutral_max_offset"])
    n = []
    for ch in range(3):
        denom = ceils[ch] - floors[ch]
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        n.append((refs[ch] - floors[ch]) / denom)
    offsets = [min(max(n[1] - n[ch], -limit), limit) for ch in range(3)]
    offsets[1] = 0.0
    return (offsets[0], offsets[1], offsets[2])


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


def calculate_wb_shifts_from_log(sampled_log_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Magenta and Yellow shifts from data in Negative Log-Density space.
    """
    r, g, b = sampled_log_rgb[:3]
    d_m = r - g
    d_y = r - b

    shift_m = density_to_cmy(d_m)
    shift_y = density_to_cmy(d_y)

    return float(shift_m), float(shift_y)
