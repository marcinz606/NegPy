import math
from typing import Any, Dict, Optional, Tuple

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.papers import (
    PaperProfile,
    compose_density_matrices,
    effective_constants,
    resolve_print_color,
    resolve_saturation_matrix,
    slope_gain,
)
from negpy.features.exposure.print_lut import lut_black_luminance
from negpy.kernel.image.validation import ensure_image


def _expit(x: Any) -> Any:
    """Numpy implementation of the logistic sigmoid function (scipy.special.expit fallback).

    expit(x) = exp(-logaddexp(0, -x)) — exact and overflow-free for any x.
    """
    return np.exp(-np.logaddexp(0.0, -x))


def _fast_sigmoid(x: float) -> float:
    """The logistic sigmoid 1 / (1 + exp(-x)), overflow-safe for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _inv_softplus_np(y: Any) -> Any:
    """Inverse of softplus: log(exp(y) - 1), stable for y > 0 (pivot solve)."""
    return np.where(y > 20.0, y, np.log(np.expm1(np.maximum(y, 1e-12))))


def separation_damping_gain(k: float, damping: float, chroma: float, ref_spread: float) -> float:
    """
    One pixel's effective dye-separation k: the frame-wide k tapered by the
    pixel's own chroma. h = (ref - c)/(ref + c) runs from 1 at grey to -1 at
    extreme separation, so at damping 1 muted color takes the full k, the
    reference spread is left at exactly 1.0 and vivid color gets 1/k — the
    sign of the change differs between the two populations, which is what a
    frame-wide matrix cannot do. Monotone in chroma for k < e**2; the clamp
    bounds the k -> 0 corner (weakly monotone, so no rank swap).

    negpy_core's print curve and exposure.wgsl mirror it.
    """
    if k <= 0.0:
        return 0.0
    h = (ref_spread - chroma) / (ref_spread + chroma)
    kf = k ** ((1.0 - damping) + damping * h)
    if kf > 3.0:
        return 3.0
    return float(kf)


def separation_damping_gain_np(k: float, damping: float, chroma: Any, ref_spread: float) -> Any:
    """Vectorized numpy twin of separation_damping_gain, for the transfer curve's
    per-pixel gain over a whole image rather than the print curve's per-pixel scalar.
    Same math; see that function for the derivation."""
    if k <= 0.0:
        return np.zeros_like(chroma, dtype=np.float32)
    h = (np.float32(ref_spread) - chroma) / (np.float32(ref_spread) + chroma)
    kf = np.power(np.float32(k), (1.0 - np.float32(damping)) + np.float32(damping) * h, dtype=np.float32)
    return np.minimum(kf, np.float32(3.0))


def tone_key_weight_np(lum: Any, e0: float, e1: float) -> Any:
    """A tone-limited mask's weight at luma `lum`, over a whole raster: a smoothstep from 0 at e0
    to 1 at e1 (see placement.key_edges). The print curve in negpy_core and exposure.wgsl mirror it."""
    t = np.clip((np.asarray(lum, dtype=np.float32) - np.float32(e0)) / np.float32(e1 - e0), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


class CharacteristicCurve:
    """
    Asymmetric H&D print curve (toe-linear-shoulder) in density space — the NumPy
    mirror of the print curve in negpy_core (print_curve.rs), used by the curve chart so the displayed
    curve matches the render. Returns density (pre-transmittance/encode). Neutral
    (no regional CMY color), since the chart shows the achromatic transfer; the
    achromatic zone density offsets (shadow/highlight ΔD) are included.
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
        paper: Optional[PaperProfile] = None,
        midtone_gamma: Optional[float] = None,
        bpc: bool = False,
        shadow_density: float = 0.0,
        highlight_density: float = 0.0,
        shadow_grade_delta: float = 0.0,
        highlight_grade_delta: float = 0.0,
        curvature: float = 0.0,
        preflash: float = 0.0,
        grade: float = 115.0,
        black_hold: float = 0.0,
    ):
        c = effective_constants(paper)
        ts = float(c["toe_shoulder_strength"])
        self.k = float(contrast)
        self.x0 = float(pivot)
        self.curvature = float(curvature)
        self.flash = max(float(preflash), 0.0)
        self.midtone_gamma = float(c["paper_midtone_gamma"]) if midtone_gamma is None else float(midtone_gamma)
        self.flash_v, self.flash_gamma = preflash_params(grade, d_min, paper, self.midtone_gamma)
        self.d_min = float(d_min)
        self.v_star = _reference_linear_value(d_min, paper)
        self.gamma_width = float(c["paper_gamma_width"])
        self.zone_sh_center = float(c["reference_print_density"]) + float(c["zone_density_shadow_offset"])
        self.zone_hi_center = float(c["reference_print_density"]) + float(c["zone_density_highlight_offset"])
        self.zone_k = float(c["zone_density_sharpness"])
        self.shadow_density = float(shadow_density)
        self.highlight_density = float(highlight_density)
        self.black_hold = float(black_hold)
        self.shadow_grade_delta = float(shadow_grade_delta)
        self.highlight_grade_delta = float(highlight_grade_delta)
        self.d_max = float(c["d_max"])
        # The BPC reference mirrors the kernel prologue: achromatic, d_min for the tint.
        self.bpc = bool(bpc)
        db = self.d_max
        if toe * ts < 0.0:
            db = self.d_max + toe * ts * float(c["toe_height"])
        self.bpc_black = 10.0**-db
        wr = float(c["toeshoulder_width_ref"])
        # toe = the shadow (upper) bound, shoulder = the highlight (lower) bound.
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
        xv = np.asarray(x, dtype=np.float64)
        if self.curvature != 0.0:
            vertex = -self.k / (2.0 * self.curvature)
            xv = np.maximum(xv, vertex) if self.curvature > 0.0 else np.minimum(xv, vertex)
        v = self.k * (xv - self.x0) + self.curvature * xv * xv
        if self.flash > 0.0:
            v = preflash_value(v, self.flash_v, self.flash, self.flash_gamma)
        if self.midtone_gamma != 0.0:
            v = v + self.midtone_gamma * self.gamma_width * np.tanh((v - self.v_star) / self.gamma_width)
        if self.shadow_grade_delta != 0.0 or self.highlight_grade_delta != 0.0:
            w_gsh = _expit(self.zone_k * (v - self.zone_sh_center))
            w_ghi = 1.0 - _expit(self.zone_k * (v - self.zone_hi_center))
            v = (
                v
                + self.shadow_grade_delta * w_gsh * (v - self.zone_sh_center)
                + self.highlight_grade_delta * w_ghi * (v - self.zone_hi_center)
            )
        if self.shadow_density != 0.0 or self.highlight_density != 0.0:
            w_zsh = _expit(self.zone_k * (v - self.zone_sh_center))
            if self.black_hold > 0.0 and self.shadow_density < 0.0:
                t = np.clip((self.d_max_eff - v) / self.black_hold, 0.0, 1.0)
                w_zsh = w_zsh * t * t * (3.0 - 2.0 * t)
            w_zhi = 1.0 - _expit(self.zone_k * (v - self.zone_hi_center))
            v = v + self.shadow_density * w_zsh + self.highlight_density * w_zhi
        v1 = self.d_min_eff + np.logaddexp(0.0, self.a_hl * (v - self.d_min_eff)) / self.a_hl
        res = self.d_max_eff - np.logaddexp(0.0, self.a_sh * (self.d_max_eff - v1)) / self.a_sh

        if self.bpc:
            t = 10.0 ** (-res)
            t = (t - self.bpc_black) / (1.0 - self.bpc_black)
            res = -np.log10(np.maximum(t, 1e-12))

        return ensure_image(res)


def black_hold_taper(exposure: Any, paper: Optional[PaperProfile] = None) -> float:
    """Hold Black's taper span for the kernels; 0 turns it off."""
    return float(effective_constants(paper)["zone_black_hold_taper"]) if exposure.hold_black else 0.0


def print_curve(
    exposure: Any,
    slope: float,
    pivot: float,
    process_mode: Optional[str] = None,
    *,
    toe: Optional[float] = None,
    shoulder: Optional[float] = None,
    toe_width: Optional[float] = None,
    shoulder_width: Optional[float] = None,
    midtone_gamma: Optional[float] = None,
    shadow_grade_delta: Optional[float] = None,
    highlight_grade_delta: Optional[float] = None,
    curvature: float = 0.0,
    highlight_density: Optional[float] = None,
    shadow_density: Optional[float] = None,
) -> CharacteristicCurve:
    """The achromatic print curve for `exposure` at `slope`/`pivot`. Each None argument takes
    the grade-coupled, trim-free value; a per-layer trace passes its own. Single source of
    truth for the chart and the step wedge.

    The paper profile is passed through to the curve, matching the render (which builds its
    constants from `effective_constants(paper)`): the RA4 papers raise d_max above the 2.3
    default, so omitting it drew a curve the print didn't have."""
    from negpy.features.exposure.papers import effective_paper_profile

    profile = effective_paper_profile(exposure.paper_profile, process_mode)
    d_min = profile.d_min if exposure.paper_dmin else 0.0
    toe_eff, shoulder_eff = grade_coupled_shape(slope, exposure.toe, exposure.shoulder)
    sg, hg = split_grade_deltas(exposure.grade, exposure.shadow_grade, exposure.highlight_grade)
    return CharacteristicCurve(
        contrast=slope,
        pivot=pivot,
        d_min=d_min,
        toe=toe_eff if toe is None else toe,
        toe_width=exposure.toe_width if toe_width is None else toe_width,
        shoulder=shoulder_eff if shoulder is None else shoulder,
        shoulder_width=exposure.shoulder_width if shoulder_width is None else shoulder_width,
        paper=profile,
        midtone_gamma=effective_midtone_gamma(profile, exposure.midtone_gamma) if midtone_gamma is None else midtone_gamma,
        bpc=not exposure.paper_black,
        shadow_density=exposure.shadow_density if shadow_density is None else shadow_density,
        highlight_density=exposure.highlight_density if highlight_density is None else highlight_density,
        shadow_grade_delta=sg[0] if shadow_grade_delta is None else shadow_grade_delta,
        highlight_grade_delta=hg[0] if highlight_grade_delta is None else highlight_grade_delta,
        curvature=curvature,
        preflash=exposure.preflash,
        grade=exposure.grade,
        black_hold=black_hold_taper(exposure, profile),
    )


def print_curve_output(curve: CharacteristicCurve, x_log_exp: Any) -> np.ndarray:
    """Curve density -> display-encoded output: the engine's 10**-D plus the working OETF.
    The chart's y axis and the step wedge's patch values both come from here, so they cannot
    drift apart."""
    from negpy.kernel.image.logic import working_oetf_encode

    d = np.asarray(curve(ensure_image(np.asarray(x_log_exp, dtype=np.float32))))
    return np.asarray(working_oetf_encode(np.power(10.0, -d).astype(np.float32))).reshape(-1)


def curve_params_from_metrics(
    exposure: Any,
    process_mode: Optional[str],
    metrics: Any,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """(slopes, pivots, curvatures) the render solved for, re-derived from the metrics it
    published — the engines keep only the raw cast refs on purpose. Green is the base curve.

    Every lookup is a .get(), so an empty metrics dict degrades to grade/density alone (the
    pre-first-render case). Bounds live under a different key per engine: CPU writes
    "final_bounds", GPU writes "log_bounds".
    """
    from negpy.features.exposure.papers import effective_paper_profile

    profile = effective_paper_profile(exposure.paper_profile, process_mode)
    d_min = profile.d_min if exposure.paper_dmin else 0.0
    anchor = metrics.get("metered_anchor") if exposure.auto_exposure else None
    bounds = metrics.get("final_bounds") or metrics.get("log_bounds")
    strength, neutral_axis_norm = cast_solve_inputs(bounds, metrics.get("neutral_axis_refs"), exposure.cast_removal_strength)
    return per_channel_curve_params(
        exposure.grade,
        exposure.density,
        exposure.auto_normalize_contrast,
        strength,
        metrics.get("norm_density_range"),
        metrics.get("textural_range"),
        d_min=d_min,
        anchor=anchor,
        paper=profile,
        neutral_axis_norm=neutral_axis_norm,
        grade_trims=(exposure.grade_trim_red, exposure.grade_trim_green, exposure.grade_trim_blue),
        shadow_point=metrics.get("shadow_point"),
        midtone_gamma=effective_midtone_gamma(profile, exposure.midtone_gamma),
    )


def per_channel_toe_shoulder(
    toe_eff: float,
    shoulder_eff: float,
    toe_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shoulder_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-layer effective toe/shoulder: grade-coupled global value + per-layer trim,
    clamped to the slider domain. Single source of truth for CPU / GPU / chart.
    """

    def _clamp(v: float) -> float:
        return min(max(v, -1.0), 1.0)

    toe3 = (_clamp(toe_eff + toe_trims[0]), _clamp(toe_eff + toe_trims[1]), _clamp(toe_eff + toe_trims[2]))
    sh3 = (
        _clamp(shoulder_eff + shoulder_trims[0]),
        _clamp(shoulder_eff + shoulder_trims[1]),
        _clamp(shoulder_eff + shoulder_trims[2]),
    )
    return toe3, sh3


def per_channel_widths(
    toe_width: float,
    shoulder_width: float,
    toe_width_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shoulder_width_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-layer effective toe/shoulder widths: global width + per-layer trim,
    clamped to the width slider domain. Single source of truth for CPU / GPU / chart.
    """

    def _clamp(v: float) -> float:
        return min(max(v, 0.1), 5.0)

    tw3 = (_clamp(toe_width + toe_width_trims[0]), _clamp(toe_width + toe_width_trims[1]), _clamp(toe_width + toe_width_trims[2]))
    sw3 = (
        _clamp(shoulder_width + shoulder_width_trims[0]),
        _clamp(shoulder_width + shoulder_width_trims[1]),
        _clamp(shoulder_width + shoulder_width_trims[2]),
    )
    return tw3, sw3


_NO_LUT = np.ones((2, 2, 2, 3), dtype=np.float32)
_NO_LUT_E_MAX = np.ones(3, dtype=np.float32)


def black_density_rgb(d_max: float, toe: Any, toe_height: float) -> Tuple[float, float, float]:
    """The density black point compensation maps to black, per channel; the kernel prologue and
    exposure.wgsl compute the same: a negative toe raises the clip point."""
    db = [d_max + t * toe_height if t < 0.0 else d_max for t in toe]
    return (db[0], db[1], db[2])


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
    midtone_gamma: Optional[float] = None,
    curvatures: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    paper: Optional[PaperProfile] = None,
    ev_map: Optional[np.ndarray] = None,
    ev_scale: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    grade_map: Optional[np.ndarray] = None,
    bpc: bool = False,
    toe_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shoulder_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    snap_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    toe_width_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shoulder_width_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shadow_density: float = 0.0,
    highlight_density: float = 0.0,
    shadow_grade_deltas: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_grade_deltas: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    dye_separation: float = 1.0,
    dye_separation_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    separation_damping: float = 0.0,
    key_alpha: Optional[np.ndarray] = None,
    key_params: Optional[np.ndarray] = None,
    grade_deltas: Optional[np.ndarray] = None,
    frame_grade: float = 115.0,
    preflash: float = 0.0,
    flash_map: Optional[np.ndarray] = None,
    diffusion: float = 0.0,
    diffusion_plane: Optional[np.ndarray] = None,
    diffusion_rect: Optional[Tuple[float, float, float, float]] = None,
    diffusion_range: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    black_hold: float = 0.0,
) -> ImageBuffer:
    """Applies the asymmetric H&D print curve per channel in log-density space.

    ev_map (H×W, stops; positive = burn) with ev_scale (see local_ev_scale) applies
    per-pixel dodge/burn as print-exposure offsets ahead of the curve.

    key_alpha (H×W×K shape alphas) with key_params (K×4: stops, ISO-R delta, e0, e1 from
    key_edges) adds K tone-limited masks, each weighted by a smoothstep from 0 at e0 to 1
    at e1 on the pixel's own luma. With them the local grade is summed in ISO-R space
    about frame_grade from grade_deltas (the unlimited masks' ISO-R plane), not grade_map.

    preflash: a uniform flash as a fraction of the threshold exposure, on the paper gamma
    of frame_grade (preflash_params).

    dye_separation(_trims): density-domain saturation, composed into the
    dye_mix slot (see resolve_saturation_matrix/compose_density_matrices).

    separation_damping: tapers that separation by each pixel's own chroma
    (see separation_damping_gain), which makes k per-pixel — so it leaves the
    dye_mix slot to the paper's coupling and runs in the kernel instead."""
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
    dye, lut = resolve_print_color(paper)
    sat_k3 = per_channel_dye_separation(dye_separation, dye_separation_trims)
    use_sep_damping = separation_damping > 0.0 and sat_k3 != (1.0, 1.0, 1.0)
    sat = None if use_sep_damping else resolve_saturation_matrix(sat_k3)
    composed = compose_density_matrices(dye, sat)
    dye_mix = np.ascontiguousarray(np.eye(3) if composed is None else composed)
    use_ev = ev_map is not None
    ev_arr = np.ascontiguousarray(ev_map.astype(np.float32)) if ev_map is not None else np.zeros((1, 1), dtype=np.float32)
    use_grade = grade_map is not None
    grade_arr = np.ascontiguousarray(grade_map.astype(np.float32)) if grade_map is not None else np.ones((1, 1), dtype=np.float32)
    use_flash = flash_map is not None
    flash_arr = np.ascontiguousarray(flash_map.astype(np.float32)) if flash_map is not None else np.zeros((1, 1), dtype=np.float32)
    use_diff = diffusion > 0.0 and diffusion_plane is not None
    diff_arr = np.ascontiguousarray(diffusion_plane.astype(np.float32)) if use_diff else np.zeros((1, 1, 3), dtype=np.float32)
    h_img, w_img = img.shape[:2]
    rect = diffusion_rect if diffusion_rect is not None else (0.0, 0.0, float(w_img), float(h_img))
    use_key = key_alpha is not None
    r_min, r_max = float(c["iso_r_min"]), float(c["iso_r_max"])
    if use_key:
        key_alpha_arr = np.ascontiguousarray(key_alpha.astype(np.float32))
        key_params_arr = np.ascontiguousarray(np.asarray(key_params, dtype=np.float64).reshape(-1, 4))
        deltas_arr = np.ascontiguousarray(grade_deltas.astype(np.float32))
    else:
        key_alpha_arr = np.zeros((1, 1, 1), dtype=np.float32)
        key_params_arr = np.zeros((0, 4), dtype=np.float64)
        deltas_arr = np.zeros((1, 1), dtype=np.float32)

    toe3, sh3 = per_channel_toe_shoulder(toe, shoulder, toe_trims, shoulder_trims)
    tw3, sw3 = per_channel_widths(toe_width, shoulder_width, toe_width_trims, shoulder_width_trims)
    flash_v, flash_gamma = preflash_params(frame_grade, d_min, paper, midtone_gamma)
    toe_k = [t * ts for t in toe3]
    d_min_rgb = paper_dmin_rgb(d_min, paper)
    lut_black = 0.0
    if lut is not None and bpc:
        lut_black = lut_black_luminance(lut, d_min_rgb, black_density_rgb(float(c["d_max"]), toe_k, float(c["toe_height"])))
    res = negpy_core.apply_print_curve(
        np.ascontiguousarray(img.astype(np.float32)),
        pivots,
        slopes,
        curvs,
        np.array(toe_k, dtype=np.float64),
        np.array([s * ts for s in sh3], dtype=np.float64),
        np.array(tw3, dtype=np.float64),
        np.array(sw3, dtype=np.float64),
        offsets,
        s_cmy,
        h_cmy,
        d_min_rgb=np.array(d_min_rgb, dtype=np.float64),
        d_max=float(c["d_max"]),
        a_toe_base=float(c["toe_sharpness_base"]),
        a_sh_base=float(c["shoulder_sharpness_base"]),
        width_ref=float(c["toeshoulder_width_ref"]),
        toe_height=float(c["toe_height"]),
        sh_height=float(c["shoulder_height"]),
        zone_center=float(c["reference_print_density"]),
        shadow_density=float(shadow_density),
        highlight_density=float(highlight_density),
        shadow_grade=np.array(shadow_grade_deltas, dtype=np.float64),
        highlight_grade=np.array(highlight_grade_deltas, dtype=np.float64),
        zone_sh_center=float(c["reference_print_density"]) + float(c["zone_density_shadow_offset"]),
        zone_hi_center=float(c["reference_print_density"]) + float(c["zone_density_highlight_offset"]),
        zone_k=float(c["zone_density_sharpness"]),
        v_star=float(v_star),
        midtone_gamma=np.array([float(midtone_gamma) + snap_trims[ch] for ch in range(3)], dtype=np.float64),
        gamma_width=float(c["paper_gamma_width"]),
        dye_mix=dye_mix,
        use_dye_mix=composed is not None,
        sep_k3=np.array(sat_k3, dtype=np.float64),
        sep_damping=float(separation_damping),
        sep_ref=float(c["separation_damping_ref_spread"]),
        use_sep_damping=use_sep_damping,
        ev_map=ev_arr,
        ev_scale=np.ascontiguousarray(np.array(ev_scale, dtype=np.float32)),
        use_ev=use_ev,
        grade_map=grade_arr,
        use_grade=use_grade,
        key_alpha=key_alpha_arr,
        key_params=key_params_arr,
        use_key=use_key,
        grade_deltas=deltas_arr,
        r0=min(max(float(frame_grade), r_min), r_max),
        r_min=r_min,
        r_max=r_max,
        flash=max(float(preflash), 0.0),
        flash_v=flash_v,
        flash_gamma=flash_gamma,
        flash_map=flash_arr,
        use_flash=use_flash,
        diff_plane=diff_arr,
        diff_rect=np.array(rect, dtype=np.float64),
        diff_mix=diffusion_mix(diffusion),
        diff_range=np.array(diffusion_range, dtype=np.float64),
        use_diff=use_diff,
        lut=_NO_LUT if lut is None else lut.table,
        lut_e_max=_NO_LUT_E_MAX if lut is None else lut.e_max,
        lut_black=lut_black,
        use_lut=lut is not None,
        bpc=bool(bpc),
        black_hold=float(black_hold),
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
    stock_matrix: Optional[np.ndarray] = None,
) -> ImageBuffer:
    """
    True log master: emit the normalized log signal directly as the code value
    (positive-oriented 1 - val), linearly remapped with headroom. No 10^-D decode
    and no sRGB OETF — the flat, log-video look, fully invertible for downstream
    editing. WB rides as an additive per-channel shift in log space; the print
    stock's density matrix (rows sum 1) mixes after it, as in the print kernel.
    """
    val = np.asarray(image, dtype=np.float32) + np.asarray(cmy_offsets, dtype=np.float32)
    if stock_matrix is not None:
        val = val @ np.asarray(stock_matrix, dtype=np.float32).T
    code = lift + gain * (1.0 - val)
    return ensure_image(np.clip(code, 0.0, 1.0).astype(np.float32))


def default_grade_range() -> float:
    """Fallback density range when none is measured: a normal negative's, scaled by K."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    return float(c["auto_grade_target"]) * float(c["auto_grade_nominal_ratio"]) * float(c["auto_grade_nominal_range"])


def grade_to_slope(grade: float, density_range: Optional[float], gain: float = 1.0) -> float:
    """
    Straight-line slope k from the grade given as an ISO R paper exposure range
    (R180 very soft ... R50 very hard; R110 ~ classic grade 2 paper). k is the
    literal H&D gamma: contrast = negative density range / paper exposure range,
    like real graded paper — k = grade_contrast_scale * range / (R/100) * gain,
    with gain the paper's slope_gain.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    rng_in = default_grade_range() if density_range is None else density_range
    er = min(max(grade, c["iso_r_min"]), c["iso_r_max"]) / 100.0
    rng = min(max(abs(float(rng_in)), 0.3), 3.5)
    k = float(c["grade_contrast_scale"]) * rng / er * float(gain)
    return float(min(max(k, c["slope_min"]), c["slope_max"]))


def per_channel_dye_separation(
    dye_separation: float,
    trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """
    Per-layer effective dye-separation k: global value + per-layer trim,
    clamped to a sane matrix-coefficient range. Mirrors
    per_channel_toe_shoulder's global+trim convention.
    """

    def _clamp(v: float) -> float:
        return min(max(v, 0.0), 3.0)

    return (
        _clamp(dye_separation + trims[0]),
        _clamp(dye_separation + trims[1]),
        _clamp(dye_separation + trims[2]),
    )


def slope_to_grade(slope: float, density_range: Optional[float], gain: float = 1.0) -> float:
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
    er = float(c["grade_contrast_scale"]) * rng * float(gain) / float(slope)
    return float(min(max(er * 100.0, c["iso_r_min"]), c["iso_r_max"]))


def effective_midtone_gamma(paper: Optional[PaperProfile], trim: float) -> float:
    """
    Paper's variable midtone gamma plus the user's additive trim. Single source
    of truth for CPU / GPU / chart.
    """
    return float(effective_constants(paper)["paper_midtone_gamma"]) + float(trim)


def per_channel_midtone_gamma(
    paper: Optional[PaperProfile],
    trim: float,
    snap_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """
    Per-layer effective midtone gamma: paper baseline + global trim + Snap trim.
    Single source of truth for CPU / GPU / chart. Unclamped: the slider domains
    keep |gamma| < 1, the kernel's monotonicity bound.
    """
    base = effective_midtone_gamma(paper, trim)
    return (base + snap_trims[0], base + snap_trims[1], base + snap_trims[2])


def _grade_trim_mult(grade: float, trim: float, c: Dict[str, Any]) -> float:
    """
    Per-layer ISO-R trim -> slope ratio: k ∝ 1/R, so a ΔR trim is the pure
    ratio R/(R+ΔR), with both grades clamped to the R ladder.
    """
    r0 = min(max(float(grade), float(c["iso_r_min"])), float(c["iso_r_max"]))
    r1 = min(max(r0 + float(trim), float(c["iso_r_min"])), float(c["iso_r_max"]))
    return r0 / r1


def local_grade_factor_map(grade_deltas: np.ndarray, grade: float) -> np.ndarray:
    """Per-pixel slope multiplier for the local-grade map, the ratio _grade_trim_mult gives
    a per-layer trim. The kernel rotates about the channel pivot, so a grade-only mask keeps
    its midtone. exposure.wgsl mirrors this."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    r_min, r_max = float(c["iso_r_min"]), float(c["iso_r_max"])
    r0 = min(max(float(grade), r_min), r_max)
    r1 = np.clip(r0 + grade_deltas.astype(np.float32), r_min, r_max)
    return (r0 / r1).astype(np.float32)


def split_grade_deltas(
    grade: float,
    shadow_grade: float,
    highlight_grade: float,
    shadow_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Split-grade ISO-R trims -> per-layer local contrast gains (multiplier - 1)
    about the zone centers: global value + per-layer trim, like grade trims.
    Single source of truth for CPU / GPU / chart. ISO-R bounds are
    paper-independent, so EXPOSURE_CONSTANTS applies.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    return (
        (
            _grade_trim_mult(grade, shadow_grade + shadow_trims[0], c) - 1.0,
            _grade_trim_mult(grade, shadow_grade + shadow_trims[1], c) - 1.0,
            _grade_trim_mult(grade, shadow_grade + shadow_trims[2], c) - 1.0,
        ),
        (
            _grade_trim_mult(grade, highlight_grade + highlight_trims[0], c) - 1.0,
            _grade_trim_mult(grade, highlight_grade + highlight_trims[1], c) - 1.0,
            _grade_trim_mult(grade, highlight_grade + highlight_trims[2], c) - 1.0,
        ),
    )


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
    Range fed to grade_to_slope. Auto Grade off: the measured floor-to-ceil range, a
    fixed paper. Auto Grade on: the paper gamma follows the negative's textural density
    scale, shrunk toward a normal negative's (Alkofer, US 4,731,671):
    effective = K * floor_ceil * ((1 - s) + s * nominal / textural),
    so the printed textural range is (1 - s) of the frame's own plus s of the norm,
    capped at max_overfill of the norm's print span.
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
    q = float(c["auto_grade_nominal_range"]) / measured
    strength = float(c["auto_grade_strength"])
    # The print span of the textural range is capped at max_overfill of a normal
    # negative's: a grade whose paper range is far shorter than the negative's clips
    # both ends (the ISO R rule read as a floor on softness).
    factor = min((1.0 - strength) + strength * q, q * float(c["auto_grade_max_overfill"]))
    return k * abs(float(floor_ceil_range)) * factor


def _reference_linear_value(d_min: float = 0.0, paper: Optional[PaperProfile] = None, target: Optional[float] = None) -> float:
    """
    Straight-line density value v* that the base shoulder+toe bounds map onto `target`
    (default reference_print_density). The reference tone is placed here so it
    prints at target, and the paper S-curve is centred here so the anchor is
    preserved. Closed form via inverse softplus at the base toe/shoulder sharpness.
    """
    c = effective_constants(paper)
    t = float(c["reference_print_density"]) if target is None else float(target)
    d_max = float(c["d_max"])
    a_hl = float(c["shoulder_sharpness_base"])  # highlight (lower) bound
    a_sh = float(c["toe_sharpness_base"])  # shadow (upper) bound
    v1 = d_max - _inv_softplus_np(a_sh * (d_max - t)) / a_sh
    return float(d_min + _inv_softplus_np(a_hl * (v1 - d_min)) / a_hl)


def _midtone_terms(d_min: float, paper: Optional[PaperProfile], midtone_gamma: Optional[float]) -> Tuple[float, float, float]:
    c = effective_constants(paper)
    mg = float(c["paper_midtone_gamma"]) if midtone_gamma is None else float(midtone_gamma)
    return mg, float(c["paper_gamma_width"]), _reference_linear_value(d_min, paper)


def _through_midtone(v: float, d_min: float = 0.0, paper: Optional[PaperProfile] = None, midtone_gamma: Optional[float] = None) -> float:
    """Straight-line value `v` after the variable midtone gamma, as the kernels apply it.
    midtone_gamma: the effective value (paper + Snap); None takes the paper's own."""
    mg, w, v_star = _midtone_terms(d_min, paper, midtone_gamma)
    return float(v + mg * w * np.tanh((v - v_star) / w))


def _straight_line_target(
    d_min: float = 0.0, paper: Optional[PaperProfile] = None, target: Optional[float] = None, midtone_gamma: Optional[float] = None
) -> float:
    """Straight-line value that prints `target` through the midtone gamma and the bounds: the
    inverse of _through_midtone at _reference_linear_value. Bisection, since the midtone term is
    monotone and moves a value by at most |mg| * width."""
    u = _reference_linear_value(d_min, paper, target)
    mg, w, v_star = _midtone_terms(d_min, paper, midtone_gamma)
    lo, hi = u - abs(mg) * w, u + abs(mg) * w
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid + mg * w * math.tanh((mid - v_star) / w) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _anchor_linear_value(
    d_min: float = 0.0, paper: Optional[PaperProfile] = None, metered: bool = False, midtone_gamma: Optional[float] = None
) -> float:
    """Straight-line value the anchor prints from: a metered anchor lands on
    anchor_target_density through the midtone gamma, an unmetered one on v*."""
    if not metered:
        return _reference_linear_value(d_min, paper)
    return _straight_line_target(d_min, paper, float(effective_constants(paper)["anchor_target_density"]), midtone_gamma)


def preflash_params(
    grade: float, d_min: float = 0.0, paper: Optional[PaperProfile] = None, midtone_gamma: Optional[float] = None
) -> Tuple[float, float]:
    """(v_th, gamma) for preflash_value. v_th is the straight-line value that prints the paper's
    threshold, d_min + preflash_threshold_density (where the ISO R range starts). gamma is the
    paper's density per log10 exposure at `grade`, so the flash is a paper exposure and does
    not depend on the scan's log range."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = effective_constants(paper)
    v_th = _straight_line_target(d_min, paper, d_min + float(c["preflash_threshold_density"]), midtone_gamma)
    er = min(max(float(grade), EXPOSURE_CONSTANTS["iso_r_min"]), EXPOSURE_CONSTANTS["iso_r_max"]) / 100.0
    return v_th, float(EXPOSURE_CONSTANTS["grade_contrast_scale"]) / er * slope_gain(paper)


def preflash_value(v: Any, v_th: float, flash: float, gamma: float) -> Any:
    """Straight-line value `v` after a uniform flash of `flash` × the threshold exposure.
    Exposures add, so the flash lifts tones near threshold and barely moves tones far above it."""
    return v_th + gamma * np.log10(10.0 ** ((v - v_th) / gamma) + flash)


def shadow_reach_slope(
    slope: float,
    anchor: float,
    shadow_point: float,
    d_min: float = 0.0,
    paper: Optional[PaperProfile] = None,
    midtone_gamma: Optional[float] = None,
    metered: bool = False,
) -> float:
    """
    Auto Grade floor on the slope: the textured dark tail at `shadow_point` must print
    at least shadow_reach_density while the anchor stays at its target, so the slope
    is raised to the straight line through both when the grade alone falls short
    (Gindele, US 7,113,649; Ajewole, US 5,046,118). Never lowered.
    """
    c = effective_constants(paper)
    span = float(shadow_point) - float(anchor)
    if span <= 1e-6:
        return slope
    v_black = _straight_line_target(d_min, paper, float(c["shadow_reach_density"]), midtone_gamma)
    needed = (v_black - _anchor_linear_value(d_min, paper, metered, midtone_gamma)) / span
    return float(min(max(slope, needed), float(c["slope_max"])))


def highlight_hold_offset(
    slope: float,
    pivot: float,
    highlight_point: float,
    d_min: float = 0.0,
    paper: Optional[PaperProfile] = None,
    preflash: float = 0.0,
    grade: float = 115.0,
    midtone_gamma: Optional[float] = None,
) -> float:
    """
    Auto Grade's soft exposure: the highlight zone burn (a highlight_density term) that
    lands the textured bright tail at `highlight_point` on highlight_hold_density when the
    straight line would print it brighter. Solved against the kernel's own zone weight at
    that tone, so the burn stays under the shoulder. Never lifts; 0 when the tone holds.
    Measured after the preflash and the midtone gamma, so the burn only tops up what they leave.
    """
    c = effective_constants(paper)
    target = float(c["highlight_hold_density"])
    if target <= 0.0:
        return 0.0
    v = float(slope) * (float(highlight_point) - float(pivot))
    if preflash > 0.0:
        v_th, gamma = preflash_params(grade, d_min, paper, midtone_gamma)
        v = float(preflash_value(v, v_th, preflash, gamma))
    v = _through_midtone(v, d_min, paper, midtone_gamma)
    v_hold = _reference_linear_value(d_min, paper, target=target)
    if v >= v_hold:
        return 0.0
    z_hi = float(c["reference_print_density"]) + float(c["zone_density_highlight_offset"])
    w = 1.0 - _fast_sigmoid(float(c["zone_density_sharpness"]) * (v - z_hi))
    return float(min((v_hold - v) / max(w, 1e-6), float(c["highlight_hold_max"])))


def shadow_hold_offset(
    slope: float,
    pivot: float,
    shadow_point: float,
    d_min: float = 0.0,
    paper: Optional[PaperProfile] = None,
    midtone_gamma: Optional[float] = None,
) -> float:
    """Auto Grade's shadow lift (a negative shadow_density), the partner of
    highlight_hold_offset, on the textured dark tail past Shadow Reach's line. A partial
    correction: the tail still prints past paper black. Never burns."""
    c = effective_constants(paper)
    strength = float(c["shadow_hold_strength"])
    if strength <= 0.0:
        return 0.0
    v = _through_midtone(float(slope) * (float(shadow_point) - float(pivot)), d_min, paper, midtone_gamma)
    v_start = _reference_linear_value(d_min, paper, target=float(c["shadow_reach_density"])) + float(c["shadow_hold_overshoot"])
    if v <= v_start:
        return 0.0
    return -float(min(strength * (v - v_start), float(c["shadow_hold_max"])))


def auto_shadow_from_metrics(exposure: Any, process_mode: Optional[str], metrics: Any) -> float:
    """The shadow hold lift the render applied, re-derived from the published metrics."""
    from negpy.features.exposure.papers import effective_paper_profile

    point = metrics.get("shadow_point")
    if not exposure.auto_normalize_contrast or point is None:
        return 0.0
    profile = effective_paper_profile(exposure.paper_profile, process_mode)
    d_min = profile.d_min if exposure.paper_dmin else 0.0
    slopes, pivots, _ = curve_params_from_metrics(exposure, process_mode, metrics)
    mg = effective_midtone_gamma(profile, exposure.midtone_gamma)
    return shadow_hold_offset(slopes[1], pivots[1], float(point), d_min=d_min, paper=profile, midtone_gamma=mg)


def auto_highlight_from_metrics(exposure: Any, process_mode: Optional[str], metrics: Any) -> float:
    """The highlight hold burn the render applied, re-derived from the published metrics
    (companion of curve_params_from_metrics for the chart, wedge and zone placement)."""
    from negpy.features.exposure.papers import effective_paper_profile

    point = metrics.get("highlight_point")
    if not exposure.auto_normalize_contrast or point is None:
        return 0.0
    profile = effective_paper_profile(exposure.paper_profile, process_mode)
    d_min = profile.d_min if exposure.paper_dmin else 0.0
    slopes, pivots, _ = curve_params_from_metrics(exposure, process_mode, metrics)
    return highlight_hold_offset(
        slopes[1],
        pivots[1],
        float(point),
        d_min=d_min,
        paper=profile,
        preflash=exposure.preflash,
        grade=exposure.grade,
        midtone_gamma=effective_midtone_gamma(profile, exposure.midtone_gamma),
    )


def compute_pivot(
    slope: float,
    density: float,
    d_min: float = 0.0,
    anchor: Optional[float] = None,
    paper: Optional[PaperProfile] = None,
    midtone_gamma: Optional[float] = None,
) -> float:
    """
    Fixed calibrated exposure: solve the curve pivot so the reference tone
    (assumed_anchor) prints at reference_print_density for the current effective
    slope, so grade changes rotate around it instead of shifting brightness.
    The density slider offsets exposure around it. A metered `anchor` (Auto
    Density) prints at anchor_target_density instead.
    """
    c = effective_constants(paper)
    ref = c["assumed_anchor"] if anchor is None else anchor
    v = _anchor_linear_value(d_min, paper, anchor is not None, midtone_gamma)
    base = ref - v / slope
    return base + (1.0 - density) * c["density_multiplier"]


def normalize_refs(
    refs: Tuple[float, float, float],
    floors: Tuple[float, float, float],
    ceils: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Per-channel reference densities -> normalized [0, 1] position in the same
    floor->ceil stretch the image is normalized with. Shared by the CPU/GPU/chart
    call sites (the Cast Removal neutral axis) so they can't drift.
    """
    epsilon = 1e-6
    out = []
    for ch in range(3):
        denom = ceils[ch] - floors[ch]
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        out.append((refs[ch] - floors[ch]) / denom)
    return (out[0], out[1], out[2])


def normalized_neutral_axis(bounds: Any, refs: Any) -> Any:
    """(midtone, shadow, highlight) neutral refs normalized against `bounds`; highlight may be
    None (2-point), or the whole thing None if either core ref is missing."""
    if bounds is None or refs is None:
        return None
    mid, shadow, highlight = refs[0], refs[1], refs[2]  # refs may carry a trailing confidence
    norm = lambda r: normalize_refs(r, bounds.floors, bounds.ceils) if r is not None else None  # noqa: E731
    return (norm(mid), norm(shadow), norm(highlight))


def effective_cast_strength(strength: float, confidence: Optional[float]) -> float:
    """Applied cast-removal strength: the neutral-reference confidence biases the
    slider (clean greys → full, ambiguous → gentler); the slider trims on top."""
    if confidence is not None:
        return confidence * strength
    return strength


def cast_solve_inputs(bounds: Any, neutral_axis_refs: Any, slider_strength: float) -> Tuple[float, Any]:
    """(effective strength, neutral_axis_norm) from raw metrics; single source of truth for
    the CPU processor and the chart."""
    neutral_axis_norm = normalized_neutral_axis(bounds, neutral_axis_refs)
    confidence = neutral_axis_refs[3] if neutral_axis_refs is not None else None
    return effective_cast_strength(slider_strength, confidence), neutral_axis_norm


def auto_base_slope(
    grade: float,
    auto_normalize_contrast: bool,
    lum_range: Optional[float],
    textural_range: Optional[float],
    shadow_point: Optional[float],
    anchor: Optional[float],
    d_min: float = 0.0,
    paper: Optional[PaperProfile] = None,
    midtone_gamma: Optional[float] = None,
) -> float:
    """Shared straight-line slope before the per-layer gammas: Grade on the effective range,
    raised to Auto Grade's Shadow Reach floor."""
    slope = grade_to_slope(grade, effective_grade_range(auto_normalize_contrast, lum_range, textural_range), slope_gain(paper))
    if auto_normalize_contrast and shadow_point is not None:
        c = effective_constants(paper)
        ref = float(c["assumed_anchor"]) if anchor is None else float(anchor)
        slope = shadow_reach_slope(
            slope, ref, shadow_point, d_min=d_min, paper=paper, midtone_gamma=midtone_gamma, metered=anchor is not None
        )
    return slope


def per_channel_curve_params(
    grade: float,
    density: float,
    auto_normalize_contrast: bool,
    strength: float,
    lum_range: Optional[float],
    textural_range: Optional[float],
    d_min: float = 0.0,
    anchor: Optional[float] = None,
    paper: Optional[PaperProfile] = None,
    neutral_axis_norm: Any = None,
    grade_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    shadow_point: Optional[float] = None,
    midtone_gamma: Optional[float] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-channel (slopes, pivots, curvatures); single source of truth for CPU/GPU/chart.
    Core: v = slope*(u - pivot) + curv*u² (curv 0 = a straight line).

    Cast Removal fits R/B to green's neutral axis (green is the reference; its pivot rides the
    luma anchor, so exposure is unchanged). neutral_axis_norm (midtone, shadow, highlight) ->
    quadratic through all three; (midtone, shadow) -> line. Off or no axis: one shared linear curve.
    """
    c = effective_constants(paper)
    # Per-channel slope multipliers for the paper dye-layer contrast crossover. The pivot is
    # re-solved per channel, so neutrals stay neutral and color diverges only away from the
    # midtone.
    cg = paper.channel_gamma if paper is not None else (1.0, 1.0, 1.0)
    if grade_trims != (0.0, 0.0, 0.0):
        # Per-layer ISO-R trims fold in as user channel_gammas, and the pivot re-solve keeps
        # the anchor neutral.
        cg = (
            cg[0] * _grade_trim_mult(grade, grade_trims[0], c),
            cg[1] * _grade_trim_mult(grade, grade_trims[1], c),
            cg[2] * _grade_trim_mult(grade, grade_trims[2], c),
        )
    slope_min = float(c["slope_min"])
    slope_max = float(c["slope_max"])
    base_slope = auto_base_slope(
        grade, auto_normalize_contrast, lum_range, textural_range, shadow_point, anchor, d_min, paper, midtone_gamma
    )

    epsilon = 1e-6

    if strength > 0.0 and neutral_axis_norm is not None:
        # A line through the green-matched midtone and shadow, plus a highlight-driven
        # curvature when present, so highlights do not extrapolate past neutral. Clamped
        # monotonic.
        mid_norm, sh_norm, hl_norm = neutral_axis_norm
        limit = float(c["midtone_cast_max_offset"])
        curv_lim = float(c["neutral_axis_curv_max_ratio"])
        m_g, s_g = float(mid_norm[1]), float(sh_norm[1])
        slope_g = min(max(base_slope * cg[1], slope_min), slope_max)
        pivot_g = compute_pivot(slope_g, density, d_min=d_min, anchor=anchor, paper=paper, midtone_gamma=midtone_gamma)
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

            # Re-pin midtone and shadow with this curvature. The mid is exact through the
            # pivot, like the two-point solve.
            du = u_m - u_s
            slope_ch = slope_g if abs(du) < epsilon else ((t_m - t_s) - curv * (u_m * u_m - u_s * u_s)) / du
            slope_ch = min(max(slope_ch * cg[ch], slope_min), slope_max)
            curv_ch = curv * cg[ch]
            pivot_ch = u_m - (t_m - curv_ch * u_m * u_m) / slope_ch if abs(slope_ch) > epsilon else pivot_g
            slopes.append(slope_ch)
            pivots.append(pivot_ch)
            curvs.append(curv_ch)
        return (slopes[0], slopes[1], slopes[2]), (pivots[0], pivots[1], pivots[2]), (curvs[0], curvs[1], curvs[2])

    # Base curve: Cast Removal off, or no neutral axis (B&W, or a frame with no grays).
    s0 = min(max(base_slope * cg[0], slope_min), slope_max)
    s1 = min(max(base_slope * cg[1], slope_min), slope_max)
    s2 = min(max(base_slope * cg[2], slope_min), slope_max)
    p0 = compute_pivot(s0, density, d_min=d_min, anchor=anchor, paper=paper, midtone_gamma=midtone_gamma)
    p1 = compute_pivot(s1, density, d_min=d_min, anchor=anchor, paper=paper, midtone_gamma=midtone_gamma)
    p2 = compute_pivot(s2, density, d_min=d_min, anchor=anchor, paper=paper, midtone_gamma=midtone_gamma)
    return (s0, s1, s2), (p0, p1, p2), (0.0, 0.0, 0.0)


def neutral_axis_affine(
    neutral_axis_norm: Any,
    strength: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-channel (gain, offset) on normalized density that lands each channel's neutral
    refs on green's — Cast Removal for the transparency transfer, which has no per-channel
    slope to re-solve. Green is identity, and so is a missing axis or zero strength.

    The clamped-deviation convention matches per_channel_curve_params, so `strength` means
    the same thing on both curves: green's ref moves a fraction of the way toward the
    channel's own, and that moved point is what maps onto green.
    """
    unit = ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    if strength <= 0.0 or neutral_axis_norm is None:
        return unit
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    mid_norm, sh_norm = neutral_axis_norm[0], neutral_axis_norm[1]
    c = EXPOSURE_CONSTANTS
    limit = float(c["midtone_cast_max_offset"])
    gain_max = float(c["cast_affine_gain_limit"])
    epsilon = 1e-6

    m_g, s_g = float(mid_norm[1]), float(sh_norm[1])
    clamp_dev = lambda g, v: g + min(max(strength * (v - g), -limit), limit)  # noqa: E731

    gains, offsets = [], []
    for ch in range(3):
        if ch == 1:
            gains.append(1.0)
            offsets.append(0.0)
            continue
        u_m = clamp_dev(m_g, float(mid_norm[ch]))
        u_s = clamp_dev(s_g, float(sh_norm[ch]))
        du = u_m - u_s
        if abs(du) < epsilon:
            gains.append(1.0)
            offsets.append(0.0)
            continue
        gain = min(max((m_g - s_g) / du, 1.0 / gain_max), gain_max)
        # Offset re-solved after the clamp, so the midtone stays exact.
        gains.append(gain)
        offsets.append(m_g - gain * u_m)
    return (gains[0], gains[1], gains[2]), (offsets[0], offsets[1], offsets[2])


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


def channel_density_ranges(bounds: Any) -> Tuple[float, float, float]:
    """Each channel's stretch range in log10 units: one normalized unit of val is this much
    density. Range 1 when bounds are None."""
    if bounds is None:
        return (1.0, 1.0, 1.0)
    out = [max(abs(bounds.ceils[ch] - bounds.floors[ch]), 1e-6) for ch in range(3)]
    return (out[0], out[1], out[2])


# A real diffuser passes much of the light unscattered, so the slider's top is half the light.
DIFFUSION_MIX_MAX = 0.5


def diffusion_mix(diffusion: float) -> float:
    """The fraction of blurred light the paper sees at a Diffusion slider value."""
    return DIFFUSION_MIX_MAX * min(max(float(diffusion), 0.0), 1.0)


def local_ev_scale(bounds: Any) -> Tuple[float, float, float]:
    """
    Normalized-space size of one dodge/burn stop per channel: log10(2) over the
    channel's stretch range (like filtration_offsets). Positive, because the map is
    exposure-signed: a positive value is a burn and must raise print exposure.
    Range 1 when bounds are None.
    """
    step = float(np.log10(2.0))
    if bounds is None:
        return (step, step, step)
    out = []
    for ch in range(3):
        out.append(step / max(abs(bounds.ceils[ch] - bounds.floors[ch]), 1e-6))
    return (out[0], out[1], out[2])


def expand_mask_plane(
    plane: Optional[np.ndarray],
    out_shape: Tuple[int, int],
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """
    The analysis-grid mask plane at render size. The plane covers the printed frame, so
    it lands back at `roi`, edge-replicated so the uncropped preview has no seam.

    Split from `contrast_mask_ev` so a caller can cache this across slider values: only
    the coefficients move with them. The GPU shader mirrors the bilinear mapping.
    """
    return _expanded(plane, out_shape, roi, 1.0)


def _expanded(
    plane: Optional[np.ndarray], out_shape: Tuple[int, int], roi: Optional[Tuple[int, int, int, int]], scale: float
) -> Optional[np.ndarray]:
    if plane is None:
        return None
    h, w = out_shape
    y1, y2, x1, x2 = (0, h, 0, w) if roi is None else roi
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(h, y2), min(w, x2)
    if y2 - y1 < 1 or x2 - x1 < 1:
        return None
    return negpy_core.expand_plane(np.ascontiguousarray(plane, dtype=np.float32), (h, w), (y1, y2, x1, x2), scale)


# Contrast Mask tone codes, after the Frontier's HyperTone: 10-bit print density with mid-gray at
# 412, reference_print_density at 450 and MASK_CODES_PER_DENSITY codes per unit. Its table is
# sign(412 - i) * (i - 412)^2 / 256 over codes 0..1023, capped at 819, and a quarter of it is added.
MASK_MID_CODE = 412.0
MASK_AIM_CODE = 450.0
MASK_CODES_PER_DENSITY = 145.0
MASK_CODE_MAX = 1023.0
MASK_TABLE_CAP = 819.0


def contrast_mask_coeffs(
    strength: float,
    centre: float,
    slopes: Tuple[float, float, float],
    pivots: Tuple[float, float, float],
    offsets: Tuple[float, float, float],
    density_range: float,
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """
    The (a, b, lo, hi, g_lo, g_hi, lim) of negpy_core.contrast_mask_stops, or None when off. Both sides
    take `strength` of the full table. The luma's code is the mean of the three
    channels' straight lines, as the Frontier reads the mean of its channels; green's sizes the stops.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    slope = float(np.mean(slopes))
    if strength <= 0.0 or slope <= 0.0 or slopes[1] <= 0.0 or density_range <= 0.0:
        return None
    c = MASK_CODES_PER_DENSITY
    ref = float(EXPOSURE_CONSTANTS["reference_print_density"])
    shift = float(np.mean([k * (o - p) for k, o, p in zip(slopes, offsets, pivots)]))
    stops_per_code = float(density_range) / (c * float(slopes[1]) * math.log10(2.0))
    gain = float(strength) * stops_per_code / (4.0 * 256.0)
    return (
        c * slope,
        MASK_AIM_CODE + c * (slope * centre + shift - ref) - MASK_MID_CODE,
        -MASK_MID_CODE,
        MASK_CODE_MAX - MASK_MID_CODE,
        gain,
        gain,
        MASK_TABLE_CAP / 4.0 * stops_per_code,
    )


def contrast_mask_ev(
    plane: Optional[np.ndarray],
    coeffs: Optional[Tuple[float, ...]],
    out_shape: Tuple[int, int],
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """The contrast mask as print-exposure stops, ready to add to the dodge/burn map."""
    if plane is None or coeffs is None:
        return None
    expanded = _expanded(plane, out_shape, roi, 1.0)
    return None if expanded is None else negpy_core.contrast_mask_stops(expanded, tuple(float(k) for k in coeffs))


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
# Slider units per mired, red-anchored so wb_cyan stays 0: Wien slopes
# c2*log10(e)*(1/lambda - 1/lambda_R) at roughly 460/550/690 nm, over a nominal effective
# print gamma and cmy_max_density. The Kelvin readout is therefore nominal, not
# colorimetric. Calibration knobs: retune by eye.
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
