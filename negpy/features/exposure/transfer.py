"""
Transparency transfer curve — the E-6 render when Normalize is off.

A slide is captured close to how it should look, so the render starts from the capture
and the controls deviate from there. At default settings the scene stage is the exact
inverse of the fixed-bounds normalization in `normalization.py` (see
TRANSFER_DENSITY_RANGE), so nothing shapes the capture; the standard display rendering
below is then applied to show it.

That last part is not optional. A bare linear-to-gamma encode is the one thing no
consumer raw converter does, and it is not what "as captured" means to anyone: measured
against a Lightroom export of the same frame it left mid-tones ~1.5 EV dark and
highlights short of display white by a factor of two, because the decode anchors to the
sensor white level and the frame was exposed below clipping. Both parts of the fix are
fixed constants, never metered — metering is what makes a bracketed set converge.

The paper H&D curve cannot serve this role. Its print character is structural, not
parametric — `d_max` floors the blacks whatever the toe slider says,
`anchor_target_density` places mid-grey at a print's mid-tone, and the midtone snap
and paper-white reference stay live at neutral settings. Neutralizing it is not
reachable from a paper profile, so a positive gets its own curve.

Controls map onto the existing Print sliders, each neutral at its current default:
  density (1.0)  -> exposure in stops
  grade (115 R)  -> contrast about a mid-grey pivot
  toe (0.0)      -> shadow roll-off      shoulder (0.0) -> highlight roll-off
  WB C/M/Y (0)   -> per-channel density offsets
  shadow/highlight_density (0.0) -> Zone Density, the print path's mid-sparing offsets,
                    re-centred onto this curve's own scale (see zone_geometry)
  dye_separation (1.0) -> density-domain saturation, applied directly to density
                    (there is no paper dye matrix here to compose it into)
  separation_damping (0.0) -> tapers that saturation by each pixel's own chroma
                    (see logic.separation_damping_gain)
"""

from typing import Optional, Tuple

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.logic import (
    _fast_sigmoid,
    effective_grade_range,
    grade_to_slope,
    per_channel_dye_separation,
    per_channel_toe_shoulder,
    per_channel_widths,
    separation_damping_gain_np,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.kernel.image.validation import ensure_image

#: Log-density window the fixed-bounds normalization maps to [0, 1]. The floor is the
#: decoder's white level (density 0), so the mapping is anchored to the capture, not to
#: frame content. That is what keeps a bracketed set rendering as a bracketed set.
TRANSFER_DENSITY_RANGE = 3.0

TRANSFER_CONSTANTS = {
    # Reference grade (ISO R) that means "no contrast change". MUST equal the grade the app
    # ships in DEFAULT_WORKSPACE_CONFIG, written there as the legacy 2.5 that
    # ExposureConfig.__post_init__ migrates, or the transfer stops being identity at
    # defaults. Mirrored rather than imported to keep this module dependency-free;
    # test_transparency_transfer.py asserts the two agree.
    "transfer_grade_ref": 100.0,
    # Stops of exposure per unit of the density slider (higher density = darker).
    "transfer_density_stops": 2.0,
    # Contrast pivot as a density: mid-grey at ~18% of the white level.
    "transfer_contrast_pivot": 0.75,
    # Knees, in density, where the roll-offs start biting. The toe works down from the shadow
    # end and the shoulder up from the highlight end.
    "transfer_toe_knee": 1.6,
    "transfer_shoulder_knee": 0.35,
    # Baseline exposure, in stops, applied before the display rendering below. The decode
    # anchors the signal to the SENSOR WHITE LEVEL (no_auto_bright, adjust_maximum_thr=0), so
    # a frame exposed below clipping arrives correspondingly dark, while every consumer raw
    # converter opens from a baseline instead. 0.7 EV is darktable's shipped default for raw
    # files, a published camera-agnostic figure rather than one fitted to any rig. Fixed,
    # never metered: metering would make a bracketed set converge again.
    "transfer_baseline_ev": 0.7,
    # Softness of both knees, in density, at the reference width. Larger blends over a wider
    # tonal span. The Toe and Shoulder Width sliders scale this about their own default, so
    # the reference must equal ExposureConfig.toe_width / shoulder_width.
    "transfer_knee_width": 0.45,
    "transfer_width_ref": 2.5,
}


def zone_geometry() -> Tuple[float, float, float]:
    """(shadow centre, highlight centre, sharpness) for Zone Density, in density.

    Derived from the print path's zone geometry by **tonal position**, not by copying its
    density numbers. The two curves do not share a density scale: a print runs d_min 0.06
    to d_max 2.3, the transfer curve runs 0 to TRANSFER_DENSITY_RANGE. Carrying the raw
    1.50 across put the shadow centre 64% of the way to black on a print but only 50% of
    the way here, so the slider reached well into the midtones on a slide — on a dusk
    frame it lifted a quarter of the picture by 0.12 and the midtones by 0.036.

    The sharpness scales with the range for the same reason, so the transition occupies
    the same share of the scale rather than the same number of decades.
    """
    c = EXPOSURE_CONSTANTS
    d_min, d_max = float(c["d_min"]), float(c["d_max"])
    span = d_max - d_min
    anchor = float(c["anchor_target_density"])
    shadow = (anchor + float(c["zone_density_shadow_offset"]) - d_min) / span
    highlight = (anchor + float(c["zone_density_highlight_offset"]) - d_min) / span
    return (
        shadow * TRANSFER_DENSITY_RANGE,
        highlight * TRANSFER_DENSITY_RANGE,
        float(c["zone_density_sharpness"]) * span / TRANSFER_DENSITY_RANGE,
    )


def wb_split_geometry() -> Tuple[float, float]:
    """(centre, sharpness) for the Shadows/Highlights white-balance split, in density.

    Same tonal-position mapping as zone_geometry(): the print path's regional-CMY
    blend (logic.py, "Regional CMY") centres on the paper's anchor density with a
    fixed sharpness of 3.0 on the print's own scale. Both carried across by fraction
    of span, not by the raw numbers, for the same reason zone_geometry gives.
    """
    c = EXPOSURE_CONSTANTS
    d_min, d_max = float(c["d_min"]), float(c["d_max"])
    span = d_max - d_min
    anchor = float(c["anchor_target_density"])
    centre = (anchor - d_min) / span * TRANSFER_DENSITY_RANGE
    sharpness = 3.0 * span / TRANSFER_DENSITY_RANGE
    return centre, sharpness


def display_rendering(scene_linear: np.ndarray) -> np.ndarray:
    """
    Scene-linear -> display-linear: the standard rendering a raw converter opens with.

    Krzysztof Narkowicz's closed-form fit to the ACES RRT + sRGB ODT — a published,
    camera-agnostic filmic curve with a real toe and shoulder, chosen over reproducing
    any one vendor's proprietary look. It is monotonic on [0, inf), maps 0 -> 0 and
    approaches 1, so scene highlights roll off toward display white instead of stopping
    wherever the sensor's white level happened to fall.

    Without this the render is a bare linear-to-gamma encode, which is the one thing no
    consumer converter does: measured against a Lightroom export of the same frame, that
    left mid-tones ~1.5 EV dark and highlights short of white by a factor of two.
    """
    x = np.maximum(np.asarray(scene_linear, dtype=np.float32), 0.0)
    num = x * (np.float32(2.51) * x + np.float32(0.03))
    den = x * (np.float32(2.43) * x + np.float32(0.59)) + np.float32(0.14)
    return np.clip(num / np.maximum(den, np.float32(1e-8)), 0.0, 1.0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, overflow-safe for large |x| (mirrors logic.py::_fast_sigmoid)."""
    return (0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float32)))).astype(np.float32)


ZONE_BLACK_TAPER = 1.0


def _black_taper(d: np.ndarray, density_range: float) -> np.ndarray:
    """Fades a shadow lift back to nothing at the bottom of the window.

    On a print, Zone Density is bounded by paper black -- a shadow burn cannot exceed
    d_max. This curve has no paper, so without a bound a lift walks the black point up
    with it and the frame simply stops having blacks. Smoothstep so the taper adds no kink.
    """
    t = np.clip((np.float32(density_range) - d) / np.float32(ZONE_BLACK_TAPER), 0.0, 1.0)
    return (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32)


def _softplus(x: np.ndarray, width: float) -> np.ndarray:
    """width * log(1 + exp(x / width)), overflow-safe: ~x for x >> width, ~0 for x << -width."""
    t = x / width
    return (width * (np.logaddexp(0.0, -np.abs(t)) + np.maximum(t, 0.0))).astype(np.float32)


def transfer_widths(config: ExposureConfig) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Per-channel knee softness, in density, from the Toe/Shoulder Width sliders. Scaled
    about transfer_width_ref so each slider's default lands exactly on transfer_knee_width.
    """
    c = TRANSFER_CONSTANTS
    base = float(c["transfer_knee_width"])
    ref = float(c["transfer_width_ref"])
    tw3, sw3 = per_channel_widths(
        float(config.toe_width),
        float(config.shoulder_width),
        (config.toe_width_trim_red, config.toe_width_trim_green, config.toe_width_trim_blue),
        (config.shoulder_width_trim_red, config.shoulder_width_trim_green, config.shoulder_width_trim_blue),
    )
    scale = base / ref
    return (
        (tw3[0] * scale, tw3[1] * scale, tw3[2] * scale),
        (sw3[0] * scale, sw3[1] * scale, sw3[2] * scale),
    )


def transfer_assumed_anchor() -> float:
    """Fraction of the fixed window a properly exposed frame's own metered midtone is
    expected to sit at -- the position Auto Density's partial metering blends toward
    (see measure_anchor_from_log's `assumed`). transfer_contrast_pivot already places
    mid-grey at this density independently; expressed as a fraction here because the
    window's own span is what the blend needs to know."""
    return float(TRANSFER_CONSTANTS["transfer_contrast_pivot"]) / TRANSFER_DENSITY_RANGE


def transfer_shadow_reach_density() -> float:
    """shadow_reach_density carried onto this curve's own scale by tonal position, like
    zone_geometry: the two curves don't share a density scale, so the raw number would
    land at a different fraction of black."""
    c = EXPOSURE_CONSTANTS
    d_min, span = float(c["d_min"]), float(c["d_max"]) - float(c["d_min"])
    return (float(c["shadow_reach_density"]) - d_min) / span * TRANSFER_DENSITY_RANGE


def transfer_highlight_hold_density() -> float:
    """highlight_hold_density carried onto this curve's own scale, same technique."""
    c = EXPOSURE_CONSTANTS
    d_min, span = float(c["d_min"]), float(c["d_max"]) - float(c["d_min"])
    return (float(c["highlight_hold_density"]) - d_min) / span * TRANSFER_DENSITY_RANGE


def transfer_auto_terms(
    exposure: ExposureConfig,
    manual_offset: float,
    manual_contrast: float,
    textural_range: Optional[float],
    anchor: Optional[float],
    shadow_point: Optional[float],
    highlight_point: Optional[float],
) -> Tuple[float, float, float]:
    """
    (exposure_offset, contrast, highlight_density_auto) with Auto Density/Auto Grade
    folded onto the manual values -- single source for CPU and GPU. Mirrors the paper
    path's own anchor placement, effective_grade_range, Shadow Reach and Highlight
    Hold, restated on this curve's plain density-linear model instead of the paper's
    toe/shoulder one. A None input (the toggle off, or a raw un-normalized slide,
    which never meters -- see is_transfer_path) leaves every term at its manual value.
    """
    c = TRANSFER_CONSTANTS
    pivot = float(c["transfer_contrast_pivot"])
    R = TRANSFER_DENSITY_RANGE

    # Auto Density: place the metered anchor at the same pivot Grade already rotates
    # about, so a later contrast change does not re-shift the brightness placed here.
    auto_offset = 0.0
    if exposure.auto_exposure and anchor is not None:
        auto_offset = float(anchor) * R - pivot
    offset = manual_offset + auto_offset

    contrast = manual_contrast
    if exposure.auto_normalize_contrast and textural_range is not None:
        effective_range = effective_grade_range(True, R, textural_range)
        k_ref = grade_to_slope(float(c["transfer_grade_ref"]), R)
        if k_ref > 1e-9:
            contrast = grade_to_slope(float(exposure.grade), effective_range) / k_ref

    # Shadow Reach: never lowers contrast, only raises it so the textured dark tail
    # still reaches shadow_reach_density. Same guard as the paper path: too little
    # span between anchor and shadow_point to solve a slope from.
    if exposure.auto_normalize_contrast and anchor is not None and shadow_point is not None:
        span = float(shadow_point) - float(anchor)
        if span > 1e-6:
            d_shadow = float(shadow_point) * R - offset
            denom = d_shadow - pivot
            if denom > 1e-6:
                needed = (transfer_shadow_reach_density() - pivot) / denom
                contrast = min(max(contrast, needed), float(EXPOSURE_CONSTANTS["slope_max"]))

    # Highlight Hold: an automatic highlight-zone burn, riding the same Zone Density
    # kernel the Shadows/Highlights Density sliders already use on this curve. Never
    # lifts; 0 once the tone already holds.
    highlight_auto = 0.0
    if exposure.auto_normalize_contrast and highlight_point is not None:
        target = transfer_highlight_hold_density()
        d_highlight = pivot + (float(highlight_point) * R - offset - pivot) * contrast
        if d_highlight < target:
            _sh_c, hi_c, k_zone = zone_geometry()
            w_hi = 1.0 - _fast_sigmoid(k_zone * (d_highlight - hi_c))
            highlight_auto = min((target - d_highlight) / max(w_hi, 1e-6), float(EXPOSURE_CONSTANTS["highlight_hold_max"]))

    return offset, contrast, highlight_auto


def transfer_curve_params(
    config: ExposureConfig,
) -> Tuple[float, float, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    (exposure_density_offset, contrast, toe3, shoulder3) for the transfer curve.

    exposure_density_offset is a density the curve subtracts, so positive = lighter.
    Returned rather than applied so the GPU can upload the identical numbers.
    """
    c = TRANSFER_CONSTANTS
    stops = (1.0 - float(config.density)) * float(c["transfer_density_stops"])
    exposure_offset = stops * float(np.log10(2.0))

    grade = float(config.grade)
    contrast = float(c["transfer_grade_ref"]) / grade if grade > 1e-6 else 1.0

    toe3, sh3 = per_channel_toe_shoulder(
        float(config.toe),
        float(config.shoulder),
        (config.toe_trim_red, config.toe_trim_green, config.toe_trim_blue),
        (config.shoulder_trim_red, config.shoulder_trim_green, config.shoulder_trim_blue),
    )
    return exposure_offset, contrast, toe3, sh3


def apply_transfer_curve(
    img_norm: ImageBuffer,
    exposure_offset: float,
    contrast: float,
    toe: Tuple[float, float, float],
    shoulder: Tuple[float, float, float],
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    toe_widths: Optional[Tuple[float, float, float]] = None,
    shoulder_widths: Optional[Tuple[float, float, float]] = None,
    density_range: float = TRANSFER_DENSITY_RANGE,
    shadow_density: float = 0.0,
    highlight_density: float = 0.0,
    shadow_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cast_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    cast_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    positive_source: bool = False,
    separation: float = 1.0,
    separation_trims: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    damping: float = 0.0,
) -> ImageBuffer:
    """
    Normalized log density -> scene-linear positive.

    With exposure_offset/cmy at 0, contrast 1, both knees 0 and the cast affine at
    (1, 0), the scene stage is an exact identity: D = density_range * n inverts the
    normalization, and 10**-D returns the capture. Every term below is written to vanish
    at its neutral value, so that is exact in float32, not approximate. The baseline gain
    and `display_rendering` then apply: they are how a raw capture is shown, not an
    adjustment of it. `positive_source` skips both and passes the scene through unshaped.

    `cast_offset` arrives already scaled by density_range (see neutral_axis_affine).

    `separation`/`separation_trims` are Dye Separation and its per-channel trims,
    applied after the curve shapes each channel and before decode. They share their
    math with the print path's resolve_saturation_matrix but not its paper crosstalk:
    this curve has no paper dye matrix to compose into, so each channel scales its own
    deviation from the frame's mean density directly rather than through a 3x3 matmul.
    `damping` is Separation Damping, tapering each channel's own k by each pixel's own
    chroma (see logic.separation_damping_gain); inert at separation 1.0, same as on the print.
    """
    c = TRANSFER_CONSTANTS
    base_width = float(c["transfer_knee_width"])
    tw3 = toe_widths or (base_width, base_width, base_width)
    sw3 = shoulder_widths or (base_width, base_width, base_width)
    pivot = float(c["transfer_contrast_pivot"])
    toe_knee = float(c["transfer_toe_knee"])
    sh_knee = float(c["transfer_shoulder_knee"])

    n = np.asarray(img_norm, dtype=np.float32)
    dens = np.empty_like(n)
    for ch in range(3):
        d = n[:, :, ch] * np.float32(density_range)

        # Cast Removal: the channel's neutral refs onto green's. First, so every control
        # below shapes the corrected signal.
        if cast_gain[ch] != 1.0 or cast_offset[ch] != 0.0:
            d = d * np.float32(cast_gain[ch]) + np.float32(cast_offset[ch])

        if exposure_offset != 0.0 or cmy_offsets[ch] != 0.0:
            # cmy_offsets arrive in normalized space, because filtration_offsets divides by the
            # channel's stretch range, so they scale back up by the same range.
            d = d - np.float32(exposure_offset) + np.float32(cmy_offsets[ch] * density_range)

        if contrast != 1.0:
            d = np.float32(pivot) + (d - np.float32(pivot)) * np.float32(contrast)

        # Shadows/Highlights WB: regional CMY, using the print path's own kernel
        # (logic.py, "Regional CMY") and geometry (wb_split_geometry).
        if shadow_cmy[ch] != 0.0 or highlight_cmy[ch] != 0.0:
            wb_c, wb_k = wb_split_geometry()
            w_sh = _sigmoid(np.float32(wb_k) * (d - np.float32(wb_c)))
            w_hi = np.float32(1.0) - w_sh
            d = d + np.float32(shadow_cmy[ch]) * w_sh + np.float32(highlight_cmy[ch]) * w_hi

        # Zone Density: mid-sparing brightness offsets, using the print path's own kernel and
        # weights (logic.py, "Zone Density (ΔD)"). Positive adds density, so it darkens: the
        # Density convention, the opposite sign to a Lightroom Shadows slider. Runs after
        # contrast and before the knees, as it does on the print.
        if shadow_density != 0.0 or highlight_density != 0.0:
            sh_c, hi_c, k = zone_geometry()
            w_sh = _sigmoid(np.float32(k) * (d - np.float32(sh_c)))
            w_sh = w_sh * _black_taper(d, density_range)
            w_hi = np.float32(1.0) - _sigmoid(np.float32(k) * (d - np.float32(hi_c)))
            d = d + np.float32(shadow_density) * w_sh + np.float32(highlight_density) * w_hi

        # Shadows sit at high density and highlights at low, so the toe compresses above its knee
        # and the shoulder below its own.
        if toe[ch] != 0.0:
            d = d - np.float32(toe[ch]) * _softplus(d - np.float32(toe_knee), tw3[ch])
        if shoulder[ch] != 0.0:
            d = d + np.float32(shoulder[ch]) * _softplus(np.float32(sh_knee) - d, sw3[ch])

        dens[:, :, ch] = d

    sep_k3 = per_channel_dye_separation(separation, separation_trims)
    if sep_k3 != (1.0, 1.0, 1.0):
        # M(k) = diag(k) + (1-k)*J (papers.resolve_saturation_matrix): each channel
        # scales its own deviation from the frame's mean density by its own k, since
        # this curve has no paper base to measure above and no dye matrix to fold the
        # per-layer trims into instead.
        mean = dens.mean(axis=2, keepdims=True)
        e = dens - mean
        if damping > 0.0:
            # Separation Damping makes each channel's k chroma-dependent per pixel, from
            # the same chroma but each channel's own k (see separation_damping_gain_np).
            chroma = np.sqrt(((e[:, :, 0] - e[:, :, 1]) ** 2 + (e[:, :, 1] - e[:, :, 2]) ** 2 + (e[:, :, 0] - e[:, :, 2]) ** 2) / 3.0)
            ref_spread = float(EXPOSURE_CONSTANTS["separation_damping_ref_spread"])
            k_eff = np.stack(
                [separation_damping_gain_np(sep_k3[ch], damping, chroma, ref_spread) for ch in range(3)],
                axis=2,
            )
            dens = mean + k_eff * e
        else:
            k3 = np.asarray(sep_k3, dtype=np.float32)
            dens = mean + k3[np.newaxis, np.newaxis, :] * e

    out = np.power(np.float32(10.0), -dens, dtype=np.float32)

    # Baseline and display rendering last, so the controls above shape the scene and this
    # only decides how the scene is shown. A finished positive sits nowhere below a sensor
    # white level, so neither belongs: it passes through at scene scale, clipped to the
    # same range display_rendering outputs.
    if positive_source:
        return ensure_image(np.clip(out, 0.0, 1.0))
    gain = np.float32(2.0 ** float(c["transfer_baseline_ev"]))
    return ensure_image(display_rendering(out * gain))


def transfer_bounds(density_range: float = TRANSFER_DENSITY_RANGE) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    The fixed (floors, ceils) the transparency path normalizes with — the decoder's
    white level down `density_range` decades, identical for every frame. Content-
    independent on purpose: measured bounds are what make two exposures of one slide
    converge on the same render.
    """
    return (0.0, 0.0, 0.0), (-density_range, -density_range, -density_range)


def is_transfer_path(process_mode: str, e6_normalize: bool, positive_source: bool = False, render_intent: Optional[str] = None) -> bool:
    """Single source of truth for the mode test, so CPU/GPU/UI cannot drift apart.

    True on an as-captured Slide (Normalize off), and on a frame marked Positive: a
    file already positivized before NegPy saw it -- a scanned print, an export from
    other software, a negative the scanner inverted itself -- has nothing left to meter
    or invert. Only a Slide config carries Positive (ProcessConfig.__post_init__); the
    flag is read here for callers that pass the fields apart."""
    from negpy.features.exposure.models import RenderIntent
    from negpy.features.process.models import ProcessMode

    if render_intent == RenderIntent.FLAT:
        return False
    if process_mode == ProcessMode.E6:
        return not e6_normalize
    return positive_source
