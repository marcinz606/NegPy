"""
Transparency transfer curve — the render of every slide.

A slide is captured close to how it should look, so the render starts from the capture
and the controls deviate from there. At default settings the scene stage is the exact
inverse of the fixed-bounds normalization in `processor.py` (see
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
`reference_print_density` places mid-grey at a print's mid-tone, and the midtone snap
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

from typing import Any, Dict, Mapping, Optional, Tuple

import negpy_core
import numpy as np

from negpy.domain.types import LUMA_R, LUMA_G, LUMA_B, ImageBuffer
from negpy.features.exposure.logic import (
    _fast_sigmoid,
    effective_grade_range,
    grade_to_slope,
    per_channel_dye_separation,
    per_channel_toe_shoulder,
    per_channel_widths,
    slope_to_grade,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.normalization import LogNegativeBounds, measure_neutral_axis_from_log
from negpy.features.process.models import ProcessConfig, per_channel_point_offsets
from negpy.kernel.image.validation import ensure_image

#: Log-density window the fixed-bounds normalization maps to [0, 1]. The floor is the
#: decoder's white level (density 0), so the mapping is anchored to the capture, not to
#: frame content. That is what keeps a bracketed set rendering as a bracketed set.
TRANSFER_DENSITY_RANGE = 3.0

#: Luma percentiles of a Positive frame's meter grid that span Cast Removal's neutral-axis window.
POSITIVE_AXIS_PERCENTILES = (1.0, 99.0)
#: Narrowest Positive window, in decades: a flatter frame takes the fixed window.
POSITIVE_AXIS_MIN_RANGE = 1e-3
#: Decades above the window's dark end that a cell must clear to count in the retry window.
POSITIVE_AXIS_FLOOR_MARGIN = 0.1

#: Dye Separation's channel cap: a soft ceiling at the higher of this density and the
#: pixel's lightest channel plus the spread below. Mirrored in transfer.wgsl.
SEPARATION_CAP_LEVEL = 2.5
SEPARATION_CAP_SPREAD = 1.5
SEPARATION_CAP_SOFTNESS = 0.15

TRANSFER_CONSTANTS = {
    # Reference grade (ISO R) that means "no contrast change". MUST equal the grade the app
    # ships in DEFAULT_WORKSPACE_CONFIG, or the transfer stops being identity at defaults.
    # Mirrored rather than imported to keep this module dependency-free;
    # test_transparency_transfer.py asserts the two agree.
    "transfer_grade_ref": 115.0,
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
    anchor = float(c["reference_print_density"])
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
    anchor = float(c["reference_print_density"])
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


ZONE_BLACK_TAPER = 1.0


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


def transfer_shadow_hold_start() -> float:
    """Where Shadow Hold starts lifting on this curve: the shadow reach density plus
    shadow_hold_overshoot, carried by tonal position like transfer_shadow_reach_density."""
    c = EXPOSURE_CONSTANTS
    span = float(c["d_max"]) - float(c["d_min"])
    return transfer_shadow_reach_density() + float(c["shadow_hold_overshoot"]) / span * TRANSFER_DENSITY_RANGE


def transfer_auto_terms(
    exposure: ExposureConfig,
    manual_offset: float,
    manual_contrast: float,
    textural_range: Optional[float],
    anchor: Optional[float],
    shadow_point: Optional[float],
    highlight_point: Optional[float],
) -> Tuple[float, float, float, float]:
    """(exposure_offset, contrast, highlight_density_auto, shadow_density_auto) with the autos folded onto the
    manual values, for CPU and GPU. The paper path's autos restated on this density-linear curve; a toggle off
    or a None meter leaves its terms manual."""
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

    # Shadow Hold: a partial shadow-zone lift on the same tail as Shadow Reach. Never burns.
    shadow_auto = 0.0
    strength = float(EXPOSURE_CONSTANTS["shadow_hold_strength"])
    if exposure.auto_normalize_contrast and shadow_point is not None and strength > 0.0:
        d_shadow = pivot + (float(shadow_point) * R - offset - pivot) * contrast
        over = d_shadow - transfer_shadow_hold_start()
        if over > 0.0:
            shadow_auto = -min(strength * over, float(EXPOSURE_CONSTANTS["shadow_hold_max"]))

    return offset, contrast, highlight_auto, shadow_auto


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


def _transfer_totals(exposure: ExposureConfig, metrics: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    """(manual offset, total offset, total contrast, highlight hold, shadow hold) for `exposure`."""
    manual_offset, manual_contrast, _, _ = transfer_curve_params(exposure)
    offset, contrast, hl_auto, sh_auto = transfer_auto_terms(
        exposure,
        manual_offset,
        manual_contrast,
        metrics.get("textural_range"),
        metrics.get("metered_anchor"),
        metrics.get("shadow_point"),
        metrics.get("highlight_point"),
    )
    return manual_offset, offset, contrast, hl_auto, sh_auto


def _offset_per_density() -> float:
    return float(TRANSFER_CONSTANTS["transfer_density_stops"]) * float(np.log10(2.0))


def transfer_shown_values(exposure: ExposureConfig, metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Transfer-curve twin of auto_sliders.print_shown_values: the manual Density, Grade
    and Shadows/Highlights Density that print what the active, metered autos print."""
    _, offset, contrast, hl_auto, sh_auto = _transfer_totals(exposure, metrics)
    shown: Dict[str, float] = {}
    if exposure.auto_exposure and metrics.get("metered_anchor") is not None:
        shown["density"] = 1.0 - offset / _offset_per_density()
    if exposure.auto_normalize_contrast and metrics.get("textural_range") is not None:
        shown["grade"] = float(TRANSFER_CONSTANTS["transfer_grade_ref"]) / contrast
    if exposure.auto_normalize_contrast and metrics.get("highlight_point") is not None:
        shown["highlight_density"] = exposure.highlight_density + hl_auto
    if exposure.auto_normalize_contrast and metrics.get("shadow_point") is not None:
        shown["shadow_density"] = exposure.shadow_density + sh_auto
    return shown


def transfer_stored_value(exposure: ExposureConfig, metrics: Mapping[str, Any], field: str, shown: float) -> float:
    """Inverse of transfer_shown_values for one field."""
    if field not in transfer_shown_values(exposure, metrics):
        return shown
    manual_offset, offset, _, hl_auto, sh_auto = _transfer_totals(exposure, metrics)
    if field == "density":
        return shown + (offset - manual_offset) / _offset_per_density()
    if field == "grade":
        c = TRANSFER_CONSTANTS
        k_ref = grade_to_slope(float(c["transfer_grade_ref"]), TRANSFER_DENSITY_RANGE)
        slope = float(c["transfer_grade_ref"]) / shown * k_ref
        return slope_to_grade(slope, effective_grade_range(True, TRANSFER_DENSITY_RANGE, metrics["textural_range"]))
    if field == "shadow_density":
        return shown - sh_auto
    return shown - hl_auto


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
    applied after the curve shapes each channel and before decode. Each channel scales
    its own deviation from a reference density by its own k. The reference is the mean of
    the channels, each softly capped at the higher of SEPARATION_CAP_LEVEL and the pixel's
    lightest channel plus SEPARATION_CAP_SPREAD, in capture density, then shaped like the
    channel itself. Nothing here bounds a channel the way the paper curve does on the
    print, and a channel far denser than the rest and near the window's dense end is
    mostly noise, which the plain mean would spread onto the other two. Grays and
    ordinary colors never reach the cap, so for them it is the plain mean.
    `damping` is Separation Damping, tapering each channel's own k by each pixel's own
    chroma (see logic.separation_damping_gain); inert at separation 1.0, same as on the print.
    """
    c = TRANSFER_CONSTANTS
    base_width = float(c["transfer_knee_width"])
    wb_c, wb_k = wb_split_geometry()
    sh_c, hi_c, zone_k = zone_geometry()
    sep_k3 = per_channel_dye_separation(separation, separation_trims)
    params = {
        "density_range": density_range,
        "cast_gain": tuple(cast_gain),
        "cast_offset": tuple(cast_offset),
        "cast_on": tuple(cast_gain[ch] != 1.0 or cast_offset[ch] != 0.0 for ch in range(3)),
        "exposure_offset": exposure_offset,
        # cmy_offsets arrive in normalized space, because filtration_offsets divides by the
        # channel's stretch range, so they scale back up by the same range.
        "cmy": tuple(cmy_offsets[ch] * density_range for ch in range(3)),
        "exposure_on": tuple(exposure_offset != 0.0 or cmy_offsets[ch] != 0.0 for ch in range(3)),
        "contrast": contrast,
        "contrast_on": contrast != 1.0,
        "pivot": float(c["transfer_contrast_pivot"]),
        "shadow_cmy": tuple(shadow_cmy),
        "highlight_cmy": tuple(highlight_cmy),
        "regional_on": tuple(shadow_cmy[ch] != 0.0 or highlight_cmy[ch] != 0.0 for ch in range(3)),
        "wb_c": wb_c,
        "wb_k": wb_k,
        "shadow_density": shadow_density,
        "highlight_density": highlight_density,
        "zone_on": shadow_density != 0.0 or highlight_density != 0.0,
        "zone_sh": sh_c,
        "zone_hi": hi_c,
        "zone_k": zone_k,
        "black_taper": ZONE_BLACK_TAPER,
        "toe": tuple(toe),
        "shoulder": tuple(shoulder),
        "toe_width": tuple(toe_widths or (base_width,) * 3),
        "shoulder_width": tuple(shoulder_widths or (base_width,) * 3),
        "toe_knee": float(c["transfer_toe_knee"]),
        "shoulder_knee": float(c["transfer_shoulder_knee"]),
        "sep_k": sep_k3,
        "sep_on": sep_k3 != (1.0, 1.0, 1.0),
        "cap_level": SEPARATION_CAP_LEVEL,
        "cap_offset": SEPARATION_CAP_SPREAD - SEPARATION_CAP_LEVEL,
        "cap_softness": SEPARATION_CAP_SOFTNESS,
        "damping": damping,
        "damping_on": damping > 0.0,
        "ref_spread": float(EXPOSURE_CONSTANTS["separation_damping_ref_spread"]),
        "positive_source": bool(positive_source),
        "gain": 2.0 ** float(c["transfer_baseline_ev"]),
    }
    return ensure_image(negpy_core.apply_transfer_curve(np.ascontiguousarray(img_norm, dtype=np.float32), params))


def transfer_point_offsets(process: ProcessConfig) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """White/Black Point on the fixed window, negated: a positive reverses the floor/ceil
    roles. Single source for the CPU base and the GPU uniform pack."""
    wp3, bp3 = per_channel_point_offsets(process)
    return (-wp3[0], -wp3[1], -wp3[2]), (-bp3[0], -bp3[1], -bp3[2])


def transfer_bounds(density_range: float = TRANSFER_DENSITY_RANGE) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    The fixed (floors, ceils) the transparency path normalizes with — the decoder's
    white level down `density_range` decades, identical for every frame. Content-
    independent on purpose: measured bounds are what make two exposures of one slide
    converge on the same render.
    """
    return (0.0, 0.0, 0.0), (-density_range, -density_range, -density_range)


def _grid_luma(grid: np.ndarray) -> np.ndarray:
    """Luma-weighted mean of the per-channel log densities, one value per grid cell."""
    luma = LUMA_R * grid[..., 0] + LUMA_G * grid[..., 1] + LUMA_B * grid[..., 2]
    return np.ascontiguousarray(luma, dtype=np.float32).ravel()


def _positive_window(bright: float, dark: float) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    if np.isfinite(dark) and np.isfinite(bright) and bright - dark > POSITIVE_AXIS_MIN_RANGE:
        return (float(bright),) * 3, (float(dark),) * 3
    return None


def transfer_axis_bounds(grid: np.ndarray, positive: bool) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    The (floors, ceils) Cast Removal's neutral-axis bands are placed in on the transfer path.
    A Positive file has no sensor white to anchor to, so its window spans the percentiles of
    its luma-weighted log density, the same on all three channels so the cast stays
    measurable. Any other frame, or a flat one, takes `transfer_bounds()`.
    """
    if positive:
        dark, bright = negpy_core.percentiles(_grid_luma(grid), list(POSITIVE_AXIS_PERCENTILES))
        window = _positive_window(bright, dark)
        if window is not None:
            return window
    return transfer_bounds()


def measure_transfer_neutral_axis(grid: np.ndarray, positive: bool) -> Optional[tuple]:
    """
    Cast Removal's neutral axis on the transfer path, measured in `transfer_axis_bounds`.
    A Positive frame that finds none is measured once more with the dark end taken over the
    cells above `POSITIVE_AXIS_FLOOR_MARGIN` of it, so a black border or clipped black that
    filled the shadow band falls outside the window.
    """
    bounds = transfer_axis_bounds(grid, positive)
    refs = measure_neutral_axis_from_log(grid, LogNegativeBounds(*bounds), None, 0.0, exclude_subjects=False)
    if refs is not None or not positive or bounds == transfer_bounds():
        return refs
    luma = _grid_luma(grid)
    above = np.ascontiguousarray(luma[luma > bounds[1][0] + POSITIVE_AXIS_FLOOR_MARGIN])
    if above.size == 0:
        return None
    window = _positive_window(bounds[0][0], negpy_core.percentiles(above, [POSITIVE_AXIS_PERCENTILES[0]])[0])
    if window is None:
        return None
    return measure_neutral_axis_from_log(grid, LogNegativeBounds(*window), None, 0.0, exclude_subjects=False)
