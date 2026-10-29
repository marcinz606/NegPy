from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

import negpy_core
import numpy as np

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R, ImageBuffer
from negpy.features.exposure.bw_filter import collapse_matrix
from negpy.features.geometry.logic import AUTOCROP_DETECT_RES, _normalize_detection_input
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.logic import percentile_from_sorted
from negpy.kernel.image.validation import ensure_image

if TYPE_CHECKING:
    from negpy.features.process.models import ProcessConfig

# Bump in any change that alters the bounds a frame measures for unchanged settings. Saved
# per-frame bounds from an older version are dropped at startup (migrations/bounds.py).
BOUNDS_VERSION = 2

# Contrast Mask blur sigma, per-cent of the analysis grid's short side. Of the grid, not of
# the render, so preview and export mask alike. Near the Frontier HyperTone blur's spread.
MASK_BLUR_PERCENT = 4.0
# Diffusion's spread, one diffuser at the lens that a printer does not vary between prints:
# (weight, sigma in per-cent of the printed frame's short side) Gaussians that fit a sum of
# radial exponentials, a halo and a bloom with a power-law tail. The weights sum to 1.
DIFFUSION_PSF = ((0.0421, 0.213), (0.1576, 0.449), (0.2723, 0.926), (0.3540, 2.017), (0.1740, 5.48))


class LogNegativeBounds:
    """
    D-min / D-max container.
    """

    def __init__(self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]):
        self.floors = floors
        self.ceils = ceils


def _rect_roi(image_shape: tuple[int, ...], rect: tuple[float, float, float, float]) -> Optional[tuple[int, int, int, int]]:
    """A normalized (x1, y1, x2, y2) rect as a pixel ROI; None for a degenerate one, so a stray click can't blank analysis."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = rect
    roi = (int(min(y1, y2) * h), int(max(y1, y2) * h), int(min(x1, x2) * w), int(max(x1, x2) * w))
    return roi if roi[1] - roi[0] >= 2 and roi[3] - roi[2] >= 2 else None


def resolve_analysis_region(
    image_shape: tuple[int, ...],
    active_roi: Optional[tuple[int, int, int, int]],
    analysis_buffer: float,
    analysis_rect: Optional[tuple[float, float, float, float]],
    meter_rect: Optional[tuple[float, float, float, float]] = None,
) -> tuple[Optional[tuple[int, int, int, int]], float]:
    """Resolve the (roi, buffer) the meters should read.

    A freehand `analysis_rect` (normalized in the transformed image) wins over the crop
    ROI + centered buffer: it maps directly to a pixel ROI and disables the symmetric
    inset. Otherwise the crop `active_roi` and the `analysis_buffer` slider apply, and with
    no crop the `meter_rect` stands in for one.
    """
    if analysis_rect is not None:
        roi = _rect_roi(image_shape, analysis_rect)
        if roi is not None:
            return roi, 0.0
    if active_roi is None and meter_rect is not None:
        return _rect_roi(image_shape, meter_rect), analysis_buffer
    return active_roi, analysis_buffer


def get_analysis_crop(img: ImageBuffer, buffer_ratio: float) -> ImageBuffer:
    """
    Returns a center crop of the image for analysis purposes.
    The buffer_ratio (0.0 to 0.25) defines how much of the border to exclude.
    """
    if buffer_ratio <= 0:
        return img

    h, w = img.shape[:2]
    safe_buffer = min(max(buffer_ratio, 0.0), 0.3)

    cut_h = int(h * safe_buffer)
    cut_w = int(w * safe_buffer)

    return img[cut_h : h - cut_h, cut_w : w - cut_w]


def _block_median_grid(img_log: ImageBuffer) -> ImageBuffer:
    """
    Block-median prefilter to a fixed target grid: isolated extremes (speculars,
    dust pinholes) vanish inside their block's median, and statistics become nearly
    resolution-invariant since the grid size is constant.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    h, w = img_log.shape[:2]
    grid = int(EXPOSURE_CONSTANTS["analysis_grid"])
    b = int(np.ceil(max(h, w) / grid))
    if b <= 1 or h < b or w < b:
        return img_log
    hb, wb = (h // b) * b, (w // b) * b
    return negpy_core.block_median_grid(np.ascontiguousarray(img_log[:hb, :wb], dtype=np.float32), b)


def prefilter_log_grid(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> ImageBuffer:
    """
    Shared meter prefilter (log10 -> crop -> block-median grid), computed once and
    fed to the *_from_log meters. Re-prefiltering it (roi=None, buffer=0) is a no-op
    (_block_median_grid early-returns when b<=1), so results stay bit-exact.
    """
    img_log = to_log_density(image)
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)
    return _block_median_grid(img_log)


def measure_clip_fractions(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> tuple[float, float, float]:
    """
    Per-channel fraction of pixels at/above the sensor-white clip level (linear
    input). In a negative scan the film base and scene shadows sit near sensor
    white, so a clipped scan silently collapses distinct densities to D=0 —
    this feeds the scan-exposure warning, not any render math.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    h, w = image.shape[:2]
    y1, y2, x1, x2 = roi if roi else (0, h, 0, w)
    y2, x2 = min(y2, h), min(x2, w)
    y1, x1 = min(y1, y2), min(x1, x2)
    if analysis_buffer > 0:
        b = min(max(analysis_buffer, 0.0), 0.3)
        cut_h, cut_w = int((y2 - y1) * b), int((x2 - x1) * b)
        y1, y2, x1, x2 = y1 + cut_h, y2 - cut_h, x1 + cut_w, x2 - cut_w
    level = float(EXPOSURE_CONSTANTS["scan_clip_level"])
    return negpy_core.clip_fractions(np.ascontiguousarray(image, dtype=np.float32), (y1, y2, x1, x2), level)


def effective_crosstalk_matrix(process: "ProcessConfig", process_mode: Optional[str]) -> Optional[np.ndarray]:
    """
    Mode-aware resolve: the configured unmix only when the profile was derived for the
    film being processed, otherwise None.

    Mirrors effective_paper_profile. A crosstalk matrix describes a dye set's unwanted
    absorptions, and C-41 and E-6 do not share one — every bundled profile is a color
    negative stock, so without this gate a slide silently gets a negative's correction.
    Legacy configs carry no `crosstalk_process` and default to C-41, which is what every
    profile that predates the field actually is.
    """

    profile_mode = str(getattr(process, "crosstalk_process", ProcessMode.C41) or ProcessMode.C41)
    if process_mode is not None and profile_mode != str(process_mode):
        return None
    return resolve_crosstalk_matrix(process.crosstalk_strength, process.crosstalk_matrix)


def resolve_crosstalk_matrix(strength: float, matrix: Optional[tuple]) -> Optional[np.ndarray]:
    """
    Effective spectral-crosstalk (dye-unmix) matrix — identity↔calibration blend
    by strength, row-normalized so neutral gray is preserved (rows redistribute
    channel differences only) — or None when off. Applied to raw NEGATIVE log
    densities before any metering/stretch; since the op is linear and
    img_log = -D, applying it to log values is exact.
    """
    if float(strength) <= 0.0:
        return None
    from negpy.features.process.models import DEFAULT_CROSSTALK_MATRIX

    m = matrix if matrix is not None else DEFAULT_CROSSTALK_MATRIX
    cal = np.array(m, dtype=np.float64).reshape(3, 3)
    applied = np.eye(3) * (1.0 - float(strength)) + cal * float(strength)
    row_sums = np.sum(applied, axis=1, keepdims=True)
    return applied / np.maximum(row_sums, 1e-6)


def unmix_log_image(img_log: ImageBuffer, matrix: Optional[np.ndarray]) -> ImageBuffer:
    """Apply the unmix matrix to a (H, W, 3) log-density image; identity when None."""
    if matrix is None:
        return img_log
    return np.einsum("hwc,kc->hwk", img_log.astype(np.float32, copy=False), matrix.astype(np.float32))


def sorted_channel_grid(img_log: ImageBuffer) -> np.ndarray:
    """
    Per-channel sorted values of a prefiltered grid, shape (n, 3). The clip percentiles
    index into it, so a clip drag re-reads one sort instead of re-partitioning the grid.
    """
    return np.sort(img_log.reshape(-1, 3), axis=0)


def to_log_density(image: ImageBuffer) -> ImageBuffer:
    """Linear scan -> log10 density clamped to 1e-6..1 (NaN to 1e-6), the domain every meter reads."""
    return negpy_core.to_log_density(np.ascontiguousarray(image, dtype=np.float32))


def measure_neutral_axis_from_log(
    img_log: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    *,
    exclude_subjects: bool = True,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """Per-channel neutral axis: median raw-log density of the lowest-chroma pixels in a highlight, midtone and shadow band.
    Pass 1 uses a loose cap so a strong but correctable cast keeps the axis; its mid+shadow correction re-ranks chroma for a
    strict pass 2. Returns (midtone, shadow, highlight or None, confidence in [0, 1]); None without a midtone or shadow set.
    `exclude_subjects` drops foliage- and sky-sector cells first. Only a per-channel stretch makes the window's neutral a
    gray, so a caller metering against the slide's fixed window passes False."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    c = EXPOSURE_CONSTANTS
    consts = (
        float(c["neutral_axis_chroma_quantile"]),
        float(c["neutral_axis_chroma_cap"]),
        float(c["neutral_axis_first_pass_cap"]),
        int(c["neutral_axis_min_pixels"]),
        tuple(float(v) for v in c["neutral_axis_highlight_band"]),
        tuple(float(v) for v in c["neutral_axis_mid_band"]),
        tuple(float(v) for v in c["neutral_axis_shadow_band"]),
        (float(c["neutral_axis_foliage_chroma"]), *(float(v) for v in c["neutral_axis_foliage_hue"])),
        (float(c["neutral_axis_sky_chroma"]), *(float(v) for v in c["neutral_axis_sky_hue"])),
        float(c["neutral_axis_confidence_n0"]),
        (float(c["neutral_axis_agreement_deadzone"]), float(c["neutral_axis_agreement_scale"])),
        (float(c["neutral_axis_width_deadzone"]), float(c["neutral_axis_width_scale"])),
    )
    grid = np.ascontiguousarray(_block_median_grid(img_log), dtype=np.float32)
    floors = tuple(float(v) for v in bounds.floors)
    ceils = tuple(float(v) for v in bounds.ceils)
    return negpy_core.neutral_axis(grid, floors, ceils, exclude_subjects, consts)


def measure_neutral_axis(
    image: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """Linear-image wrapper around measure_neutral_axis_from_log."""
    img_log = to_log_density(image)
    return measure_neutral_axis_from_log(img_log, bounds, roi, analysis_buffer)


_OPEN_LIGHT_GAP = 0.08  # decades a brighter, smaller group must clear the base to be bare light


def measure_base_log(image: ImageBuffer) -> Optional[Tuple[float, float, float]]:
    """Per-channel log10 transmittance of the clear base: the median of the flat base blocks in
    the film box's margin on the uncropped frame. None when the scan shows no film box, the
    margin holds too few flat blocks, or the base is darker than the picture, because a
    negative's base is its thinnest film. The blocks group by brightness, the way negdensity.py
    reads the base; the film box comes from the crop's detector at its own resolution, whose
    edge walk is calibrated there."""
    det, _ = _normalize_detection_input(image, AUTOCROP_DETECT_RES)
    base = negpy_core.measure_base_log(np.ascontiguousarray(det, dtype=np.float32))
    return None if base is None else (base[0], base[1], base[2])


def log_from_val(val_rgb: Tuple[float, float, float], bounds: LogNegativeBounds) -> Tuple[float, float, float]:
    """A normalized-log triplet back to per-channel log10 transmittance, through the window it
    was normalized with. The trims cancel: the window that made the value unmakes it."""
    r, g, b = (float(f) + float(v) * (float(c) - float(f)) for v, f, c in zip(val_rgb, bounds.floors, bounds.ceils))
    return (r, g, b)


def density_above_base(
    base_log: Optional[Tuple[float, float, float]],
    bounds: LogNegativeBounds,
    prefiltered: np.ndarray,
    unmix: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float, float]]:
    """(Dmin, Dmed, Dmax) above the clear base, luminance-weighted: the thin bound and dense bound
    the frame is normalized with (before the trims; the roll's under Roll Analysis) and the metered
    median. `unmix` is the meters' crosstalk matrix, applied to the base so all four share one
    space. None without a base. Negatives only: the ceiling is the thin end."""
    if base_log is None:
        return None
    base_rgb = unmix_log_image(np.asarray(base_log, dtype=np.float32).reshape(1, 1, 3), unmix).reshape(3)
    w = np.array([LUMA_R, LUMA_G, LUMA_B], dtype=np.float64)
    base = float(np.dot(w, np.asarray(base_rgb, dtype=np.float64)))
    thin = float(np.dot(w, np.asarray(bounds.ceils, dtype=np.float64)))
    dense = float(np.dot(w, np.asarray(bounds.floors, dtype=np.float64)))
    median = negpy_core.luma_median(np.ascontiguousarray(prefiltered, dtype=np.float32))
    return (base - thin, base - median, base - dense)


def base_scan_density(base_log: Optional[Tuple[float, float, float]], unmix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Per-channel scan density of the clear base after the crosstalk unmix: the space the scan
    Parade plots. None without a base."""
    if base_log is None:
        return None
    return -unmix_log_image(np.asarray(base_log, dtype=np.float32).reshape(1, 1, 3), unmix).reshape(3).astype(np.float64)


def luminance_density_range(bounds: LogNegativeBounds) -> float:
    """
    Single global density range as a Rec.709 luminance weighting of the
    per-channel ranges. Replaces the green-only range so frames with a strong
    single-channel cast don't swing the slope as hard, while green still
    dominates so calibrated grade behaviour barely shifts. abs() keeps it
    sign-safe for a slide's reversed (f > c) window.
    """
    rr = abs(bounds.ceils[0] - bounds.floors[0])
    rg = abs(bounds.ceils[1] - bounds.floors[1])
    rb = abs(bounds.ceils[2] - bounds.floors[2])
    return float(LUMA_R * rr + LUMA_G * rg + LUMA_B * rb)


def _activity_gate() -> Tuple[int, float, float]:
    """The activity gate shared by the anchor and textural-range meters: the grid is tiled into
    sectors of 2x2 blocks of activity_block cells, and a sector votes when its four block means
    span more than activity_gate_density (Boyack & Juenger, US 5,724,456). Block means average
    grain out, and a flat sector beside a textured one stays out because only its own blocks are
    compared. Every cell votes when fewer than activity_min_fraction of the sectors pass."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    return int(c["activity_block"]), float(c["activity_gate_density"]), float(c["activity_min_fraction"])


def _meter_norm(bounds: LogNegativeBounds) -> list:
    """Per channel (floor, span, and whether each is a float64 operand): NumPy normalizes in
    float64 when the bounds hold NumPy float64 scalars, in float32 for Python floats."""
    epsilon = 1e-6
    out = []
    for ch in range(3):
        f = bounds.floors[ch]
        denom = bounds.ceils[ch] - f
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        wide = (np.result_type(np.float32, f) == np.float64, np.result_type(np.float32, denom) == np.float64)
        out.append((float(f), bool(wide[0]), float(denom), bool(wide[1])))
    return out


def measure_shadow_point_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """Normalized luma of the textured dark tail (shadow_reach_percentile), the tone
    Auto Grade's shadow reach prints at paper black."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)
    grid = np.ascontiguousarray(_block_median_grid(img_log), dtype=np.float32)
    return float(
        negpy_core.textured_percentile(grid, _meter_norm(bounds), _activity_gate(), float(EXPOSURE_CONSTANTS["shadow_reach_percentile"]))
    )


def measure_highlight_point_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """Normalized luma of the textured bright tail (highlight_hold_percentile), the tone
    Auto Grade's highlight hold keeps off paper white."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)
    grid = np.ascontiguousarray(_block_median_grid(img_log), dtype=np.float32)
    return float(
        negpy_core.textured_percentile(grid, _meter_norm(bounds), _activity_gate(), float(EXPOSURE_CONSTANTS["highlight_hold_percentile"]))
    )


def measure_anchor_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    assumed: Optional[float] = None,
) -> float:
    """
    Per-frame exposure anchor: where this negative's midtone sits in [0, 1],
    replacing the fixed assumed_anchor. Read over the textured cells of the
    block-median grid, as the mean of the trimmed window and its midpoint averaged
    (Boyack & Juenger, US 5,724,456), so a skewed histogram is placed by its
    detail-bearing span rather than by its median.

    Partial metering: the anchor moves only anchor_meter_strength of the way from
    `assumed` toward the metered median, so a deliberately low-key (dark) or
    high-key (bright) scene keeps most of its intended key instead of being
    forced to mid-gray, while gross mis-exposure is still pulled toward correct.
    A linear pull (no key-dependent amplification) keeps it predictable. Finally
    clamped to `assumed` +/- anchor_meter_band as a hard safety guard.

    `assumed` defaults to EXPOSURE_CONSTANTS["assumed_anchor"], calibrated for the
    paper path's own per-frame-adaptive bounds. The transfer path's `bounds` are
    fixed rather than adaptive, so its expected anchor position is a different
    number (transfer_assumed_anchor) in the same [0, 1] units; pass it explicitly.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    grid = np.ascontiguousarray(_block_median_grid(img_log), dtype=np.float32)
    measured = negpy_core.textured_anchor(grid, _meter_norm(bounds), _activity_gate(), float(EXPOSURE_CONSTANTS["anchor_trim_clip"]))

    a = float(EXPOSURE_CONSTANTS["assumed_anchor"]) if assumed is None else float(assumed)
    strength = float(EXPOSURE_CONSTANTS["anchor_meter_strength"])
    band = float(EXPOSURE_CONSTANTS["anchor_meter_band"])
    anchor = a + strength * (measured - a)
    return float(min(max(anchor, a - band), a + band))


def measure_anchor(
    image: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_anchor_from_log.
    """
    img_log = to_log_density(image)
    return measure_anchor_from_log(img_log, bounds, roi, analysis_buffer)


def measure_textural_range_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Per-frame textural density range: the P10-P90 luminance spread of the
    prefiltered log image, in log10-density units. This is the *useful* scene
    range that grade selection fits to paper — block-median prefiltering and the
    inner percentiles reject speculars / film-base / dust, so it is far more
    outlier-robust than the floor-to-ceil extreme range.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    grid = np.ascontiguousarray(_block_median_grid(img_log), dtype=np.float32)
    return negpy_core.textured_range(grid, _activity_gate(), float(EXPOSURE_CONSTANTS["textural_range_clip"]))


def measure_textural_range(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_textural_range_from_log.
    """
    img_log = to_log_density(image)
    return measure_textural_range_from_log(img_log, roi, analysis_buffer)


def normalized_roi(roi: Optional[Tuple[int, int, int, int]], shape: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    """A pixel ROI as fractions of `shape`, replayable on any downsampled copy. None
    and a full-frame ROI both stay None."""
    if roi is None:
        return None
    h, w = shape
    y1, y2, x1, x2 = roi
    if (y1, x1, y2, x2) == (0, 0, h, w):
        return None
    return (y1 / float(h), y2 / float(h), x1 / float(w), x2 / float(w))


def geometry_kwargs(geometry: Any, distortion_k1: float) -> Dict[str, Any]:
    """The geometry an analysis-grid plane replays, as `analysis_grid`'s keyword arguments,
    so every plane on both engines replays the same fields."""
    return {
        "rotation": geometry.rotation,
        "fine_rotation": geometry.fine_rotation,
        "flip_horizontal": geometry.flip_horizontal,
        "flip_vertical": geometry.flip_vertical,
        "distortion_k1": distortion_k1,
        "converge_v": geometry.converge_v,
        "converge_h": geometry.converge_h,
    }


def analysis_grid(
    image: ImageBuffer,
    rotation: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    distortion_k1: float = 0.0,
    converge_v: float = 0.0,
    converge_h: float = 0.0,
    roi_norm: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """The frame before geometry, downsampled to the analysis grid, geometry replayed and cropped to `roi_norm`
    ((y1, y2, x1, x2) fractions; a rebate blurred in prints as a vignette). Both engines call this on the same
    array, and the Contrast Mask and Diffusion planes are blurred from it."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS
    from negpy.features.geometry.logic import apply_fine_rotation, apply_keystone, apply_radial_distortion

    h, w = image.shape[:2]
    grid = int(EXPOSURE_CONSTANTS["analysis_grid"])
    if max(h, w) > grid:
        scale = grid / float(max(h, w))
        image = negpy_core.resize_area(np.ascontiguousarray(image, dtype=np.float32), (max(1, round(w * scale)), max(1, round(h * scale))))

    if rotation:
        image = np.rot90(image, k=rotation)
    if flip_horizontal:
        image = np.fliplr(image)
    if flip_vertical:
        image = np.flipud(image)
    image = np.ascontiguousarray(image)
    if fine_rotation != 0.0:
        image = apply_fine_rotation(image, fine_rotation)
    if distortion_k1 != 0.0:
        image = apply_radial_distortion(image, distortion_k1)
    image = apply_keystone(image, converge_v, converge_h)

    if roi_norm is not None:
        gh, gw = image.shape[:2]
        y1 = max(0, min(gh - 1, int(round(roi_norm[0] * gh))))
        y2 = max(y1 + 1, min(gh, int(round(roi_norm[1] * gh))))
        x1 = max(0, min(gw - 1, int(round(roi_norm[2] * gw))))
        x2 = max(x1 + 1, min(gw, int(round(roi_norm[3] * gw))))
        image = np.ascontiguousarray(image[y1:y2, x1:x2])
    return image


def contrast_mask_plane(
    image: ImageBuffer,
    bounds: LogNegativeBounds,
    unmix: Optional[np.ndarray],
    rotation: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    distortion_k1: float = 0.0,
    converge_v: float = 0.0,
    converge_h: float = 0.0,
    roi_norm: Optional[Tuple[float, float, float, float]] = None,
    bw_weights: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, float]:
    """The zero-mean blurred luma plane the Contrast Mask reads, on `analysis_grid`, plus the val it was
    centred on. `bw_weights` collapses the val first, as a B&W print does the pixel."""
    image = analysis_grid(image, rotation, fine_rotation, flip_horizontal, flip_vertical, distortion_k1, converge_v, converge_h, roi_norm)

    # A frame that never metered normalizes to huge values; no stretch, no mask.
    if luminance_density_range(bounds) < 1e-6:
        return np.zeros(image.shape[:2], dtype=np.float32), 0.5

    val = normalize_log_image(unmix_log_image(prefilter_log_grid(image, None, 0.0), unmix), bounds)
    if bw_weights is not None:
        val = negpy_core.matmul_3x3(np.ascontiguousarray(val, dtype=np.float32), collapse_matrix(bw_weights))
    sigma = MASK_BLUR_PERCENT * 0.01 * min(val.shape[:2])
    plane, centre = negpy_core.contrast_mask_tail(np.ascontiguousarray(val, dtype=np.float32), sigma)
    return plane, float(centre)


def diffusion_grid(
    image: ImageBuffer,
    unmix: Optional[np.ndarray],
    rotation: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    distortion_k1: float = 0.0,
    converge_v: float = 0.0,
    converge_h: float = 0.0,
    roi_norm: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """The unmixed light blurred by `DIFFUSION_PSF`, as log density on `analysis_grid`. It takes no bounds:
    the print's normalization is a per-channel gain on that light."""
    image = analysis_grid(image, rotation, fine_rotation, flip_horizontal, flip_vertical, distortion_k1, converge_v, converge_h, roi_norm)
    unmixed = np.ascontiguousarray(unmix_log_image(prefilter_log_grid(image, None, 0.0), unmix), dtype=np.float32)
    short = 0.01 * min(unmixed.shape[:2])
    return negpy_core.diffusion_tail(unmixed, [(w, s * short) for w, s in DIFFUSION_PSF])


def diffusion_plane(
    grid: np.ndarray, bounds: LogNegativeBounds, bw_weights: Optional[Tuple[float, float, float]] = None
) -> Optional[np.ndarray]:
    """`diffusion_grid` in the print's normalization, for the kernel to mix as light. `bw_weights` collapses it,
    as a B&W print does the pixel. None when the frame never metered."""
    if luminance_density_range(bounds) < 1e-6:
        return None
    val = normalize_log_image(grid, bounds)
    if bw_weights is not None:
        val = negpy_core.matmul_3x3(np.ascontiguousarray(val, dtype=np.float32), collapse_matrix(bw_weights))
    return np.ascontiguousarray(val, dtype=np.float32)


def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Log -> ~0.0-1.0 (linear stretch, unclamped: out-of-bounds densities are rolled off by
    the downstream characteristic curve). Supports both f < c (negative) and f > c (positive).
    """
    floors = [float(v) for v in np.asarray(bounds.floors, dtype=np.float32)]
    ceils = [float(v) for v in np.asarray(bounds.ceils, dtype=np.float32)]
    return ensure_image(negpy_core.normalize_log_image(np.ascontiguousarray(img_log, dtype=np.float32), floors, ceils))


def _sample_log_bounds(
    img_log: np.ndarray,
    percentile_clip: float,
    base: float,
    sorted_grid: Optional[np.ndarray] = None,
) -> tuple[list, list]:
    """
    Per-channel (floors, ceils) at one clip level. `base` is the robust baseline
    clip added on top of the slider value; negative slider values expand outward
    by a log-density margin instead.
    """
    if percentile_clip >= 0:
        clip = max(0.00001, min(50.0, percentile_clip + base))
        margin = 0.0
    else:
        # Margin mode expands from the same robust basis, so the slider stays continuous through
        # its neutral position.
        clip = base
        margin = -percentile_clip
    p_low, p_high = np.float64(clip), np.float64(100.0 - clip)

    if sorted_grid is not None:
        floors = [float(v) for v in percentile_from_sorted(sorted_grid, p_low)]
        ceils = [float(v) for v in percentile_from_sorted(sorted_grid, p_high)]
    else:
        per_channel = negpy_core.channel_percentiles(np.ascontiguousarray(img_log, dtype=np.float32), [float(p_low), float(p_high)])
        floors = [lo for lo, _ in per_channel]
        ceils = [hi for _, hi in per_channel]

    if margin > 0.0:
        # Expand outward; per-channel sign handles both f < c and f > c.
        for ch in range(3):
            if ceils[ch] >= floors[ch]:
                floors[ch] -= margin
                ceils[ch] += margin
            else:
                floors[ch] += margin
                ceils[ch] -= margin

    return floors, ceils


def _same_pixel_color_floor_refs(
    img_log: ImageBuffer,
    luma_floors: list,
    luma_ceils: list,
    base_refs: Tuple[float, float, float],
    color_clip: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Dense-end (print-white) color refs from one shared pixel set: the luma-extreme
    band's lowest-chroma subset, chroma measured base-anchored (offsets from the
    thin-end refs, per-channel span as provisional gamma, refined once from the
    band medians). Independent per-channel percentiles read a different scene
    object per channel, so colored highlight content masquerades as film cast;
    a shared chroma-gated set cannot. The thin end needs no such treatment —
    density on real film is bounded below by base, anchoring per-channel ceils.
    None when the band's neutral set is too small or too chromatic (caller falls
    back to the percentile pass).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    constants = (
        float(c["neutral_axis_chroma_quantile"]),
        float(c["neutral_axis_chroma_cap"]),
        float(c["neutral_axis_first_pass_cap"]),
        int(c["neutral_axis_min_pixels"]),
        float(c["color_bounds_band_width"]),
    )
    refs = negpy_core.same_pixel_color_floor_refs(
        np.ascontiguousarray(img_log, dtype=np.float32),
        [float(v) for v in luma_floors],
        [float(v) for v in luma_ceils],
        [float(v) for v in base_refs],
        float(color_clip),
        constants,
    )
    return None if refs is None else (refs[0], refs[1], refs[2])


def analyze_log_exposure_bounds(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    percentile_clip: float = 0.0,
    color_clip: float = 0.0,
    unmix: Optional[np.ndarray] = None,
) -> LogNegativeBounds:
    """
    Performs full analysis pass on a linear image to find density floors/ceils.

    Two independent axes are sampled and recombined:
      - percentile_clip (luma): drives the overall black/white-point luminance and
        span (ceil-floor) — i.e. dynamic range / highlight headroom. Sampled at the
        gentle base_luma_clip baseline; slider semantics are:
          > 0  clips the histogram tails (added on top of the baseline clip).
          = 0  robust extremes (block-median prefilter + baseline clip).
          < 0  outward headroom: bounds pushed BEYOND the robust extremes by the margin.
      - color_clip (color): the absolute per-tail clip percentile for the per-channel
        color deviation (white balance / orange-mask cast). A tighter (larger) clip
        gives a more robust channel balance; a gentler (smaller) clip samples nearer
        the extremes. Default neutral is base_color_clip.
    The per-channel refs come from the color sampling, and the luma sampling sets how
    far past them the window reaches, so the cast clip is tunable without compressing
    highlights. Identical channels (mono) give zero deviation at any clip.
    """
    img_log = to_log_density(image)
    img_log = unmix_log_image(img_log, unmix)
    return analyze_log_exposure_bounds_from_log(img_log, roi, analysis_buffer, percentile_clip, color_clip)


def analyze_log_exposure_bounds_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    percentile_clip: float = 0.0,
    color_clip: float = 0.0,
    sorted_grid: Optional[np.ndarray] = None,
) -> LogNegativeBounds:
    """Log-image core of analyze_log_exposure_bounds (skips the log10).

    ``sorted_grid`` is `sorted_channel_grid(img_log)` from a caller that already holds it;
    it stands in for both percentile passes, so it is only valid when no ROI or analysis
    buffer is left to apply here.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi or analysis_buffer > 0:
        sorted_grid = None
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]

    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    base_luma = float(EXPOSURE_CONSTANTS["base_luma_clip"])

    floors, ceils = _sample_log_bounds(img_log, percentile_clip, base_luma, sorted_grid)

    # Color pass: per-channel refs, widened to the luma window. The
    # ceils (thin end, base-anchored) come from per-channel percentiles at color_clip. The
    # floors (dense end, scene content) prefer the same-pixel chroma-gated band refs and fall
    # back to the percentile pass when the band holds no trustworthy neutrals, and always for
    # margin-mode clips.
    c_floors, c_ceils = _sample_log_bounds(img_log, color_clip, 0.0, sorted_grid)
    if color_clip >= 0:
        sp = _same_pixel_color_floor_refs(img_log, floors, ceils, (c_ceils[0], c_ceils[1], c_ceils[2]), color_clip)
        if sp is not None:
            c_floors = [sp[0], sp[1], sp[2]]
    spans = [c_floors[ch] - c_ceils[ch] for ch in range(3)]
    total = sum(spans)
    if all(s * total > 0 for s in spans):
        # The luma window reaches past the color refs by one shared fraction of each channel's own
        # span: a reach in log units tints a film whose channels differ in density range. A ratio
        # of sums, not a mean of per-channel ratios, so a near-flat channel cannot inflate it.
        reach_f = sum(floors[ch] - c_floors[ch] for ch in range(3)) / total
        reach_c = sum(ceils[ch] - c_ceils[ch] for ch in range(3)) / total
        # The window sits off the luma mean by the base's mean-minus-median channel offset, the
        # placement the print constants are tuned to.
        shift = (sum(c_ceils) - 3.0 * sorted(c_ceils)[1]) / total
        floors = [c_floors[ch] + (reach_f + shift) * spans[ch] for ch in range(3)]
        ceils = [c_ceils[ch] + (reach_c + shift) * spans[ch] for ch in range(3)]
    else:
        mean_lf, mean_lc = sum(floors) / 3.0, sum(ceils) / 3.0
        mean_cf, mean_cc = sorted(c_floors)[1], sorted(c_ceils)[1]
        floors = [mean_lf + (c_floors[ch] - mean_cf) for ch in range(3)]
        ceils = [mean_lc + (c_ceils[ch] - mean_cc) for ch in range(3)]

    return LogNegativeBounds(
        (floors[0], floors[1], floors[2]),
        (ceils[0], ceils[1], ceils[2]),
    )


def mix_luma_color_bounds(luma_src: LogNegativeBounds, color_src: LogNegativeBounds) -> LogNegativeBounds:
    """
    Luma-weighted centre+range from one bounds, per-channel color cast from
    another. Keeps the color source's per-channel shape but shifts it so the
    result's luma-weighted centre and range (the brightness/anchor and the H&D
    slope drivers — see luminance_density_range) match the luma source. So
    color-average moves only the per-channel cast, never contrast/brightness.
    Identity when luma_src is color_src (mirrors analyze_log_exposure_bounds'
    recombination), which also keeps a persisted self-mix from stacking edits.
    """
    if luma_src == color_src:
        return luma_src
    w = (LUMA_R, LUMA_G, LUMA_B)
    centre = lambda b: sum(w[c] * (b.floors[c] + b.ceils[c]) / 2.0 for c in range(3))  # noqa: E731
    rng = lambda b: sum(w[c] * (b.ceils[c] - b.floors[c]) for c in range(3))  # noqa: E731
    dC = centre(luma_src) - centre(color_src)
    dR = rng(luma_src) - rng(color_src)
    df, dc = dC - dR / 2.0, dC + dR / 2.0
    cf, cc = color_src.floors, color_src.ceils
    return LogNegativeBounds(
        (cf[0] + df, cf[1] + df, cf[2] + df),
        (cc[0] + dc, cc[1] + dc, cc[2] + dc),
    )


# Luma-free color distance (log density, 6-D over both bounds) from the pool's median past
# which a frame is an outlier: shot under a different light, or scanned with a different
# per-channel balance. Fixed rather than scaled to the pool's spread: a roll that mixes two
# groups has no single spread to scale by.
POOL_OUTLIER_DISTANCE = 0.3


def pool_frame_bounds(floors: np.ndarray, ceils: np.ndarray) -> tuple[LogNegativeBounds, np.ndarray]:
    """
    Pools (N, 3) per-frame bounds into one baseline and an outlier mask. Outliers are whole
    frames whose luma-free color is far from the median (N >= 3 only). A color offset
    also shifts luma through G, so the baseline pools the inliers alone: luma is the median
    of their luma-weighted floor and ceil, color the mean of their luma-free offsets. When
    every frame is an outlier, all of them pool.
    """
    floors = np.asarray(floors, dtype=np.float64)
    ceils = np.asarray(ceils, dtype=np.float64)
    w = np.array([LUMA_R, LUMA_G, LUMA_B])
    luma_f, luma_c = floors @ w, ceils @ w
    chroma = np.hstack([floors - luma_f[:, None], ceils - luma_c[:, None]])

    outliers = np.zeros(len(chroma), dtype=bool)
    if len(chroma) >= 3:
        outliers = np.linalg.norm(chroma - np.median(chroma, axis=0), axis=1) > POOL_OUTLIER_DISTANCE
    pool = ~outliers if (~outliers).any() else np.ones_like(outliers)

    pooled_chroma = chroma[pool].mean(axis=0)
    f = float(np.median(luma_f[pool])) + pooled_chroma[:3]
    c = float(np.median(luma_c[pool])) + pooled_chroma[3:]
    return LogNegativeBounds((float(f[0]), float(f[1]), float(f[2])), (float(c[0]), float(c[1]), float(c[2]))), outliers


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def _axis_curve_residuals(axis: tuple, pooled: tuple) -> np.ndarray:
    """(mid, shadow) x (R, B) distance of *axis* from *pooled*'s R/B-vs-G curve (a quadratic
    through its bands, a line without a highlight), each read at *axis*'s own green."""
    bands = [b for b in (pooled[0], pooled[1], pooled[2]) if b is not None]
    g = np.array([b[1] for b in bands])
    out = np.zeros((2, 2))
    for j, ch in enumerate((0, 2)):
        coef = np.polyfit(g, np.array([b[ch] for b in bands]), len(bands) - 1) if np.ptp(g) > 1e-6 else None
        for i in (0, 1):
            ref_g = axis[i][1]
            fit = float(np.polyval(coef, ref_g)) if coef is not None else bands[0][ch] - bands[0][1] + ref_g
            out[i, j] = axis[i][ch] - fit
    return out


def pool_neutral_axis(axes: list, outliers: np.ndarray) -> Optional[tuple]:
    """
    Pools per-frame neutral axes (measure_neutral_axis_from_log's shape, or None) into one,
    skipping bounds outliers: a confidence-weighted median per band and channel. The
    highlight band pools only when at least half the contributing frames have one. None
    when no inlier frame has an axis with non-zero confidence.

    A fifth element is the offset weight blend_neutral_axis gives a frame's own midtone
    offset: 1 - noise / spread of the frames' offsets from the pooled curve. The curve's shape
    is the film's, so the spread of the frames' shapes (shadow minus midtone offset, halved
    as a difference of two meters) estimates the meter noise. Zero below three frames.
    """
    pool = [a for a, out in zip(axes, outliers) if a is not None and not out and a[3] > 0.0]
    if not pool:
        return None
    conf = np.array([a[3] for a in pool])

    def band(i: int, frames: list, w: np.ndarray) -> tuple[float, float, float]:
        refs = np.array([a[i] for a in frames])
        return (_weighted_median(refs[:, 0], w), _weighted_median(refs[:, 1], w), _weighted_median(refs[:, 2], w))

    with_hl = [a for a in pool if a[2] is not None]
    highlight = band(2, with_hl, np.array([a[3] for a in with_hl])) if 2 * len(with_hl) >= len(pool) else None
    pooled = (band(0, pool, conf), band(1, pool, conf), highlight, float(np.median(conf)))

    weight = 0.0
    if len(pool) >= 3:
        res = np.array([_axis_curve_residuals(a, pooled) for a in pool])
        spread = float(res[:, 0, :].var(axis=0).mean())
        noise = 0.5 * float((res[:, 1, :] - res[:, 0, :]).var(axis=0).mean())
        weight = float(np.clip(1.0 - noise / spread, 0.0, 1.0)) if spread > 0.0 else 0.0
    return (*pooled, weight)


#: The Frontier's cast-hue strength (ez171autoc-nega.dat 0x4D18), piecewise linear in the cast's hue
#: atan2(B - (R+G)/2, (R-G)·√3/2) in negative density: -30 red, 30 magenta, 90 blue, -90 yellow.
_CAST_HUE_DEGREES = (-180.0, -120.0, -90.0, -30.0, 0.0, 30.0, 100.0, 140.0, 180.0)
_CAST_HUE_STRENGTH = (0.3, 0.3, 0.6, 0.6, 0.0, 0.0, 1.0, 1.0, 0.3)


def _cast_hue_strength(dr: float, db: float) -> float:
    """Share of a frame's own cast that Cast Removal removes, from its (R, B) log offsets at equal green.
    A magenta cast stays and a blue one goes, as on the Frontier."""
    hue = np.degrees(np.arctan2(0.5 * dr - db, -0.5 * np.sqrt(3.0) * dr))
    return float(np.interp(hue, _CAST_HUE_DEGREES, _CAST_HUE_STRENGTH))


def blend_neutral_axis(own: Optional[tuple], pooled: tuple) -> tuple:
    """The pooled axis shifted by this frame's own midtone offset from the pooled curve, scaled
    by the pool's offset weight, the frame's confidence and the offset's cast-hue strength: the
    roll's shape, this frame's level. No own axis, or a zero weight, is the pooled axis itself."""
    weight = pooled[4] if len(pooled) > 4 else 0.0
    k = weight * own[3] if own is not None else 0.0
    if k <= 0.0:
        return pooled[:4]
    dr, db = _axis_curve_residuals(own, pooled)[0]
    k *= _cast_hue_strength(dr, db)
    shift = lambda b: (b[0] + k * dr, b[1], b[2] + k * db) if b is not None else None  # noqa: E731
    return (shift(pooled[0]), shift(pooled[1]), shift(pooled[2]), pooled[3])


#: Detect Scenes: two frames are under one light when their midtone grays, read at equal green,
#: are within this many log D (about 8 CC) of each other.
SCENE_LIGHT_DISTANCE = 0.08
#: A frame's axis below this confidence has no trustworthy grays and joins no scene.
SCENE_MIN_CONFIDENCE = 0.2
#: A found scene keeps only frames whose own light is within this many log D of the scene's median.
SCENE_MEMBER_RADIUS = 0.05
#: Weights of the neighbours 1, 2 and 3 frames away in scan order.
SCENE_ORDER_REACH = (1.5, 0.9, 0.45)


def _smooth_along_scan_order(light: np.ndarray) -> np.ndarray:
    """Each frame's light averaged with its scan-order neighbours under the same light, weighted by
    how much more neighbours agree than any two frames: strips scanned out of order get no weight."""
    n = len(light)
    if n < 4:
        return light
    steps = np.linalg.norm(np.diff(light, axis=0), axis=1)
    pairs = np.linalg.norm(light[:, None] - light[None, :], axis=-1)[np.triu_indices(n, 1)]
    spread = float(np.median(pairs))
    evidence = float(np.clip(1.0 - np.median(steps) / spread, 0.0, 1.0)) if spread > 0.0 else 0.0
    out = light.copy()
    for x in range(n):
        total, weight = light[x].copy(), 1.0
        for k, reach in enumerate(SCENE_ORDER_REACH, 1):
            for y in (x - k, x + k):
                if 0 <= y < n and np.linalg.norm(light[x] - light[y]) < SCENE_LIGHT_DISTANCE:
                    total += evidence * reach * light[y]
                    weight += evidence * reach
        out[x] = total / weight
    return out


def _complete_linkage(points: np.ndarray, cut: float) -> list[list[int]]:
    """Groups in which every pair of points is closer than *cut*, merged closest first."""
    d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    groups: list[list[int]] = [[i] for i in range(len(points))]
    while len(points) > 1:
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[i, j] >= cut:
            break
        groups[i] += groups[j]
        groups[j] = []
        d[i, :] = d[:, i] = np.maximum(d[i], d[j])
        d[i, i] = np.inf
        d[j, :] = d[:, j] = np.inf
    return [g for g in groups if g]


def detect_light_groups(axes: list) -> list[list[int]]:
    """Detect Scenes: indices of each group of frames under one light other than the roll's own, two
    frames or more, largest first. *axes* are the frames' neutral axes in scan order (None without
    one). A frame's light is its midtone offset from the pooled axis at its own green, so the
    subject's colors do not count; the largest group stays with the roll baseline."""
    usable = [i for i, a in enumerate(axes) if a is not None and a[3] > SCENE_MIN_CONFIDENCE]
    if len(usable) < 2:
        return []
    frames = [axes[i] for i in usable]
    pooled = pool_neutral_axis(frames, np.zeros(len(frames), dtype=bool))
    if pooled is None:
        return []
    light = np.array([_axis_curve_residuals(a, pooled)[0] for a in frames])
    groups = _complete_linkage(_smooth_along_scan_order(light), SCENE_LIGHT_DISTANCE)
    groups.sort(key=lambda g: (-len(g), min(g)))
    found = []
    for g in groups[1:]:
        center = np.median(light[g], axis=0)
        g = [j for j in g if np.linalg.norm(light[j] - center) < SCENE_MEMBER_RADIUS]
        if len(g) >= 2:
            found.append(g)
    found.sort(key=lambda g: (-len(g), min(g)))
    return [sorted(usable[j] for j in g) for g in found]


def resolve_bounds(process, analyze_fn) -> LogNegativeBounds:
    """Final bounds for rendering. See resolve_bounds_detailed for the per-frame base."""
    return resolve_bounds_detailed(process, analyze_fn)[0]


def resolve_bounds_detailed(process, analyze_fn) -> tuple[LogNegativeBounds, LogNegativeBounds]:
    """
    Returns (final, base): the final mixed bounds to render with, and the per-frame
    base (local/analyzed) to persist. Persist the base, not the mix — re-feeding a
    mix as the next base stacks edits (mean-vs-median drift; color-only roll).
    Picks luma + color from the roll baseline (locked) or the per-frame base, then
    mixes. analyze_fn() supplies the base and is called only when actually needed.
    """
    roll_luma = process.use_luma_average and process.is_locked_initialized
    roll_color = process.use_color_average and process.is_locked_initialized
    locked = LogNegativeBounds(process.locked_floors, process.locked_ceils)
    if roll_luma and roll_color:
        return locked, locked
    base = LogNegativeBounds(process.local_floors, process.local_ceils) if process.is_local_initialized else analyze_fn()
    final = mix_luma_color_bounds(locked if roll_luma else base, locked if roll_color else base)
    return final, base


def luma_source_bounds(process, base: LogNegativeBounds) -> LogNegativeBounds:
    """
    Bounds the luma/exposure reading (metered anchor) must come from: the roll
    baseline when luma-average is on, else the per-frame base. The anchor is a
    luma-weighted percentile and so reacts non-linearly to the per-channel cast;
    measuring it here, not on the final mix, keeps brightness independent of the
    color-average toggle (mix_luma_color_bounds already pins centre+range).
    """
    if process.use_luma_average and process.is_locked_initialized:
        return LogNegativeBounds(process.locked_floors, process.locked_ceils)
    return base
