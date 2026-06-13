from typing import Optional, Tuple

import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R, ImageBuffer
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.validation import ensure_image


@njit(cache=True, fastmath=True)
def _normalize_log_image_jit(img_log: np.ndarray, floors: np.ndarray, ceils: np.ndarray) -> np.ndarray:
    """
    Log -> ~0.0-1.0 (Linear stretch, unclamped: out-of-bounds densities are
    rolled off by the downstream characteristic curve).
    Supports both f < c (Negative) and f > c (Positive) mapping.
    """
    h, w, c = img_log.shape
    res = np.empty_like(img_log)
    epsilon = 1e-6

    for y in range(h):
        for x in range(w):
            for ch in range(3):
                f = floors[ch]
                c_val = ceils[ch]
                delta = c_val - f

                denom = delta
                if abs(delta) < epsilon:
                    if delta >= 0:
                        denom = epsilon
                    else:
                        denom = -epsilon

                res[y, x, ch] = (img_log[y, x, ch] - f) / denom
    return res


class LogNegativeBounds:
    """
    D-min / D-max container.
    """

    def __init__(self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]):
        self.floors = floors
        self.ceils = ceils


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
    if b > 1 and h >= b and w >= b:
        hb, wb = (h // b) * b, (w // b) * b
        blocks = img_log[:hb, :wb].reshape(hb // b, b, wb // b, b, img_log.shape[2])
        img_log = np.median(blocks, axis=(1, 3))
    return img_log


def measure_shadow_refs_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Per-channel shadow reference density: a high percentile of the prefiltered
    log image — the tones just inside print black (thin negative side for C-41).
    Channel differences here are the residual shadow cast that auto
    shadow-neutral cancels.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)
    p = float(EXPOSURE_CONSTANTS["shadow_neutral_percentile"])
    refs = [float(np.percentile(img_log[:, :, ch], p)) for ch in range(3)]
    return (refs[0], refs[1], refs[2])


def measure_shadow_log_refs(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Linear-image wrapper around measure_shadow_refs_from_log.
    """
    epsilon = 1e-6
    img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))
    return measure_shadow_refs_from_log(img_log, roi, analysis_buffer)


def luminance_density_range(bounds: LogNegativeBounds) -> float:
    """
    Single global density range as a Rec.709 luminance weighting of the
    per-channel ranges. Replaces the green-only range so frames with a strong
    single-channel cast don't swing the slope as hard, while green still
    dominates so calibrated grade behaviour barely shifts. abs() keeps it
    sign-safe for E6's reversed (f > c) bounds.
    """
    rr = abs(bounds.ceils[0] - bounds.floors[0])
    rg = abs(bounds.ceils[1] - bounds.floors[1])
    rb = abs(bounds.ceils[2] - bounds.floors[2])
    return float(LUMA_R * rr + LUMA_G * rg + LUMA_B * rb)


def measure_anchor_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Per-frame exposure anchor: where this negative's midtone sits in [0, 1],
    replacing the fixed assumed_anchor. Block-median prefiltered (speculars/dust
    rejected).

    Partial metering: the anchor moves only anchor_meter_strength of the way from
    assumed_anchor toward the metered median, so a deliberately low-key (dark) or
    high-key (bright) scene keeps most of its intended key instead of being
    forced to mid-gray, while gross mis-exposure is still pulled toward correct.
    Finally clamped to assumed_anchor +/- anchor_meter_band as a hard safety guard.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    epsilon = 1e-6
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    norm = np.empty_like(img_log)
    for ch in range(3):
        f = bounds.floors[ch]
        denom = bounds.ceils[ch] - f
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        norm[:, :, ch] = (img_log[:, :, ch] - f) / denom

    lum = LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2]
    p = float(EXPOSURE_CONSTANTS["anchor_meter_percentile"])
    measured = float(np.percentile(lum, p))

    assumed = float(EXPOSURE_CONSTANTS["assumed_anchor"])
    strength = float(EXPOSURE_CONSTANTS["anchor_meter_strength"])
    anchor = assumed + strength * (measured - assumed)
    band = float(EXPOSURE_CONSTANTS["anchor_meter_band"])
    return float(min(max(anchor, assumed - band), assumed + band))


def measure_anchor(
    image: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_anchor_from_log.
    """
    epsilon = 1e-6
    img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))
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

    img_log = _block_median_grid(img_log)

    lum = LUMA_R * img_log[:, :, 0] + LUMA_G * img_log[:, :, 1] + LUMA_B * img_log[:, :, 2]
    clip = float(EXPOSURE_CONSTANTS["textural_range_clip"])
    lo, hi = np.percentile(lum, [clip, 100.0 - clip])
    return float(abs(hi - lo))


def measure_textural_range(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_textural_range_from_log.
    """
    epsilon = 1e-6
    img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))
    return measure_textural_range_from_log(img_log, roi, analysis_buffer)


def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Stretches log-data to fit [0, 1].
    """
    floors = np.ascontiguousarray(np.array(bounds.floors, dtype=np.float32))
    ceils = np.ascontiguousarray(np.array(bounds.ceils, dtype=np.float32))

    return ensure_image(_normalize_log_image_jit(np.ascontiguousarray(img_log.astype(np.float32)), floors, ceils))


def analyze_log_exposure_bounds(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    process_mode: str = ProcessMode.C41,
    e6_normalize: bool = True,
    percentile_clip: float = 0.0,
) -> LogNegativeBounds:
    """
    Performs full analysis pass on a linear image to find density floors/ceils.
    percentile_clip controls how the bounds are sampled:
      > 0  clips the histogram tails (e.g. 0.0001 = nearly no clipping; 1.0 = clip 1% per tail),
           added on top of the robust baseline clip.
      = 0  robust extremes: a block-median prefilter rejects isolated outliers (speculars,
           dust) and base_drange_clip excludes small coherent extreme populations.
      < 0  outward headroom: bounds are pushed BEYOND the robust extremes by margin = -percentile_clip
           (in log-density units), leaving lifted blacks / unclipped highlights (gentler than 0).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    epsilon = 1e-6
    img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]

    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    base = float(EXPOSURE_CONSTANTS["base_drange_clip"])
    if percentile_clip >= 0:
        clip = max(0.00001, min(1.0, percentile_clip + base))
        margin = 0.0
    else:
        # Margin mode expands from the same robust basis so the slider stays
        # continuous through its neutral position.
        clip = base
        margin = -percentile_clip
    p_low, p_high = np.float64(clip), np.float64(100.0 - clip)
    fixed_range = 3.0

    if process_mode == ProcessMode.E6:
        p_low, p_high = p_high, p_low
        fixed_range = -3.0

    floors = []
    for ch in range(3):
        floors.append(float(np.percentile(img_log[:, :, ch], p_low)))

    ceils = []
    for ch in range(3):
        data = img_log[:, :, ch]
        if process_mode != ProcessMode.E6 or e6_normalize:
            c = np.percentile(data, p_high)
            ceils.append(float(c))
        else:
            ceils.append(float(floors[ch] + fixed_range))

    if margin > 0.0:
        # Expand outward; per-channel sign handles both f < c and f > c (E6).
        for ch in range(3):
            if ceils[ch] >= floors[ch]:
                floors[ch] -= margin
                ceils[ch] += margin
            else:
                floors[ch] += margin
                ceils[ch] -= margin

    return LogNegativeBounds(
        (floors[0], floors[1], floors[2]),
        (ceils[0], ceils[1], ceils[2]),
    )
