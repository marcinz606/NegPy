import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R, ImageBuffer
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.validation import ensure_image

# Above this size the block-median is threaded over row strips (np.median frees the GIL).
_BLOCK_MEDIAN_PARALLEL_MIN_PIXELS = 2_000_000


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
    if b <= 1 or h < b or w < b:
        return img_log

    hb, wb = (h // b) * b, (w // b) * b
    arr = img_log[:hb, :wb]
    grid_rows, c = hb // b, arr.shape[2]

    def _median(rows: np.ndarray) -> np.ndarray:
        return np.median(rows.reshape(rows.shape[0] // b, b, wb // b, b, c), axis=(1, 3))

    workers = min(os.cpu_count() or 1, grid_rows)
    if workers < 2 or hb * wb < _BLOCK_MEDIAN_PARALLEL_MIN_PIXELS:
        return _median(arr)

    # Block-aligned strips -> per-cell median identical to the single pass.
    rows_per = -(-grid_rows // workers)
    strips = [arr[i * b : min(grid_rows, i + rows_per) * b] for i in range(0, grid_rows, rows_per)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_median, strips))
    return np.concatenate(parts, axis=0)


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
    epsilon = 1e-6
    img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))
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

    img = image
    if roi:
        y1, y2, x1, x2 = roi
        img = img[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img = get_analysis_crop(img, analysis_buffer)
    # Stride-subsampled: a warning metric doesn't need every pixel.
    img = img[::4, ::4]
    level = float(EXPOSURE_CONSTANTS["scan_clip_level"])
    clipped = np.mean(img >= level, axis=(0, 1))
    return (float(clipped[0]), float(clipped[1]), float(clipped[2]))


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


def measure_neutral_axis_from_log(
    img_log: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """
    Per-channel neutral axis: median raw-log density at a highlight, a midtone and a shadow
    luma band, over each band's lowest-chroma pixels. The relative chroma quantile finds the
    near-neutral population through the residual cast and rejects saturated content
    (foliage/skin) that would otherwise pull the balance green. Returns (midtone, shadow,
    highlight, confidence) — highlight is None when that band has no trustworthy neutral set
    (callers then fit a 2-point line); confidence in [0,1] rates how tight the midtone grey set
    is (drives Auto Cast Removal). None overall when midtone or shadow is missing (shadow tie).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)
    norm = normalize_log_image(img_log, bounds)
    luma = LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2]
    chroma = norm.max(axis=2) - norm.min(axis=2)

    flat_log = img_log.reshape(-1, 3)
    luma_f = luma.reshape(-1)
    chroma_f = chroma.reshape(-1)

    c = EXPOSURE_CONSTANTS
    q = float(c["neutral_axis_chroma_quantile"])
    cap = float(c["neutral_axis_chroma_cap"])
    min_px = int(c["neutral_axis_min_pixels"])

    def _band_refs(lo: float, hi: float) -> Optional[Tuple[Tuple[float, float, float], float]]:
        band = (luma_f >= lo) & (luma_f <= hi)
        if int(band.sum()) < min_px:
            return None
        band_chroma = chroma_f[band]
        thr = float(np.quantile(band_chroma, q))
        idx = np.nonzero(band)[0][band_chroma <= thr]
        near_neutral_chroma = float(np.median(chroma_f[idx])) if idx.size else cap
        if idx.size < min_px or near_neutral_chroma > cap:
            return None
        refs = (
            float(np.median(flat_log[idx, 0])),
            float(np.median(flat_log[idx, 1])),
            float(np.median(flat_log[idx, 2])),
        )
        return (refs, near_neutral_chroma)

    hb = c["neutral_axis_highlight_band"]
    mb = c["neutral_axis_mid_band"]
    sb = c["neutral_axis_shadow_band"]
    mid = _band_refs(float(mb[0]), float(mb[1]))
    shadow = _band_refs(float(sb[0]), float(sb[1]))
    if mid is None or shadow is None:
        return None
    highlight = _band_refs(float(hb[0]), float(hb[1]))
    # 1 when the midtone grey set is spectrally tight, 0 near the trust cap.
    confidence = float(np.clip(1.0 - mid[1] / cap, 0.0, 1.0))
    return (mid[0], shadow[0], highlight[0] if highlight is not None else None, confidence)


def measure_neutral_axis(
    image: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """Linear-image wrapper around measure_neutral_axis_from_log."""
    epsilon = 1e-6
    img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))
    return measure_neutral_axis_from_log(img_log, bounds, roi, analysis_buffer)


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
    A linear pull (no key-dependent amplification) keeps it predictable. Finally
    clamped to assumed_anchor +/- anchor_meter_band as a hard safety guard.
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
    band = float(EXPOSURE_CONSTANTS["anchor_meter_band"])
    anchor = assumed + strength * (measured - assumed)
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


def _sample_log_bounds(
    img_log: np.ndarray,
    percentile_clip: float,
    base: float,
    process_mode: str,
    e6_normalize: bool,
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
        # Margin mode expands from the same robust basis so the slider stays
        # continuous through its neutral position.
        clip = base
        margin = -percentile_clip
    p_low, p_high = np.float64(clip), np.float64(100.0 - clip)
    fixed_range = 3.0

    if process_mode == ProcessMode.E6:
        p_low, p_high = p_high, p_low
        fixed_range = -3.0

    floors = [float(np.percentile(img_log[:, :, ch], p_low)) for ch in range(3)]

    ceils = []
    for ch in range(3):
        data = img_log[:, :, ch]
        if process_mode != ProcessMode.E6 or e6_normalize:
            ceils.append(float(np.percentile(data, p_high)))
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

    return floors, ceils


def analyze_log_exposure_bounds(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    process_mode: str = ProcessMode.C41,
    e6_normalize: bool = True,
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
      - color_clip (colour): the absolute per-tail clip percentile for the per-channel
        colour deviation (white balance / orange-mask cast). A tighter (larger) clip
        gives a more robust channel balance; a gentler (smaller) clip samples nearer
        the extremes. Default neutral is base_color_clip.
    The luminance centre+span comes from the luma sampling, the per-channel colour
    offsets from the colour sampling, so the cast clip is tunable without compressing
    highlights. Identical channels (mono) give zero deviation at any clip.
    """
    epsilon = 1e-6
    img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))
    img_log = unmix_log_image(img_log, unmix)
    return analyze_log_exposure_bounds_from_log(img_log, roi, analysis_buffer, process_mode, e6_normalize, percentile_clip, color_clip)


def analyze_log_exposure_bounds_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    process_mode: str = ProcessMode.C41,
    e6_normalize: bool = True,
    percentile_clip: float = 0.0,
    color_clip: float = 0.0,
) -> LogNegativeBounds:
    """Log-image core of analyze_log_exposure_bounds (skips the log10)."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]

    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    base_luma = float(EXPOSURE_CONSTANTS["base_luma_clip"])

    floors, ceils = _sample_log_bounds(img_log, percentile_clip, base_luma, process_mode, e6_normalize)

    # Colour pass: per-channel deviation sampled at its own absolute clip percentile
    # (color_clip), recombined onto the luma mean centre+span. Tightening the colour
    # clip tightens channel balance / cast removal without touching the luma span.
    c_floors, c_ceils = _sample_log_bounds(img_log, color_clip, 0.0, process_mode, e6_normalize)
    mean_lf, mean_lc = sum(floors) / 3.0, sum(ceils) / 3.0
    mean_cf, mean_cc = sorted(c_floors)[1], sorted(c_ceils)[1]
    floors = [mean_lf + (c_floors[ch] - mean_cf) for ch in range(3)]
    ceils = [mean_lc + (c_ceils[ch] - mean_cc) for ch in range(3)]

    return LogNegativeBounds(
        (floors[0], floors[1], floors[2]),
        (ceils[0], ceils[1], ceils[2]),
    )


def mix_luma_colour_bounds(luma_src: LogNegativeBounds, colour_src: LogNegativeBounds) -> LogNegativeBounds:
    """
    Luma-weighted centre+range from one bounds, per-channel colour cast from
    another. Keeps the colour source's per-channel shape but shifts it so the
    result's luma-weighted centre and range (the brightness/anchor and the H&D
    slope drivers — see luminance_density_range) match the luma source. So
    colour-average moves only the per-channel cast, never contrast/brightness.
    Identity when luma_src is colour_src (mirrors analyze_log_exposure_bounds'
    recombination), which also keeps a persisted self-mix from stacking edits.
    """
    if luma_src == colour_src:
        return luma_src
    w = (LUMA_R, LUMA_G, LUMA_B)
    centre = lambda b: sum(w[c] * (b.floors[c] + b.ceils[c]) / 2.0 for c in range(3))  # noqa: E731
    rng = lambda b: sum(w[c] * (b.ceils[c] - b.floors[c]) for c in range(3))  # noqa: E731
    dC = centre(luma_src) - centre(colour_src)
    dR = rng(luma_src) - rng(colour_src)
    df, dc = dC - dR / 2.0, dC + dR / 2.0
    cf, cc = colour_src.floors, colour_src.ceils
    return LogNegativeBounds(
        (cf[0] + df, cf[1] + df, cf[2] + df),
        (cc[0] + dc, cc[1] + dc, cc[2] + dc),
    )


def resolve_bounds(process, analyze_fn) -> LogNegativeBounds:
    """Final bounds for rendering. See resolve_bounds_detailed for the per-frame base."""
    return resolve_bounds_detailed(process, analyze_fn)[0]


def resolve_bounds_detailed(process, analyze_fn) -> tuple[LogNegativeBounds, LogNegativeBounds]:
    """
    Returns (final, base): the final mixed bounds to render with, and the per-frame
    base (local/analyzed) to persist. Persist the base, not the mix — re-feeding a
    mix as the next base stacks edits (mean-vs-median drift; colour-only roll).
    Picks luma + colour from the roll baseline (locked) or the per-frame base, then
    mixes. analyze_fn() supplies the base and is called only when actually needed.
    """
    roll_luma = process.use_luma_average and process.is_locked_initialized
    roll_colour = process.use_colour_average and process.is_locked_initialized
    locked = LogNegativeBounds(process.locked_floors, process.locked_ceils)
    if roll_luma and roll_colour:
        return locked, locked
    base = LogNegativeBounds(process.local_floors, process.local_ceils) if process.is_local_initialized else analyze_fn()
    final = mix_luma_colour_bounds(locked if roll_luma else base, locked if roll_colour else base)
    return final, base


def luma_source_bounds(process, base: LogNegativeBounds) -> LogNegativeBounds:
    """
    Bounds the luma/exposure reading (metered anchor) must come from: the roll
    baseline when luma-average is on, else the per-frame base. The anchor is a
    luma-weighted percentile and so reacts non-linearly to the per-channel cast;
    measuring it here, not on the final mix, keeps brightness independent of the
    colour-average toggle (mix_luma_colour_bounds already pins centre+range).
    """
    if process.use_luma_average and process.is_locked_initialized:
        return LogNegativeBounds(process.locked_floors, process.locked_ceils)
    return base
