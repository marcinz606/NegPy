import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

import cv2
import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R, ROI, ImageBuffer
from negpy.features.geometry.logic import AUTOCROP_DETECT_RES, _normalize_detection_input, film_and_picture
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.logic import percentile_from_sorted
from negpy.kernel.image.validation import ensure_image

if TYPE_CHECKING:
    from negpy.features.process.models import ProcessConfig

# Bump in any change that alters the bounds a frame measures for unchanged settings. Saved
# per-frame bounds from an older version are dropped at startup (migrations/bounds.py).
BOUNDS_VERSION = 1

# Above this size the block-median is threaded over row strips (np.median frees the GIL).
_BLOCK_MEDIAN_PARALLEL_MIN_PIXELS = 2_000_000

# Contrast-mask spacer, per-cent of the analysis grid's short side. Of the grid, not of
# the render, so preview and export mask alike. Under the minimum the mask stops being
# unsharp and collapses into a plain (1-g) reduction, which is Grade.
MASK_SPACER_DEFAULT = 4.0
MASK_SPACER_MIN = 2.0
MASK_SPACER_MAX = 6.0
# Diffusion's blur sigma in per-cent of the printed frame's short side: one diffuser at the
# lens, whose spread a printer does not vary between prints.
DIFFUSION_SPREAD = 2.0


@njit(cache=True, nogil=True, fastmath=True)
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
    arr = img_log[:hb, :wb]
    grid_rows, c = hb // b, arr.shape[2]

    def _median(rows: np.ndarray) -> np.ndarray:
        v = rows.reshape(rows.shape[0] // b, b, wb // b, b, c)
        if b == 2:
            # Median of 4 = (sum - min - max) / 2; strided views, no partition.
            p00, p01, p10, p11 = v[:, 0, :, 0], v[:, 0, :, 1], v[:, 1, :, 0], v[:, 1, :, 1]
            s = p00.astype(np.float64) + p01 + p10 + p11
            mn = np.minimum(np.minimum(p00, p01), np.minimum(p10, p11))
            mx = np.maximum(np.maximum(p00, p01), np.maximum(p10, p11))
            return ((s - mn - mx) * 0.5).astype(rows.dtype, copy=False)
        return np.median(v, axis=(1, 3))

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
    """
    Linear scan -> log10 density, the domain every meter reads.

    fmin/fmax rather than clip: they drop a NaN in favour of the bound, so one clamp covers
    the NaN and infinity fixup as well.
    """
    epsilon = 1e-6
    return np.log10(np.fmin(np.fmax(image, epsilon), 1.0))


def _rms_chroma(triplets: np.ndarray) -> np.ndarray:
    """Distance from the neutral axis in normalized-density space (pairwise RMS):
    rotation-symmetric around grey, so the near-neutral ranking is hue-uniform
    (max-min scores an opposed R/B split double a same-side deviation)."""
    r, g, b = triplets[..., 0], triplets[..., 1], triplets[..., 2]
    return np.sqrt(((r - g) ** 2 + (g - b) ** 2 + (r - b) ** 2) / 3.0)


def _subject_cells(norm_f: np.ndarray) -> np.ndarray:
    """Cells in the foliage or the sky sector. In a frame with no grays the lowest-chroma cells are
    desaturated leaves or pale sky, and neutralizing them prints magenta or yellow (a printer's
    subject failure)."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    dr = norm_f[:, 0] - norm_f[:, 1]
    db = norm_f[:, 2] - norm_f[:, 1]
    t, s = (dr + db) / 2.0, (dr - db) / 2.0
    hue = np.degrees(np.arctan2(-s, t))
    chroma = np.hypot(t, s)
    cells = np.zeros(chroma.shape, dtype=bool)
    for name in ("foliage", "sky"):
        lo, hi = c[f"neutral_axis_{name}_hue"]
        cells |= (chroma > float(c[f"neutral_axis_{name}_chroma"])) & (hue > float(lo)) & (hue < float(hi))
    return cells


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

    img_log = _block_median_grid(img_log)
    norm = normalize_log_image(img_log, bounds)
    luma = LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2]

    flat_log = img_log.reshape(-1, 3)
    norm_f = norm.reshape(-1, 3)
    luma_f = luma.reshape(-1)
    if exclude_subjects:
        keep = ~_subject_cells(norm_f)
        flat_log, norm_f, luma_f = flat_log[keep], norm_f[keep], luma_f[keep]
    chroma_f = _rms_chroma(norm_f)

    c = EXPOSURE_CONSTANTS
    q = float(c["neutral_axis_chroma_quantile"])
    cap = float(c["neutral_axis_chroma_cap"])
    pass1_cap = float(c["neutral_axis_first_pass_cap"])
    min_px = int(c["neutral_axis_min_pixels"])
    epsilon = 1e-6

    def _band_refs(
        lo: float, hi: float, chroma_vals: np.ndarray, cap_val: float
    ) -> Optional[Tuple[Tuple[float, float, float], float, int, float]]:
        band = (luma_f >= lo) & (luma_f <= hi)
        if int(band.sum()) < min_px:
            return None
        band_chroma = chroma_vals[band]
        thr = float(np.quantile(band_chroma, q))
        keep = band_chroma <= thr
        idx = np.nonzero(band)[0][keep]
        near_neutral_chroma = float(np.median(band_chroma[keep])) if idx.size else cap_val
        if idx.size < min_px or near_neutral_chroma > cap_val:
            return None
        # One gather for all three channels: the per-channel fancy index is the cost here.
        sel = flat_log[idx]
        # R and B are green plus the set's median offset from green. Independent channel
        # medians can each fall in a different cluster of a set that holds two colors.
        green = float(np.median(sel[:, 1]))
        refs = (green + float(np.median(sel[:, 0] - sel[:, 1])), green, green + float(np.median(sel[:, 2] - sel[:, 1])))
        offsets = (norm_f[idx][:, (0, 2)] - norm_f[idx][:, 1:2]).astype(np.float64)
        p10, p90 = np.percentile(offsets, (10.0, 90.0), axis=0)
        return (refs, near_neutral_chroma, int(idx.size), float(np.max(p90 - p10)))

    def _norm_ref(refs: Tuple[float, float, float]) -> Tuple[float, float, float]:
        out = []
        for ch in range(3):
            denom = bounds.ceils[ch] - bounds.floors[ch]
            if abs(denom) < epsilon:
                denom = epsilon if denom >= 0 else -epsilon
            out.append((refs[ch] - bounds.floors[ch]) / denom)
        return (out[0], out[1], out[2])

    hb = c["neutral_axis_highlight_band"]
    mb = c["neutral_axis_mid_band"]
    sb = c["neutral_axis_shadow_band"]
    mid1 = _band_refs(float(mb[0]), float(mb[1]), chroma_f, pass1_cap)
    sh1 = _band_refs(float(sb[0]), float(sb[1]), chroma_f, pass1_cap)
    if mid1 is None or sh1 is None:
        return None

    nm, ns = _norm_ref(mid1[0]), _norm_ref(sh1[0])
    corrected = norm_f.copy()
    for ch in (0, 2):
        du = nm[ch] - ns[ch]
        if abs(du) < epsilon:
            a, b = 1.0, nm[1] - nm[ch]
        else:
            a = (nm[1] - ns[1]) / du
            b = nm[1] - a * nm[ch]
        corrected[:, ch] = a * norm_f[:, ch] + b
    chroma2_f = _rms_chroma(corrected)

    mid = _band_refs(float(mb[0]), float(mb[1]), chroma2_f, cap)
    shadow = _band_refs(float(sb[0]), float(sb[1]), chroma2_f, cap)
    if mid is None or shadow is None:
        return None
    highlight = _band_refs(float(hb[0]), float(hb[1]), chroma2_f, cap)

    # Confidence: the corrected tightness of the grey sets, times the midtone sample size,
    # times the mid/shadow deviation agreement, where a dead zone passes plausible crossover.
    n0 = float(c["neutral_axis_confidence_n0"])
    dead = float(c["neutral_axis_agreement_deadzone"])
    scale = float(c["neutral_axis_agreement_scale"])
    tight = float(np.clip(1.0 - max(mid[1], shadow[1]) / cap, 0.0, 1.0))
    size_term = mid[2] / (mid[2] + n0)
    dm, ds = _norm_ref(mid[0]), _norm_ref(shadow[0])
    spread = max(abs((dm[ch] - dm[1]) - (ds[ch] - ds[1])) for ch in (0, 2))
    agree = 1.0 - min(max(spread - dead, 0.0) / scale, 1.0)
    width = max(mid[3], shadow[3])
    single = 1.0 - min(max(width - float(c["neutral_axis_width_deadzone"]), 0.0) / float(c["neutral_axis_width_scale"]), 1.0)
    confidence = float(np.clip(tight * size_term * agree * single, 0.0, 1.0))
    return (mid[0], shadow[0], highlight[0] if highlight is not None else None, confidence)


def measure_neutral_axis(
    image: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """Linear-image wrapper around measure_neutral_axis_from_log."""
    img_log = to_log_density(image)
    return measure_neutral_axis_from_log(img_log, bounds, roi, analysis_buffer)


#: The clear base is read from flat blocks in the film box's margin, grouped by brightness, the
#: way negdensity.py reads it. The film box comes from the crop's detector at its own resolution,
#: whose edge walk is calibrated there.
_BASE_MARGIN = 0.15  # the band inside the film box, per side, that holds rebate, perforations and picture edge
_BASE_BLOCK = 8  # block size at detection resolution
_FLAT_CV = 0.03  # a block whose std over mean is under this is flat film, not edge print or a gradient
_GROUP_GAP = 0.04  # decades between neighbouring blocks that split two brightness groups
_OPEN_LIGHT_GAP = 0.08  # decades a brighter, smaller group must clear the base to be bare light
_MIN_BASE_BLOCKS = 5
_CLIP_LEVEL = 0.98  # a block with a pixel at sensor white is light the scan clipped, never film


def _margin_band(shape: Tuple[int, int], film_roi: ROI) -> np.ndarray:
    """Inside the film box, outside its central core: where the rebate and the perforations lie."""
    fy1, fy2, fx1, fx2 = film_roi
    band = np.zeros(shape, bool)
    band[fy1:fy2, fx1:fx2] = True
    my, mx = int((fy2 - fy1) * _BASE_MARGIN), int((fx2 - fx1) * _BASE_MARGIN)
    band[fy1 + my : fy2 - my, fx1 + mx : fx2 - mx] = False
    return band


def _base_blocks(det: np.ndarray, band: np.ndarray) -> Optional[np.ndarray]:
    """Per-channel means of the band's flat blocks that are clear base, or None without enough.

    Flat blocks are grouped by brightness; the largest group is the base. A smaller group
    well above it is bare light through a perforation or past the film's edge. A flat
    picture region in the band that outnumbers the base is taken for it."""
    b = _BASE_BLOCK
    hb, wb = det.shape[0] // b, det.shape[1] // b
    blocks = det[: hb * b, : wb * b].reshape(hb, b, wb, b, 3)
    mean = blocks.mean(axis=(1, 3))
    lum = mean.mean(axis=2)
    inside = band[: hb * b, : wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3)) >= 0.9
    unclipped = blocks.max(axis=(1, 3, 4)) < _CLIP_LEVEL
    flat = inside & unclipped & (lum > 1e-4) & (blocks.std(axis=(1, 3)).mean(axis=2) / np.maximum(lum, 1e-6) < _FLAT_CV)
    if int(flat.sum()) < _MIN_BASE_BLOCKS:
        return None
    cand = mean[flat]
    log_lum = np.log10(np.maximum(cand.mean(axis=1), 1e-6))
    order = np.argsort(log_lum)[::-1]  # brightest first
    groups: list[np.ndarray] = []
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or log_lum[order[i - 1]] - log_lum[order[i]] > _GROUP_GAP:
            groups.append(order[start:i])
            start = i
    min_n = max(_MIN_BASE_BLOCKS, int(0.02 * len(order)))
    big = [g for g in groups if len(g) >= min_n]
    if not big:
        return None
    return cand[max(big, key=len)]


def measure_base_log(image: ImageBuffer) -> Optional[Tuple[float, float, float]]:
    """Per-channel log10 transmittance of the clear base: the median of the flat base blocks in
    the film box's margin on the uncropped frame. None when the scan shows no film box, the
    margin holds too few flat blocks, or the base is darker than the picture, because a
    negative's base is its thinnest film."""
    det, _ = _normalize_detection_input(image, AUTOCROP_DETECT_RES)
    det = np.ascontiguousarray(det, dtype=np.float32)
    film_roi, gate = film_and_picture(det)
    if film_roi is None:
        return None
    base = _base_blocks(det, _margin_band(det.shape[:2], film_roi))
    if base is None:
        return None
    gy1, gy2, gx1, gx2 = gate
    picture = det[gy1:gy2, gx1:gx2].reshape(-1, 3)
    if np.median(base.mean(axis=1)) <= np.median(picture.mean(axis=1)):
        return None
    r, g, b = (float(np.log10(max(float(v), 1e-6))) for v in np.median(base, axis=0))
    return (r, g, b)


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
    median = float(np.median(np.asarray(prefiltered, dtype=np.float64).reshape(-1, 3) @ w))
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


def _textured_cells(lum: np.ndarray) -> np.ndarray:
    """
    Cells of the prefiltered luma grid that carry detail. The grid is tiled into
    sectors of 2x2 blocks of activity_block cells; a sector votes when its four block
    means span more than activity_gate_density (Boyack & Juenger, US 5,724,456). Block
    means average grain out, and a flat sector beside a textured one stays out because
    only its own blocks are compared. Falls back to every cell when too few pass.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    b = int(c["activity_block"])
    h, w = lum.shape
    hs, ws = (h // (2 * b)) * 2 * b, (w // (2 * b)) * 2 * b
    if hs == 0 or ws == 0:
        return lum.ravel()
    core = lum[:hs, :ws]
    blocks = core.reshape(hs // b, b, ws // b, b).mean(axis=(1, 3))
    sectors = blocks.reshape(hs // (2 * b), 2, ws // (2 * b), 2)
    active = (sectors.max(axis=(1, 3)) - sectors.min(axis=(1, 3))) > float(c["activity_gate_density"])
    if active.mean() < float(c["activity_min_fraction"]):
        return lum.ravel()
    mask = np.repeat(np.repeat(active, 2 * b, axis=0), 2 * b, axis=1)
    return core[mask]


def _textured_norm_luma(grid: ImageBuffer, bounds: LogNegativeBounds) -> np.ndarray:
    """Luma of the grid normalized against `bounds`, textured cells only."""
    epsilon = 1e-6
    norm = np.empty_like(grid)
    for ch in range(3):
        f = bounds.floors[ch]
        denom = bounds.ceils[ch] - f
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        norm[:, :, ch] = (grid[:, :, ch] - f) / denom
    return _textured_cells(LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2])


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
    lum = _textured_norm_luma(_block_median_grid(img_log), bounds)
    return float(np.percentile(lum, float(EXPOSURE_CONSTANTS["shadow_reach_percentile"])))


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
    lum = _textured_norm_luma(_block_median_grid(img_log), bounds)
    return float(np.percentile(lum, float(EXPOSURE_CONSTANTS["highlight_hold_percentile"])))


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

    lum = _textured_norm_luma(_block_median_grid(img_log), bounds)
    clip = float(EXPOSURE_CONSTANTS["anchor_trim_clip"])
    lo, hi = np.percentile(lum, [clip, 100.0 - clip])
    inner = lum[(lum >= lo) & (lum <= hi)]
    measured = 0.5 * (float(inner.mean()) + 0.5 * (float(lo) + float(hi)))

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

    img_log = _block_median_grid(img_log)

    lum = _textured_cells(LUMA_R * img_log[:, :, 0] + LUMA_G * img_log[:, :, 1] + LUMA_B * img_log[:, :, 2])
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
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

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
    spacer: float = MASK_SPACER_DEFAULT,
) -> Tuple[np.ndarray, float]:
    """The zero-mean blurred low-gamma plane an unsharp mask is built from, on `analysis_grid`, plus the val it
    was centred on (the Analysis chart draws the mask's band from it). The gamma's sign picks the polarity.
    `spacer` is per-cent of the grid: the scale above which tones are masked."""
    image = analysis_grid(image, rotation, fine_rotation, flip_horizontal, flip_vertical, distortion_k1, converge_v, converge_h, roi_norm)

    # A frame that never metered normalizes to huge values; no stretch, no mask.
    if luminance_density_range(bounds) < 1e-6:
        return np.zeros(image.shape[:2], dtype=np.float32), 0.5

    val = normalize_log_image(unmix_log_image(prefilter_log_grid(image, None, 0.0), unmix), bounds)
    lum = LUMA_R * val[:, :, 0] + LUMA_G * val[:, :, 1] + LUMA_B * val[:, :, 2]
    sigma = min(max(spacer, MASK_SPACER_MIN), MASK_SPACER_MAX) * 0.01 * min(lum.shape[:2])
    blurred = cv2.GaussianBlur(np.ascontiguousarray(lum, dtype=np.float32), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    centre = float(blurred.mean())
    return blurred - centre, centre


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
    """The unmixed light blurred by `DIFFUSION_SPREAD`, as log density on `analysis_grid`. It takes no bounds:
    the print's normalization is a per-channel gain on that light."""
    image = analysis_grid(image, rotation, fine_rotation, flip_horizontal, flip_vertical, distortion_k1, converge_v, converge_h, roi_norm)
    light = np.ascontiguousarray(10.0 ** unmix_log_image(prefilter_log_grid(image, None, 0.0), unmix), dtype=np.float32)
    sigma = DIFFUSION_SPREAD * 0.01 * min(light.shape[:2])
    return np.log10(cv2.GaussianBlur(light, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE))


def diffusion_plane(grid: np.ndarray, bounds: LogNegativeBounds, panchromatic: bool = False) -> Optional[np.ndarray]:
    """`diffusion_grid` in the print's normalization, for the kernel to mix as light. `panchromatic` collapses it
    to luma, as a B&W print does the pixel. None when the frame never metered."""
    if luminance_density_range(bounds) < 1e-6:
        return None
    val = normalize_log_image(grid, bounds)
    if panchromatic:
        lum = LUMA_R * val[:, :, 0] + LUMA_G * val[:, :, 1] + LUMA_B * val[:, :, 2]
        val = np.dstack([lum, lum, lum])
    return np.ascontiguousarray(val, dtype=np.float32)


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

    def _pct(p) -> list:
        if sorted_grid is not None:
            return [float(v) for v in percentile_from_sorted(sorted_grid, p)]
        return [float(np.percentile(img_log[:, :, ch], p)) for ch in range(3)]

    floors = _pct(p_low)
    ceils = _pct(p_high)

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
    q = float(c["neutral_axis_chroma_quantile"])
    cap = float(c["neutral_axis_chroma_cap"])
    min_px = int(c["neutral_axis_min_pixels"])
    width = float(c["color_bounds_band_width"])
    epsilon = 1e-6

    flat = img_log.reshape(-1, 3).astype(np.float64)
    base = np.asarray(base_refs, dtype=np.float64)
    norm = np.empty_like(flat)
    for ch in range(3):
        denom = luma_ceils[ch] - luma_floors[ch]
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        norm[:, ch] = (flat[:, ch] - luma_floors[ch]) / denom
    luma = LUMA_R * norm[:, 0] + LUMA_G * norm[:, 1] + LUMA_B * norm[:, 2]

    clip = max(0.00001, min(50.0 - width, float(color_clip)))
    lo, hi = np.percentile(luma, [clip, clip + width])
    band = (luma >= lo) & (luma <= hi)
    if int(band.sum()) < min_px:
        return None
    d = flat[band] - base[None, :]

    def _select(gamma: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        g = gamma.copy()
        g[np.abs(g) < epsilon] = epsilon
        chroma = _rms_chroma(d / g[None, :])
        thr = float(np.quantile(chroma, q))
        sel = chroma <= thr
        if int(sel.sum()) < min_px:
            return None
        return sel, float(np.median(chroma[sel]))

    spans = np.array([luma_floors[ch] - base_refs[ch] for ch in range(3)], dtype=np.float64)
    first = _select(spans)
    # Pass-1 loose cap: a homogeneous colored cluster would otherwise be self-normalized to
    # zero chroma by pass 2 and read as neutral.
    if first is None or first[1] > float(c["neutral_axis_first_pass_cap"]):
        return None
    provisional = np.median(d[first[0]], axis=0)
    if np.any(np.abs(provisional) < epsilon):
        return None
    second = _select(provisional)
    if second is None or second[1] > cap:
        return None
    refs = base + np.median(d[second[0]], axis=0)
    return (float(refs[0]), float(refs[1]), float(refs[2]))


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


def blend_neutral_axis(own: Optional[tuple], pooled: tuple) -> tuple:
    """The pooled axis shifted by this frame's own midtone offset from the pooled curve, scaled
    by the pool's offset weight and the frame's confidence: the roll's shape, this frame's
    level. No own axis, or a zero weight, is the pooled axis itself."""
    weight = pooled[4] if len(pooled) > 4 else 0.0
    k = weight * own[3] if own is not None else 0.0
    if k <= 0.0:
        return pooled[:4]
    dr, db = _axis_curve_residuals(own, pooled)[0]
    shift = lambda b: (b[0] + k * dr, b[1], b[2] + k * db) if b is not None else None  # noqa: E731
    return (shift(pooled[0]), shift(pooled[1]), shift(pooled[2]), pooled[3])


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
