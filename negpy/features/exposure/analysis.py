"""Analysis-panel histogram/zone math. Pure NumPy, no Qt."""

from functools import lru_cache
from typing import Any, Optional, Tuple

import numpy as np

from negpy.kernel.image.logic import get_luminance, working_oetf_encode

# Mirrored by the chart's x-range (PhotometricCurveWidget._X_MIN/_X_MAX) and
# the WGSL literals in shaders/density_hist.wgsl — keep in lock step.
DENSITY_HIST_BINS = 120
DENSITY_HIST_RANGE = (-0.1, 1.1)

# Mirrors metrics.wgsl / HISTOGRAM_BINS in gpu_engine.py.
OUTPUT_HIST_BINS = 256

# Full-res exports run through the same normalization stage — cap the cost.
_MAX_HIST_SAMPLES = 2_000_000


def output_histogram(buffer: Any) -> Optional[np.ndarray]:
    """(4, 256) [R, G, B, L] counts from a (4, 256) bin array or an H×W×3 encoded buffer."""
    if buffer is None:
        return None
    buffer = np.asarray(buffer)
    if buffer.ndim == 2 and buffer.shape == (4, OUTPUT_HIST_BINS):
        return buffer.astype(float)
    if buffer.ndim != 3 or buffer.shape[-1] != 3:
        return None
    if buffer.shape[0] > 500:
        buffer = buffer[::4, ::4]
    buffer = np.ascontiguousarray(buffer.astype(np.float32, copy=False))
    lum = get_luminance(buffer)
    rows = [np.histogram(buffer[..., c], bins=OUTPUT_HIST_BINS, range=(0, 1))[0] for c in range(3)]
    rows.append(np.histogram(lum, bins=OUTPUT_HIST_BINS, range=(0, 1))[0])
    return np.stack(rows).astype(float)


def output_clip_fractions(bins: np.ndarray) -> Tuple[float, float]:
    """Worst-channel share of the black / white bin (shadow, highlight clipping)."""
    bins = np.asarray(bins, dtype=float)
    totals = np.maximum(bins[:3].sum(axis=1), 1.0)
    return float((bins[:3, 0] / totals).max()), float((bins[:3, -1] / totals).max())


# Zone ruler is piecewise-linear in encoded space (0 = black, V = 18% gray,
# X = white) — stops-per-zone can't reach the top zones on a print.
ZONE_COUNT = 10
ZONE_MID_REFLECTANCE = 0.18
ZONE_EMPTY = 0.005
ZONE_LOADED = 0.02


@lru_cache(maxsize=1)
def _mid_gray_encoded() -> float:
    return float(working_oetf_encode(np.asarray([ZONE_MID_REFLECTANCE], dtype=np.float32))[0])


def zone_of_encoded(enc: Any) -> Any:
    """Print zone (0..10) of a display-encoded lightness; scalar or ndarray."""
    enc = np.clip(enc, 0.0, 1.0)
    mid = _mid_gray_encoded()
    return np.where(enc <= mid, 5.0 * enc / mid, 5.0 + 5.0 * (enc - mid) / (1.0 - mid))


def zone_occupancy(l_bins: np.ndarray) -> np.ndarray:
    """Fold display-encoded luma histogram bins into ZONE_COUNT occupancy fractions."""
    l_bins = np.asarray(l_bins, dtype=float)
    out = np.zeros(ZONE_COUNT)
    total = float(l_bins.sum())
    if total <= 0.0:
        return out
    centers = (np.arange(l_bins.size) + 0.5) / l_bins.size
    cells = np.minimum(np.asarray(zone_of_encoded(centers)).astype(int), ZONE_COUNT - 1)
    np.add.at(out, cells, l_bins)
    return out / total


def zone_warnings(occ: np.ndarray) -> Tuple[bool, bool]:
    """(shadow, highlight): a texture zone pair is empty while its extreme holds mass."""
    shadow = float(occ[2] + occ[3]) < ZONE_EMPTY and float(occ[0] + occ[1]) > ZONE_LOADED
    highlight = float(occ[7] + occ[8]) < ZONE_EMPTY and float(occ[9]) > ZONE_LOADED
    return shadow, highlight


def density_histogram(normalized_log: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Luma occupancy of the val domain; out-of-range mass lands in the edge bins.
    `roi` is the crop rect (y1, y2, x1, x2) in the same frame as `normalized_log`."""
    img = normalized_log
    if roi is not None:
        y1, y2, x1, x2 = roi
        img = img[y1:y2, x1:x2]
    step = max(1, round(np.sqrt(img.shape[0] * img.shape[1] / _MAX_HIST_SAMPLES)))
    if step > 1:
        img = img[::step, ::step]
    val = get_luminance(np.ascontiguousarray(img))
    lo, hi = DENSITY_HIST_RANGE
    hist, _ = np.histogram(np.clip(val, lo, hi), bins=DENSITY_HIST_BINS, range=DENSITY_HIST_RANGE)
    return hist.astype(np.float64)
