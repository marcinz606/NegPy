"""Analysis-panel histogram/zone math. Pure NumPy/OpenCV, no Qt."""

from functools import lru_cache
from typing import Any, List, Optional, Tuple

import cv2
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
    step = max(1, round(np.sqrt(buffer.shape[0] * buffer.shape[1] / _MAX_HIST_SAMPLES)))
    if step > 1:
        buffer = buffer[::step, ::step]
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


ZONE_GRID_CELLS = 24  # cells along the frame's long edge


def zone_grid(rgb: Any, cells: int = ZONE_GRID_CELLS) -> Optional[np.ndarray]:
    """Integer Adams zone (0..10) per cell of a fixed grid over a display-encoded H×W×3
    frame; None if the frame isn't one. Cells are square-ish whatever the aspect, and the
    area-average over a cell is what damps grain — no extra smoothing needed.

    The frame is sRGB- rather than Adobe-RGB-encoded on the soft-proof/splash path; the
    ruler hinge differs by ~0.04 zone there, under the rounding step, so we don't branch.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] < 3 or min(rgb.shape[:2]) < 2:
        return None
    h, w = rgb.shape[:2]
    # Slice first: INTER_AREA straight off a full-res HQ buffer costs ~50 ms.
    step = max(1, min(h, w) // (4 * cells))
    small = np.ascontiguousarray(rgb[::step, ::step, :3], dtype=np.float32)
    scale = cells / max(small.shape[:2])
    sh, sw = small.shape[:2]
    small = cv2.resize(small, (max(1, round(sw * scale)), max(1, round(sh * scale))), interpolation=cv2.INTER_AREA)
    z = np.nan_to_num(zone_of_encoded(get_luminance(small)))
    return np.rint(z).astype(np.int32)


def zone_region_labels(zones: np.ndarray) -> List[Tuple[int, int, int]]:
    """One label anchor per contiguous same-zone region: [(col, row, zone)].

    Anchors are snapped to a cell the region actually owns, so a concave region's numeral
    can't land outside it (its centroid can).
    """
    out: List[Tuple[int, int, int]] = []
    for zone in np.unique(zones):
        count, comp, _, centroids = cv2.connectedComponentsWithStats((zones == zone).astype(np.uint8), connectivity=4)
        for i in range(1, count):  # 0 is the background (everything not this zone)
            ys, xs = np.nonzero(comp == i)
            cx, cy = centroids[i]
            j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
            out.append((int(xs[j]), int(ys[j]), int(zone)))
    return out


# Test strip ladder: a fixed absolute grid, not an offset around the current
# settings, so a strip printed off one frame is comparable to the next. Columns
# darken left to right, rows soften top to bottom — the two diagonals then read as
# the darkroom's light/dark and soft/hard axes. Named strip_* rather than
# test_strip_*: pytest collects any test_-prefixed callable a test module imports.
STRIP_DENSITIES = (0.4, 0.7, 1.0, 1.3, 1.6, 1.9)
STRIP_GRADES = (55.0, 80.0, 105.0, 130.0, 155.0, 180.0)


def strip_cells() -> List[Tuple[int, int, float, float]]:
    """(row, col, density, grade) for every patch, row-major."""
    return [(r, c, d, g) for r, g in enumerate(STRIP_GRADES) for c, d in enumerate(STRIP_DENSITIES)]


def _strip_bounds(extent: int, divisions: int, index: int) -> Tuple[int, int]:
    return round(extent * index / divisions), round(extent * (index + 1) / divisions)


def strip_mosaic(tiles: List[np.ndarray]) -> np.ndarray:
    """One frame assembled from `strip_cells()`-ordered renders, each contributing only
    its own patch. Bounds are rounded from the same fractions on both sides of a seam,
    so patches tile exactly — no gap, no overlap."""
    cells = strip_cells()
    if len(tiles) != len(cells):
        raise ValueError(f"expected {len(cells)} tiles, got {len(tiles)}")
    out = np.empty_like(tiles[0])
    h, w = out.shape[:2]
    for (row, col, _, _), tile in zip(cells, tiles):
        if tile.shape != out.shape:
            raise ValueError(f"tile shape {tile.shape} != {out.shape}")
        y0, y1 = _strip_bounds(h, len(STRIP_GRADES), row)
        x0, x1 = _strip_bounds(w, len(STRIP_DENSITIES), col)
        out[y0:y1, x0:x1] = tile[y0:y1, x0:x1]
    return out


def strip_patch_rect(h: int, w: int, row: int, col: int) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of one patch inside an h×w frame — the bounds `strip_mosaic` filled."""
    y0, y1 = _strip_bounds(h, len(STRIP_GRADES), row)
    x0, x1 = _strip_bounds(w, len(STRIP_DENSITIES), col)
    return x0, y0, x1, y1


def strip_cell_at(nx: float, ny: float) -> Tuple[int, int]:
    """Content-normalized position (0..1) -> (row, col)."""
    rows, cols = len(STRIP_GRADES), len(STRIP_DENSITIES)
    return (
        int(np.clip(int(ny * rows), 0, rows - 1)),
        int(np.clip(int(nx * cols), 0, cols - 1)),
    )


def strip_nearest_cell(density: float, grade: float) -> Tuple[int, int]:
    """(row, col) of the patch closest to the settings currently in force."""
    col = int(np.argmin([abs(d - density) for d in STRIP_DENSITIES]))
    row = int(np.argmin([abs(g - grade) for g in STRIP_GRADES]))
    return row, col


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
