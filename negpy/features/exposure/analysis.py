"""Analysis-panel histogram/zone math. Pure NumPy/OpenCV, no Qt."""

from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from negpy.kernel.image.logic import get_luminance, rgb_to_lab_working, working_oetf_decode, working_oetf_encode

# Mirrored by the chart's x-range (PhotometricCurveWidget._X_MIN/_X_MAX) and the WGSL
# literals in shaders/density_hist.wgsl. Keep them in lock step.
DENSITY_HIST_BINS = 120
DENSITY_HIST_RANGE = (-0.1, 1.1)

# Mirrors metrics.wgsl / HISTOGRAM_BINS in gpu_engine.py.
OUTPUT_HIST_BINS = 256

# Joint RGB histogram: 32 bins per axis, mirrored as the array length and BINS constant
# in color_hist.wgsl. A color is in or out of gamut as a triple rather than per channel,
# which is what the marginal RGBL histogram cannot answer.
COLOR_HIST_BINS = 32

# Waveform: print luma per image column, (bins, cols) with row 0 = black. waveform.wgsl mirrors this.
WAVEFORM_COLS = 256
WAVEFORM_BINS = 128
# Per-cell R, G, B sums are fixed point (u32 atomics on the GPU): encoded value x this scale.
WAVEFORM_COLOR_SCALE = 1023

# Vectorscope: CIELAB a*b* in hue x log-chroma bins, so linear and log displays read one
# histogram; the last chroma bin holds all past SCOPE_MAX_CHROMA. vectorscope.wgsl mirrors this.
SCOPE_HUE_BINS = 180
SCOPE_CHROMA_BINS = 64
SCOPE_MAX_CHROMA = 60.0
SCOPE_LOG_KNEE = 4.0  # chroma where the log axis turns from linear to compressed
# The scope reads a distribution, so the CPU twin bins a strided sample.
_SCOPE_MAX_SAMPLES = 262_144

# Full-res exports run through the same normalization stage, so cap the cost.
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
    if buffer.dtype == np.uint8:
        # An 8-bit frame (the camera live view) bins by value: the same bins as the float
        # path's floor(v / 255 * 256).
        rows = [np.bincount(buffer[..., c].ravel(), minlength=OUTPUT_HIST_BINS) for c in range(3)]
        lum = get_luminance(np.ascontiguousarray(buffer.astype(np.float32) / 255.0))
        rows.append(
            np.bincount(np.minimum((lum * OUTPUT_HIST_BINS).astype(np.int64), OUTPUT_HIST_BINS - 1).ravel(), minlength=OUTPUT_HIST_BINS)
        )
        return np.stack(rows).astype(float)
    buffer = np.ascontiguousarray(buffer.astype(np.float32, copy=False))
    lum = get_luminance(buffer)
    rows = [np.histogram(buffer[..., c], bins=OUTPUT_HIST_BINS, range=(0, 1))[0] for c in range(3)]
    rows.append(np.histogram(lum, bins=OUTPUT_HIST_BINS, range=(0, 1))[0])
    return np.stack(rows).astype(float)


def color_histogram(buffer: Any) -> Optional[np.ndarray]:
    """(32, 32, 32) joint RGB counts from an H×W×3 encoded buffer, or a ready grid
    passed through. CPU counterpart of color_hist.wgsl."""
    if buffer is None:
        return None
    buffer = np.asarray(buffer)
    if buffer.ndim == 3 and buffer.shape == (COLOR_HIST_BINS,) * 3:
        return buffer.astype(float)
    if buffer.ndim != 3 or buffer.shape[-1] != 3:
        return None
    step = max(1, round(np.sqrt(buffer.shape[0] * buffer.shape[1] / _MAX_HIST_SAMPLES)))
    if step > 1:
        buffer = buffer[::step, ::step]
    q = np.clip((np.asarray(buffer, dtype=np.float32) * COLOR_HIST_BINS).astype(np.int32), 0, COLOR_HIST_BINS - 1)
    flat = (q[..., 0] * COLOR_HIST_BINS + q[..., 1]) * COLOR_HIST_BINS + q[..., 2]
    counts = np.bincount(flat.ravel(), minlength=COLOR_HIST_BINS**3)
    return counts.reshape((COLOR_HIST_BINS,) * 3).astype(float)


def gamut_fraction(color_hist: Any, out_of_gamut: Any) -> Optional[float]:
    """Share of the frame the output profile cannot print, from the joint histogram and a
    boolean (32, 32, 32) gamut mask. None when either is missing.

    Quantized to the histogram grid, so a color one bin inside the boundary counts as
    printable. It answers "how much of this frame", not "is this pixel".
    """
    if color_hist is None or out_of_gamut is None:
        return None
    hist = np.asarray(color_hist, dtype=np.float64)
    mask = np.asarray(out_of_gamut, dtype=bool)
    if hist.shape != mask.shape:
        return None
    total = float(hist.sum())
    return float(hist[mask].sum() / total) if total > 0 else None


def _waveform_cells(buffer: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(sampled H×W×3 float32 buffer, flat bin * WAVEFORM_COLS + col index per pixel)."""
    if buffer is None:
        return None
    buffer = np.asarray(buffer)
    if buffer.ndim != 3 or buffer.shape[-1] < 3 or min(buffer.shape[:2]) < 1:
        return None
    step = max(1, round(np.sqrt(buffer.shape[0] * buffer.shape[1] / _MAX_HIST_SAMPLES)))
    if step > 1:
        buffer = buffer[::step, ::step]
    rgb = np.ascontiguousarray(buffer[..., :3], dtype=np.float32)
    lum = get_luminance(rgb)
    w = lum.shape[1]
    cols = np.minimum(np.arange(w) * WAVEFORM_COLS // w, WAVEFORM_COLS - 1)
    bins = np.clip((np.nan_to_num(lum) * WAVEFORM_BINS).astype(np.int32), 0, WAVEFORM_BINS - 1)
    return rgb, (bins * WAVEFORM_COLS + cols[None, :]).ravel()


def waveform(buffer: Any) -> Optional[np.ndarray]:
    """(WAVEFORM_BINS, WAVEFORM_COLS) luma counts per image column of an H×W×3 encoded
    buffer. CPU counterpart of waveform.wgsl."""
    cells = _waveform_cells(buffer)
    if cells is None:
        return None
    return np.bincount(cells[1], minlength=WAVEFORM_BINS * WAVEFORM_COLS).reshape(WAVEFORM_BINS, WAVEFORM_COLS).astype(float)


def waveform_color(buffer: Any) -> Optional[np.ndarray]:
    """(3, WAVEFORM_BINS, WAVEFORM_COLS) fixed-point sums of the encoded R, G, B in each
    waveform cell. CPU counterpart of the sums in waveform.wgsl."""
    cells = _waveform_cells(buffer)
    if cells is None:
        return None
    rgb, flat = cells
    fixed = np.round(np.clip(np.nan_to_num(rgb.reshape(-1, 3)), 0.0, 1.0) * WAVEFORM_COLOR_SCALE)
    sums = [np.bincount(flat, weights=fixed[:, c], minlength=WAVEFORM_BINS * WAVEFORM_COLS) for c in range(3)]
    return np.stack(sums).reshape(3, WAVEFORM_BINS, WAVEFORM_COLS)


def scope_radius(chroma: Any, log: bool) -> Any:
    """Chroma -> 0..1 of the scope's radius (unclipped)."""
    c = np.asarray(chroma, dtype=np.float64)
    if log:
        return np.log1p(c / SCOPE_LOG_KNEE) / np.log1p(SCOPE_MAX_CHROMA / SCOPE_LOG_KNEE)
    return c / SCOPE_MAX_CHROMA


def scope_chroma(radius: Any, log: bool) -> Any:
    """Exact inverse of scope_radius."""
    r = np.asarray(radius, dtype=np.float64)
    if log:
        return np.expm1(r * np.log1p(SCOPE_MAX_CHROMA / SCOPE_LOG_KNEE)) * SCOPE_LOG_KNEE
    return r * SCOPE_MAX_CHROMA


def _scope_bins(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    hue = np.floor((np.arctan2(b, a) + np.pi) / (2 * np.pi) * SCOPE_HUE_BINS).astype(np.int64) % SCOPE_HUE_BINS
    radius = scope_radius(np.hypot(a, b), log=True)
    chroma = np.clip(np.floor(radius * SCOPE_CHROMA_BINS), 0, SCOPE_CHROMA_BINS - 1).astype(np.int64)
    return hue * SCOPE_CHROMA_BINS + chroma


def vectorscope_histogram(buffer: Any) -> Optional[np.ndarray]:
    """(SCOPE_HUE_BINS, SCOPE_CHROMA_BINS) a*b* counts of an H×W×3 encoded buffer. CPU
    counterpart of vectorscope.wgsl."""
    if buffer is None:
        return None
    buffer = np.asarray(buffer)
    if buffer.ndim != 3 or buffer.shape[-1] < 3 or min(buffer.shape[:2]) < 1:
        return None
    step = max(1, round(np.sqrt(buffer.shape[0] * buffer.shape[1] / _SCOPE_MAX_SAMPLES)))
    if step > 1:
        buffer = buffer[::step, ::step]
    encoded = np.clip(np.nan_to_num(np.asarray(buffer[..., :3], dtype=np.float32)), 0.0, 1.0)
    lab = rgb_to_lab_working(np.asarray(working_oetf_decode(encoded)))
    flat = _scope_bins(lab[..., 1].ravel().astype(np.float64), lab[..., 2].ravel().astype(np.float64))
    counts = np.bincount(flat, minlength=SCOPE_HUE_BINS * SCOPE_CHROMA_BINS)
    return counts.reshape(SCOPE_HUE_BINS, SCOPE_CHROMA_BINS).astype(float)


def scope_point(rgb: Sequence[float]) -> Tuple[float, float, float]:
    """(luma, a*, b*) of one display-encoded pixel: where it lands on the waveform and the vectorscope."""
    encoded = np.clip(np.asarray([rgb], dtype=np.float32), 0.0, 1.0)
    luma = float(get_luminance(encoded)[0])
    lab = rgb_to_lab_working(np.asarray(working_oetf_decode(encoded)))[0]
    return luma, float(lab[1]), float(lab[2])


@lru_cache(maxsize=4)
def _scope_lookup(size: int, log: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Polar bin of each display pixel (-1 outside the disc), each bin's pixel count, and for
    the bins no pixel samples (near neutral, finer than a pixel) the flat pixel they fold into."""
    u = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(u, u[::-1])
    radius = np.hypot(x, y)
    chroma = scope_chroma(np.minimum(radius, 1.0), log)
    hue = np.arctan2(y, x)
    idx = _scope_bins(chroma * np.cos(hue), chroma * np.sin(hue))
    idx[radius > 1.0] = -1
    area = np.bincount(idx[idx >= 0], minlength=SCOPE_HUE_BINS * SCOPE_CHROMA_BINS).astype(float)

    orphans = np.flatnonzero(area == 0)
    centre_hue = ((orphans // SCOPE_CHROMA_BINS) + 0.5) / SCOPE_HUE_BINS * 2 * np.pi - np.pi
    centre_c = scope_chroma(((orphans % SCOPE_CHROMA_BINS) + 0.5) / SCOPE_CHROMA_BINS, log=True)
    r = np.minimum(scope_radius(centre_c, log), 1.0)
    col = np.clip(((r * np.cos(centre_hue) + 1.0) / 2.0 * size).astype(np.int64), 0, size - 1)
    row = np.clip(((1.0 - r * np.sin(centre_hue)) / 2.0 * size).astype(np.int64), 0, size - 1)
    return idx, area, orphans, row * size + col


def vectorscope_density(counts: Any, size: int, log: bool) -> np.ndarray:
    """(size, size) pixels per display pixel, +a right and +b up. Each polar bin's count is
    spread over the display pixels it covers, so a bin's on-screen area does not change its brightness."""
    idx, area, orphans, orphan_px = _scope_lookup(size, log)
    flat = np.asarray(counts, dtype=float).ravel()
    out = np.zeros(idx.shape)
    inside = idx >= 0
    picked = idx[inside]
    out[inside] = flat[picked] / np.maximum(area[picked], 1.0)
    np.add.at(out.reshape(-1), orphan_px, flat[orphans])
    return out


def output_clip_fractions(bins: np.ndarray) -> Tuple[float, float]:
    """Worst-channel share of the black / white bin (shadow, highlight clipping)."""
    bins = np.asarray(bins, dtype=float)
    totals = np.maximum(bins[:3].sum(axis=1), 1.0)
    return float((bins[:3, 0] / totals).max()), float((bins[:3, -1] / totals).max())


# The zone ruler is piecewise-linear in encoded space (0 = black, V = 18% gray,
# X = white), because stops-per-zone cannot reach the top zones on a print.
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


def encoded_of_zone(zone: float) -> float:
    """Exact inverse of zone_of_encoded; kept beside it so the ruler can't fork."""
    z = min(max(float(zone), 0.0), 10.0)
    mid = _mid_gray_encoded()
    if z <= 5.0:
        return mid * z / 5.0
    return mid + (1.0 - mid) * (z - 5.0) / 5.0


def zone_occupancy(l_bins: np.ndarray) -> np.ndarray:
    """Fold display-encoded luma histogram bins into the share of each zone 0..X, rounded to the
    nearest zone as zone_field rounds, so the strip and the Zone Overlay count alike."""
    l_bins = np.asarray(l_bins, dtype=float)
    out = np.zeros(ZONE_COUNT + 1)
    total = float(l_bins.sum())
    if total <= 0.0:
        return out
    centers = (np.arange(l_bins.size) + 0.5) / l_bins.size
    cells = np.clip(np.rint(np.asarray(zone_of_encoded(centers))), 0, ZONE_COUNT).astype(int)
    np.add.at(out, cells, l_bins)
    return out / total


def zone_warnings(occ: np.ndarray) -> Tuple[bool, bool]:
    """(shadow, highlight): a texture zone pair is empty while its extreme pair holds mass."""
    shadow = float(occ[2] + occ[3]) < ZONE_EMPTY and float(occ[0] + occ[1]) > ZONE_LOADED
    highlight = float(occ[7] + occ[8]) < ZONE_EMPTY and float(occ[9] + occ[10]) > ZONE_LOADED
    return shadow, highlight


MASK_MAX_EDGE = 1024  # canvas masks are drawn scaled to the view, so finer adds nothing


def _mask_preview(rgb: Any, max_edge: int) -> Optional[np.ndarray]:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] < 3 or min(rgb.shape[:2]) < 1:
        return None
    step = max(1, -(-max(rgb.shape[:2]) // max_edge))
    return np.ascontiguousarray(rgb[::step, ::step, :3], dtype=np.float32)


def clip_masks(rgb: Any, max_edge: int = MASK_MAX_EDGE) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(shadow, highlight): pixels with any channel in the output histogram's black or white
    bin, the bins the Clipping row and the chart's clip marks count."""
    small = _mask_preview(rgb, max_edge)
    if small is None:
        return None
    edge = 1.0 / OUTPUT_HIST_BINS
    return np.any(small < edge, axis=-1), np.any(small >= 1.0 - edge, axis=-1)


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


ZONE_FIELD_MAX_EDGE = 768  # drawn scaled to the view; finer only sharpens grain


def zone_field(rgb: Any, max_edge: int = ZONE_FIELD_MAX_EDGE) -> Optional[np.ndarray]:
    """Integer Adams zone (0..10) per pixel of a display-encoded H×W×3 frame, at most
    `max_edge` on the long side; None if the frame isn't one. Smoothed first, so grain
    does not split one tone into speckled zones."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] < 3 or min(rgb.shape[:2]) < 2:
        return None
    h, w = rgb.shape[:2]
    # Slice first: INTER_AREA straight off a full-res HQ buffer is slow.
    step = max(1, max(h, w) // (2 * max_edge))
    small = np.ascontiguousarray(rgb[::step, ::step, :3], dtype=np.float32)
    scale = min(1.0, max_edge / max(small.shape[:2]))
    sh, sw = small.shape[:2]
    lum = get_luminance(cv2.resize(small, (max(1, round(sw * scale)), max(1, round(sh * scale))), interpolation=cv2.INTER_AREA))
    lum = cv2.GaussianBlur(np.ascontiguousarray(lum, dtype=np.float32), (0, 0), 2.0)
    zones = np.clip(np.rint(np.nan_to_num(zone_of_encoded(lum))), 0, ZONE_COUNT).astype(np.uint8)
    return cv2.medianBlur(zones, 5)


def zone_shares(zones: np.ndarray) -> np.ndarray:
    """Fraction of the frame in each zone 0..10."""
    return np.bincount(zones.ravel(), minlength=ZONE_COUNT + 1)[: ZONE_COUNT + 1] / max(1, zones.size)


def zone_anchors(zones: np.ndarray, min_share: float = 0.01) -> List[Tuple[int, int, int, float]]:
    """(x, y, zone, inset) per region holding at least `min_share` of the frame, at its most
    interior pixel; `inset` is that pixel's distance to the region's edge. The frame edge
    counts as an edge, so a numeral never sits half off the picture."""
    out: List[Tuple[int, int, int, float]] = []
    min_area = min_share * zones.size
    for zone in np.unique(zones):
        count, comp, stats, _ = cv2.connectedComponentsWithStats((zones == zone).astype(np.uint8), connectivity=4)
        for i in range(1, count):  # 0 is the background (everything not this zone)
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            mask = (comp == i).astype(np.uint8)
            mask[0, :] = mask[-1, :] = 0
            mask[:, 0] = mask[:, -1] = 0
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
            out.append((int(x), int(y), int(zone), float(dist[y, x])))
    return out


# Absolute ladders, centred on the defaults and inside the sliders' travel (density 0-2,
# grade R50-R180). Named strip_*, not test_strip_*: pytest collects any test_-prefixed
# callable a test module imports.
STRIP_DENSITIES = (0.4, 0.7, 1.0, 1.3, 1.6)
STRIP_GRADES = (75.0, 95.0, 115.0, 135.0, 155.0)


STRIP_GRID = (len(STRIP_GRADES), len(STRIP_DENSITIES))  # (rows, cols)

# Ring-around rungs: absolute filtration centred on neutral, like the strip's ladders, so
# a ring printed off one frame is comparable to the next and the mosaic is invariant to
# the filtration in force. 1.0 slider = 20cc (see filtration_offsets), so the step is 2cc
# and the outer rung 4cc. Calibration knobs.
RING_CC_STEP = 0.1
RING_CC_PER_UNIT = 20.0
RING_GRID = (5, 5)


def strip_center() -> Tuple[float, float]:
    """(density, grade) of the middle patch; a proof's memo key pins these, so the mosaic is one cache entry."""
    return STRIP_DENSITIES[len(STRIP_DENSITIES) // 2], STRIP_GRADES[len(STRIP_GRADES) // 2]


def strip_cells() -> List[Tuple[int, int, float, float]]:
    """(row, col, density, grade) for every patch, row-major."""
    return [(r, c, d, g) for r, g in enumerate(STRIP_GRADES) for c, d in enumerate(STRIP_DENSITIES)]


def strip_overrides() -> List[dict]:
    """ExposureConfig field overrides per patch, row-major over STRIP_GRID."""
    return [{"density": d, "grade": g} for _, _, d, g in strip_cells()]


def ring_rungs() -> Tuple[float, ...]:
    """The absolute wb values one axis steps through, centred on neutral."""
    mid = RING_GRID[0] // 2
    return tuple(round((i - mid) * RING_CC_STEP, 6) for i in range(RING_GRID[0]))


def ring_cells() -> List[Tuple[int, int, float, float]]:
    """(row, col, wb_magenta, wb_yellow) row-major. Rows step magenta, columns yellow, cyan
    stays 0. Absolute, so the centre patch is neutral rather than whatever is dialled in."""
    rungs = ring_rungs()
    rows, cols = RING_GRID
    return [(r, c, rungs[r], rungs[c]) for r in range(rows) for c in range(cols)]


def ring_overrides() -> List[dict]:
    """Per-patch ExposureConfig overrides. Only the two color-head fields, so a replace()
    built from these cannot disturb density, grade or cyan."""
    return [{"wb_magenta": m, "wb_yellow": y} for _, _, m, y in ring_cells()]


def ring_cc(index: int) -> float:
    """A rung's filtration in cc, as the axis labels show it."""
    return ring_rungs()[index] * RING_CC_PER_UNIT


def ring_nearest_cell(magenta: float, yellow: float) -> Tuple[int, int]:
    """(row, col) of the patch closest to the filtration in force."""
    rungs = ring_rungs()
    return (
        int(np.argmin([abs(m - magenta) for m in rungs])),
        int(np.argmin([abs(y - yellow) for y in rungs])),
    )


def _strip_bounds(extent: int, divisions: int, index: int) -> Tuple[int, int]:
    return round(extent * index / divisions), round(extent * (index + 1) / divisions)


def strip_mosaic(tiles: List[np.ndarray], grid: Tuple[int, int]) -> np.ndarray:
    """One frame assembled from row-major renders over `grid`, each contributing only its own
    patch. Both sides of a seam round the same fraction, so patches tile exactly."""
    rows, cols = grid
    if len(tiles) != rows * cols:
        raise ValueError(f"expected {rows * cols} tiles, got {len(tiles)}")
    out = np.empty_like(tiles[0])
    h, w = out.shape[:2]
    for i, tile in enumerate(tiles):
        if tile.shape != out.shape:
            raise ValueError(f"tile shape {tile.shape} != {out.shape}")
        row, col = divmod(i, cols)
        y0, y1 = _strip_bounds(h, rows, row)
        x0, x1 = _strip_bounds(w, cols, col)
        out[y0:y1, x0:x1] = tile[y0:y1, x0:x1]
    return out


def strip_patch_rect(h: int, w: int, row: int, col: int, grid: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of one patch inside an h×w frame — the bounds `strip_mosaic` filled."""
    rows, cols = grid
    y0, y1 = _strip_bounds(h, rows, row)
    x0, x1 = _strip_bounds(w, cols, col)
    return x0, y0, x1, y1


def strip_cell_at(nx: float, ny: float, grid: Tuple[int, int]) -> Tuple[int, int]:
    """Content-normalized position (0..1) -> (row, col)."""
    rows, cols = grid
    return (
        int(np.clip(int(ny * rows), 0, rows - 1)),
        int(np.clip(int(nx * cols), 0, cols - 1)),
    )


def strip_nearest_cell(density: float, grade: float) -> Tuple[int, int]:
    """(row, col) of the patch closest to the settings currently in force."""
    col = int(np.argmin([abs(d - density) for d in STRIP_DENSITIES]))
    row = int(np.argmin([abs(g - grade) for g in STRIP_GRADES]))
    return row, col


def _rotation_index(grid: Tuple[int, int], rotation: int) -> np.ndarray:
    """Row-major indices of `grid` after `rotation` quarter-turns CCW."""
    rows, cols = grid
    return np.rot90(np.arange(rows * cols).reshape(rows, cols), rotation % 4)


def proof_grid(grid: Tuple[int, int], rotation: int) -> Tuple[int, int]:
    """(rows, cols) after `rotation` quarter-turns. Both ladders are square today, so this only
    bites if one changes length."""
    rows, cols = grid
    return (cols, rows) if rotation % 2 else (rows, cols)


def rotate_grid(items: Sequence[Any], grid: Tuple[int, int], rotation: int) -> List[Any]:
    """Row-major `items` re-placed by `rotation` quarter-turns CCW — np.rot90's k, the same
    convention as GeometryConfig.rotation."""
    return [items[i] for i in _rotation_index(grid, rotation).ravel()]


def rotated_cell(base: Tuple[int, int], grid: Tuple[int, int], rotation: int) -> Tuple[int, int]:
    """Where a base (row, col) lands after the rotation."""
    flat = _rotation_index(grid, rotation).ravel()
    pos = int(np.flatnonzero(flat == base[0] * grid[1] + base[1])[0])
    return divmod(pos, proof_grid(grid, rotation)[1])


# Stouffer T2115: 21 steps 0.15 D apart, so 20 intervals and 3.00 D total, which is
# exactly the val domain's [0,1] on a scan whose density range is 3.0. The geometry is
# absolute like the test strip's ladders, so two frames' wedges compare. The physical
# truth rides in the labels, which report this scan's own density per step.
WEDGE_STEPS = 21
WEDGE_STEP_DENSITY = 0.15
WEDGE_SEPARATION = 0.004  # encoded delta below which neighbouring patches read as one tone


def wedge_vals() -> np.ndarray:
    """The 21 steps as val-domain log exposures: clear step (prints paper black) first,
    densest step (prints paper white) last — the Analysis chart's x-axis direction."""
    return np.linspace(1.0, 0.0, WEDGE_STEPS)


def wedge_step_density(norm_density_range: Optional[float]) -> float:
    """Density per step on this scan: its range spread over the 20 intervals. Falls back to
    the nominal T2115 increment when the range hasn't been measured yet."""
    if not norm_density_range:
        return WEDGE_STEP_DENSITY
    return abs(float(norm_density_range)) / (WEDGE_STEPS - 1)


def wedge_usable_span(enc: Any) -> Optional[Tuple[int, int]]:
    """(first, last) step still separating from its neighbour — the paper's usable scale.
    None when the whole wedge reads as one tone.

    Read off the patches themselves rather than from d_max_eff, so toe and shoulder lift,
    black-point compensation, the paper profile and split grade are all accounted for
    without a second model to keep in sync.
    """
    idx = np.flatnonzero(np.abs(np.diff(np.asarray(enc, dtype=float))) > WEDGE_SEPARATION)
    return (int(idx[0]), int(idx[-1] + 1)) if idx.size else None


def loupe_acutance(patch: Any) -> float:
    """Edge energy of a display-encoded H×W×3 patch: RMS Laplacian of its luma, ×100.

    A Laplacian rather than a plain standard deviation — σ reads *high* on a smooth gradient,
    which is the opposite of sharp, so a σ-based figure calls a soft ramp crisp. Resolution
    dependent by nature: compare regions of one frame, not one frame against another.
    """
    patch = np.asarray(patch)
    if patch.ndim != 3 or patch.shape[-1] < 3 or min(patch.shape[:2]) < 3:
        return 0.0
    lum = get_luminance(np.ascontiguousarray(patch[..., :3], dtype=np.float32))
    return float(np.sqrt(np.mean(cv2.Laplacian(lum, cv2.CV_32F) ** 2)) * 100.0)


def density_histogram(normalized_log: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """(4, DENSITY_HIST_BINS) [R, G, B, L] val-domain occupancy; out-of-range mass lands in
    the edge bins. `roi` is the crop rect (y1, y2, x1, x2) in the same frame as `normalized_log`."""
    img = normalized_log
    if roi is not None:
        y1, y2, x1, x2 = roi
        img = img[y1:y2, x1:x2]
    step = max(1, round(np.sqrt(img.shape[0] * img.shape[1] / _MAX_HIST_SAMPLES)))
    if step > 1:
        img = img[::step, ::step]
    img = np.ascontiguousarray(img)
    lum = get_luminance(img)
    lo, hi = DENSITY_HIST_RANGE
    rows = [np.histogram(np.clip(img[..., c], lo, hi), bins=DENSITY_HIST_BINS, range=DENSITY_HIST_RANGE)[0] for c in range(3)]
    rows.append(np.histogram(np.clip(lum, lo, hi), bins=DENSITY_HIST_BINS, range=DENSITY_HIST_RANGE)[0])
    return np.stack(rows).astype(np.float64)
