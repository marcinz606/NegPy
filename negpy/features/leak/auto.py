"""Auto: find a frame's leaks and propose a shape over each, for the user to adjust."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from negpy.features.leak.bake import film_dmax, leak_scan, leak_unmix
from negpy.features.leak.estimate import CAP_FILM_FRACTION, REBATE_LIT, edge_rows, lone_gap, spike_floor
from negpy.features.leak.film import FilmResponse, read_film
from negpy.features.leak.models import LeakShape
from negpy.features.leak.rebate import rebate_along, rebate_lit, rebate_sides
from negpy.features.leak.shapes import new_band, new_edge, same_line, snap_edge

# The excess map is read in _RADON_CELL_PX cells over the frame's _RADON_TYPICAL percentile spike and
# smoothed by _RADON_SMOOTH cells; lines step by _RADON_ANGLE_STEP degrees and count only where they
# cross _RADON_MIN_SPAN of the longest line's cells. A line is a leak when the rebate it crosses is
# lit; one that crosses no rebate needs _RADON_MIN_Z robust deviations over its angle's lines and a
# lift of _RADON_LONE_LIFT. No line under _AUTO_MIN_LEAK is read. Edges and ends go where the
# evidence falls under _AUTO_FADE of the peak; at most _RADON_MAX_BANDS.
_RADON_CELL_PX = 16
_RADON_TYPICAL = 40.0
_RADON_SMOOTH = 2.0
_RADON_ANGLE_STEP = 2.0
_RADON_MIN_SPAN = 0.35
_RADON_MIN_Z = 6.0
_RADON_LONE_LIFT = 50.0
_RADON_MAX_BANDS = 3
# Only the strongest lines are candidates: a leak ranks among them, while past them some weak line
# meets a frame number or a lit gap in the rebate by chance. A line that crosses no rebate has only
# its strength to go on and must rank higher still.
_RADON_TRIES = 16
_RADON_LONE_TRIES = 6
_AUTO_MIN_LEAK = 0.5
_AUTO_FADE = 0.25
_AUTO_MIN_CELLS = 6
# Edges: each picture side is read _EDGE_PROBE_REACH long sides deep. A run of at least
# _EDGE_MIN_CELLS cells is a leak when lit rebate the picture carries reads _EDGE_MIN_LEAK there;
# its reach is its cells' _EDGE_REACH_PERCENTILE reach, at least _EDGE_MIN_REACH.
_EDGE_PROBE_REACH = 0.3
_EDGE_MIN_CELLS = 3
_EDGE_GAP_CELLS = 3
_EDGE_MIN_LEAK = 0.5
_EDGE_REACH_PERCENTILE = 90.0
_EDGE_MIN_REACH = 0.03


def _excess_map(small: np.ndarray, film: np.ndarray, response: FilmResponse) -> Tuple[np.ndarray, np.ndarray, int]:
    """(each cell's shadow spike over the frame's typical one, smoothed to a leak's scale, which cells
    hold film, the cell size in work pixels), over the whole film, rebate included."""
    h, w = small.shape[:2]
    cell = _RADON_CELL_PX
    gh, gw = max(4, h // cell), max(4, w // cell)
    u = np.maximum(response.excess(small), 0.0).mean(axis=-1, keepdims=True)
    floor = np.full((gh, gw), np.nan, dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            ys, xs = slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell)
            keep = film[ys, xs]
            if keep.mean() >= CAP_FILM_FRACTION:
                floor[i, j] = spike_floor(u[ys, xs][keep])[0]
    seen = ~np.isnan(floor)
    if seen.sum() < _AUTO_MIN_CELLS:
        return np.zeros((gh, gw), np.float32), seen, cell
    excess = np.where(seen, np.log1p(np.maximum(floor, 0.0)) - float(np.nanpercentile(np.log1p(floor), _RADON_TYPICAL)), 0.0)
    mask = seen.astype(np.float32)
    blur_v = cv2.GaussianBlur((excess * mask).astype(np.float32), (0, 0), _RADON_SMOOTH)
    blur_m = cv2.GaussianBlur(mask, (0, 0), _RADON_SMOOTH)
    return np.where(blur_m > 0.2, blur_v / np.maximum(blur_m, 1e-6), 0.0).astype(np.float32), blur_m > 0.2, cell


def _sinogram(excess: np.ndarray, valid: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radon transform as line means: for each angle, the map rotated so lines run down the columns,
    and each column's mean over the cells it crosses; columns crossing too few cells are blank."""
    gh, gw = excess.shape
    diag = int(np.ceil(np.hypot(gh, gw))) + 2
    pad_v = np.zeros((diag, diag), np.float32)
    pad_m = np.zeros((diag, diag), np.float32)
    oy, ox = (diag - gh) // 2, (diag - gw) // 2
    pad_v[oy : oy + gh, ox : ox + gw] = excess
    pad_m[oy : oy + gh, ox : ox + gw] = valid
    center = (diag / 2.0, diag / 2.0)
    sums = np.zeros((angles.size, diag), np.float32)
    counts = np.zeros((angles.size, diag), np.float32)
    for k, angle in enumerate(angles):
        rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        sums[k] = cv2.warpAffine(pad_v, rot, (diag, diag), flags=cv2.INTER_LINEAR).sum(axis=0)
        counts[k] = cv2.warpAffine(pad_m, rot, (diag, diag), flags=cv2.INTER_LINEAR).sum(axis=0)
    span = counts.max(axis=1, keepdims=True)
    means = np.where(counts >= _RADON_MIN_SPAN * np.maximum(span, 1.0), sums / np.maximum(counts, 1e-6), np.nan)
    return means, counts


def auto_bands(img: np.ndarray, geometry=None, dmax: float = 0.0, unmix: Optional[np.ndarray] = None) -> List[LeakShape]:
    """Bands over the frame's leaks, strongest first; empty when none stands out.

    Each film cell's shadow spike over the frame's typical one, smoothed to a leak's scale, is the
    evidence; its Radon transform, the mean along every straight line, peaks on a straight leak.
    Of the _RADON_TRIES strongest lines, one counts when the rebate is lit where it crosses it; one
    that crosses no rebate must be among the _RADON_LONE_TRIES strongest, stand _RADON_MIN_Z robust
    deviations above the other lines at its angle and lift the shadows by _RADON_LONE_LIFT. Its
    profile across sets the two edges, and the map along it sets where the band starts and stops."""
    reading = read_film(img, geometry, dmax, unmix)
    if reading is None:
        return []
    small, film, rect = reading.small, reading.film, reading.rect
    sides = rebate_sides(reading.profiles, reading.response, reading.lines) if rect is not None else []
    excess, valid, cell = _excess_map(small, film, reading.response)
    if not valid.any():
        return []
    angles = np.arange(0.0, 180.0, _RADON_ANGLE_STEP)
    means, _ = _sinogram(excess, valid, angles)
    if not np.isfinite(means).any():
        return []
    h, w = small.shape[:2]
    gh, gw = excess.shape
    diag = means.shape[1]
    long_side = float(max(h, w))
    mid = np.array([(gw - 1) / 2.0, (gh - 1) / 2.0])
    order = np.argsort(np.where(np.isfinite(means), -means, np.inf), axis=None)
    bands: List[LeakShape] = []
    found: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for rank, flat in enumerate(order):
        k, col = np.unravel_index(int(flat), means.shape)
        peak = float(means[k, col])
        if rank >= _RADON_TRIES or not np.isfinite(peak) or np.expm1(max(peak, 0.0)) < _AUTO_MIN_LEAK or len(bands) >= _RADON_MAX_BANDS:
            break
        # The rotation sends p to the column (p - center) . (cos, sin), so a column is the line
        # through that offset along the normal, running along (sin, -cos).
        theta = np.deg2rad(angles[k])
        d = np.array([np.sin(theta), -np.cos(theta)])
        normal = np.array([np.cos(theta), np.sin(theta)])
        point = mid + (col - (diag - 1) / 2.0) * normal
        steps = np.arange(-diag, diag, 0.5)
        line = point[None, :] + steps[:, None] * d[None, :]
        inside = (line[:, 0] >= 0) & (line[:, 0] <= gw - 1) & (line[:, 1] >= 0) & (line[:, 1] <= gh - 1)
        if not inside.any():
            continue
        cells = line[inside]
        middle = cells.mean(axis=0)
        if any(same_line(middle, d, prev) for prev in found):
            continue
        finite = means[k][np.isfinite(means[k])]
        center = float(np.median(finite))
        spread = max(float(1.4826 * np.median(np.abs(finite - center))), 1e-3)
        lit = rebate_lit((point + 0.5) * cell, d, rect, sides) if rect is not None else None
        if lit is None:
            lone = rank < _RADON_LONE_TRIES and (peak - center) / spread >= _RADON_MIN_Z and np.expm1(peak) >= _RADON_LONE_LIFT
            if not lone:
                continue
        elif not lit:
            continue
        lifted = np.where(np.isfinite(means[k]), means[k], center) - center
        lo = col
        while lo > 0 and lifted[lo - 1] > _AUTO_FADE * lifted[col]:
            lo -= 1
        hi = col
        while hi < diag - 1 and lifted[hi + 1] > _AUTO_FADE * lifted[col]:
            hi += 1
        values = excess[np.round(cells[:, 1]).astype(int), np.round(cells[:, 0]).astype(int)]
        strong = np.flatnonzero(values > _AUTO_FADE * float(values.max()))
        a = (cells[strong[0]] + 0.5) * cell
        b = (cells[strong[-1]] + 0.5) * cell
        # Columns grow along the normal, the left of d; the band's left is left of a -> b.
        left_w = max((hi - col + 0.5) * cell, cell) / long_side
        right_w = max((col - lo + 0.5) * cell, cell) / long_side
        pa, pb = (float(a[0] / w), float(a[1] / h)), (float(b[0] / w), float(b[1] / h))
        band = new_band(pa, pb, float(left_w), float(right_w)) if (b - a) @ d >= 0 else new_band(pb, pa, float(right_w), float(left_w))
        bands.append(band)
        found.append((middle, d, float(max(hi - lo, 2))))
    return bands


def _side_edges(reading, ends) -> List[LeakShape]:
    """Edges over the runs of cells along one picture side where lit rebate carries on into the picture."""
    h, w = reading.small.shape[:2]
    out: List[LeakShape] = []
    # Only lit rebate vouches for an edge leak: a side whose rebate is bare is not read.
    lit = rebate_along(
        np.array([ends[0][0] * w, ends[0][1] * h]),
        np.array([ends[1][0] * w, ends[1][1] * h]),
        reading.rect,
        reading.profiles,
        reading.response,
        8,
    )
    if lit is None or float(np.median(lit.max(axis=-1))) < REBATE_LIT:
        return []
    rows = edge_rows(reading, new_edge(ends[0], ends[1], _EDGE_PROBE_REACH))
    if rows is None:
        return []
    strong = rows.supported & (np.nan_to_num(rows.amplitude, nan=0.0) >= _EDGE_MIN_LEAK)
    # Tongues leave short gaps in a leak along a side; a gap of _EDGE_GAP_CELLS or fewer joins it.
    read = strong.copy()
    for i in np.flatnonzero(~read):
        left, right = read[max(0, i - _EDGE_GAP_CELLS) : i], read[i + 1 : i + 1 + _EDGE_GAP_CELLS]
        strong[i] = left.any() and right.any()
    runs = []
    start = None
    for i in range(rows.n + 1):
        if i < rows.n and strong[i]:
            start = i if start is None else start
            continue
        if start is not None:
            runs.append((start, i))
        start = None
    if not runs:
        return []
    # One leak along a side crosses in stretches; runs closer than the side's strong length are one,
    # and one counts where a run of _EDGE_MIN_CELLS or more holds it.
    total = sum(b - a for a, b in runs)
    merged = [[*runs[0], runs[0][1] - runs[0][0]]]
    for lo, hi in runs[1:]:
        if lo - merged[-1][1] <= total:
            merged[-1][1] = hi
            merged[-1][2] = max(merged[-1][2], hi - lo)
        else:
            merged.append([lo, hi, hi - lo])
    for lo, hi, longest in merged:
        if longest < _EDGE_MIN_CELLS:
            continue
        pa = np.add(ends[0], np.subtract(ends[1], ends[0]) * lo / rows.n)
        pb = np.add(ends[0], np.subtract(ends[1], ends[0]) * hi / rows.n)
        extent = rows.extent[lo:hi][strong[lo:hi]]
        reach = (
            float(np.nanpercentile(extent, _EDGE_REACH_PERCENTILE)) * _EDGE_PROBE_REACH if np.isfinite(extent).any() else _EDGE_PROBE_REACH
        )
        out.append(new_edge((float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1])), max(reach, _EDGE_MIN_REACH)))
    return out


def auto_edges(img: np.ndarray, geometry=None, dmax: float = 0.0, unmix: Optional[np.ndarray] = None) -> List[LeakShape]:
    """Edge leaks in from the picture's sides: each side is read as one long edge leak, and every
    run of cells along it where lit rebate carries on into the picture becomes an edge over that
    run. Rebate holds no scene, so it is the leak's own evidence; the picture alone reads a bright
    subject near an edge the same as a leak, and those are left to draw by hand. Rebate lit along a
    side whose light the picture does not carry is edge fog and reads no leak."""
    reading = read_film(img, geometry, dmax, unmix)
    if reading is None or reading.rect is None:
        return []
    h, w = reading.small.shape[:2]
    y0, y1, x0, x1 = reading.rect
    box = (x0 / w, y0 / h, x1 / w, y1 / h)
    found: List[LeakShape] = []
    for a, b in (
        ((box[0], box[1]), (box[2], box[1])),
        ((box[0], box[3]), (box[2], box[3])),
        ((box[0], box[1]), (box[0], box[3])),
        ((box[2], box[1]), (box[2], box[3])),
    ):
        ends = snap_edge(a, b, box)
        if ends is None:
            continue
        side = _side_edges(reading, ends)
        if not side and lone_gap(reading, new_edge(ends[0], ends[1], _EDGE_PROBE_REACH)) > 0:
            # A crop set past the frame line shows lit rebate where no leak crosses: an edge clears it.
            side = [new_edge(ends[0], ends[1], _EDGE_MIN_REACH)]
        found += side
    return found


def find_leaks(img: np.ndarray, config) -> List[LeakShape]:
    """Every leak Auto proves on a frame under its edit: edges first, then bands."""
    scan, dmax, unmix = leak_scan(img, config), film_dmax(config), leak_unmix(config)
    return auto_edges(scan, config.geometry, dmax, unmix) + auto_bands(scan, config.geometry, dmax, unmix)
