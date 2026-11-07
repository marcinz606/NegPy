"""The picture box and the rebate outside it. Unexposed rebate holds no scene, so where a leak
crosses it, its excess exposure is the leak alone."""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from negpy.features.leak.film import WORK_SIZE, FilmResponse, Rect

# Rebate strip depth outside the picture, as a share of the long side: a crop set inside the
# picture still reaches rebate, and past the frame gap the next frame is never thinner than the gap.
_STRIP_DEPTH = 0.08
# A strip shallower than this share of the long side is the crop's own edge, not rebate.
_MIN_STRIP_DEPTH = 0.01
# A crossing is read in the rebate _LINE_NEAR to _LINE_FAR work pixels outside the frame line: past the
# scan's blur at the line, short of the sprocket holes and the edge print beyond them, which a leak
# that crosses the line lights and the edge print never reaches.
_LINE_NEAR = 3
_LINE_FAR = 8
# Profile bin width along a side, in work pixels, and the film pixels a bin needs.
_BIN_PX = 8
_MIN_BIN_PIXELS = 12
# A side is read when this share of its bins holds film.
_MIN_SIDE_COVERAGE = 0.25
# The thin end of a bin: rebate is the thinnest film in a strip, picture holds leak plus scene.
_THIN_PERCENTILE = 5.0
# A rebate side counts only with a median under _REBATE_BARE (excess exposure); a crossing is read
# over _SUPPORT_HALF profile bins either side and is lit when it lifts _SUPPORT_LIFT over the side's
# median and _SUPPORT_Z of its spread, which is at least _SUPPORT_MIN_SPREAD.
_REBATE_BARE = 0.5
_SUPPORT_HALF = 3
_SUPPORT_LIFT = 2.0
_SUPPORT_Z = 6.0
_SUPPORT_MIN_SPREAD = 0.05


def picture_rect(small: np.ndarray, geometry) -> Optional[Rect]:
    """The picture as (y0, y1, x0, x1) in *small*'s pixels: the crop, else the auto crop's image
    area, mapped back through the transforms the render applies before it."""
    if geometry is None:
        return None
    from negpy.features.geometry.logic import apply_fine_rotation, apply_keystone, apply_radial_distortion, resolve_autocrop_rect

    rect = geometry.crop_rect
    if rect is None:
        rect = resolve_autocrop_rect(small, geometry, WORK_SIZE)
    if rect is None:
        return None
    h, w = small.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.dstack([xx, yy, np.zeros_like(xx)])
    if geometry.rotation != 0:
        coords = np.rot90(coords, k=geometry.rotation)
    if geometry.flip_horizontal:
        coords = np.fliplr(coords)
    if geometry.flip_vertical:
        coords = np.flipud(coords)
    coords = np.ascontiguousarray(coords)
    if geometry.fine_rotation != 0.0:
        coords = apply_fine_rotation(coords, geometry.fine_rotation)
    if geometry.distortion_k1 != 0.0:
        coords = apply_radial_distortion(coords, geometry.distortion_k1)
    coords = apply_keystone(coords, geometry.converge_v, geometry.converge_h)
    th, tw = coords.shape[:2]
    x1, y1, x2, y2 = rect
    inside = coords[
        int(round(min(y1, y2) * th)) : int(round(max(y1, y2) * th)),
        int(round(min(x1, x2) * tw)) : int(round(max(x1, x2) * tw)),
    ].reshape(-1, 3)
    if inside.shape[0] == 0:
        return None
    x0, xe = float(inside[:, 0].min()), float(inside[:, 0].max())
    y0, ye = float(inside[:, 1].min()), float(inside[:, 1].max())
    box = (
        int(np.clip(round(y0), 0, h)),
        int(np.clip(round(ye) + 1, 0, h)),
        int(np.clip(round(x0), 0, w)),
        int(np.clip(round(xe) + 1, 0, w)),
    )
    if box[1] - box[0] < 8 or box[3] - box[2] < 8:
        return None
    return box


def _side_strips(rect: Rect, shape: Tuple[int, int]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """(side, strip rows, strip cols) for each side with room for rebate outside the picture."""
    y0, y1, x0, x1 = rect
    h, w = shape
    depth = max(2, round(_STRIP_DEPTH * max(h, w)))
    shallow = max(4, round(_MIN_STRIP_DEPTH * max(h, w)))
    sides = [
        ("top", np.arange(max(0, y0 - depth), y0), np.arange(x0, x1)),
        ("bottom", np.arange(y1, min(h, y1 + depth)), np.arange(x0, x1)),
        ("left", np.arange(y0, y1), np.arange(max(0, x0 - depth), x0)),
        ("right", np.arange(y0, y1), np.arange(x1, min(w, x1 + depth))),
    ]
    return [
        (name, rows, cols)
        for name, rows, cols in sides
        if (rows.size if name in ("top", "bottom") else cols.size) >= shallow and min(rows.size, cols.size) >= 2
    ]


def _profile(small: np.ndarray, film: np.ndarray, rows: np.ndarray, cols: np.ndarray, along_x: bool) -> Optional[np.ndarray]:
    """The thin end of film transmittance in bins along a side, (bins, 3), gaps filled; None when
    too little of the side is film."""
    block = small[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    keep = film[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    if not along_x:
        block, keep = block.transpose(1, 0, 2), keep.T
    length = block.shape[1]
    edges = np.linspace(0, length, max(2, round(length / _BIN_PX)) + 1).round().astype(int)
    values = np.full((edges.size - 1, 3), np.nan)
    for k in range(edges.size - 1):
        mask = keep[:, edges[k] : edges[k + 1]]
        if mask.sum() >= _MIN_BIN_PIXELS:
            values[k] = np.percentile(block[:, edges[k] : edges[k + 1]][mask], 100.0 - _THIN_PERCENTILE, axis=0)
    found = ~np.isnan(values[:, 0])
    if found.mean() < _MIN_SIDE_COVERAGE:
        return None
    centers = (edges[:-1] + edges[1:]) / 2.0
    return np.stack([np.interp(centers, centers[found], values[found, c]) for c in range(3)], axis=-1)


def read_profiles(small: np.ndarray, film: np.ndarray, rect: Rect, line: bool = False) -> Dict[str, np.ndarray]:
    """Each readable side's rebate profile, the thin end of film transmittance in bins along it; with
    *line*, of the rebate _LINE_NEAR to _LINE_FAR pixels outside the frame line alone."""
    profiles = {}
    for name, rows, cols in _side_strips(rect, small.shape[:2]):
        if line:
            near = slice(_LINE_NEAR, _LINE_FAR)
            if name == "top":
                rows = rows[::-1][near][::-1]
            elif name == "bottom":
                rows = rows[near]
            elif name == "left":
                cols = cols[::-1][near][::-1]
            else:
                cols = cols[near]
            if rows.size < 2 or cols.size < 2:
                continue
        p = _profile(small, film, rows, cols, along_x=name in ("top", "bottom"))
        if p is not None:
            profiles[name] = p
    return profiles


def refine_base(profiles: Dict[str, np.ndarray], base: np.ndarray) -> np.ndarray:
    """Unleaked rebate is bare base; film_base reads only the frame interior, whose thinnest shadow
    can sit above it."""
    if not profiles:
        return base
    bins = np.concatenate(list(profiles.values()))
    return np.maximum(base, bins[np.argmax((bins / base).min(axis=-1))]).astype(np.float32)


class RebateSide(NamedTuple):
    name: str
    u: np.ndarray  # (bins, 3) excess exposure per layer
    median: np.ndarray  # per layer
    spread: float  # of the strongest layer
    # The rebate just outside the frame line, (bins, 3) excess exposure less its median; None unread.
    line: Optional[np.ndarray] = None


def rebate_sides(
    profiles: Dict[str, np.ndarray], response: FilmResponse, lines: Optional[Dict[str, np.ndarray]] = None
) -> List[RebateSide]:
    """Each side that is bare rebate, in excess exposure per layer, with the rebate just outside its
    frame line from *lines*. A side whose median is not near bare base is picture past a crop set
    inside it."""
    sides = []
    for name, p in profiles.items():
        u = np.maximum(response.excess(p), 0.0)
        strongest = u.max(axis=-1)
        median = float(np.median(strongest))
        if median > _REBATE_BARE:
            continue
        spread = max(
            float(1.4826 * np.median(np.abs(strongest - median))), float(np.percentile(strongest, 75)) - median, _SUPPORT_MIN_SPREAD
        )
        line = None
        if lines is not None and name in lines:
            near = np.maximum(response.excess(lines[name]), 0.0)
            line = near - np.median(near, axis=0)
        sides.append(RebateSide(name, u, np.median(u, axis=0), spread, line))
    return sides


def rebate_crossing(point: np.ndarray, direction: np.ndarray, rect: Rect, sides: List[RebateSide]) -> Optional[Optional[np.ndarray]]:
    """Where a line crosses bare rebate: the leak there per layer when the rebate is lit, an empty
    array when it is not; None when the line crosses no bare rebate. The subject never reaches the
    rebate, so a lit crossing is a leak's own evidence, and its color is the leak's. A crossing is
    read as the median over _SUPPORT_HALF bins either side, so a printed frame number is not a leak,
    and counts when its strongest layer lifts _SUPPORT_LIFT over the side's median and _SUPPORT_Z
    of its spread. Where the rebate just outside the frame line can be read, it must lift
    _SUPPORT_LIFT too, and its lift is the leak: edge print past the sprocket holes never reaches
    the line, and reads in its own color."""
    y0, y1, x0, x1 = rect
    crossed = False
    for side in sides:
        if side.name in ("top", "bottom") and abs(direction[1]) > 1e-3:
            edge = (y0 - 1) if side.name == "top" else y1
            at = point[0] + (edge - point[1]) / direction[1] * direction[0]
            span = (x0, x1)
        elif side.name in ("left", "right") and abs(direction[0]) > 1e-3:
            edge = (x0 - 1) if side.name == "left" else x1
            at = point[1] + (edge - point[0]) / direction[0] * direction[1]
            span = (y0, y1)
        else:
            continue
        if not span[0] <= at < span[1]:
            continue
        crossed = True
        i = int((at - span[0]) / (span[1] - span[0]) * side.u.shape[0])
        lift = np.median(side.u[max(0, i - _SUPPORT_HALF) : i + _SUPPORT_HALF + 1], axis=0) - side.median
        strongest = float(lift.max())
        if strongest >= _SUPPORT_LIFT and strongest / side.spread >= _SUPPORT_Z:
            if side.line is None:
                return np.maximum(lift, 0.0)
            j = int((at - span[0]) / (span[1] - span[0]) * side.line.shape[0])
            near = np.median(side.line[max(0, j - _SUPPORT_HALF) : j + _SUPPORT_HALF + 1], axis=0)
            if float(near.max()) >= _SUPPORT_LIFT:
                return np.maximum(near, 0.0)
    return np.zeros(0) if crossed else None


def rebate_lit(point: np.ndarray, direction: np.ndarray, rect: Rect, sides: List[RebateSide]) -> Optional[bool]:
    """Whether the rebate is lit where a line crosses it; None when it crosses no bare rebate."""
    found = rebate_crossing(point, direction, rect, sides)
    return None if found is None else found.size > 0


def edge_side(a: np.ndarray, b: np.ndarray, rect: Rect) -> str:
    """The picture side a stretch a -> b (work pixels) lies along."""
    y0, y1, x0, x1 = rect
    if abs(b[0] - a[0]) >= abs(b[1] - a[1]):
        return "top" if abs((a[1] + b[1]) / 2.0 - y0) <= abs((a[1] + b[1]) / 2.0 - y1) else "bottom"
    return "left" if abs((a[0] + b[0]) / 2.0 - x0) <= abs((a[0] + b[0]) / 2.0 - x1) else "right"


def rebate_along(
    a: np.ndarray, b: np.ndarray, rect: Rect, profiles: Dict[str, np.ndarray], response: FilmResponse, n: int
) -> Optional[np.ndarray]:
    """(n, 3) excess exposure per layer in the rebate outside a stretch of picture edge a -> b (work
    pixels), resampled to n cells along it; None with no rebate there. Rebate holds no scene, so lit
    rebate along an edge is the leak at the edge."""
    name = edge_side(a, b, rect)
    if name not in profiles:
        return None
    u = np.maximum(response.excess(profiles[name]), 0.0)
    y0, y1, x0, x1 = rect
    lo, hi = (x0, x1) if name in ("top", "bottom") else (y0, y1)
    start, stop = (a[0], b[0]) if name in ("top", "bottom") else (a[1], b[1])
    along = (np.linspace(start, stop, n) - lo) / max(hi - lo, 1) * u.shape[0] - 0.5
    bins = np.arange(u.shape[0])
    return np.stack([np.interp(along, bins, u[:, c]) for c in range(3)], axis=-1)
