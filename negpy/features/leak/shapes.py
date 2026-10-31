"""Leak shape geometry on the raw frame: outlines the canvas draws and grabs, and the per-pixel
frame the estimate measures in."""

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from negpy.features.leak.models import LeakKind, LeakShape, Point

# A drawn band starts NEW_BAND_WIDTH long sides wide on each side; no side is narrower than
# _MIN_BAND_WIDTH.
NEW_BAND_WIDTH = 0.06
# A drawn edge leak starts NEW_EDGE_REACH long sides deep.
NEW_EDGE_REACH = 0.15
_MIN_BAND_WIDTH = 0.005
# Bands this close to parallel, with the middle of one within the other's reach, cover one leak.
_SAME_BAND_DEG = 35.0


def new_band(a: Point, b: Point, left: float = NEW_BAND_WIDTH, right: float = NEW_BAND_WIDTH) -> LeakShape:
    return LeakShape(kind=LeakKind.BAND, points=(a, b), width=(left, right))


def new_edge(a: Point, b: Point, reach: float = NEW_EDGE_REACH) -> LeakShape:
    """An edge leak in from the picture edge a -> b, reaching *reach* into the picture on its left."""
    return LeakShape(kind=LeakKind.EDGE, points=(a, b), width=(reach, 0.0))


def new_spot(center: Point, a: Point, b: Point) -> LeakShape:
    """A spot with its *center* and the ends of its two axes, as an oval drawn on the canvas."""
    return LeakShape(kind=LeakKind.SPOT, points=(center, a, b))


def new_paint(points, size: float) -> LeakShape:
    """A painted leak of one brush stroke, *points* raw-normalized and *size* its diameter at
    HEAL_SIZE_REF."""
    return LeakShape(kind=LeakKind.PAINT, strokes=((tuple(points), float(size)),))


def with_stroke(paint: LeakShape, points, size: float) -> LeakShape:
    """The painted leak grown by one more brush stroke."""
    return replace(paint, strokes=paint.strokes + ((tuple(points), float(size)),))


def paint_weight(paint: LeakShape, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """(h, w) share of a painted leak at each work pixel: 1 deep inside the brushed region, fading to
    0 at its edge over its Falloff's share of the region's own depth, its deepest pixel's distance
    from the edge; None with nothing painted on the frame."""
    import negpy_core

    from negpy.features.retouch.logic import exclusion_cover

    cover = exclusion_cover([(points, size) for points, size in paint.strokes], shape)
    if not cover.any():
        return None
    inside = negpy_core.distance_transform(np.ascontiguousarray(cover, dtype=np.uint8), 2, 5)
    reach = max(float(inside.max()) * float(np.clip(paint.falloff, 0.0, 1.0)), 1.0)
    t = np.clip(inside / reach, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def handle_count(shape: LeakShape) -> int:
    """A band has its line's ends and an edge each side; an edge leak its ends and its reach; a spot
    its center and the end of each axis."""
    return 4 if shape.kind is LeakKind.BAND else 3


def spot_outline(spot: LeakShape) -> List[Point]:
    """The spot's edge, raw-normalized: the canvas draws it."""
    from negpy.features.local.logic import outline_points
    from negpy.features.local.models import MaskShape

    return outline_points(MaskShape.OVAL, list(spot.points[:3]))


def spot_with_handle(spot: LeakShape, handle: int, point: Point) -> LeakShape:
    """The spot with one handle moved to *point*: the center moves the whole spot, an axis end sets
    that axis."""
    pts = list(spot.points[:3])
    if handle == 0:
        dx, dy = point[0] - pts[0][0], point[1] - pts[0][1]
        return replace(spot, points=tuple((x + dx, y + dy) for x, y in pts))
    pts[handle] = point
    return replace(spot, points=tuple(pts))


def spot_frame(spot: LeakShape, shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(each work pixel's distance from the spot's center over its oval, 1 on its edge, and its angle
    around the center in the oval's axes); None for an oval with no area."""
    h, w = shape
    (cx, cy), (px, py), (qx, qy) = ((x * w, y * h) for x, y in spot.points[:3])
    axes = np.array([[px - cx, qx - cx], [py - cy, qy - cy]])
    if abs(float(np.linalg.det(axes))) < 4.0:
        return None
    inv = np.linalg.inv(axes)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = xs - cx, ys - cy
    a = inv[0, 0] * rx + inv[0, 1] * ry
    b = inv[1, 0] * rx + inv[1, 1] * ry
    return np.sqrt(a * a + b * b).astype(np.float32), np.arctan2(b, a).astype(np.float32)


def same_leak(a: LeakShape, b: LeakShape, shape: Tuple[int, int]) -> bool:
    """Whether two shapes on a frame of *shape* (h, w) cover the same leak: two bands or edges by
    same_band, two spots whose centers lie within the larger one's reach."""
    if LeakKind.PAINT in (a.kind, b.kind):
        return False
    if (a.kind is LeakKind.SPOT) != (b.kind is LeakKind.SPOT):
        return False
    if a.kind is not LeakKind.SPOT:
        return same_band(a, b, shape)
    h, w = shape

    def reach(spot: LeakShape) -> float:
        (cx, cy), *ends = spot.points[:3]
        return max(float(np.hypot((x - cx) * w, (y - cy) * h)) for x, y in ends)

    (ax, ay), (bx, by) = a.points[0], b.points[0]
    return float(np.hypot((ax - bx) * w, (ay - by) * h)) <= max(reach(a), reach(b))


def snap_edge(a: Point, b: Point, rect: Tuple[float, float, float, float]) -> Optional[Tuple[Point, Point]]:
    """A drawn stretch moved onto the nearest side of the picture box (x0, y0, x1, y1, raw-normalized),
    its ends ordered so the picture lies on its left; None when the stretch runs across the box."""
    x0, y0, x1, y1 = rect
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    horizontal = abs(b[0] - a[0]) >= abs(b[1] - a[1])
    if horizontal:
        y = y0 if abs(my - y0) <= abs(my - y1) else y1
        lo, hi = sorted((float(np.clip(a[0], x0, x1)), float(np.clip(b[0], x0, x1))))
        if hi - lo <= 1e-6:
            return None
        # Left of a line's direction (dx, dy) is (-dy, dx): below a rightward line, rows running down.
        return ((lo, y), (hi, y)) if y == y0 else ((hi, y), (lo, y))
    x = x0 if abs(mx - x0) <= abs(mx - x1) else x1
    lo, hi = sorted((float(np.clip(a[1], y0, y1)), float(np.clip(b[1], y0, y1))))
    if hi - lo <= 1e-6:
        return None
    return ((x, hi), (x, lo)) if x == x0 else ((x, lo), (x, hi))


def _band_values(band: LeakShape) -> Tuple[float, float, float, float, float, float]:
    (x0, y0), (x1, y1) = band.points[:2]
    return x0, y0, x1, y1, band.width[0], band.width[1]


def band_outline(band: LeakShape, shape: Tuple[int, int]) -> Tuple[Point, Point, Point, Point, Point, Point]:
    """(a, b, left edge at a, left edge at b, right edge at a, right edge at b), raw-normalized, for
    a band on a frame of *shape* (h, w): the canvas draws and grabs these."""
    h, w = shape
    x0, y0, x1, y1, left, right = _band_values(band)
    dx, dy = (x1 - x0) * w, (y1 - y0) * h
    length = max(float(np.hypot(dx, dy)), 1e-6)
    long_side = float(max(h, w))
    nx, ny = -dy / length, dx / length

    def shift(x: float, y: float, dist: float) -> Point:
        return (x + nx * dist * long_side / w, y + ny * dist * long_side / h)

    return (x0, y0), (x1, y1), shift(x0, y0, left), shift(x1, y1, left), shift(x0, y0, -right), shift(x1, y1, -right)


def band_with_handle(band: LeakShape, handle: int, point: Point, shape: Tuple[int, int]) -> LeakShape:
    """The band with one handle moved to *point* (raw-normalized): 0 and 1 move the line's ends,
    2 and 3 set the left and right widths from the point's distance to the line."""
    h, w = shape
    x0, y0, x1, y1, left, right = _band_values(band)
    if handle == 0:
        return replace(band, points=(point, (x1, y1)))
    if handle == 1:
        return replace(band, points=((x0, y0), point))
    dx, dy = (x1 - x0) * w, (y1 - y0) * h
    length = max(float(np.hypot(dx, dy)), 1e-6)
    across = ((point[0] - x0) * w * (-dy / length) + (point[1] - y0) * h * (dx / length)) / float(max(h, w))
    width = max(abs(across), _MIN_BAND_WIDTH)
    return replace(band, width=(width, right) if handle == 2 else (left, width))


def band_frame(band: LeakShape, shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """(along, side, length, taper) per work pixel for a band. along runs 0..length on the center
    line; side is 0 on it, +1 on the left edge and -1 on the right one, as left of the line's
    direction; taper is the falloff past the line's ends, the mean of the two widths."""
    h, w = shape
    x0, y0, x1, y1, left, right = _band_values(band)
    long_side = float(max(h, w))
    dx, dy = (x1 - x0) * w, (y1 - y0) * h
    length = float(np.hypot(dx, dy))
    wl, wr = max(left * long_side, 1.0), max(right * long_side, 1.0)
    if length < 4.0:
        return None
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = xs - x0 * w, ys - y0 * h
    along = (rx * dx + ry * dy) / length
    across = (rx * -dy + ry * dx) / length
    side = np.where(across >= 0.0, across / wl, across / wr)
    return along, side, length, 0.5 * (wl + wr)


def band_falloff(side: np.ndarray, falloff: float) -> np.ndarray:
    """The leak's share across a band: 1 on its line, 0 at either edge, flat at the edges so no seam
    shows. *falloff* 0 holds a shoulder and drops late; 1 drops early into a long soft tail, peaking
    sharply on the line above 0.75."""
    shape = 4.0 ** (1.0 - 2.0 * float(np.clip(falloff, 0.0, 1.0)))
    t = np.clip(np.abs(side), 0.0, 1.0) ** shape
    return np.cos(0.5 * np.pi * t) ** 2


def same_band(a: LeakShape, b: LeakShape, shape: Tuple[int, int]) -> bool:
    """Whether two bands on a frame of *shape* (h, w) cover the same leak: near-parallel, and the
    middle of one within the other's wider side of its line."""
    h, w = shape
    long_side = float(max(h, w))

    def frame(band: LeakShape) -> Tuple[np.ndarray, np.ndarray, float]:
        x0, y0, x1, y1, left, right = _band_values(band)
        d = np.array([(x1 - x0) * w, (y1 - y0) * h])
        d = d / max(float(np.hypot(*d)), 1e-6)
        middle = np.array([(x0 + x1) / 2.0 * w, (y0 + y1) / 2.0 * h])
        return middle, d, max(left, right) * long_side

    ma, da, ra = frame(a)
    mb, db, rb = frame(b)
    return same_line(ma, da, (mb, db, max(ra, rb)))


def same_line(middle: np.ndarray, d: np.ndarray, prev: Tuple[np.ndarray, np.ndarray, float]) -> bool:
    """A line near-parallel to a found band whose middle lies within its reach is the same leak."""
    p_middle, p_d, p_reach = prev
    parallel = abs(float(d @ p_d)) >= np.cos(np.deg2rad(_SAME_BAND_DEG))
    apart = abs(float((middle - p_middle) @ np.array([-p_d[1], p_d[0]])))
    return parallel and apart <= p_reach
