"""Composition guide geometry for the crop tool overlay.

Qt-free; all shapes are polylines in pixel space (0,0)-(w,h) so angles stay aspect-true.
"""

import math
from enum import StrEnum
from typing import List, Tuple

Point = Tuple[float, float]
Polyline = List[Point]

PHI = (1.0 + math.sqrt(5.0)) / 2.0
_SPIRAL_ARCS = 10
_ARC_SAMPLES = 12
_GRID_CELLS = 8


class CropGuide(StrEnum):
    THIRDS = "thirds"
    PHI = "phi"
    DIAGONALS = "diagonals"
    TRIANGLES = "triangles"
    SPIRAL = "spiral"
    ARMATURE = "armature"
    DIAGONAL_METHOD = "diagonal_method"
    GRID = "grid"
    OFF = "off"


GUIDE_LABELS: dict[CropGuide, str] = {
    CropGuide.THIRDS: "Thirds",
    CropGuide.PHI: "Phi Grid",
    CropGuide.DIAGONALS: "Diagonals",
    CropGuide.TRIANGLES: "Golden Triangles",
    CropGuide.SPIRAL: "Golden Spiral",
    CropGuide.ARMATURE: "Armature",
    CropGuide.DIAGONAL_METHOD: "Diagonal Method",
    CropGuide.GRID: "Grid",
    CropGuide.OFF: "Off",
}

ORIENTATION_COUNT: dict[CropGuide, int] = {
    CropGuide.SPIRAL: 8,
    CropGuide.TRIANGLES: 2,
}


def _fraction_lines(w: float, h: float, fractions: List[float]) -> List[Polyline]:
    lines: List[Polyline] = []
    for f in fractions:
        lines.append([(w * f, 0.0), (w * f, h)])
    for f in fractions:
        lines.append([(0.0, h * f), (w, h * f)])
    return lines


def _grid(w: float, h: float) -> List[Polyline]:
    step = min(w, h) / _GRID_CELLS
    lines: List[Polyline] = []
    eps = step * 1e-6  # keep float-rounded edge lines off the border
    xs = {w / 2.0}
    k = 1
    while w / 2.0 + k * step < w - eps:
        xs.add(w / 2.0 + k * step)
        xs.add(w / 2.0 - k * step)
        k += 1
    ys = {h / 2.0}
    k = 1
    while h / 2.0 + k * step < h - eps:
        ys.add(h / 2.0 + k * step)
        ys.add(h / 2.0 - k * step)
        k += 1
    for x in sorted(xs):
        lines.append([(x, 0.0), (x, h)])
    for y in sorted(ys):
        lines.append([(0.0, y), (w, y)])
    return lines


def _diagonals(w: float, h: float) -> List[Polyline]:
    return [
        [(0.0, 0.0), (w, h)],
        [(w, 0.0), (0.0, h)],
        [(w / 2.0, 0.0), (w / 2.0, h)],
        [(0.0, h / 2.0), (w, h / 2.0)],
    ]


def _project_foot(px: float, py: float, dx: float, dy: float) -> Point:
    """Foot of the perpendicular from (px,py) onto the line through origin with direction (dx,dy)."""
    t = (px * dx + py * dy) / (dx * dx + dy * dy)
    return (t * dx, t * dy)


def _triangles(w: float, h: float, orientation: int) -> List[Polyline]:
    if orientation % 2 == 0:
        f1 = _project_foot(w, 0.0, w, h)
        f2 = _project_foot(0.0, h, w, h)
        return [[(0.0, 0.0), (w, h)], [(w, 0.0), f1], [(0.0, h), f2]]
    # orientation 1 mirrored across x
    fx1, fy1 = _project_foot(w, 0.0, w, h)
    fx2, fy2 = _project_foot(0.0, h, w, h)
    return [[(w, 0.0), (0.0, h)], [(0.0, 0.0), (w - fx1, fy1)], [(w, h), (w - fx2, fy2)]]


def _ray_to_boundary(px: float, py: float, dx: float, dy: float, w: float, h: float) -> Point:
    """Endpoint where the ray from (px,py) along (dx,dy) first exits the rect."""
    t_best = math.inf
    if dx > 0:
        t_best = min(t_best, (w - px) / dx)
    elif dx < 0:
        t_best = min(t_best, -px / dx)
    if dy > 0:
        t_best = min(t_best, (h - py) / dy)
    elif dy < 0:
        t_best = min(t_best, -py / dy)
    return (px + dx * t_best, py + dy * t_best)


def _armature(w: float, h: float) -> List[Polyline]:
    # Bouleau's 14 lines: diagonals, corner reciprocals, corner-to-far-midpoint lines.
    lines: List[Polyline] = [
        [(0.0, 0.0), (w, h)],
        [(w, 0.0), (0.0, h)],
    ]
    reciprocals = [
        ((0.0, 0.0), (h, w)),
        ((w, 0.0), (-h, w)),
        ((w, h), (-h, -w)),
        ((0.0, h), (h, -w)),
    ]
    for (px, py), (dx, dy) in reciprocals:
        lines.append([(px, py), _ray_to_boundary(px, py, dx, dy, w, h)])
    mid_top, mid_bottom = (w / 2.0, 0.0), (w / 2.0, h)
    mid_left, mid_right = (0.0, h / 2.0), (w, h / 2.0)
    lines += [
        [(0.0, 0.0), mid_right],
        [(0.0, 0.0), mid_bottom],
        [(w, 0.0), mid_left],
        [(w, 0.0), mid_bottom],
        [(w, h), mid_left],
        [(w, h), mid_top],
        [(0.0, h), mid_right],
        [(0.0, h), mid_top],
    ]
    return lines


def _diagonal_method(w: float, h: float) -> List[Polyline]:
    m = min(w, h)
    return [
        [(0.0, 0.0), (m, m)],
        [(w, 0.0), (w - m, m)],
        [(w, h), (w - m, h - m)],
        [(0.0, h), (m, h - m)],
    ]


def _spiral_unit_points() -> List[Point]:
    """Fibonacci quarter-arc spiral inside the unit golden rectangle (1 x 1/phi), y-down."""
    pts: List[Point] = []
    x, y, w, h = 0.0, 0.0, 1.0, 1.0 / PHI
    angle = math.pi
    for i in range(_SPIRAL_ARCS):
        phase = i % 4
        if phase == 0:  # left
            s = h
            cx, cy = x + s, y + s
            x, w = x + s, w - s
        elif phase == 1:  # top
            s = w
            cx, cy = x, y + s
            y, h = y + s, h - s
        elif phase == 2:  # right
            s = h
            cx, cy = x + w - s, y
            w = w - s
        else:  # bottom
            s = w
            cx, cy = x + s, y + h - s
            h = h - s
        for j in range(_ARC_SAMPLES + 1):
            t = angle + (math.pi / 2.0) * (j / _ARC_SAMPLES)
            pts.append((cx + s * math.cos(t), cy + s * math.sin(t)))
        angle += math.pi / 2.0
    return pts


def _orient_unit(pts: List[Point], orientation: int) -> List[Point]:
    """Apply one of 8 symmetries (4 rotations x mirror) of the unit square."""
    rot = orientation % 4
    mirror = (orientation // 4) % 2
    out: List[Point] = []
    for px, py in pts:
        if mirror:
            px = 1.0 - px
        for _ in range(rot):
            px, py = py, 1.0 - px
        out.append((px, py))
    return out


def _spiral(w: float, h: float, orientation: int) -> List[Polyline]:
    unit = [(px, py * PHI) for px, py in _spiral_unit_points()]  # golden rect -> unit square
    oriented = _orient_unit(unit, orientation)
    return [[(px * w, py * h) for px, py in oriented]]


def guide_shapes(guide: CropGuide, w: float, h: float, orientation: int = 0) -> List[Polyline]:
    """Polylines for `guide` in pixel space (0,0)-(w,h); a 2-point polyline is a segment."""
    if w <= 0 or h <= 0 or guide == CropGuide.OFF:
        return []
    if guide == CropGuide.THIRDS:
        return _fraction_lines(w, h, [1.0 / 3.0, 2.0 / 3.0])
    if guide == CropGuide.PHI:
        return _fraction_lines(w, h, [1.0 / PHI**2, 1.0 / PHI])
    if guide == CropGuide.DIAGONALS:
        return _diagonals(w, h)
    if guide == CropGuide.TRIANGLES:
        return _triangles(w, h, orientation)
    if guide == CropGuide.SPIRAL:
        return _spiral(w, h, orientation)
    if guide == CropGuide.ARMATURE:
        return _armature(w, h)
    if guide == CropGuide.DIAGONAL_METHOD:
        return _diagonal_method(w, h)
    return _grid(w, h)
