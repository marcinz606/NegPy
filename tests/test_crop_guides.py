import math

import pytest

from negpy.desktop.view.canvas.crop_guides import (
    GUIDE_LABELS,
    ORIENTATION_COUNT,
    PHI,
    CropGuide,
    guide_shapes,
)

SIZES = [(600.0, 400.0), (500.0, 500.0), (650.0, 240.0), (240.0, 650.0)]


def _points(shapes):
    return [p for poly in shapes for p in poly]


@pytest.mark.parametrize("guide", list(CropGuide))
@pytest.mark.parametrize("w,h", SIZES)
def test_all_points_inside_rect(guide, w, h):
    for orientation in range(8):
        for x, y in _points(guide_shapes(guide, w, h, orientation)):
            assert -1e-6 <= x <= w + 1e-6
            assert -1e-6 <= y <= h + 1e-6


def test_off_and_degenerate_empty():
    assert guide_shapes(CropGuide.OFF, 600, 400) == []
    assert guide_shapes(CropGuide.THIRDS, 0, 400) == []
    assert guide_shapes(CropGuide.THIRDS, 600, 0) == []


def test_thirds_fractions():
    shapes = guide_shapes(CropGuide.THIRDS, 300, 300)
    xs = sorted({p[0] for poly in shapes[:2] for p in poly})
    assert xs == [100.0, 200.0]
    assert len(shapes) == 4


def test_phi_fractions():
    shapes = guide_shapes(CropGuide.PHI, 1000, 1000)
    xs = sorted({poly[0][0] for poly in shapes[:2]})
    assert xs[0] == pytest.approx(1000 / PHI**2, abs=1e-6)
    assert xs[1] == pytest.approx(1000 / PHI, abs=1e-6)
    assert xs[0] == pytest.approx(381.966, abs=1e-3)


def test_diagonals_endpoints():
    w, h = 600.0, 400.0
    shapes = guide_shapes(CropGuide.DIAGONALS, w, h)
    assert [(0, 0), (w, h)] in [[tuple(p) for p in poly] for poly in shapes]
    assert [(w, 0), (0, h)] in [[tuple(p) for p in poly] for poly in shapes]
    assert [(w / 2, 0), (w / 2, h)] in [[tuple(p) for p in poly] for poly in shapes]
    assert [(0, h / 2), (w, h / 2)] in [[tuple(p) for p in poly] for poly in shapes]


@pytest.mark.parametrize("w,h", SIZES)
@pytest.mark.parametrize("orientation", [0, 1])
def test_triangles_perpendicular(w, h, orientation):
    diag, seg1, seg2 = guide_shapes(CropGuide.TRIANGLES, w, h, orientation)
    dx, dy = diag[1][0] - diag[0][0], diag[1][1] - diag[0][1]
    for seg in (seg1, seg2):
        sx, sy = seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]
        assert abs(sx * dx + sy * dy) < 1e-6 * (w * w + h * h)
        fx, fy = seg[1][0] - diag[0][0], seg[1][1] - diag[0][1]
        assert abs(fx * dy - fy * dx) < 1e-6 * (w * w + h * h)


def test_triangles_orientation_period_two():
    a = guide_shapes(CropGuide.TRIANGLES, 600, 400, 0)
    b = guide_shapes(CropGuide.TRIANGLES, 600, 400, 2)
    c = guide_shapes(CropGuide.TRIANGLES, 600, 400, 1)
    assert a == b
    assert a != c


@pytest.mark.parametrize("w,h", SIZES)
def test_armature_fourteen_lines(w, h):
    shapes = guide_shapes(CropGuide.ARMATURE, w, h)
    assert len(shapes) == 14
    diag_tlbr = (w, h)
    diag_trbl = (-w, h)
    # Reciprocals (shapes 2..5) are perpendicular to the diagonal not through their corner.
    for poly, diag in zip(shapes[2:6], [diag_trbl, diag_tlbr, diag_trbl, diag_tlbr]):
        dx, dy = poly[1][0] - poly[0][0], poly[1][1] - poly[0][1]
        assert abs(dx * diag[0] + dy * diag[1]) < 1e-6 * (w * w + h * h)


def test_diagonal_method_45_degrees():
    for w, h in SIZES:
        shapes = guide_shapes(CropGuide.DIAGONAL_METHOD, w, h)
        assert len(shapes) == 4
        for (x1, y1), (x2, y2) in shapes:
            assert abs(abs(x2 - x1) - abs(y2 - y1)) < 1e-9
            assert abs(x2 - x1) == pytest.approx(min(w, h))


def test_grid_cells_square():
    shapes = guide_shapes(CropGuide.GRID, 640.0, 480.0)
    xs = sorted({poly[0][0] for poly in shapes if poly[0][0] == poly[1][0]})
    ys = sorted({poly[0][1] for poly in shapes if poly[0][1] == poly[1][1]})
    step = 480.0 / 8
    assert all(b - a == pytest.approx(step) for a, b in zip(xs, xs[1:]))
    assert all(b - a == pytest.approx(step) for a, b in zip(ys, ys[1:]))


def test_spiral_single_continuous_polyline():
    (poly,) = guide_shapes(CropGuide.SPIRAL, 600, 400)
    assert len(poly) > 100
    diag = math.hypot(600, 400)
    for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
        assert math.hypot(x2 - x1, y2 - y1) < diag * 0.2


def test_spiral_orientations_distinct_period_eight():
    variants = [tuple(guide_shapes(CropGuide.SPIRAL, 600, 400, o)[0]) for o in range(8)]
    assert len(set(variants)) == 8
    assert guide_shapes(CropGuide.SPIRAL, 600, 400, 8) == guide_shapes(CropGuide.SPIRAL, 600, 400, 0)


def test_orientation_invariant_guides():
    for guide in (CropGuide.THIRDS, CropGuide.PHI, CropGuide.DIAGONALS, CropGuide.ARMATURE, CropGuide.DIAGONAL_METHOD, CropGuide.GRID):
        base = guide_shapes(guide, 600, 400, 0)
        for o in range(1, 8):
            assert guide_shapes(guide, 600, 400, o) == base


def test_labels_cover_all_guides():
    assert set(GUIDE_LABELS) == set(CropGuide)
    for guide, count in ORIENTATION_COUNT.items():
        assert count > 1
        assert guide in GUIDE_LABELS
