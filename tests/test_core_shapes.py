import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(15)
_POINT_SETS = [_RNG.integers(-50, 500, (n, 2)).astype(np.int32) for n in (1, 2, 3, 4, 7, 12, 39)] + [
    np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], np.int32),
    np.array([[0, 0], [5, 0], [10, 0]], np.int32),
    np.array([[3, 3], [3, 3], [3, 3]], np.int32),
]


@pytest.mark.parametrize("index", range(len(_POINT_SETS)))
def test_hull_rect_and_box_match_cv2(index):
    pts = _POINT_SETS[index]
    assert np.array_equal(np.array(negpy_core.convex_hull(pts), np.int32).reshape(-1, 2), cv2.convexHull(pts).reshape(-1, 2))
    want = cv2.minAreaRect(pts)
    got = negpy_core.min_area_rect(pts)
    assert np.array_equal(np.float32([*want[0], *want[1], want[2]]), np.float32([*got[0], *got[1], got[2]]))
    assert np.array_equal(np.array(negpy_core.box_points(want), np.float32), cv2.boxPoints(want))


@pytest.mark.parametrize("index", range(len(_POINT_SETS)))
@pytest.mark.parametrize("shape", [(400, 500), (60, 70)])
def test_fill_convex_poly_matches_cv2(index, shape):
    quad = np.rint(cv2.boxPoints(cv2.minAreaRect(_POINT_SETS[index]))).astype(np.int32)
    want = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(want, quad, 1)
    assert np.array_equal(negpy_core.fill_convex_poly(shape, quad), want)
