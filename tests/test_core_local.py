"""fill_poly and stroke_cover against cv2 on random shapes, and a hash pin of the local-mask rasteriser."""

import cv2
import numpy as np
from core_oracle import cv2_oracle
from core_pin import digest

import negpy_core
from negpy.features.local.logic import rasterise
from negpy.features.local.models import MaskShape

pytestmark = cv2_oracle


def test_fill_poly_matches_cv2_on_random_polygons():
    rng = np.random.default_rng(0)
    for _ in range(500):
        h, w = (int(v) for v in rng.integers(5, 90, 2))
        n = int(rng.integers(3, 30))
        pts = np.stack([rng.integers(-20, w + 20, n), rng.integers(-20, h + 20, n)], 1).astype(np.int32)
        want = np.zeros((h, w), np.uint8)
        cv2.fillPoly(want, [pts], 255)
        np.testing.assert_array_equal(negpy_core.fill_poly((h, w), pts), want)


def test_rasterise_pin():
    cases = [
        (MaskShape.POLYGON, ((0.2, 0.2), (0.5, 0.15), (0.6, 0.45), (0.25, 0.5)), 6.0, False),
        (MaskShape.OVAL, ((0.6, 0.6), (0.85, 0.6), (0.6, 0.8)), 3.5, True),
        (MaskShape.GRADIENT, ((0.5, 0.0), (0.4, 0.7)), 0.0, False),
        (MaskShape.POLYGON, ((0.1, 0.9), (0.9, 0.1), (0.9, 0.9), (0.1, 0.1)), 0.0, False),
    ]
    got = [rasterise(shape, pts, 150, 210, sigma, invert) for shape, pts, sigma, invert in cases]
    assert digest(got) == "51449962febf32a1347b6bd79fb6eb013700cfdd5f7b29d25ed89264887b51f9"


def test_stroke_cover_matches_cv2_on_random_strokes():
    rng = np.random.default_rng(1)
    for _ in range(800):
        h, w = (int(v) for v in rng.integers(4, 90, 2))
        n = int(rng.integers(1, 7))
        pts = np.stack([rng.integers(-30, w + 30, n), rng.integers(-30, h + 30, n)], 1).astype(np.int32)
        radius, thickness = int(rng.integers(0, 25)), int(rng.integers(1, 50))
        want = np.zeros((h, w), np.uint8)
        if n > 1:
            cv2.polylines(want, [pts], False, 1, thickness=thickness)
        for cx, cy in pts:
            cv2.circle(want, (int(cx), int(cy)), radius, 1, -1)
        np.testing.assert_array_equal(negpy_core.stroke_cover((h, w), [(pts, radius, thickness)]), want)
