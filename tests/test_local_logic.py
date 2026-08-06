import unittest

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import smooth_polyline
from negpy.features.local.logic import compute_local_maps
from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask


def _center_square_mask(stops: float, feather: float = 0.0) -> PolygonMask:
    """Polygon covering the central 50% of the frame."""
    return PolygonMask(
        vertices=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
        stops=stops,
        feather=feather,
    )


def _ev(cfg: LocalAdjustmentsConfig, h: int = 100, w: int = 100) -> np.ndarray:
    return compute_local_maps(cfg, h, w, (h, w))[:, :, 0]


def _grades(cfg: LocalAdjustmentsConfig, h: int = 100, w: int = 100) -> np.ndarray:
    return compute_local_maps(cfg, h, w, (h, w))[:, :, 1]


class TestComputeEvMap(unittest.TestCase):
    """Plane 0 is the shared CPU/GPU primitive — per-pixel print exposure in stops,
    positive = burn."""

    def test_all_zeros_when_empty(self) -> None:
        ev = _ev(LocalAdjustmentsConfig())
        np.testing.assert_array_equal(ev, np.zeros((100, 100), dtype=np.float32))

    def test_interior_equals_the_masks_stops(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[5, 5]), 0.0, places=5)

    def test_a_dodge_is_negative(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(-1.5),))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), -1.5, places=5)

    def test_overlapping_masks_are_additive(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(0.5), _center_square_mask(0.75)))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), 1.25, places=5)

    def test_degenerate_mask_skipped(self) -> None:
        """A mask with fewer than 3 vertices is ignored."""
        cfg = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=((0.4, 0.4), (0.6, 0.6)), stops=-1.0),))
        ev = _ev(cfg)
        np.testing.assert_array_equal(ev, np.zeros((100, 100), dtype=np.float32))

    def test_feathered_mask_stays_within_its_stops(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0, feather=0.05),))
        ev = _ev(cfg)
        self.assertEqual(ev.shape, (100, 100))
        self.assertGreaterEqual(float(ev.min()), 0.0)
        self.assertLessEqual(float(ev.max()), 1.0 + 1e-5)
        self.assertGreater(float(ev[50, 50]), 0.9)


class TestGradePlane(unittest.TestCase):
    """Plane 1 carries the local grade in ISO-R points — same rasterisation as the EV plane."""

    def test_zero_without_local_grade(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        np.testing.assert_array_equal(_grades(cfg), np.zeros((100, 100), dtype=np.float32))

    def test_interior_equals_delta_and_exterior_is_clean(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=_center_square_mask(0.0).vertices, stops=0.0, grade=-30.0),))
        grades = _grades(cfg)
        self.assertAlmostEqual(float(grades[50, 50]), -30.0, places=4)
        self.assertAlmostEqual(float(grades[5, 5]), 0.0, places=5)

    def test_overlapping_masks_are_additive(self) -> None:
        mask = _center_square_mask(0.0)
        cfg = LocalAdjustmentsConfig(
            masks=(
                PolygonMask(vertices=mask.vertices, stops=0.0, grade=-10.0),
                PolygonMask(vertices=mask.vertices, stops=0.0, grade=-15.0),
            )
        )
        self.assertAlmostEqual(float(_grades(cfg)[50, 50]), -25.0, places=4)

    def test_exposure_and_grade_ride_the_same_alpha(self) -> None:
        """One mask, both values: the feathered edge must weight them identically."""
        cfg = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=_center_square_mask(0.0).vertices, stops=1.0, feather=0.05, grade=-20.0),))
        maps = compute_local_maps(cfg, 100, 100, (100, 100))
        alpha_ev = maps[:, :, 0] / 1.0
        alpha_grade = maps[:, :, 1] / -20.0
        np.testing.assert_allclose(alpha_ev, alpha_grade, atol=1e-6)


class TestSmoothPolyline(unittest.TestCase):
    """Mask outlines and heal paths are always drawn as a Catmull-Rom curve."""

    def test_short_input_returned_unchanged(self) -> None:
        self.assertEqual(smooth_polyline([(0.0, 0.0), (1.0, 1.0)]), [(0.0, 0.0), (1.0, 1.0)])

    def test_closed_interpolates_control_points(self) -> None:
        sq = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
        out = smooth_polyline(sq, closed=True, samples_per_seg=8)
        self.assertEqual(len(out), 4 * 8)  # denser than the 4 control points
        for i, p in enumerate(sq):  # each control point is the t=0 sample of its segment
            self.assertAlmostEqual(out[i * 8][0], p[0])
            self.assertAlmostEqual(out[i * 8][1], p[1])

    def test_open_keeps_endpoints(self) -> None:
        line = [(0.1, 0.1), (0.5, 0.2), (0.9, 0.1)]
        out = smooth_polyline(line, closed=False, samples_per_seg=8)
        self.assertEqual(out[0], (0.1, 0.1))
        self.assertEqual(out[-1], (0.9, 0.1))
        self.assertGreater(len(out), len(line))

    def test_smoothed_square_mask_still_fills_interior(self) -> None:
        # Smoothing is unconditional in compute_local_maps; the interior/exterior
        # invariants the pipeline relies on must survive it.
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[5, 5]), 0.0, places=5)


class TestLocalSerialization(unittest.TestCase):
    def test_roundtrip_preserves_masks(self) -> None:
        """to_dict -> from_flat_dict preserves polygon mask fields."""
        mask = PolygonMask(
            vertices=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
            stops=-0.4,
            feather=0.03,
            grade=-15.0,
        )
        cfg = WorkspaceConfig(local=LocalAdjustmentsConfig(masks=(mask,)))

        restored = WorkspaceConfig.from_flat_dict(cfg.to_dict())

        self.assertEqual(len(restored.local.masks), 1)
        out = restored.local.masks[0]
        self.assertEqual(tuple(out.vertices), mask.vertices)
        self.assertAlmostEqual(out.stops, -0.4)
        self.assertAlmostEqual(out.feather, 0.03)
        self.assertAlmostEqual(out.grade, -15.0)

    def test_legacy_mask_migrates_brightness_signed_strength_to_stops(self) -> None:
        """Pre-flip saves stored `strength` (positive = dodge); the same mask must come
        back as exposure-signed stops, at the frame's grade."""
        legacy = {"local_masks": {"masks": [{"vertices": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "strength": 0.4, "feather": 0.03}]}}
        mask = WorkspaceConfig.from_flat_dict(legacy).local.masks[0]
        self.assertAlmostEqual(mask.stops, -0.4)
        self.assertEqual(mask.grade, 0.0)

    def test_a_legacy_burn_migrates_to_a_positive_burn(self) -> None:
        legacy = {"local_masks": {"masks": [{"vertices": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "strength": -1.0}]}}
        self.assertAlmostEqual(WorkspaceConfig.from_flat_dict(legacy).local.masks[0].stops, 1.0)


if __name__ == "__main__":
    unittest.main()
