import unittest

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import smooth_polyline
from negpy.features.local.logic import compute_local_ev_map
from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask


def _center_square_mask(strength: float, feather: float = 0.0) -> PolygonMask:
    """Polygon covering the central 50% of the frame."""
    return PolygonMask(
        vertices=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
        strength=strength,
        feather=feather,
    )


class TestComputeEvMap(unittest.TestCase):
    """The EV map is the shared CPU/GPU primitive — per-pixel print-exposure stops."""

    def test_all_zeros_when_empty(self) -> None:
        ev = compute_local_ev_map(LocalAdjustmentsConfig(), 100, 100, (100, 100))
        np.testing.assert_array_equal(ev, np.zeros((100, 100), dtype=np.float32))

    def test_interior_equals_strength(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        ev = compute_local_ev_map(cfg, 100, 100, (100, 100))
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[5, 5]), 0.0, places=5)

    def test_burn_is_negative(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(-1.5),))
        ev = compute_local_ev_map(cfg, 100, 100, (100, 100))
        self.assertAlmostEqual(float(ev[50, 50]), -1.5, places=5)

    def test_overlapping_masks_are_additive(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(0.5), _center_square_mask(0.75)))
        ev = compute_local_ev_map(cfg, 100, 100, (100, 100))
        self.assertAlmostEqual(float(ev[50, 50]), 1.25, places=5)

    def test_degenerate_mask_skipped(self) -> None:
        """A mask with fewer than 3 vertices is ignored."""
        cfg = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=((0.4, 0.4), (0.6, 0.6)), strength=1.0),))
        ev = compute_local_ev_map(cfg, 100, 100, (100, 100))
        np.testing.assert_array_equal(ev, np.zeros((100, 100), dtype=np.float32))

    def test_feathered_mask_stays_within_strength(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0, feather=0.05),))
        ev = compute_local_ev_map(cfg, 100, 100, (100, 100))
        self.assertEqual(ev.shape, (100, 100))
        self.assertGreaterEqual(float(ev.min()), 0.0)
        self.assertLessEqual(float(ev.max()), 1.0 + 1e-5)
        self.assertGreater(float(ev[50, 50]), 0.9)


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
        # Smoothing is unconditional in compute_local_ev_map; the interior/exterior
        # invariants the pipeline relies on must survive it.
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        ev = compute_local_ev_map(cfg, 100, 100, (100, 100))
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[5, 5]), 0.0, places=5)


class TestLocalSerialization(unittest.TestCase):
    def test_roundtrip_preserves_masks(self) -> None:
        """to_dict -> from_flat_dict preserves polygon mask fields."""
        mask = PolygonMask(
            vertices=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
            strength=0.4,
            feather=0.03,
        )
        cfg = WorkspaceConfig(local=LocalAdjustmentsConfig(masks=(mask,)))

        restored = WorkspaceConfig.from_flat_dict(cfg.to_dict())

        self.assertEqual(len(restored.local.masks), 1)
        out = restored.local.masks[0]
        self.assertEqual(tuple(out.vertices), mask.vertices)
        self.assertAlmostEqual(out.strength, 0.4)
        self.assertAlmostEqual(out.feather, 0.03)


if __name__ == "__main__":
    unittest.main()
