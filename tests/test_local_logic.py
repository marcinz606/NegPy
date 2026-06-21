import unittest

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.local.logic import apply_local_adjustments, compute_local_factor_map
from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask


def _center_square_mask(strength: float, feather: float = 0.0) -> PolygonMask:
    """Polygon covering the central 50% of the frame."""
    return PolygonMask(
        vertices=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
        strength=strength,
        feather=feather,
    )


class TestApplyLocalAdjustments(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        self.shape = (100, 100)

    def test_noop_when_no_masks(self) -> None:
        """Empty config returns the input image."""
        res = apply_local_adjustments(self.img, LocalAdjustmentsConfig(), self.shape)
        np.testing.assert_array_equal(res, self.img)

    def test_dodge_brightens_inside_corners_unchanged(self) -> None:
        """Positive strength brightens the masked region; outside stays put."""
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        res = apply_local_adjustments(self.img, cfg, self.shape)
        self.assertGreater(float(res[50, 50, 0]), 0.5)
        self.assertAlmostEqual(float(res[5, 5, 0]), 0.5, places=5)

    def test_burn_darkens_inside(self) -> None:
        """Negative strength darkens the masked region (2^-1 -> 0.25)."""
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(-1.0),))
        res = apply_local_adjustments(self.img, cfg, self.shape)
        self.assertAlmostEqual(float(res[50, 50, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(res[5, 5, 0]), 0.5, places=5)

    def test_degenerate_mask_skipped(self) -> None:
        """A mask with fewer than 3 vertices is ignored."""
        cfg = LocalAdjustmentsConfig(masks=(PolygonMask(vertices=((0.4, 0.4), (0.6, 0.6)), strength=1.0),))
        res = apply_local_adjustments(self.img, cfg, self.shape)
        np.testing.assert_array_equal(res, self.img)

    def test_output_clipped_to_unit_range(self) -> None:
        """Large strength is clipped into [0, 1]."""
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(5.0),))
        res = apply_local_adjustments(self.img, cfg, self.shape)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_feathered_mask_runs(self) -> None:
        """Default feather path produces a valid in-range result."""
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0, feather=0.05),))
        res = apply_local_adjustments(self.img, cfg, self.shape)
        self.assertEqual(res.shape, self.img.shape)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)


class TestComputeFactorMap(unittest.TestCase):
    """The factor map is the shared CPU/GPU primitive — multiply the image by it."""

    def test_all_ones_when_empty(self) -> None:
        factor = compute_local_factor_map(LocalAdjustmentsConfig(), 100, 100, (100, 100))
        np.testing.assert_array_equal(factor, np.ones((100, 100), dtype=np.float32))

    def test_interior_equals_two_pow_strength(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        factor = compute_local_factor_map(cfg, 100, 100, (100, 100))
        self.assertAlmostEqual(float(factor[50, 50]), 2.0, places=5)
        self.assertAlmostEqual(float(factor[5, 5]), 1.0, places=5)

    def test_matches_apply_local_adjustments(self) -> None:
        """apply_local_adjustments must equal clip(img * factor_map)."""
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(-1.0, feather=0.04),))
        factor = compute_local_factor_map(cfg, 100, 100, (100, 100))
        expected = np.clip(img * factor[..., np.newaxis], 0.0, 1.0)
        got = apply_local_adjustments(img, cfg, (100, 100))
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)


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
