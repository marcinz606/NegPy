import unittest
import numpy as np
from negpy.features.finish.logic import CARRIER_SAMPLES, apply_carrier, apply_vignette, carrier_profiles


class TestVignette(unittest.TestCase):
    def _gradient_image(self) -> np.ndarray:
        """100x100 mid-gray image for reliable vignette testing."""
        return np.full((100, 100, 3), 0.5, dtype=np.float32)

    def test_noop_when_stops_zero(self) -> None:
        """Zero stops returns image unchanged."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=0.0, size=0.5)
        np.testing.assert_array_equal(res, img)

    def test_output_shape_and_range(self) -> None:
        """Output keeps same shape and stays in [0, 1]."""
        img = self._gradient_image()
        for stops in [-1.0, 1.0, -2.0, 2.0]:
            for size in [0.0, 0.5, 1.0]:
                res = apply_vignette(img, stops, size)
                self.assertEqual(res.shape, img.shape)
                self.assertGreaterEqual(float(res.min()), 0.0)
                self.assertLessEqual(float(res.max()), 1.0)

    def test_burn_darkens_corners(self) -> None:
        """Positive stops (burn) darkens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=1.0, size=0.5)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertLess(corner_luma, center_luma)

    def test_dodge_brightens_corners(self) -> None:
        """Negative stops (hold back) brightens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=-1.0, size=0.5)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertGreater(corner_luma, center_luma)

    def test_burn_is_exposure_exact(self) -> None:
        """A fully-covered corner at +1 stop halves the linear value."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=1.0, size=0.5)
        np.testing.assert_allclose(res[0, 0], img[0, 0] * 0.5, atol=1e-5)

    def test_center_unaffected(self) -> None:
        """Center pixel should be unchanged regardless of stops."""
        img = self._gradient_image()
        for stops in [-2.0, -1.0, 1.0, 2.0]:
            res = apply_vignette(img, stops, size=0.5)
            np.testing.assert_array_almost_equal(res[50, 50], img[50, 50], decimal=5)

    def test_small_size_localizes_to_corners(self) -> None:
        """Small size keeps the center untouched while still burning corners."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=2.0, size=0.1)
        center_luma = float(res[50, 50].mean())
        self.assertAlmostEqual(center_luma, 0.5, delta=0.01)
        corner_luma = float(res[0, 0].mean())
        self.assertLess(corner_luma, center_luma)

    def test_size_one_affects_entire_image(self) -> None:
        """Size=1 means the burn covers the entire image — center is affected too."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=2.0, size=1.0)
        center_luma = float(res[50, 50].mean())
        self.assertLess(center_luma, 0.5)

    def test_non_square_image(self) -> None:
        """Works correctly on non-square images."""
        img = np.full((50, 200, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5)
        self.assertEqual(res.shape, img.shape)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_circular_falloff_invariant(self) -> None:
        """Pixels equidistant from center receive identical vignette weight."""
        h, w = 100, 200
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5)

        c00 = res[0, 0]
        c0w = res[0, w - 1]
        ch0 = res[h - 1, 0]
        chw = res[h - 1, w - 1]
        np.testing.assert_allclose(c00, c0w, atol=1e-5)
        np.testing.assert_allclose(c00, ch0, atol=1e-5)
        np.testing.assert_allclose(c00, chw, atol=1e-5)

        mid_x = w // 2
        mid_y = h // 2
        np.testing.assert_allclose(res[0, mid_x], res[h - 1, mid_x], atol=1e-5)
        np.testing.assert_allclose(res[mid_y, 0], res[mid_y, w - 1], atol=1e-5)

    def test_radial_edges_weaker_than_corners(self) -> None:
        """Roundness 0: edge midpoints sit inside the radial falloff, so they
        burn less than corners."""
        h, w = 100, 100
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5, roundness=0.0)
        self.assertGreater(float(res[0, w // 2].mean()), float(res[0, 0].mean()))

    def test_rectangular_edges_match_corners(self) -> None:
        """Roundness 1: the burn follows the frame — edge midpoints and corners
        receive the same weight (card burn along each edge)."""
        h, w = 100, 200
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5, roundness=1.0)
        np.testing.assert_allclose(res[0, w // 2], res[0, 0], atol=1e-5)
        np.testing.assert_allclose(res[h // 2, 0], res[0, 0], atol=1e-5)


class TestCarrier(unittest.TestCase):
    def _image(self) -> np.ndarray:
        return np.full((100, 150, 3), 0.5, dtype=np.float32)

    def test_noop_when_width_zero(self) -> None:
        img = self._image()
        res = apply_carrier(img, width_px=0.0, rough=0.5)
        np.testing.assert_array_equal(res, img)

    def test_edges_black_interior_untouched(self) -> None:
        img = self._image()
        res = apply_carrier(img, width_px=5.0, rough=0.5)
        self.assertEqual(float(res[0, 75].max()), 0.0)
        self.assertEqual(float(res[99, 75].max()), 0.0)
        self.assertEqual(float(res[50, 0].max()), 0.0)
        self.assertEqual(float(res[50, 149].max()), 0.0)
        np.testing.assert_array_equal(res[50, 75], img[50, 75])

    def test_deterministic(self) -> None:
        img = self._image()
        a = apply_carrier(img, width_px=5.0, rough=1.0)
        b = apply_carrier(img, width_px=5.0, rough=1.0)
        np.testing.assert_array_equal(a, b)

    def test_rough_jitters_edge(self) -> None:
        """With roughness the rebate width varies along an edge; without, it doesn't."""
        img = self._image()
        smooth = apply_carrier(img, width_px=6.0, rough=0.0)
        rough = apply_carrier(img, width_px=6.0, rough=1.0)
        row_smooth = smooth[:, :, 0].sum(axis=0)[20:130]
        row_rough = rough[:, :, 0].sum(axis=0)[20:130]
        self.assertLess(float(np.std(row_smooth)), 1e-4)
        self.assertGreater(float(np.std(row_rough)), 1e-3)

    def test_profiles_shape_and_range(self) -> None:
        p = carrier_profiles()
        self.assertEqual(p.shape, (4, CARRIER_SAMPLES))
        self.assertLessEqual(float(np.abs(p).max()), 1.0)
        self.assertEqual(p.dtype, np.float32)

    def test_soft_penumbra(self) -> None:
        """The rebate-to-image transition spans multiple pixels, not one."""
        img = np.full((200, 300, 3), 1.0, dtype=np.float32)
        res = apply_carrier(img, width_px=20.0, rough=0.0)
        cut = res[:, 150, 0]
        partial = np.sum((cut > 0.05) & (cut < 0.95))
        self.assertGreaterEqual(int(partial), 4)

    def test_small_image_does_not_crash(self) -> None:
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        res = apply_carrier(img, width_px=10.0, rough=1.0)
        self.assertEqual(res.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
