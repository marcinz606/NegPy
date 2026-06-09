import unittest
import numpy as np
from negpy.features.finish.logic import apply_vignette


class TestVignette(unittest.TestCase):
    def _gradient_image(self) -> np.ndarray:
        """100x100 mid-gray image for reliable vignette testing."""
        return np.full((100, 100, 3), 0.5, dtype=np.float32)

    def test_noop_when_strength_zero(self) -> None:
        """Strength 0.0 returns image unchanged."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=0.0, size=0.5)
        np.testing.assert_array_equal(res, img)

    def test_output_shape_and_range(self) -> None:
        """Output keeps same shape and stays in [0, 1]."""
        img = self._gradient_image()
        for strength in [-0.5, 0.5, -1.0, 1.0]:
            for size in [0.0, 0.5, 1.0]:
                res = apply_vignette(img, strength, size)
                self.assertEqual(res.shape, img.shape)
                self.assertGreaterEqual(float(res.min()), 0.0)
                self.assertLessEqual(float(res.max()), 1.0)

    def test_darken_corners_darker_than_center(self) -> None:
        """Negative strength darkens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=-1.0, size=0.5)
        # Corner pixel (0,0) should be darker than center (50,50)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertLess(corner_luma, center_luma)

    def test_brighten_corners_brighter_than_center(self) -> None:
        """Positive strength brightens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=1.0, size=0.5)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertGreater(corner_luma, center_luma)

    def test_center_unaffected(self) -> None:
        """Center pixel should be unchanged regardless of strength."""
        img = self._gradient_image()
        for strength in [-1.0, -0.5, 0.5, 1.0]:
            res = apply_vignette(img, strength, size=0.5)
            np.testing.assert_array_almost_equal(res[50, 50], img[50, 50], decimal=5)

    def test_small_size_localizes_to_corners(self) -> None:
        """Small size keeps the center untouched while still darkening corners."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=-1.0, size=0.1)
        center_luma = float(res[50, 50].mean())
        self.assertAlmostEqual(center_luma, 0.5, delta=0.01)
        corner_luma = float(res[0, 0].mean())
        self.assertLess(corner_luma, center_luma)

    def test_size_one_affects_entire_image(self) -> None:
        """Size=1 means vignette covers entire image — center is affected too."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=-1.0, size=1.0)
        # Center should be darkened too
        center_luma = float(res[50, 50].mean())
        self.assertLess(center_luma, 0.5)

    def test_non_square_image(self) -> None:
        """Works correctly on non-square images."""
        img = np.full((50, 200, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, strength=-1.0, size=0.5)
        self.assertEqual(res.shape, img.shape)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_circular_falloff_invariant(self) -> None:
        """Pixels equidistant from center receive identical vignette weight."""
        h, w = 100, 200
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, strength=-1.0, size=0.5)

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


if __name__ == "__main__":
    unittest.main()
