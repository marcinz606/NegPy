import unittest
import numpy as np
import cv2
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    apply_clahe,
    apply_glow_and_halation,
    apply_output_sharpening,
    apply_saturation,
    apply_spectral_crosstalk,
    apply_vibrance,
)


class TestLabLogic(unittest.TestCase):
    def test_spectral_crosstalk(self) -> None:
        """Matrix should mix channels."""
        img = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)
        # Identity matrix
        matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        res = apply_spectral_crosstalk(img, 1.0, matrix)
        assert np.allclose(res, img)

        # Swap R and G
        matrix_swap = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        res_swap = apply_spectral_crosstalk(img, 1.0, matrix_swap)
        assert np.allclose(res_swap[0, 0], [0.5, 1.0, 0.0])

    def test_clahe(self) -> None:
        """CLAHE should modify image."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        res = apply_clahe(img, 1.0)
        assert res.shape == img.shape
        # Should be different
        assert not np.allclose(res, img)

    def test_output_sharpening(self) -> None:
        """Sharpening should increase local variance."""
        # Create a simple square
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[25:75, 25:75, :] = 0.5

        res = apply_output_sharpening(img, amount=1.0, scale_factor=1.0)

        # Sharpening should increase variance on edges
        self.assertGreater(np.var(res), np.var(img))

    def test_saturation(self) -> None:
        """Saturation should modify color intensity."""
        # Pure Red (H=0, S=1, V=1)
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = 1.0

        # Reduce saturation to 0 (Greyscale)
        desat = apply_saturation(img, 0.0)

        # Desaturated pure primary should result in high value (white in this context for HSV)
        self.assertAlmostEqual(desat[0, 0, 0], 1.0)
        self.assertAlmostEqual(desat[0, 0, 1], 1.0)
        self.assertAlmostEqual(desat[0, 0, 2], 1.0)

        # Increase saturation of a pale color
        # Pale Red: R=1.0, G=0.5, B=0.5
        img2 = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img2[:, :, 0] = 1.0

        sat = apply_saturation(img2, 2.0)
        # S should become 1.0 -> Pure Red
        self.assertAlmostEqual(sat[0, 0, 0], 1.0, delta=1e-5)
        self.assertAlmostEqual(sat[0, 0, 1], 0.0, delta=1e-5)
        self.assertAlmostEqual(sat[0, 0, 2], 0.0, delta=1e-5)

    def test_vibrance(self) -> None:
        """Vibrance should increase saturation of pale colors more than vibrant ones."""
        # Pale color
        img_pale = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img_pale[:, :, 0] = 0.6

        # Vibrant color
        img_vibrant = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img_vibrant[:, :, 0] = 1.0

        res_pale = apply_vibrance(img_pale, 1.5)
        res_vibrant = apply_vibrance(img_vibrant, 1.5)

        # Calculate saturation increase
        def get_sat(rgb):
            c = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            return np.mean(c[:, :, 1])

        sat_gain_pale = get_sat(res_pale) - get_sat(img_pale)
        sat_gain_vibrant = get_sat(res_vibrant) - get_sat(img_vibrant)

        self.assertGreater(sat_gain_pale, sat_gain_vibrant)

    def test_chroma_denoise(self) -> None:
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] += np.random.normal(0, 5, (100, 100))
        img_noisy = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        res = apply_chroma_denoise(img_noisy, radius=2.0)
        res_lab = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        np.testing.assert_array_almost_equal(lab[:, :, 0], res_lab[:, :, 0], decimal=0)
        self.assertLess(float(np.var(res_lab[:, :, 1])), float(np.var(lab[:, :, 1])))


class TestGlowAndHalation(unittest.TestCase):
    def _highlight_image(self) -> np.ndarray:
        """100x100 image with a bright white spot in the centre on a dark background."""
        img = np.full((100, 100, 3), 0.1, dtype=np.float32)
        img[40:60, 40:60, :] = 1.0
        return img

    def test_noop_when_both_zero(self) -> None:
        """No change when both amounts are 0.0."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=0.0)
        np.testing.assert_array_equal(res, img)

    def test_output_shape_and_range(self) -> None:
        """Output keeps the same shape and stays in [0, 1]."""
        img = self._highlight_image()
        for glow, hal in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            res = apply_glow_and_halation(img, glow, hal)
            self.assertEqual(res.shape, img.shape)
            self.assertGreaterEqual(float(res.min()), 0.0)
            self.assertLessEqual(float(res.max()), 1.0)

    def test_glow_brightens_dark_area_near_highlight(self) -> None:
        """Glow should increase brightness in the dark area neighbouring the highlight."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0)
        # Dark border just outside the bright spot should be brighter after glow
        dark_before = float(img[35, 35, 0])
        dark_after = float(res[35, 35, 0])
        self.assertGreater(dark_after, dark_before)

    def test_glow_all_channels_equally(self) -> None:
        """Glow bloom should be approximately equal across R, G, B channels."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0)
        # Check a dark pixel near the highlight
        delta = res[30, 50] - img[30, 50]
        # All three channels should have gained roughly the same amount
        self.assertAlmostEqual(float(delta[0]), float(delta[1]), delta=0.05)
        self.assertAlmostEqual(float(delta[1]), float(delta[2]), delta=0.05)

    def test_halation_red_dominant(self) -> None:
        """Halation scatter should add more red than blue to dark pixels near highlights."""
        img = self._highlight_image()
        res = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=1.0)
        delta = res[30, 50] - img[30, 50]
        self.assertGreater(float(delta[0]), float(delta[2]))

    def test_scale_factor_affects_spread(self) -> None:
        """A larger scale factor should spread the bloom further from the highlight."""
        img = self._highlight_image()
        res_small = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0, scale_factor=0.5)
        res_large = apply_glow_and_halation(img, glow_amount=1.0, halation_strength=0.0, scale_factor=2.0)
        # scale=0.5 → kernel radius ~7px; scale=2.0 → kernel radius ~30px.
        # Pixel at row 28 is ~12px above the highlight edge (row 40), so it should
        # receive bloom with scale=2.0 but not with scale=0.5.
        far_small = float(res_small[28, 50, 0])
        far_large = float(res_large[28, 50, 0])
        self.assertGreater(far_large, far_small)

    def test_combined_brighter_than_individual(self) -> None:
        """Applying both glow and halation should be at least as bright as either alone."""
        img = self._highlight_image()
        res_glow = apply_glow_and_halation(img, glow_amount=0.5, halation_strength=0.0)
        res_hal = apply_glow_and_halation(img, glow_amount=0.0, halation_strength=0.5)
        res_both = apply_glow_and_halation(img, glow_amount=0.5, halation_strength=0.5)
        self.assertGreaterEqual(float(res_both[30, 50, 0]), float(res_glow[30, 50, 0]))
        self.assertGreaterEqual(float(res_both[30, 50, 0]), float(res_hal[30, 50, 0]))


if __name__ == "__main__":
    unittest.main()
