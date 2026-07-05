import unittest
import cv2
import numpy as np
from negpy.features.toning.logic import (
    apply_chemical_toning,
    apply_split_toning,
)


class TestChemicalToning(unittest.TestCase):
    """Density-driven chemical toners on the linear print: selenium converts the
    densest silver first (Dmax boost, cool shadows); sepia bleach-redevelop
    converts the thinnest silver first (warm highlights, shadows hold)."""

    @staticmethod
    def _gray(t: float) -> np.ndarray:
        return np.full((4, 4, 3), t, dtype=np.float32)

    @staticmethod
    def _density(res: np.ndarray, ch: int) -> float:
        return float(-np.log10(max(float(res[0, 0, ch]), 1e-6)))

    def test_zero_strength_is_identity(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        res = apply_chemical_toning(img, selenium_strength=0.0, sepia_strength=0.0)
        np.testing.assert_array_equal(res, img)

    def test_selenium_deepens_shadows(self):
        """Selenium adds density where silver is dense — blacks get deeper."""
        dark = self._gray(0.05)  # D ~ 1.3
        res = apply_chemical_toning(dark, selenium_strength=1.0, sepia_strength=0.0)
        self.assertLess(float(res.mean()), 0.05)

    def test_selenium_converts_densest_first(self):
        """Density gain grows with input density; highlights barely move."""
        d_dark_in, d_light_in = -np.log10(0.05), -np.log10(0.9)
        res_dark = apply_chemical_toning(self._gray(0.05), selenium_strength=1.0, sepia_strength=0.0)
        res_light = apply_chemical_toning(self._gray(0.9), selenium_strength=1.0, sepia_strength=0.0)
        gain_dark = self._density(res_dark, 1) - d_dark_in
        gain_light = self._density(res_light, 1) - d_light_in
        self.assertGreater(gain_dark, gain_light * 10)
        self.assertAlmostEqual(gain_light, 0.0, places=3)

    def test_selenium_cools_shadows(self):
        """Green gains the most density -> magenta/eggplant cast in the shadows."""
        res = apply_chemical_toning(self._gray(0.05), selenium_strength=1.0, sepia_strength=0.0)
        self.assertLess(float(res[0, 0, 1]), float(res[0, 0, 0]))  # G darker than R
        self.assertLess(float(res[0, 0, 1]), float(res[0, 0, 2]))  # G darker than B

    def test_sepia_warms_highlights(self):
        """Converted silver -> warm sulfide dye: red lifts, blue drops."""
        light = self._gray(0.6)
        res = apply_chemical_toning(light, selenium_strength=0.0, sepia_strength=1.0)
        self.assertGreater(float(res[0, 0, 0]), 0.6)  # R lighter (warm)
        self.assertLess(float(res[0, 0, 2]), 0.6)  # B denser

    def test_sepia_converts_thinnest_first(self):
        """Bleach eats the thinnest silver first — highlights tone, shadows hold
        (the classic split-sepia look at partial strength)."""
        res_light = apply_chemical_toning(self._gray(0.6), selenium_strength=0.0, sepia_strength=1.0)
        res_dark = apply_chemical_toning(self._gray(0.01), selenium_strength=0.0, sepia_strength=1.0)
        warmth_light = float(res_light[0, 0, 0] - res_light[0, 0, 2])
        warmth_dark = float(res_dark[0, 0, 0] - res_dark[0, 0, 2])
        self.assertGreater(warmth_light, 0.01)
        self.assertAlmostEqual(warmth_dark, 0.0, places=3)

    def test_paper_white_stays_white(self):
        """No silver at paper white — nothing to tone."""
        white = self._gray(1.0)
        res = apply_chemical_toning(white, selenium_strength=1.0, sepia_strength=1.0)
        np.testing.assert_allclose(res, white, atol=1e-3)

    def test_output_range_combined(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        res = apply_chemical_toning(img, selenium_strength=1.0, sepia_strength=1.0)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_slider_max_saturates_conversion(self):
        """Sliders go to 2.0 — conversion caps at all-silver-toned, output stays
        sane and monotone with strength."""
        dark = self._gray(0.05)
        res_1 = apply_chemical_toning(dark, selenium_strength=1.0, sepia_strength=0.0)
        res_2 = apply_chemical_toning(dark, selenium_strength=2.0, sepia_strength=0.0)
        self.assertGreaterEqual(float(res_2.min()), 0.0)
        self.assertLessEqual(float(res_2.max()), 1.0)
        self.assertLessEqual(float(res_2.mean()), float(res_1.mean()))  # longer bath, deeper


class TestSplitToning(unittest.TestCase):
    def test_noop_at_zero_strength(self):
        """Zero strengths → output identical to input."""
        img = np.random.rand(20, 20, 3).astype(np.float32)
        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=0.0, highlight_hue=30.0, highlight_strength=0.0)
        np.testing.assert_array_almost_equal(img, res)

    def test_shadow_tint_affects_shadows_more_than_highlights(self):
        """Shadow tint should shift chroma in dark pixels more than bright pixels."""
        # Dark pixel (shadow) vs bright pixel (highlight)
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[0:5, :, :] = 0.05  # shadows
        img[5:10, :, :] = 0.95  # highlights

        res = apply_split_toning(img, shadow_hue=0.0, shadow_strength=1.0, highlight_hue=0.0, highlight_strength=0.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        chroma_change_shadow = np.mean(np.abs(lab_out[0:5, :, 1:] - lab_in[0:5, :, 1:]))
        chroma_change_highlight = np.mean(np.abs(lab_out[5:10, :, 1:] - lab_in[5:10, :, 1:]))

        self.assertGreater(chroma_change_shadow, chroma_change_highlight)

    def test_highlight_tint_affects_highlights_more_than_shadows(self):
        """Highlight tint should shift chroma in bright pixels more than dark pixels."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[0:5, :, :] = 0.05  # shadows
        img[5:10, :, :] = 0.95  # highlights

        res = apply_split_toning(img, shadow_hue=0.0, shadow_strength=0.0, highlight_hue=90.0, highlight_strength=1.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        chroma_change_shadow = np.mean(np.abs(lab_out[0:5, :, 1:] - lab_in[0:5, :, 1:]))
        chroma_change_highlight = np.mean(np.abs(lab_out[5:10, :, 1:] - lab_in[5:10, :, 1:]))

        self.assertGreater(chroma_change_highlight, chroma_change_shadow)

    def test_shadow_hue_direction(self):
        """Hue 0° pushes a* positive (magenta); hue 180° pushes a* negative (green)."""
        img = np.full((10, 10, 3), 0.1, dtype=np.float32)  # dark shadows

        res_magenta = apply_split_toning(img, shadow_hue=0.0, shadow_strength=1.0)
        res_green = apply_split_toning(img, shadow_hue=180.0, shadow_strength=1.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_magenta = cv2.cvtColor(res_magenta, cv2.COLOR_RGB2LAB)
        lab_green = cv2.cvtColor(res_green, cv2.COLOR_RGB2LAB)

        # Hue 0° → a* increases (magenta direction)
        self.assertGreater(float(np.mean(lab_magenta[:, :, 1])), float(np.mean(lab_in[:, :, 1])))
        # Hue 180° → a* decreases (green direction)
        self.assertLess(float(np.mean(lab_green[:, :, 1])), float(np.mean(lab_in[:, :, 1])))

    def test_luminance_preserved(self):
        """Split toning should not significantly alter luminance."""
        img = np.random.rand(20, 20, 3).astype(np.float32)

        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=1.0, highlight_hue=30.0, highlight_strength=1.0)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        # L* change should be small (within 3 Lab units on average)
        mean_L_change = float(np.mean(np.abs(lab_out[:, :, 0] - lab_in[:, :, 0])))
        self.assertLess(mean_L_change, 3.0)

    def test_output_range(self):
        """Output should stay in [0, 1]."""
        img = np.random.rand(20, 20, 3).astype(np.float32)
        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=1.0, highlight_hue=30.0, highlight_strength=1.0)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_bw_image_gets_tinted(self):
        """A neutral gray (B&W) image should acquire chroma after split toning."""
        img = np.full((10, 10, 3), 0.1, dtype=np.float32)  # neutral gray shadow

        res = apply_split_toning(img, shadow_hue=195.0, shadow_strength=0.8)

        lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

        # Chroma (distance from neutral in a*b* plane) should increase
        chroma_in = np.sqrt(lab_in[:, :, 1] ** 2 + lab_in[:, :, 2] ** 2)
        chroma_out = np.sqrt(lab_out[:, :, 1] ** 2 + lab_out[:, :, 2] ** 2)
        self.assertGreater(float(np.mean(chroma_out)), float(np.mean(chroma_in)))


if __name__ == "__main__":
    unittest.main()
