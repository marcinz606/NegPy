import unittest
import cv2
import numpy as np
from negpy.features.toning.logic import (
    simulate_paper_substrate,
    apply_chemical_toning,
    apply_split_toning,
    PAPER_PROFILES,
)


class TestToningLogic(unittest.TestCase):
    def test_simulate_paper_substrate_none(self):
        """Identity check for 'None' profile."""
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        res = simulate_paper_substrate(img, "None")
        np.testing.assert_array_almost_equal(img, res)

    def test_simulate_paper_substrate_tint(self):
        """Verify tint application."""
        img = np.full((10, 10, 3), 1.0, dtype=np.float32)  # White input
        res = simulate_paper_substrate(img, "Warm Fiber")

        profile = PAPER_PROFILES["Warm Fiber"]
        expected_tint = np.array(profile.tint, dtype=np.float32)
        np.testing.assert_array_almost_equal(res[0, 0], expected_tint, decimal=2)

    def test_apply_chemical_toning_selenium(self):
        """Selenium targets shadows (low luma)."""
        # Create a gradient from 0 to 1
        img = np.linspace(0, 1, 100).reshape((10, 10, 1)).repeat(3, axis=2).astype(np.float32)

        res = apply_chemical_toning(img, selenium_strength=1.0, sepia_strength=0.0)

        # Selenium color is [0.85, 0.75, 0.85] (cool/dark)
        # It affects low lum (1 - lum_val)
        # Shadow (img=0.1) should be changed more than highlight (img=0.9)
        diff_shadow = np.abs(res[1, 0, 0] - img[1, 0, 0])
        diff_highlight = np.abs(res[9, 0, 0] - img[9, 0, 0])

        self.assertGreater(diff_shadow, diff_highlight)

    def test_apply_chemical_toning_sepia(self):
        """Sepia targets midtones (warm shift)."""
        img = np.full((10, 10, 3), 0.6, dtype=np.float32)
        res = apply_chemical_toning(img, selenium_strength=0.0, sepia_strength=1.0)

        # Sepia color is [1.1, 0.99, 0.825]
        # Midtones around 0.6 are affected by exp(-((lum-0.6)**2)/0.08)
        # Check that red increased and blue decreased
        self.assertGreater(res[0, 0, 0], img[0, 0, 0])
        self.assertLess(res[0, 0, 2], img[0, 0, 2])


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
