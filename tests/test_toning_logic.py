import unittest
import numpy as np
from src.features.toning.logic import (
    simulate_paper_substrate,
    apply_chemical_toning,
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
        res = simulate_paper_substrate(img, "Antique Ivory")

        profile = PAPER_PROFILES["Antique Ivory"]
        expected_tint = np.array(profile.tint, dtype=np.float32)
        np.testing.assert_array_almost_equal(res[0, 0], expected_tint, decimal=2)

    def test_apply_chemical_toning_selenium(self):
        """Selenium targets shadows (low luma)."""
        # Create a gradient from 0 to 1
        img = (
            np.linspace(0, 1, 100)
            .reshape((10, 10, 1))
            .repeat(3, axis=2)
            .astype(np.float32)
        )

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


if __name__ == "__main__":
    unittest.main()
