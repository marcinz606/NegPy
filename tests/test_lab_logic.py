import unittest
import numpy as np
from src.features.lab.logic import (
    apply_chroma_noise_removal,
    apply_output_sharpening,
)


class TestLabLogic(unittest.TestCase):
    def test_chroma_noise_removal_shadows(self):
        """Verify that chroma noise removal affects shadows more than highlights."""
        # Create a dark image with chroma noise (random a, b)
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[0:50, :, :] = 0.1  # Dark half
        img[50:100, :, :] = 0.9  # Bright half

        # Add "noise" in R and B channels to create chroma variation
        noise = np.random.rand(100, 100, 3).astype(np.float32) * 0.1
        img += noise
        img = np.clip(img, 0, 1)

        res = apply_chroma_noise_removal(img, strength_input=1.0)

        # In dark areas, it should be smoother (less variance)
        var_orig_dark = np.var(img[0:50, :, :])
        var_res_dark = np.var(res[0:50, :, :])

        # In bright areas, it should be less affected
        var_orig_bright = np.var(img[50:100, :, :])
        var_res_bright = np.var(res[50:100, :, :])

        self.assertLess(var_res_dark, var_orig_dark)
        # The reduction in bright areas should be smaller than in dark areas
        red_dark = var_orig_dark - var_res_dark
        red_bright = var_orig_bright - var_res_bright
        self.assertGreater(red_dark, red_bright)

    def test_output_sharpening(self):
        """Verify that sharpening increases local variance (edge contrast)."""
        # Create a simple square
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[25:75, 25:75, :] = 0.5

        res = apply_output_sharpening(img, amount=1.0)

        # Sharpening should increase variance on edges
        self.assertGreater(np.var(res), np.var(img))


if __name__ == "__main__":
    unittest.main()
