import unittest
import numpy as np
from src.features.exposure.logic import apply_characteristic_curve
from src.features.lab.logic import (
    apply_spectral_crosstalk,
    apply_clahe,
)


class TestAnalogSimulation(unittest.TestCase):
    def test_cmy_offset_inverse_relationship(self):
        """
        Yellow offset should reduce Blue output (Positive process).
        """
        # Create a neutral grey input (log density space)
        img_log = np.full((10, 10, 3), 0.5, dtype=np.float32)
        params = (0.5, 1.0)  # pivot=0.5, slope=1.0

        # Base run with no offsets
        res_base = apply_characteristic_curve(
            img_log, params, params, params, cmy_offsets=(0, 0, 0)
        )

        # Run with Yellow Offset (CMY: index 2 is Yellow)
        res_yellow = apply_characteristic_curve(
            img_log, params, params, params, cmy_offsets=(0, 0, 0.5)
        )

        # Blue channel is index 2
        blue_base = np.mean(res_base[:, :, 2])
        blue_yellow = np.mean(res_yellow[:, :, 2])

        self.assertLess(
            blue_yellow,
            blue_base,
            "Increasing Yellow Offset should reduce Blue output transmittance.",
        )

    def test_crosstalk_unmixing(self):
        """
        Normalized matrix should preserve neutral grey.
        """
        # Neutral input in density space
        img_dens = np.full((10, 10, 3), 0.5, dtype=np.float32)

        # A matrix that would normally darken/lighten the image
        custom_matrix = [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        res = apply_spectral_crosstalk(img_dens, 1.0, custom_matrix)

        # With row normalization, [0.5, 0.5, 0.5] @ [2,0,0 / sum] -> 0.5
        # So the output should still be 0.5
        self.assertAlmostEqual(np.mean(res), 0.5)

    def test_clahe_preserves_color(self):
        """
        CLAHE should not skew channel ratios (color shift).
        """
        # Create a colorized image
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[:, :, 0] = 0.6  # Reddish
        img[:, :, 1] = 0.4
        img[:, :, 2] = 0.2

        res = apply_clahe(img, 0.5)

        # Check that ratio of channels is roughly preserved
        mean_orig = np.mean(img, axis=(0, 1))
        mean_res = np.mean(res, axis=(0, 1))

        ratio_orig = mean_orig[0] / mean_orig[1]
        ratio_res = mean_res[0] / mean_res[1]

        self.assertAlmostEqual(ratio_orig, ratio_res, delta=0.1)


if __name__ == "__main__":
    unittest.main()
