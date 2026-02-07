import unittest
import numpy as np
from negpy.features.exposure.normalization import analyze_log_exposure_bounds, normalize_log_image
from negpy.features.exposure.logic import apply_characteristic_curve
from negpy.features.process.models import ProcessMode


class TestE6Mode(unittest.TestCase):
    def test_e6_inversion_and_normalization(self):
        """Verify that E-6 mode correctly inverts and normalizes positive signals."""
        # 1. Create synthetic data:
        # Pixel 0: Black (0.1)
        # Pixel 1: White (0.9)
        img = np.array([[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]], dtype=np.float32)

        # 2. Analyze bounds in E-6 mode
        # E-6 inverts input: 1.0 - 0.9 = 0.1 (Highlight), 1.0 - 0.1 = 0.9 (Shadow)
        # Log values: log10(0.1) = -1.0, log10(0.9) = -0.045
        bounds = analyze_log_exposure_bounds(img, process_mode=ProcessMode.E6)

        # Floors (Inverted Highlights) should be ~ -1.0
        # Ceils (Inverted Shadows) should be ~ -0.045
        self.assertAlmostEqual(bounds.floors[0], -1.0, delta=0.1)
        self.assertAlmostEqual(bounds.ceils[0], -0.045, delta=0.1)

        # 3. Process normalization
        epsilon = 1e-6
        img_log = np.log10(np.clip(1.0 - img, epsilon, 1.0))
        norm = normalize_log_image(img_log, bounds)

        # Normalized result: 0.0=Highlight, 1.0=Shadow (Pseudo-Negative space)
        self.assertAlmostEqual(norm[0, 1, 0], 0.0, delta=0.05)
        self.assertAlmostEqual(norm[0, 0, 0], 1.0, delta=0.05)

    def test_e6_curve_parity(self):
        """Verify that E-6 mode produces a positive image using the pseudo-negative pipeline."""
        # Input is a normalized pseudo-negative pixel (0.0=Highlight, 1.0=Shadow)
        img_norm = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)

        # Neutral curve
        params = (0.0, 1.0)
        res = apply_characteristic_curve(img_norm, params, params, params, mode=2)

        # Pixel 0 (Highlight norm=0.0) -> Sigmoid(0) -> Density 2.0 -> Transmittance 0.01
        # Pixel 1 (Shadow norm=1.0) -> Sigmoid(1*slope) -> Density > 2.0 -> Transmittance < 0.01
        self.assertGreater(res[0, 0, 0], res[0, 1, 0])

        # Verify slope boost (mode=2 uses 1.15x slope)
        res_c41 = apply_characteristic_curve(img_norm, params, params, params, mode=0)

        # At norm=1.0 (Shadow), E-6 should be darker due to higher slope
        self.assertLess(res[0, 1, 0], res_c41[0, 1, 0])


if __name__ == "__main__":
    unittest.main()
