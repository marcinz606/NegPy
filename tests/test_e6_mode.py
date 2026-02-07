import unittest
import numpy as np
from negpy.features.exposure.normalization import analyze_log_exposure_bounds, normalize_log_image
from negpy.features.exposure.logic import apply_characteristic_curve
from negpy.features.process.models import ProcessMode


class TestE6Mode(unittest.TestCase):
    def test_e6_native_normalization(self):
        img = np.array([[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]], dtype=np.float32)

        bounds = analyze_log_exposure_bounds(img, process_mode=ProcessMode.E6)

        self.assertAlmostEqual(bounds.floors[0], -0.045, delta=0.1)
        self.assertAlmostEqual(bounds.ceils[0], -1.0, delta=0.1)

        epsilon = 1e-6
        img_log = np.log10(np.clip(img, epsilon, 1.0))
        norm = normalize_log_image(img_log, bounds)

        self.assertAlmostEqual(norm[0, 1, 0], 0.0, delta=0.05)
        self.assertAlmostEqual(norm[0, 0, 0], 1.0, delta=0.05)

    def test_e6_curve_parity(self):
        img_norm = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)

        params = (0.0, 1.0)
        res = apply_characteristic_curve(img_norm, params, params, params, mode=2)

        self.assertGreater(res[0, 0, 0], res[0, 1, 0])


if __name__ == "__main__":
    unittest.main()