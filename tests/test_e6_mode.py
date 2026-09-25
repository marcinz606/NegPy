import unittest
import numpy as np
from negpy.features.exposure.logic import apply_characteristic_curve


class TestE6Mode(unittest.TestCase):
    def test_e6_curve_parity(self):
        img_norm = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)

        params = (0.0, 1.0)
        res = apply_characteristic_curve(img_norm, params, params, params)

        self.assertGreater(res[0, 0, 0], res[0, 1, 0])


if __name__ == "__main__":
    unittest.main()
