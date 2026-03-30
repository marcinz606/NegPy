import unittest

import numpy as np

from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    cmy_to_density,
    density_to_cmy,
)


class TestExposureLogic(unittest.TestCase):
    def test_apply_characteristic_curve_identity(self):
        """
        Verify math for neutral/flat settings.
        """
        img = np.full((10, 10, 3), 0.0, dtype=np.float32)  # Log space 0.0
        # If pivot=0, diff=0, sigmoid(0)=0.5.
        # d_max=4.0 -> density=2.0
        # transmittance = 10^-2.0 = 0.01
        # final = 0.01 ^ (1/2.2)
        params = (0.0, 1.0)
        res = apply_characteristic_curve(img, params, params, params)
        self.assertAlmostEqual(res[0, 0, 0], 0.01 ** (1 / 2.2), delta=0.01)

    def test_exposure_shift(self):
        """Check density shift direction."""
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)

        res1 = apply_characteristic_curve(img, (0.5, 2.0), (0.5, 2.0), (0.5, 2.0))
        res2 = apply_characteristic_curve(img, (0.6, 2.0), (0.6, 2.0), (0.6, 2.0))

        # Higher pivot -> lower diff -> lower density -> higher transmittance
        self.assertGreater(float(np.mean(res2)), float(np.mean(res1)))

    def test_cmy_conversions(self):
        """Verify unit conversion roundtrip."""
        val = 0.5
        dens = cmy_to_density(val, log_range=1.0)
        self.assertEqual(dens, 0.075)

        val_back = density_to_cmy(dens, log_range=1.0)
        self.assertAlmostEqual(val, val_back)

    def test_calculate_wb_shifts(self):
        """Verify WB shift calculation (neutralizing tint)."""
        from negpy.features.exposure.logic import calculate_wb_shifts

        # R=0.5, G=0.6, B=0.4 (Green cast, low Blue)
        sampled = np.array([0.5, 0.6, 0.4])
        dm, dy = calculate_wb_shifts(sampled)

        # dM = log10(0.6)-log10(0.5) > 0
        # dY = log10(0.4)-log10(0.5) < 0
        self.assertGreater(dm, 0)
        self.assertLess(dy, 0)

    def test_toe_shoulder_direction(self):
        """Verify that positive toe/shoulder values brighten the image (hybrid lift/recovery)."""
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        params = (0.5, 1.0)

        res_neutral = apply_characteristic_curve(img, params, params, params)
        res_toe = apply_characteristic_curve(img, params, params, params, toe=1.0)
        res_shoulder = apply_characteristic_curve(img, params, params, params, shoulder=1.0)

        # Positive toe lifts shadows -> brighter
        # Positive shoulder recovers highlights -> darker (in positive sense, but here it shifts exposure)
        # Wait, if shoulder > 0 it increases density (recovers highlights), so it should be DARKER.
        # If toe > 0 it decreases density (lifts shadows), so it should be BRIGHTER.

        self.assertGreater(float(np.mean(res_toe)), float(np.mean(res_neutral)))
        self.assertLess(float(np.mean(res_shoulder)), float(np.mean(res_neutral)))

    def test_regional_cmy(self):
        """Verify that regional CMY affects the output."""
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        params = (0.5, 1.0)

        res_neutral = apply_characteristic_curve(img, params, params, params)
        # Apply Cyan to shadows (Cyan in density space decreases R)
        # R = R_dens + offset. Transmittance = 10^-R. So more cyan -> lower R transmittance.
        res_shadow_cyan = apply_characteristic_curve(img, params, params, params, shadow_cmy=(1.0, 0.0, 0.0))

        self.assertLess(float(res_shadow_cyan[0, 0, 0]), float(res_neutral[0, 0, 0]))
        self.assertAlmostEqual(float(res_shadow_cyan[0, 0, 1]), float(res_neutral[0, 0, 1]), places=5)


if __name__ == "__main__":
    unittest.main()
