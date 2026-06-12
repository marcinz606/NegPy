import unittest

import numpy as np

from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    cmy_to_density,
    density_to_cmy,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS


class TestExposureLogic(unittest.TestCase):
    def test_apply_characteristic_curve_identity(self):
        """
        Verify math for neutral/flat settings.
        """
        img = np.full((10, 10, 3), 0.0, dtype=np.float32)  # Log space 0.0
        # If pivot=0, diff=0, sigmoid(0)=0.5 -> density = asymptote / 2,
        # then the soft Dmax shoulder; transmittance = 10^-density;
        # final = sRGB OETF(transmittance).
        params = (0.0, 1.0)
        res = apply_characteristic_curve(img, params, params, params)
        beta = EXPOSURE_CONSTANTS["dmax_shoulder"]
        d = EXPOSURE_CONSTANTS["curve_asymptote"] * 0.5 ** EXPOSURE_CONSTANTS["paper_toe_nu"]
        d -= np.logaddexp(0.0, beta * (d - EXPOSURE_CONSTANTS["d_max"])) / beta
        t = 10.0**-d
        self.assertAlmostEqual(res[0, 0, 0], 1.055 * t ** (1 / 2.4) - 0.055, delta=0.01)

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
        self.assertEqual(dens, 0.1)  # 0.5 * cmy_max_density(0.2) / 1.0

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
        """Verify toe/shoulder act in their zones; the midtone (pivot) stays anchored."""
        params = (0.5, 4.0)

        # Shadow zone (high input = dense print): positive toe lifts -> brighter.
        img_shadow = np.full((10, 10, 3), 0.9, dtype=np.float32)
        res_neutral = apply_characteristic_curve(img_shadow, params, params, params)
        res_toe = apply_characteristic_curve(img_shadow, params, params, params, toe=1.0)
        self.assertGreater(float(np.mean(res_toe)), float(np.mean(res_neutral)))

        # Highlight zone (low input = bright print): positive shoulder compresses -> darker.
        img_highlight = np.full((10, 10, 3), 0.1, dtype=np.float32)
        res_neutral_hl = apply_characteristic_curve(img_highlight, params, params, params)
        res_shoulder = apply_characteristic_curve(img_highlight, params, params, params, shoulder=1.0)
        self.assertLess(float(np.mean(res_shoulder)), float(np.mean(res_neutral_hl)))

        # Midtone anchor: value at the pivot is invariant under shoulder.
        img_mid = np.full((10, 10, 3), 0.5, dtype=np.float32)
        res_mid = apply_characteristic_curve(img_mid, params, params, params)
        res_mid_s = apply_characteristic_curve(img_mid, params, params, params, shoulder=-1.0)
        np.testing.assert_array_almost_equal(res_mid, res_mid_s, decimal=5)

        # Highlight anchor: toe (density-domain, anchored at D=0) leaves
        # bright print tones untouched (true highlights, well above the
        # shadow-lever onset).
        img_bright = np.full((10, 10, 3), 0.0, dtype=np.float32)
        res_neutral_b = apply_characteristic_curve(img_bright, params, params, params)
        res_b_toe = apply_characteristic_curve(img_bright, params, params, params, toe=1.0)
        np.testing.assert_array_almost_equal(res_neutral_b, res_b_toe, decimal=2)

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
