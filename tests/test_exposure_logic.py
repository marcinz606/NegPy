import unittest

import numpy as np

from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    cmy_to_density,
    density_to_cmy,
    kelvin_to_wb,
    wb_to_kelvin,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.process.logic import linear_raw_token
from negpy.features.process.models import ProcessConfig


class TestExposureLogic(unittest.TestCase):
    def test_apply_characteristic_curve_identity(self):
        """At the pivot (line value 0) the toe-linear-shoulder curve gives a
        density set by the base toe/shoulder bounds; verify the closed form."""
        img = np.full((10, 10, 3), 0.0, dtype=np.float32)
        params = (0.0, 1.0)  # pivot 0, slope 1 -> line value v = 0
        # midtone_gamma=0: isolate the bare toe/shoulder closed form (the S-curve
        # shapes around the reference value, not the pivot).
        res = apply_characteristic_curve(img, params, params, params, midtone_gamma=0.0)

        a_hl = EXPOSURE_CONSTANTS["shoulder_sharpness_base"]  # highlight (lower) bound
        a_sh = EXPOSURE_CONSTANTS["toe_sharpness_base"]  # shadow (upper) bound
        d_max = EXPOSURE_CONSTANTS["d_max"]
        v1 = np.logaddexp(0.0, a_hl * 0.0) / a_hl
        d = d_max - np.logaddexp(0.0, a_sh * (d_max - v1)) / a_sh
        # Exposure stage now outputs linear reflectance (transmittance = 10^-D);
        # the OETF moved to the engine output.
        t = 10.0**-d
        self.assertAlmostEqual(res[0, 0, 0], t, delta=0.01)

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

    def test_wb_to_kelvin_neutral_is_reference(self):
        from negpy.features.exposure.logic import TEMP_REF_KELVIN

        self.assertAlmostEqual(wb_to_kelvin(0.0, 0.0), TEMP_REF_KELVIN, places=6)

    def test_kelvin_wb_roundtrip(self):
        for k in (4000.0, 5500.0, 8000.0, 11000.0):
            m, y = kelvin_to_wb(k, 0.0, 0.0)
            self.assertAlmostEqual(wb_to_kelvin(m, y), k, places=4)

    def test_kelvin_warm_direction_and_ratio(self):
        m, y = kelvin_to_wb(4000.0, 0.0, 0.0)
        self.assertGreater(m, 0.0)
        self.assertGreater(y, 0.0)
        # Along the locus both channels move by k * dmu, so the ratio is exact.
        self.assertAlmostEqual(y / m, 0.0057 / 0.0029, places=6)

    def test_kelvin_to_wb_preserves_tint(self):
        # Moving temperature away and back recovers the original pair exactly.
        m0, y0 = 0.3, -0.2
        k0 = wb_to_kelvin(m0, y0)
        m1, y1 = kelvin_to_wb(4500.0, m0, y0)
        m2, y2 = kelvin_to_wb(k0, m1, y1)
        self.assertAlmostEqual(m2, m0, places=6)
        self.assertAlmostEqual(y2, y0, places=6)

    def test_kelvin_projection_idempotent(self):
        m1, y1 = kelvin_to_wb(4500.0, 0.1, 0.05)
        m2, y2 = kelvin_to_wb(4500.0, m1, y1)
        self.assertAlmostEqual(m1, m2, places=9)
        self.assertAlmostEqual(y1, y2, places=9)

    def test_kelvin_extreme_wb_clamps_in_mired_domain(self):
        from negpy.features.exposure.logic import TEMP_MAX_KELVIN, TEMP_MIN_KELVIN

        # A full-cool pair pushes mu negative; the mired-domain clamp must
        # land on the cool end, not wrap to the warm one.
        self.assertAlmostEqual(wb_to_kelvin(-1.0, -1.0), TEMP_MAX_KELVIN, places=6)
        self.assertAlmostEqual(wb_to_kelvin(1.0, 1.0), TEMP_MIN_KELVIN, places=6)

    def test_kelvin_orthogonal_tint_moves_dont_change_readout(self):
        from negpy.features.exposure.logic import _TEMP_K_MAGENTA, _TEMP_K_YELLOW

        k_before = wb_to_kelvin(0.1, 0.2)
        t = 0.15
        k_after = wb_to_kelvin(0.1 + _TEMP_K_YELLOW * t, 0.2 - _TEMP_K_MAGENTA * t)
        self.assertAlmostEqual(k_before, k_after, places=6)

    def test_toe_shoulder_direction(self):
        """Toe rolls shadows (high input), shoulder rolls highlights (low input),
        each leaving the other end untouched (film/print convention)."""
        params = (0.5, 4.0)

        # Shadow zone (high input = dense print): positive toe -> lighter (lifted blacks).
        img_shadow = np.full((10, 10, 3), 0.9, dtype=np.float32)
        res_neutral_sh = apply_characteristic_curve(img_shadow, params, params, params)
        res_toe = apply_characteristic_curve(img_shadow, params, params, params, toe=1.0)
        self.assertGreater(float(np.mean(res_toe)), float(np.mean(res_neutral_sh)))

        # Highlight zone (low input = bright print): positive shoulder -> darker (compressed).
        img_highlight = np.full((10, 10, 3), 0.1, dtype=np.float32)
        res_neutral_hl = apply_characteristic_curve(img_highlight, params, params, params)
        res_shoulder = apply_characteristic_curve(img_highlight, params, params, params, shoulder=1.0)
        self.assertLess(float(np.mean(res_shoulder)), float(np.mean(res_neutral_hl)))

        # Independence: toe leaves bright highlights put; shoulder leaves deep shadows put.
        res_hl_toe = apply_characteristic_curve(img_highlight, params, params, params, toe=1.0)
        np.testing.assert_allclose(res_hl_toe, res_neutral_hl, atol=0.015)
        res_sh_sh = apply_characteristic_curve(img_shadow, params, params, params, shoulder=1.0)
        np.testing.assert_allclose(res_sh_sh, res_neutral_sh, atol=0.015)

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


class TestAdaptiveMeteringStrength(unittest.TestCase):
    def test_strength_increases_with_deviation(self):
        from negpy.features.exposure.normalization import LogNegativeBounds, measure_anchor_from_log

        c = EXPOSURE_CONSTANTS
        assumed = float(c["assumed_anchor"])

        def _anchor_for_luminance(norm_val: float) -> float:
            floor, ceil = -2.0, -0.5
            log_val = floor + norm_val * (ceil - floor)
            img_log = np.full((64, 64, 3), log_val, dtype=np.float32)
            bounds = LogNegativeBounds(floors=(floor, floor, floor), ceils=(ceil, ceil, ceil))
            return measure_anchor_from_log(img_log, bounds)

        a_slight = _anchor_for_luminance(assumed + 0.05)
        a_large = _anchor_for_luminance(assumed + float(c["anchor_meter_band"]))
        dev_slight = abs(a_slight - assumed)
        dev_large = abs(a_large - assumed)
        self.assertGreater(dev_large, dev_slight)
        a_extreme = _anchor_for_luminance(assumed + 0.5)
        self.assertLessEqual(abs(a_extreme - assumed), float(c["anchor_meter_band"]) + 1e-6)


class TestLinearRawToken(unittest.TestCase):
    def test_token_differs_by_mode(self):
        """Toggling Linear RAW must change the source identity, else the per-source
        auto-meter cache (bounds + neutral-axis cast) goes stale across the toggle (#355)."""
        on = linear_raw_token(ProcessConfig(linear_raw=True))
        off = linear_raw_token(ProcessConfig(linear_raw=False))
        self.assertNotEqual(on, off)


if __name__ == "__main__":
    unittest.main()
