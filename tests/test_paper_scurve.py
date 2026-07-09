import unittest

import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
    apply_characteristic_curve,
    compute_pivot,
    grade_to_slope,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS


class TestPaperSCurve(unittest.TestCase):
    """Variable-gamma paper S-curve: steeper midtones, anchor preserved, flat exempt."""

    def _slope_pivot(self):
        slope = grade_to_slope(115.0, 1.3)
        pivot = compute_pivot(slope, density=1.0, d_min=0.0)
        return slope, pivot

    def test_anchor_preserved(self):
        # The reference tone must still print at anchor_target_density with the
        # S-curve on (it is centred on the reference value, shape(0)=0).
        target = EXPOSURE_CONSTANTS["anchor_target_density"]
        slope, pivot = self._slope_pivot()
        curve = CharacteristicCurve(contrast=slope, pivot=pivot)
        anchor = EXPOSURE_CONSTANTS["assumed_anchor"]
        printed = float(curve(np.array([[anchor]], dtype=np.float32))[0, 0])
        self.assertAlmostEqual(printed, target, places=3)

    def test_midtone_steeper_than_straight_line(self):
        # Local contrast around the reference tone is higher with the S-curve on
        # than with midtone_gamma=0 (the plain straight line + softplus).
        slope, pivot = self._slope_pivot()
        anchor = EXPOSURE_CONSTANTS["assumed_anchor"]
        eps = 0.02
        ramp = np.array([[[anchor - eps] * 3, [anchor + eps] * 3]], dtype=np.float32)

        on = apply_characteristic_curve(ramp, (pivot, slope), (pivot, slope), (pivot, slope))
        off = apply_characteristic_curve(ramp, (pivot, slope), (pivot, slope), (pivot, slope), midtone_gamma=0.0)
        spread_on = float(on[0, 0, 0] - on[0, 1, 0])
        spread_off = float(off[0, 0, 0] - off[0, 1, 0])
        # Same sign, larger magnitude with the S-curve.
        self.assertGreater(abs(spread_on), abs(spread_off))

    def test_gamma_trim_steepens_midtones(self):
        # A positive user trim (effective_midtone_gamma) raises local midtone
        # contrast above the paper baseline; anchor and endpoints stay put.
        from negpy.features.exposure.logic import effective_midtone_gamma

        slope, pivot = self._slope_pivot()
        anchor = EXPOSURE_CONSTANTS["assumed_anchor"]
        eps = 0.02
        ramp = np.array([[[anchor - eps] * 3, [anchor + eps] * 3]], dtype=np.float32)
        args = ((pivot, slope), (pivot, slope), (pivot, slope))

        base = apply_characteristic_curve(ramp, *args)
        trimmed = apply_characteristic_curve(ramp, *args, midtone_gamma=effective_midtone_gamma(None, 0.3))
        spread_base = float(base[0, 0, 0] - base[0, 1, 0])
        spread_trim = float(trimmed[0, 0, 0] - trimmed[0, 1, 0])
        self.assertGreater(abs(spread_trim), abs(spread_base))

        # Anchor preserved (tanh centred on v_star); the ends stay bounded by
        # toe/shoulder (eased, not pinned — a small residual shift is inherent).
        for value, delta in ((anchor, 0.002), (0.0, 0.03), (1.0, 0.03)):
            patch = np.full((1, 1, 3), value, dtype=np.float32)
            a = float(apply_characteristic_curve(patch, *args)[0, 0, 0])
            b = float(apply_characteristic_curve(patch, *args, midtone_gamma=effective_midtone_gamma(None, 0.3))[0, 0, 0])
            self.assertAlmostEqual(a, b, delta=delta, msg=f"value={value}")

    def test_zero_trim_is_identity(self):
        from negpy.features.exposure.logic import effective_midtone_gamma

        self.assertEqual(effective_midtone_gamma(None, 0.0), EXPOSURE_CONSTANTS["paper_midtone_gamma"])

    def test_flat_disables_shape(self):
        # midtone_gamma=0 must reproduce the un-shaped curve exactly (flat master path).
        slope, pivot = self._slope_pivot()
        ramp = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16, 1).repeat(3, axis=2)
        a = apply_characteristic_curve(ramp, (pivot, slope), (pivot, slope), (pivot, slope), midtone_gamma=0.0)
        b = apply_characteristic_curve(ramp, (pivot, slope), (pivot, slope), (pivot, slope), midtone_gamma=0.0)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
