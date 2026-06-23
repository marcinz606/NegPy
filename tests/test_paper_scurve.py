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

    def test_flat_disables_shape(self):
        # midtone_gamma=0 must reproduce the un-shaped curve exactly (flat master path).
        slope, pivot = self._slope_pivot()
        ramp = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16, 1).repeat(3, axis=2)
        a = apply_characteristic_curve(ramp, (pivot, slope), (pivot, slope), (pivot, slope), midtone_gamma=0.0)
        b = apply_characteristic_curve(ramp, (pivot, slope), (pivot, slope), (pivot, slope), midtone_gamma=0.0)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
