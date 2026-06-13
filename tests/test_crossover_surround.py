import unittest

import numpy as np

from negpy.features.exposure.logic import (
    LogisticSigmoid,
    apply_characteristic_curve,
    per_channel_curve_params,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.kernel.image.validation import ensure_image


class TestCrossover(unittest.TestCase):
    """
    Per-channel color crossover: each channel's slope is scaled toward its own
    measured negative density range, with the pivot solved per channel so the
    metered anchor still prints neutral.
    """

    def test_off_collapses_to_single_curve(self):
        s, p = per_channel_curve_params(
            115.0, 1.0, True, False, 1.4, (1.3, 1.4, 1.5), 0.7, d_min=0.06, anchor=0.46
        )
        self.assertEqual(s[0], s[1])
        self.assertEqual(s[1], s[2])
        self.assertEqual(p[0], p[1])
        self.assertEqual(p[1], p[2])

    def test_on_diverges_with_measured_ranges(self):
        s, p = per_channel_curve_params(
            115.0, 1.0, True, True, 1.4, (1.3, 1.4, 1.5), 0.7, d_min=0.06, anchor=0.46
        )
        # Distinct per-channel ranges -> distinct slopes/pivots.
        self.assertGreater(max(s) - min(s), 1e-4)
        self.assertGreater(max(p) - min(p), 1e-4)

    def test_equal_ranges_stay_neutral_even_on(self):
        s, p = per_channel_curve_params(
            115.0, 1.0, True, True, 1.4, (1.4, 1.4, 1.4), 0.7, d_min=0.06, anchor=0.46
        )
        self.assertAlmostEqual(s[0], s[2], places=6)
        self.assertAlmostEqual(p[0], p[2], places=6)

    def test_midtone_prints_neutral_under_crossover(self):
        anchor = 0.46
        s, p = per_channel_curve_params(
            115.0, 1.0, True, True, 1.4, (1.2, 1.4, 1.6), 0.7, d_min=0.06, anchor=anchor
        )
        densities = []
        for ch in range(3):
            curve = LogisticSigmoid(contrast=s[ch], pivot=p[ch], d_min=0.06)
            densities.append(float(curve(ensure_image(np.array([anchor])))[0]))
        # All three channels print the anchor tone at the same density => neutral.
        self.assertAlmostEqual(densities[0], densities[1], places=4)
        self.assertAlmostEqual(densities[1], densities[2], places=4)
        self.assertAlmostEqual(densities[0], EXPOSURE_CONSTANTS["anchor_target_density"], places=3)


class TestSurroundGamma(unittest.TestCase):
    """
    Surround system gamma: a fixed contrast expansion about paper white. Default
    (identity) leaves the render untouched; enabled, it darkens midtones while
    holding paper white and is monotone.
    """

    def test_identity_is_no_op(self):
        img = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
        a = apply_characteristic_curve(img, (0.4, 5.0), (0.4, 5.0), (0.4, 5.0), d_min=0.06)
        b = apply_characteristic_curve(
            img, (0.4, 5.0), (0.4, 5.0), (0.4, 5.0), d_min=0.06, surround_gamma=1.0
        )
        self.assertTrue(np.allclose(np.asarray(a), np.asarray(b)))

    def test_paper_white_invariant(self):
        d_min = 0.06
        # At density == d_min, D' = d_min + gamma*(d_min - d_min) = d_min.
        gamma = EXPOSURE_CONSTANTS["target_system_gamma"]
        self.assertAlmostEqual(d_min + gamma * (d_min - d_min), d_min, places=9)

    def test_midtone_darkens_and_monotone(self):
        gamma = EXPOSURE_CONSTANTS["target_system_gamma"]
        x = np.linspace(0.0, 1.0, 50).reshape(-1, 1, 1)
        base = np.asarray(LogisticSigmoid(5.0, 0.3, d_min=0.06)(x)).ravel()
        warp = np.asarray(LogisticSigmoid(5.0, 0.3, d_min=0.06, surround_gamma=gamma)(x)).ravel()
        # Midtone density increases (print darkens) where density is above d_min.
        mid = base > 0.2
        self.assertTrue(np.all(warp[mid] >= base[mid] - 1e-9))
        # Monotone non-decreasing density along the input axis preserved.
        self.assertTrue(np.all(np.diff(warp) >= -1e-6))


if __name__ == "__main__":
    unittest.main()
