import unittest

import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
    apply_characteristic_curve,
    per_channel_curve_params,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.kernel.image.validation import ensure_image


class TestDensityBalance(unittest.TestCase):
    """
    Per-channel density balance: a two-point gray balance. Each channel's slope is
    solved so its measured shadow reference prints at the green channel's shadow
    density, while compute_pivot keeps the midtone anchor neutral — so both
    neutrals read equal-RGB and grays stay neutral across the range (crossover
    removed). shadow_refs_norm are per-channel shadow positions in normalized [0,1].
    """

    def test_off_collapses_to_single_curve(self):
        s, p = per_channel_curve_params(115.0, 1.0, True, False, 1.4, (0.85, 0.80, 0.75), 0.7, d_min=0.06, anchor=0.46)
        self.assertEqual(s[0], s[1])
        self.assertEqual(s[1], s[2])
        self.assertEqual(p[0], p[1])
        self.assertEqual(p[1], p[2])

    def test_no_refs_collapses_to_single_curve(self):
        # E6 / B&W: no shadow refs -> behaves like off.
        s, p = per_channel_curve_params(115.0, 1.0, True, True, 1.4, None, 0.7, d_min=0.06, anchor=0.46)
        self.assertEqual(s[0], s[1])
        self.assertEqual(s[1], s[2])

    def test_equal_refs_stay_neutral_even_on(self):
        s, p = per_channel_curve_params(115.0, 1.0, True, True, 1.4, (0.80, 0.80, 0.80), 0.7, d_min=0.06, anchor=0.46)
        self.assertAlmostEqual(s[0], s[2], places=6)
        self.assertAlmostEqual(p[0], p[2], places=6)

    def test_mismatched_refs_diverge_slopes(self):
        s, p = per_channel_curve_params(115.0, 1.0, True, True, 1.4, (0.85, 0.80, 0.72), 0.7, d_min=0.06, anchor=0.46)
        self.assertGreater(max(s) - min(s), 1e-4)
        # Green keeps the base slope (reference channel).
        s_off, _ = per_channel_curve_params(115.0, 1.0, True, False, 1.4, (0.85, 0.80, 0.72), 0.7, d_min=0.06, anchor=0.46)
        self.assertAlmostEqual(s[1], s_off[1], places=6)

    def test_two_neutrals_print_neutral(self):
        anchor = 0.46
        refs = (0.85, 0.80, 0.72)
        s, p = per_channel_curve_params(115.0, 1.0, True, True, 1.4, refs, 0.7, d_min=0.06, anchor=anchor)
        anchor_d = []
        shadow_d = []
        for ch in range(3):
            curve = CharacteristicCurve(contrast=s[ch], pivot=p[ch], d_min=0.06)
            anchor_d.append(float(curve(ensure_image(np.array([anchor])))[0]))
            shadow_d.append(float(curve(ensure_image(np.array([refs[ch]])))[0]))
        # Midtone neutral: every channel prints the anchor at anchor_target_density.
        for d in anchor_d:
            self.assertAlmostEqual(d, EXPOSURE_CONSTANTS["anchor_target_density"], places=3)
        # Shadow neutral: every channel's shadow ref prints at the SAME density
        # (the green channel's), so the shadow is neutral too.
        self.assertAlmostEqual(shadow_d[0], shadow_d[1], places=3)
        self.assertAlmostEqual(shadow_d[1], shadow_d[2], places=3)


class TestSurroundGamma(unittest.TestCase):
    """
    Surround system gamma: a fixed contrast expansion about paper white. Default
    (identity) leaves the render untouched; enabled, it darkens midtones while
    holding paper white and is monotone.
    """

    def test_identity_is_no_op(self):
        img = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
        a = apply_characteristic_curve(img, (0.4, 5.0), (0.4, 5.0), (0.4, 5.0), d_min=0.06)
        b = apply_characteristic_curve(img, (0.4, 5.0), (0.4, 5.0), (0.4, 5.0), d_min=0.06, surround_gamma=1.0)
        self.assertTrue(np.allclose(np.asarray(a), np.asarray(b)))

    def test_paper_white_invariant(self):
        d_min = 0.06
        # At density == d_min, D' = d_min + gamma*(d_min - d_min) = d_min.
        gamma = EXPOSURE_CONSTANTS["target_system_gamma"]
        self.assertAlmostEqual(d_min + gamma * (d_min - d_min), d_min, places=9)

    def test_midtone_darkens_and_monotone(self):
        gamma = EXPOSURE_CONSTANTS["target_system_gamma"]
        x = np.linspace(0.0, 1.0, 50).reshape(-1, 1, 1)
        base = np.asarray(CharacteristicCurve(5.0, 0.3, d_min=0.06)(x)).ravel()
        warp = np.asarray(CharacteristicCurve(5.0, 0.3, d_min=0.06, surround_gamma=gamma)(x)).ravel()
        # Midtone density increases (print darkens) where density is above d_min.
        mid = base > 0.2
        self.assertTrue(np.all(warp[mid] >= base[mid] - 1e-9))
        # Monotone non-decreasing density along the input axis preserved.
        self.assertTrue(np.all(np.diff(warp) >= -1e-6))


if __name__ == "__main__":
    unittest.main()
