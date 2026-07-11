import unittest

import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
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
        s, p, _ = per_channel_curve_params(115.0, 1.0, True, False, 1.4, (0.85, 0.80, 0.75), 0.7, d_min=0.06, anchor=0.46)
        self.assertEqual(s[0], s[1])
        self.assertEqual(s[1], s[2])
        self.assertEqual(p[0], p[1])
        self.assertEqual(p[1], p[2])

    def test_no_refs_collapses_to_single_curve(self):
        # E6 / B&W: no shadow refs -> behaves like off.
        s, p, _ = per_channel_curve_params(115.0, 1.0, True, True, 1.4, None, 0.7, d_min=0.06, anchor=0.46)
        self.assertEqual(s[0], s[1])
        self.assertEqual(s[1], s[2])

    def test_equal_refs_stay_neutral_even_on(self):
        s, p, _ = per_channel_curve_params(115.0, 1.0, True, True, 1.4, (0.80, 0.80, 0.80), 0.7, d_min=0.06, anchor=0.46)
        self.assertAlmostEqual(s[0], s[2], places=6)
        self.assertAlmostEqual(p[0], p[2], places=6)

    def test_mismatched_refs_diverge_slopes(self):
        s, p, _ = per_channel_curve_params(115.0, 1.0, True, True, 1.4, (0.85, 0.80, 0.72), 0.7, d_min=0.06, anchor=0.46)
        self.assertGreater(max(s) - min(s), 1e-4)
        # Green keeps the base slope (reference channel).
        s_off, _, _ = per_channel_curve_params(115.0, 1.0, True, False, 1.4, (0.85, 0.80, 0.72), 0.7, d_min=0.06, anchor=0.46)
        self.assertAlmostEqual(s[1], s_off[1], places=6)

    def test_two_neutrals_print_neutral(self):
        anchor = 0.46
        refs = (0.85, 0.80, 0.72)
        s, p, _ = per_channel_curve_params(115.0, 1.0, True, True, 1.4, refs, 0.7, d_min=0.06, anchor=anchor)
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


if __name__ == "__main__":
    unittest.main()
