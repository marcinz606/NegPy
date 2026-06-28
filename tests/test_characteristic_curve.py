"""Contract tests for the asymmetric H&D print curve (toe-linear-shoulder).

These pin the *new* curve semantics independent of the exact calibrated
constants: monotone, asymptotes to paper white / paper black, and the toe and
shoulder sliders act on independent ends (highlights vs shadows).
"""

import unittest

import numpy as np

from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    compute_pivot,
    grade_to_slope,
    slope_to_grade,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode


def _ramp(n=257):
    x = np.linspace(0.0, 1.0, n).astype(np.float32)
    return x, np.stack([x, x, x], axis=-1)[None, :, :]


def _curve(toe=0.0, shoulder=0.0, grade=115.0, density=1.0, lum_range=1.3):
    x, ramp = _ramp()
    d_min = EXPOSURE_CONSTANTS["d_min"]
    slope = grade_to_slope(grade, lum_range)
    pivot = compute_pivot(slope, density=density, d_min=d_min)
    out = apply_characteristic_curve(
        ramp,
        (pivot, slope),
        (pivot, slope),
        (pivot, slope),
        toe=toe,
        shoulder=shoulder,
        d_min=d_min,
    )
    # Stage outputs linear; compare in the displayed space where goldens are calibrated.
    return x, np.asarray(working_oetf_encode(np.asarray(out)))[0, :, 0].astype(np.float64)


def _output_to_density(s):
    # Decode the displayed curve value (working TRC) back to transmittance, then density.
    t = np.asarray(working_oetf_decode(np.asarray(s, dtype=np.float32)))
    return -np.log10(np.clip(t, 1e-12, None))


class TestCurveShape(unittest.TestCase):
    def test_monotone_decreasing(self):
        """Print brightness decreases monotonically with input (highlights->shadows)."""
        for grade in (50.0, 115.0, 180.0):
            _, out = _curve(grade=grade)
            self.assertTrue(np.all(np.diff(out) <= 1e-6), f"non-monotone at grade {grade}")

    def test_endpoints_reach_paper_white_and_black(self):
        """x->0 prints near paper white (D~d_min); x->1 near paper black (D~d_max)."""
        _, out = _curve()
        d = _output_to_density(out)
        self.assertLess(abs(d[0] - EXPOSURE_CONSTANTS["d_min"]), 0.25, "highlight not near paper white")
        self.assertLess(abs(d[-1] - EXPOSURE_CONSTANTS["d_max"]), 0.4, "shadow not near paper black")


class TestToeShoulderIndependence(unittest.TestCase):
    """The defining new behaviour: toe shapes ONLY shadows, shoulder ONLY highlights
    (film/print convention)."""

    def test_toe_acts_on_shadows_not_highlights(self):
        x, base = _curve()
        _, toed = _curve(toe=1.0)
        hi = x < 0.2  # highlights (paper-white end)
        sh = x > 0.8  # shadows (paper-black end)
        self.assertGreater(np.max(np.abs(toed[sh] - base[sh])), 0.02, "toe did not affect shadows")
        np.testing.assert_allclose(toed[hi], base[hi], atol=0.01, err_msg="toe leaked into highlights")

    def test_shoulder_acts_on_highlights_not_shadows(self):
        x, base = _curve()
        _, sh_out = _curve(shoulder=1.0)
        hi = x < 0.2
        sh = x > 0.8
        self.assertGreater(np.max(np.abs(sh_out[hi] - base[hi])), 0.02, "shoulder did not affect highlights")
        np.testing.assert_allclose(sh_out[sh], base[sh], atol=0.01, err_msg="shoulder leaked into shadows")

    def test_toe_positive_lifts_shadows(self):
        x, base = _curve()
        _, toed = _curve(toe=1.0)
        sh = x > 0.85
        self.assertGreater(float(np.mean(toed[sh])), float(np.mean(base[sh])))

    def test_shoulder_positive_darkens_highlights(self):
        x, base = _curve()
        _, sh_out = _curve(shoulder=1.0)
        hi = x < 0.15
        self.assertLess(float(np.mean(sh_out[hi])), float(np.mean(base[hi])))


class TestCalibration(unittest.TestCase):
    def test_default_curve_shape(self):
        """Pin the default tone reproduction so edits don't drift. Golden is the
        displayed curve in the working-space TRC (ProPhoto ROMM gamma) at x = 0, .25,
        .5, .75, 1."""
        x, out = _curve()
        idx = [0, 64, 128, 192, 256]
        golden = [0.904, 0.739, 0.327, 0.114, 0.059]
        for i, g in zip(idx, golden):
            self.assertAlmostEqual(out[i], g, delta=0.03, msg=f"x={x[i]:.2f}")


class TestPivotAndGrade(unittest.TestCase):
    def test_reference_prints_at_target_grade_invariant(self):
        """Reference tone lands on anchor_target_density for any grade (rotation center)."""
        d_min = EXPOSURE_CONSTANTS["d_min"]
        x_ref = EXPOSURE_CONSTANTS["assumed_anchor"]
        target = EXPOSURE_CONSTANTS["anchor_target_density"]
        ref_img = np.full((4, 4, 3), x_ref, dtype=np.float32)
        for grade in (60.0, 115.0, 170.0):
            slope = grade_to_slope(grade, 1.3)
            pivot = compute_pivot(slope, density=1.0, d_min=d_min)
            out = working_oetf_encode(
                np.asarray(apply_characteristic_curve(ref_img, (pivot, slope), (pivot, slope), (pivot, slope), d_min=d_min))
            )
            d_out = float(_output_to_density(float(out[0, 0, 0])))
            self.assertAlmostEqual(d_out, target, places=3, msg=f"grade={grade}")

    def test_grade_slope_roundtrip(self):
        for grade in (50.0, 90.0, 115.0, 150.0, 180.0):
            r = 1.4
            self.assertAlmostEqual(slope_to_grade(grade_to_slope(grade, r), r), grade, places=2)


if __name__ == "__main__":
    unittest.main()
