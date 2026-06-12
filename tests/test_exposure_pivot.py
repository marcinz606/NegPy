import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import _expit, compute_pivot, grade_to_slope
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.processor import PhotometricProcessor


def _srgb_oetf(t: float) -> float:
    if t <= 0.0031308:
        return 12.92 * t
    return 1.055 * t ** (1.0 / 2.4) - 0.055


class TestComputePivot(unittest.TestCase):
    def test_reference_tone_prints_at_target(self):
        """The assumed reference tone must land exactly on anchor_target_density."""
        asym = EXPOSURE_CONSTANTS["curve_asymptote"]
        nu = EXPOSURE_CONSTANTS["paper_toe_nu"]
        x_ref = EXPOSURE_CONSTANTS["assumed_anchor"]
        slope = grade_to_slope(110.0, 1.3)
        pivot = compute_pivot(slope, density=1.0)
        density_out = asym * _expit(slope * (x_ref - pivot)) ** nu
        self.assertAlmostEqual(density_out, EXPOSURE_CONSTANTS["anchor_target_density"], places=5)

    def test_reference_tone_prints_at_target_with_dmin(self):
        """The Dmin floor must not shift the reference tone off target."""
        asym, d_min = EXPOSURE_CONSTANTS["curve_asymptote"], EXPOSURE_CONSTANTS["d_min"]
        nu = EXPOSURE_CONSTANTS["paper_toe_nu"]
        x_ref = EXPOSURE_CONSTANTS["assumed_anchor"]
        slope = grade_to_slope(110.0, 1.3)
        pivot = compute_pivot(slope, density=1.0, d_min=d_min)
        density_out = d_min + (asym - d_min) * _expit(slope * (x_ref - pivot)) ** nu
        self.assertAlmostEqual(density_out, EXPOSURE_CONSTANTS["anchor_target_density"], places=5)

    def test_grade_does_not_shift_reference_tone(self):
        """Grade changes rotate around the assumed reference tone."""
        asym = EXPOSURE_CONSTANTS["curve_asymptote"]
        x_ref = EXPOSURE_CONSTANTS["assumed_anchor"]
        outputs = []
        for grade in (160.0, 110.0, 60.0):
            slope = grade_to_slope(grade, 1.3)
            pivot = compute_pivot(slope, density=1.0)
            outputs.append(asym * _expit(slope * (x_ref - pivot)))
        self.assertAlmostEqual(outputs[0], outputs[1], places=5)
        self.assertAlmostEqual(outputs[1], outputs[2], places=5)

    def test_density_slider_shifts_exposure(self):
        """Higher density = lower pivot = denser (darker) print."""
        slope = grade_to_slope(110.0, 1.3)
        p_light = compute_pivot(slope, density=0.5)
        p_dark = compute_pivot(slope, density=1.5)
        self.assertGreater(p_light, p_dark)

    def test_pivot_is_deterministic(self):
        """Same sliders -> same pivot, regardless of image content (no metering)."""
        slope = grade_to_slope(115.0, 1.3)
        self.assertEqual(compute_pivot(slope, density=1.0), compute_pivot(slope, density=1.0))


class TestEndToEndExposure(unittest.TestCase):
    def test_reference_pixel_prints_at_target_brightness(self):
        """Full CPU path: the assumed reference tone must come out at the
        target print density regardless of grade."""
        config = WorkspaceConfig().exposure
        ctx = PipelineContext(scale_factor=1.0, original_size=(8, 8), process_mode="C41")
        ctx.metrics["norm_density_range"] = 1.3

        x_ref = EXPOSURE_CONSTANTS["assumed_anchor"]
        img = np.full((8, 8, 3), x_ref, dtype=np.float32)
        expected = _srgb_oetf(10.0 ** -EXPOSURE_CONSTANTS["anchor_target_density"])

        for grade in (130.0, 110.0, 70.0):
            res = PhotometricProcessor(replace(config, grade=grade)).process(img, ctx)
            self.assertAlmostEqual(float(res[0, 0, 0]), expected, places=4, msg=f"grade={grade}")

    def test_skewed_negative_reaches_paper_black(self):
        """Regression: film-toe-compressed shadows must still print near paper
        black — a symmetric L=d_max curve starved them at ~0.17 sRGB."""
        config = WorkspaceConfig().exposure
        ctx = PipelineContext(scale_factor=1.0, original_size=(8, 8), process_mode="C41")
        ctx.metrics["norm_density_range"] = 1.6

        img = np.full((8, 8, 3), 1.0, dtype=np.float32)  # deepest measured shadow
        res = PhotometricProcessor(config).process(img, ctx)
        self.assertLessEqual(float(res[0, 0, 0]), 0.16)


if __name__ == "__main__":
    unittest.main()
