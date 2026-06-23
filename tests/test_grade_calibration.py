import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import grade_to_slope
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor


class TestGradeToSlope(unittest.TestCase):
    def test_iso_r_exposure_range(self):
        # ISO R110 -> exposure range 1.1; the straight-line slope is the H&D gamma
        # = grade_contrast_scale * negative density range / paper exposure range.
        scale = EXPOSURE_CONSTANTS["grade_contrast_scale"]
        self.assertAlmostEqual(grade_to_slope(110.0, 1.3), scale * 1.3 / 1.1, places=5)

    def test_slope_scales_linearly_with_range(self):
        # Contrast is proportional to the negative density range at fixed grade.
        self.assertAlmostEqual(grade_to_slope(110.0, 2.0) / grade_to_slope(110.0, 1.0), 2.0, places=5)

    def test_missing_range_uses_typical(self):
        from negpy.features.exposure.logic import default_grade_range

        self.assertAlmostEqual(grade_to_slope(110.0, None), grade_to_slope(110.0, default_grade_range()), places=6)

    def test_lower_r_is_steeper(self):
        self.assertGreater(grade_to_slope(70.0, 1.3), grade_to_slope(130.0, 1.3))

    def test_flat_negative_lowers_slope(self):
        self.assertLess(grade_to_slope(110.0, 0.9), grade_to_slope(110.0, 1.3))

    def test_negative_range_uses_abs(self):
        self.assertAlmostEqual(grade_to_slope(110.0, -1.3), grade_to_slope(110.0, 1.3), places=6)

    def test_clamped_at_extremes(self):
        self.assertAlmostEqual(grade_to_slope(180.0, 0.01), EXPOSURE_CONSTANTS["slope_min"], places=6)
        self.assertAlmostEqual(grade_to_slope(50.0, 10.0), EXPOSURE_CONSTANTS["slope_max"], places=6)

    def test_grade_clamped_to_r_bounds(self):
        # Raw values outside the slider range behave as the nearest bound.
        self.assertAlmostEqual(grade_to_slope(10.0, 1.3), grade_to_slope(50.0, 1.3), places=6)
        self.assertAlmostEqual(grade_to_slope(250.0, 1.3), grade_to_slope(180.0, 1.3), places=6)


class TestLegacyGradeMigration(unittest.TestCase):
    def test_legacy_grade_converted_with_old_ladder(self):
        # Old 0-5 paper grades map via R = 150 - 20*G so saved edits keep their look.
        self.assertAlmostEqual(ExposureConfig(grade=2.5).grade, 100.0, places=6)
        self.assertAlmostEqual(ExposureConfig(grade=0.0).grade, 150.0, places=6)
        self.assertAlmostEqual(ExposureConfig(grade=5.0).grade, 50.0, places=6)

    def test_default_grade_is_iso_r(self):
        self.assertAlmostEqual(ExposureConfig().grade, 115.0, places=6)

    def test_roundtrip_keeps_iso_r_values(self):
        config = WorkspaceConfig(exposure=ExposureConfig(grade=95.0))
        restored = WorkspaceConfig.from_flat_dict(config.to_dict())
        self.assertAlmostEqual(restored.exposure.grade, 95.0, places=6)


class TestDensityRangeMetric(unittest.TestCase):
    def setUp(self):
        self.config = WorkspaceConfig()

    def _context(self):
        return PipelineContext(scale_factor=1.0, original_size=(100, 100), process_mode="C41")

    def test_metric_set_on_local_bounds(self):
        process = replace(self.config.process, local_floors=(-2.0, -1.5, -1.0), local_ceils=(-0.1, -0.3, -0.5))
        ctx = self._context()
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        NormalizationProcessor(process).process(img, ctx)
        # Luminance-weighted (Rec.709) over per-channel ranges 1.9, 1.2, 0.5.
        expected = 0.2126 * 1.9 + 0.7152 * 1.2 + 0.0722 * 0.5
        self.assertAlmostEqual(ctx.metrics["norm_density_range"], expected, places=5)

    def test_metric_set_on_locked_bounds(self):
        process = replace(
            self.config.process,
            use_roll_average=True,
            locked_floors=(-2.2, -2.2, -2.2),
            locked_ceils=(-0.2, -0.2, -0.2),
        )
        ctx = self._context()
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        NormalizationProcessor(process).process(img, ctx)
        self.assertAlmostEqual(ctx.metrics["norm_density_range"], 2.0, places=5)

    def test_metric_set_on_analyzed_bounds(self):
        ctx = self._context()
        img = np.random.default_rng(0).uniform(0.01, 0.9, (32, 32, 3)).astype(np.float32)
        NormalizationProcessor(self.config.process).process(img, ctx)
        self.assertIn("norm_density_range", ctx.metrics)
        self.assertGreater(ctx.metrics["norm_density_range"], 0.0)


class TestCalibratedGradeOutput(unittest.TestCase):
    def setUp(self):
        # Physical (range-coupled) mode: auto contrast/exposure off.
        self.config = WorkspaceConfig(exposure=ExposureConfig(auto_normalize_contrast=False, auto_exposure=False))

    def _run(self, density_range):
        ctx = PipelineContext(scale_factor=1.0, original_size=(100, 100), process_mode="C41")
        if density_range is not None:
            ctx.metrics["norm_density_range"] = density_range
        img = np.full((8, 8, 3), 0.3, dtype=np.float32)
        return PhotometricProcessor(self.config.exposure).process(img, ctx)

    def test_typical_range_matches_no_metric_baseline(self):
        from negpy.features.exposure.logic import default_grade_range

        np.testing.assert_array_almost_equal(self._run(default_grade_range()), self._run(None))

    def test_flat_negative_renders_flatter(self):
        res_flat = self._run(0.9)
        res_ref = self._run(1.3)
        # Value 0.3 is below the pivot (print-bright side); a lower slope pulls
        # the output closer to the pivot's mid density, i.e. darker here.
        self.assertLess(float(res_flat[0, 0, 0]), float(res_ref[0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
