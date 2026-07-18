import unittest

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure import models as exposure_models
from negpy.features.exposure.logic import effective_grade_range
from negpy.features.exposure.models import (
    DEFAULT_TARGETS,
    EXPOSURE_CONSTANTS,
    TUNABLE_TARGETS,
    apply_targets,
)
from negpy.features.exposure.normalization import LogNegativeBounds, measure_anchor_from_log
from negpy.services.rendering.engine import DarkroomEngine


class TargetsTestCase(unittest.TestCase):
    """EXPOSURE_CONSTANTS is a process-global; every test must put it back."""

    def tearDown(self) -> None:
        apply_targets(DEFAULT_TARGETS)


class TestApplyTargets(TargetsTestCase):
    def test_writes_known_keys_and_bumps_revision(self):
        before = exposure_models.TARGETS_REVISION
        apply_targets({"auto_grade_target": 0.7})
        self.assertEqual(EXPOSURE_CONSTANTS["auto_grade_target"], 0.7)
        self.assertGreater(exposure_models.TARGETS_REVISION, before)

    def test_ignores_unknown_keys(self):
        apply_targets({"not_a_target": 123.0, "d_max": 9.9})
        self.assertNotIn("not_a_target", EXPOSURE_CONSTANTS)
        self.assertEqual(EXPOSURE_CONSTANTS["d_max"], 2.3)

    def test_defaults_match_shipped_constants(self):
        self.assertEqual(set(DEFAULT_TARGETS), set(TUNABLE_TARGETS))
        for key, value in DEFAULT_TARGETS.items():
            self.assertEqual(EXPOSURE_CONSTANTS[key], value)

    def test_defaults_sit_inside_their_slider_range(self):
        for key, (lo, hi) in TUNABLE_TARGETS.items():
            self.assertGreaterEqual(DEFAULT_TARGETS[key], lo, key)
            self.assertLessEqual(DEFAULT_TARGETS[key], hi, key)


class TestGradeTargetsTakeEffect(TargetsTestCase):
    def test_contrast_target_scales_the_range(self):
        base = float(effective_grade_range(True, 1.6, 0.8) or 0.0)
        apply_targets({"auto_grade_target": DEFAULT_TARGETS["auto_grade_target"] * 2})
        self.assertAlmostEqual(float(effective_grade_range(True, 1.6, 0.8) or 0.0), base * 2, places=6)

    def test_adaptation_strength_zero_ignores_the_scene(self):
        apply_targets({"auto_grade_strength": 0.0})
        flat = float(effective_grade_range(True, 1.2, 1.0) or 0.0)
        dense = float(effective_grade_range(True, 2.8, 0.6) or 0.0)
        self.assertAlmostEqual(flat, dense, places=6)


class TestDensityTargetsTakeEffect(TargetsTestCase):
    def _anchor(self) -> float:
        # A ramp so the metered median sits well away from assumed_anchor.
        img_log = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64, 1)
        img_log = np.repeat(img_log, 3, axis=2)
        bounds = LogNegativeBounds(floors=(0.0, 0.0, 0.0), ceils=(1.0, 1.0, 1.0))
        return measure_anchor_from_log(img_log, bounds)

    def test_metering_strength_zero_pins_the_assumed_key(self):
        apply_targets({"anchor_meter_strength": 0.0})
        self.assertAlmostEqual(self._anchor(), EXPOSURE_CONSTANTS["assumed_anchor"], places=6)

    def test_metering_strength_moves_the_anchor(self):
        apply_targets({"anchor_meter_strength": 0.0})
        pinned = self._anchor()
        apply_targets({"anchor_meter_strength": 1.0})
        self.assertNotAlmostEqual(self._anchor(), pinned, places=4)

    def test_metering_band_still_clamps(self):
        band = 0.02
        apply_targets({"anchor_meter_strength": 1.0, "anchor_meter_band": band})
        assumed = float(EXPOSURE_CONSTANTS["assumed_anchor"])
        self.assertLessEqual(abs(self._anchor() - assumed), band + 1e-6)


class TestRenderCacheInvalidation(TargetsTestCase):
    """Retuned targets aren't in any config hash — the render must still refresh."""

    def test_engine_rerenders_after_apply_targets(self):
        engine = DarkroomEngine()
        rng = np.random.default_rng(0)
        img = rng.random((64, 64, 3), dtype=np.float32) * 0.5 + 0.25
        settings = WorkspaceConfig()

        first = engine.process(img, settings, source_hash="targets")
        cached = engine.process(img, settings, source_hash="targets")
        np.testing.assert_array_equal(first, cached)

        apply_targets({"anchor_target_density": DEFAULT_TARGETS["anchor_target_density"] + 0.25})
        retuned = engine.process(img, settings, source_hash="targets")
        self.assertFalse(np.array_equal(cached, retuned))


if __name__ == "__main__":
    unittest.main()
