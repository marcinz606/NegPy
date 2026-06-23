import unittest

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.logic import (
    CharacteristicCurve,
    compute_pivot,
    effective_grade_range,
    grade_to_slope,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    measure_anchor_from_log,
    measure_textural_range_from_log,
)
from negpy.features.exposure.processor import PhotometricProcessor


def _context(density_range):
    ctx = PipelineContext(scale_factor=1.0, original_size=(100, 100), process_mode="C41")
    ctx.metrics["norm_density_range"] = density_range
    return ctx


class TestAutoNormalizeContrast(unittest.TestCase):
    """Auto contrast: the curve must be image-independent (range ignored)."""

    def _run(self, exposure, density_range):
        ctx = _context(density_range)
        img = np.full((8, 8, 3), 0.4, dtype=np.float32)
        return PhotometricProcessor(exposure).process(img, ctx)

    def test_slope_independent_of_range_when_on(self):
        # Dense (large range) and flat (small range) negatives must get the
        # same slope when auto contrast is on — that is the whole point.
        exp = ExposureConfig(auto_normalize_contrast=True)
        dense = self._run(exp, 2.4)
        flat = self._run(exp, 0.7)
        np.testing.assert_array_almost_equal(dense, flat)

    def test_matches_fixed_reference_slope(self):
        exp = ExposureConfig(auto_normalize_contrast=True)
        on = self._run(exp, 2.4)
        # Fixed-reference slope == grade_to_slope(grade, None).
        ref_exp = ExposureConfig(auto_normalize_contrast=False)
        ref = self._run(ref_exp, None)
        np.testing.assert_array_almost_equal(on, ref)

    def test_off_still_tracks_range(self):
        exp = ExposureConfig(auto_normalize_contrast=False)
        dense = self._run(exp, 2.4)
        flat = self._run(exp, 0.7)
        self.assertFalse(np.allclose(dense, flat))


class TestEffectiveGradeRange(unittest.TestCase):
    def test_physical_returns_floor_ceil(self):
        self.assertEqual(effective_grade_range(False, 1.7, 0.9), 1.7)
        self.assertIsNone(effective_grade_range(False, None, 0.9))

    def test_auto_damped_printed_contrast(self):
        # effective = K * (nominal + strength * (ratio - nominal)).
        k = EXPOSURE_CONSTANTS["auto_grade_target"]
        nominal = EXPOSURE_CONSTANTS["auto_grade_nominal_ratio"]
        strength = EXPOSURE_CONSTANTS["auto_grade_strength"]
        ratio = 1.6 / 0.8
        expected = k * (nominal + strength * (ratio - nominal))
        self.assertAlmostEqual(effective_grade_range(True, 1.6, 0.8), expected, places=6)

    def test_auto_strength_zero_is_fixed(self):
        # strength 0 collapses to the nominal default regardless of ratio.
        import negpy.features.exposure.logic as logic_mod

        orig = EXPOSURE_CONSTANTS["auto_grade_strength"]
        EXPOSURE_CONSTANTS["auto_grade_strength"] = 0.0
        try:
            self.assertAlmostEqual(effective_grade_range(True, 2.4, 0.6), logic_mod.default_grade_range(), places=6)
        finally:
            EXPOSURE_CONSTANTS["auto_grade_strength"] = orig

    def test_auto_constant_for_constant_ratio(self):
        # A normal frame's floor_ceil/textural ratio is what sets contrast, not
        # the absolute range: same ratio -> same effective range (no swing).
        a = effective_grade_range(True, 1.6, 0.8)  # ratio 2.0
        b = effective_grade_range(True, 2.4, 1.2)  # ratio 2.0
        self.assertAlmostEqual(a, b, places=6)

    def test_auto_speculars_boost_not_soften(self):
        # Speculars inflate floor_ceil while textural stays put -> higher effective
        # range (more slope), recovering compressed midtones instead of softening.
        clean = effective_grade_range(True, 1.6, 0.8)
        specular = effective_grade_range(True, 2.4, 0.8)
        self.assertGreater(specular, clean)

    def test_auto_degenerate_flat_is_capped(self):
        # Near-zero textural can't divide to infinity; capped for the slope clamp.
        self.assertLessEqual(effective_grade_range(True, 1.6, 0.0), 3.5 + 1e-6)

    def test_auto_no_textural_falls_back_to_default(self):
        from negpy.features.exposure.logic import default_grade_range

        self.assertAlmostEqual(effective_grade_range(True, 2.4, None), default_grade_range(), places=6)


class TestMeasureTexturalRange(unittest.TestCase):
    def test_uniform_image_is_zero(self):
        img_log = np.full((16, 16, 3), -1.0, dtype=np.float32)
        self.assertAlmostEqual(measure_textural_range_from_log(img_log), 0.0, places=5)

    def test_tracks_spread(self):
        # Half the pixels at log -1.5, half at -0.5 → P10..P90 spans ~1.0.
        col = np.where(np.arange(64) < 32, -1.5, -0.5).astype(np.float32)
        img_log = np.repeat(col[None, :, None], 64, axis=0).repeat(3, axis=2)
        rng = measure_textural_range_from_log(img_log)
        self.assertAlmostEqual(rng, 1.0, places=2)

    def test_positive_for_reversed_e6_style(self):
        # Inverted densities must still yield a positive span.
        col = np.where(np.arange(64) < 32, -0.3, -1.7).astype(np.float32)
        img_log = np.repeat(col[None, :, None], 64, axis=0).repeat(3, axis=2)
        self.assertGreater(measure_textural_range_from_log(img_log), 0.0)


class TestMeasureAnchor(unittest.TestCase):
    BOUNDS = LogNegativeBounds(floors=(-2.0, -2.0, -2.0), ceils=(0.0, 0.0, 0.0))

    def _measure(self, log_val):
        img_log = np.full((16, 16, 3), log_val, dtype=np.float32)
        return measure_anchor_from_log(img_log, self.BOUNDS)

    def test_tracks_midtone_partial(self):
        # Linear partial metering: correction = strength * (norm - assumed).
        assumed = EXPOSURE_CONSTANTS["assumed_anchor"]
        strength = EXPOSURE_CONSTANTS["anchor_meter_strength"]

        def expected(norm):
            return assumed + strength * (norm - assumed)

        self.assertAlmostEqual(self._measure(-1.2), expected(0.4), places=4)  # within band
        self.assertAlmostEqual(self._measure(-0.9), expected(0.55), places=4)  # within band
        self.assertNotAlmostEqual(self._measure(-1.2), self._measure(-0.9), places=3)

    def test_partial_preserves_key(self):
        # A low-key (dark) frame's anchor leans dark but is pulled toward assumed
        # (not all the way to the raw median), by a fixed fraction of the distance.
        assumed = EXPOSURE_CONSTANTS["assumed_anchor"]
        strength = EXPOSURE_CONSTANTS["anchor_meter_strength"]
        norm = 0.4
        low = self._measure(-1.2)  # raw norm 0.4 < assumed
        self.assertAlmostEqual(low - assumed, strength * (norm - assumed), places=5)

    def test_clamped_to_band(self):
        band = EXPOSURE_CONSTANTS["anchor_meter_band"]
        assumed = EXPOSURE_CONSTANTS["assumed_anchor"]
        # Extreme frames stay within assumed +/- band (hard safety clamp); the
        # gentle linear pull keeps the common case well inside the band.
        hi = self._measure(-0.02)  # norm ~0.99
        lo = self._measure(-1.98)  # norm ~0.01
        self.assertGreater(hi, assumed)
        self.assertLess(lo, assumed)
        self.assertLessEqual(hi, assumed + band + 1e-6)
        self.assertGreaterEqual(lo, assumed - band - 1e-6)

    def test_e6_reversed_bounds(self):
        # E6 normalizes with floors > ceils; anchor must stay finite and in band.
        bounds = LogNegativeBounds(floors=(0.0, 0.0, 0.0), ceils=(-2.0, -2.0, -2.0))
        img_log = np.full((16, 16, 3), -1.0, dtype=np.float32)
        a = measure_anchor_from_log(img_log, bounds)
        band = EXPOSURE_CONSTANTS["anchor_meter_band"]
        assumed = EXPOSURE_CONSTANTS["assumed_anchor"]
        self.assertTrue(assumed - band - 1e-6 <= a <= assumed + band + 1e-6)


class TestAutoTogglesAcrossModes(unittest.TestCase):
    """The toggles must render valid output in C41, B&W and E6 (CPU path)."""

    def _render(self, mode, exposure):
        from dataclasses import replace

        from negpy.domain.models import WorkspaceConfig
        from negpy.features.process.models import ProcessConfig, ProcessMode
        from negpy.services.rendering.engine import DarkroomEngine

        mode_enum = {"C41": ProcessMode.C41, "BW": ProcessMode.BW, "E6": ProcessMode.E6}[mode]
        settings = replace(
            WorkspaceConfig(),
            process=replace(ProcessConfig(), process_mode=mode_enum),
            exposure=exposure,
        )
        img = np.random.default_rng(7).uniform(0.02, 0.9, (48, 48, 3)).astype(np.float32)
        return DarkroomEngine().process(img, settings, f"mode_{mode}")

    def test_valid_and_active_in_each_mode(self):
        for mode in ("C41", "BW", "E6"):
            base = self._render(mode, ExposureConfig(auto_exposure=False, auto_normalize_contrast=False))
            auto = self._render(mode, ExposureConfig(auto_exposure=True, auto_normalize_contrast=True))
            self.assertTrue(np.all(np.isfinite(auto)), mode)
            self.assertGreaterEqual(float(auto.min()), 0.0, mode)
            self.assertLessEqual(float(auto.max()), 1.0, mode)
            # The toggles must actually change the render.
            self.assertFalse(np.allclose(base, auto), mode)


class TestAnchorPivotRoundTrip(unittest.TestCase):
    def test_metered_anchor_prints_at_target(self):
        # compute_pivot must place the curve so the anchor tone prints at
        # anchor_target_density (density slider neutral, no paper Dmin).
        target = EXPOSURE_CONSTANTS["anchor_target_density"]
        for anchor in (0.40, 0.46, 0.55):
            slope = grade_to_slope(115.0, 1.3)
            pivot = compute_pivot(slope, density=1.0, d_min=0.0, anchor=anchor)
            curve = CharacteristicCurve(contrast=slope, pivot=pivot)
            printed = float(curve(np.array([[anchor]], dtype=np.float32))[0, 0])
            self.assertAlmostEqual(printed, target, places=3)


if __name__ == "__main__":
    unittest.main()
