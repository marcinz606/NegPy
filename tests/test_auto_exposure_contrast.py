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
from negpy.features.process.models import ProcessMode


def _context(density_range):
    ctx = PipelineContext(scale_factor=1.0, original_size=(100, 100), process_mode=ProcessMode.C41)
    ctx.metrics["norm_density_range"] = density_range
    return ctx


class TestAutoNormalizeContrast(unittest.TestCase):
    """Auto Grade: paper grade follows the negative's textural density scale, partially."""

    def _run(self, exposure, density_range, textural=None):
        ctx = _context(density_range)
        if textural is not None:
            ctx.metrics["textural_range"] = textural
        img = np.full((8, 8, 3), 0.4, dtype=np.float32)
        return PhotometricProcessor(exposure).process(img, ctx)

    def test_wider_textural_scale_prints_softer(self):
        # Same normalized tone, same bounds: the frame whose detail spans more density
        # gets a softer grade (Ilford/Kodak: ISO R follows the negative density range).
        exp = ExposureConfig(auto_normalize_contrast=True)
        contrasty = self._run(exp, 1.6, textural=1.3)
        normal = self._run(exp, 1.6, textural=0.9)
        self.assertFalse(np.allclose(contrasty, normal))

    def test_no_textural_falls_back_to_fixed_reference(self):
        exp = ExposureConfig(auto_normalize_contrast=True)
        on = self._run(exp, 2.4)
        ref_exp = ExposureConfig(auto_normalize_contrast=False)
        ref = self._run(ref_exp, None)
        np.testing.assert_array_almost_equal(on, ref)

    def test_off_still_tracks_range(self):
        exp = ExposureConfig(auto_normalize_contrast=False)
        dense = self._run(exp, 2.4)
        flat = self._run(exp, 0.7)
        self.assertFalse(np.allclose(dense, flat))


class TestEffectiveGradeRange(unittest.TestCase):
    """
    effective = K * lum_range * ((1 - s) + s * nominal / textural): the paper gamma is
    shrunk toward the population norm (Alkofer, US 4,731,671), so the printed textural
    range is a blend of the frame's own and a normal negative's.
    """

    def _with(self, **over):
        import contextlib

        @contextlib.contextmanager
        def cm():
            orig = {k: EXPOSURE_CONSTANTS[k] for k in over}
            EXPOSURE_CONSTANTS.update(over)
            try:
                yield
            finally:
                EXPOSURE_CONSTANTS.update(orig)

        return cm()

    def test_physical_returns_floor_ceil(self):
        self.assertEqual(effective_grade_range(False, 1.7, 0.9), 1.7)
        self.assertIsNone(effective_grade_range(False, None, 0.9))

    def test_auto_blends_toward_population_norm(self):
        k = EXPOSURE_CONSTANTS["auto_grade_target"]
        nominal = EXPOSURE_CONSTANTS["auto_grade_nominal_range"]
        s = EXPOSURE_CONSTANTS["auto_grade_strength"]
        expected = k * 1.6 * ((1.0 - s) + s * nominal / 1.2)
        self.assertAlmostEqual(effective_grade_range(True, 1.6, 1.2), expected, places=6)

    def test_normal_negative_is_target_times_range(self):
        nominal = EXPOSURE_CONSTANTS["auto_grade_nominal_range"]
        k = EXPOSURE_CONSTANTS["auto_grade_target"]
        self.assertAlmostEqual(effective_grade_range(True, 1.8, nominal), k * 1.8, places=6)

    def test_strength_zero_is_a_fixed_paper(self):
        # The grade tracks the negative exactly like Auto Grade off, scaled by K.
        with self._with(auto_grade_strength=0.0):
            k = EXPOSURE_CONSTANTS["auto_grade_target"]
            self.assertAlmostEqual(effective_grade_range(True, 2.4, 0.6), k * 2.4, places=6)
            self.assertAlmostEqual(effective_grade_range(True, 2.4, 1.0), k * 2.4, places=6)

    def test_strength_one_prints_every_textural_range_alike(self):
        # Full normalization: slope * textural / lum_range is the same for every frame.
        with self._with(auto_grade_strength=1.0):
            a = effective_grade_range(True, 1.6, 0.8)
            b = effective_grade_range(True, 2.4, 1.5)
            self.assertAlmostEqual(a * 0.8 / 1.6, b * 1.5 / 2.4, places=6)

    def test_wider_textural_softens_narrower_hardens(self):
        normal = effective_grade_range(True, 1.6, 0.9)
        self.assertLess(effective_grade_range(True, 1.6, 1.3), normal)
        self.assertGreater(effective_grade_range(True, 1.6, 0.5), normal)

    def test_overfill_is_capped(self):
        # A textural range far past the norm (a rebate-polluted meter, an extreme scene)
        # cannot print past auto_grade_max_overfill of a normal negative's print span.
        k = EXPOSURE_CONSTANTS["auto_grade_target"]
        nominal = EXPOSURE_CONSTANTS["auto_grade_nominal_range"]
        cap = EXPOSURE_CONSTANTS["auto_grade_max_overfill"]
        self.assertAlmostEqual(effective_grade_range(True, 2.4, 2.0), k * 2.4 * (nominal / 2.0) * cap, places=6)

    def test_auto_degenerate_flat_is_capped(self):
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

    def test_statistic_is_mean_and_midpoint_of_trimmed_window(self):
        # Boyack & Juenger (US 5,724,456): place the average of the trimmed window's mean
        # and its midpoint, not the median. Columns 0.2/0.8/0.2/0.5 normalized: median
        # 0.35, mean 0.425, P5-P95 midpoint 0.5 -> 0.4625.
        assumed = EXPOSURE_CONSTANTS["assumed_anchor"]
        strength = EXPOSURE_CONSTANTS["anchor_meter_strength"]
        cols = np.array([-1.6, -0.4, -1.6, -1.0], dtype=np.float32)[np.arange(64) % 4]
        img = np.repeat(cols[None, :, None], 64, axis=0).repeat(3, axis=2)
        self.assertAlmostEqual(measure_anchor_from_log(img, self.BOUNDS), assumed + strength * (0.4625 - assumed), places=4)

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


def _bordered_texture(levels, border, size=128, inner=96):
    """Textured interior (columns cycling over `levels`) inside a flat border, like a
    frame inside its rebate. Cells are pixels here: no block median runs at this size,
    and the interior sits on the gate's sector grid."""
    img = np.full((size, size, 3), border, dtype=np.float32)
    o = (size - inner) // 2
    cols = np.array(levels, dtype=np.float32)[np.arange(inner) % len(levels)]
    img[o : o + inner, o : o + inner, :] = cols[None, :, None]
    return img


class TestActivityGate(unittest.TestCase):
    """Both meters read textured cells only, so a flat rebate, border or sky does not vote."""

    BOUNDS = LogNegativeBounds(floors=(-2.0, -2.0, -2.0), ceils=(0.0, 0.0, 0.0))

    def test_textural_range_ignores_flat_border(self):
        img = _bordered_texture([-1.5, -1.0, -0.5], border=-0.1)
        self.assertAlmostEqual(measure_textural_range_from_log(img), 1.0, places=2)

    def test_anchor_ignores_flat_border(self):
        assumed = EXPOSURE_CONSTANTS["assumed_anchor"]
        strength = EXPOSURE_CONSTANTS["anchor_meter_strength"]
        img = _bordered_texture([-1.5, -1.0, -0.5], border=-0.1)
        # Interior normalizes to 0.25/0.5/0.75; the border alone would read 0.95.
        self.assertAlmostEqual(measure_anchor_from_log(img, self.BOUNDS), assumed + strength * (0.5 - assumed), places=2)

    def test_flat_frame_falls_back_to_every_cell(self):
        img = np.full((128, 128, 3), -1.2, dtype=np.float32)
        assumed = EXPOSURE_CONSTANTS["assumed_anchor"]
        strength = EXPOSURE_CONSTANTS["anchor_meter_strength"]
        self.assertAlmostEqual(measure_anchor_from_log(img, self.BOUNDS), assumed + strength * (0.4 - assumed), places=4)
        self.assertAlmostEqual(measure_textural_range_from_log(img), 0.0, places=5)


class TestAutoTogglesAcrossModes(unittest.TestCase):
    """The toggles must render valid output in every process mode (CPU path)."""

    def _render(self, mode, exposure, normalize=True):
        from dataclasses import replace

        from negpy.domain.models import WorkspaceConfig
        from negpy.features.process.models import ProcessConfig
        from negpy.services.rendering.engine import DarkroomEngine

        settings = replace(
            WorkspaceConfig(),
            # Transparency defaults Normalize off, which is the transfer path — pinned on
            # here so this stays a test of the print path in all three modes.
            process=replace(ProcessConfig(), process_mode=mode, e6_normalize=normalize),
            exposure=exposure,
        )
        img = np.random.default_rng(7).uniform(0.02, 0.9, (48, 48, 3)).astype(np.float32)
        return DarkroomEngine().process(img, settings, f"mode_{mode}_{normalize}")

    def test_valid_and_active_in_each_mode(self):
        for mode in ProcessMode:
            base = self._render(mode, ExposureConfig(auto_exposure=False, auto_normalize_contrast=False))
            auto = self._render(mode, ExposureConfig(auto_exposure=True, auto_normalize_contrast=True))
            self.assertTrue(np.all(np.isfinite(auto)), mode)
            self.assertGreaterEqual(float(auto.min()), 0.0, mode)
            self.assertLessEqual(float(auto.max()), 1.0, mode)
            # The toggles must actually change the render.
            self.assertFalse(np.allclose(base, auto), mode)

    def test_inert_on_the_transparency_transfer(self):
        """The complement: with Normalize off there is no metered stretch to grade against,
        so both toggles must be no-ops — which is why the sidebar hides them there."""
        base = self._render(ProcessMode.E6, ExposureConfig(auto_exposure=False, auto_normalize_contrast=False), normalize=False)
        auto = self._render(ProcessMode.E6, ExposureConfig(auto_exposure=True, auto_normalize_contrast=True), normalize=False)
        self.assertTrue(np.allclose(base, auto))


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
