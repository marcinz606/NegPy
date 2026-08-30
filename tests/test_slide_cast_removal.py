"""
Cast Removal on a transparency.

What these pin down:
  - the affine solve is the identity wherever it has nothing to say (green, no axis,
    zero strength, a degenerate axis), so a slide that starts at 0 renders as the
    capture;
  - at full strength it lands a channel's neutral refs on green's;
  - the gain clamp bounds the correction;
  - a slide starts at 0 by every route into E-6: a saved edit, autodetect and the
    mode switch.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.migrations import _SHIPPED_CAST_STRENGTH, migrate_flat_config
from negpy.features.exposure.logic import neutral_axis_affine
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from negpy.features.exposure.transfer import TRANSFER_CONSTANTS, display_rendering
from negpy.features.process.models import ProcessMode, cast_removal_for_mode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _axis(mid, shadow, highlight=None):
    return (mid, shadow, highlight)


class TestAffineSolve(unittest.TestCase):
    UNIT = ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    def test_no_axis_or_no_strength_is_the_identity(self):
        axis = _axis((0.4, 0.5, 0.6), (0.7, 0.8, 0.9))
        self.assertEqual(neutral_axis_affine(None, 1.0), self.UNIT)
        self.assertEqual(neutral_axis_affine(axis, 0.0), self.UNIT)

    def test_green_is_always_the_identity(self):
        gain, offset = neutral_axis_affine(_axis((0.4, 0.5, 0.6), (0.7, 0.8, 0.9)), 1.0)
        self.assertEqual((gain[1], offset[1]), (1.0, 0.0))

    def test_a_neutral_frame_is_left_alone(self):
        gain, offset = neutral_axis_affine(_axis((0.5, 0.5, 0.5), (0.8, 0.8, 0.8)), 1.0)
        for ch in range(3):
            self.assertAlmostEqual(gain[ch], 1.0, places=6)
            self.assertAlmostEqual(offset[ch], 0.0, places=6)

    def test_a_degenerate_axis_is_the_identity(self):
        """Midtone and shadow on top of each other carry no slope to solve for."""
        self.assertEqual(neutral_axis_affine(_axis((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 1.0), self.UNIT)

    def test_full_strength_lands_the_refs_on_green(self):
        mid, shadow = (0.44, 0.50, 0.57), (0.76, 0.80, 0.85)
        gain, offset = neutral_axis_affine(_axis(mid, shadow), 1.0)
        for ch in (0, 2):
            self.assertAlmostEqual(gain[ch] * mid[ch] + offset[ch], mid[1], places=5)
            self.assertAlmostEqual(gain[ch] * shadow[ch] + offset[ch], shadow[1], places=5)

    def test_partial_strength_moves_partway(self):
        """Half strength must correct, but less than full — the slider has to be a dial."""
        mid, shadow = (0.44, 0.50, 0.57), (0.76, 0.80, 0.85)
        full = neutral_axis_affine(_axis(mid, shadow), 1.0)
        half = neutral_axis_affine(_axis(mid, shadow), 0.5)
        for ch in (0, 2):
            err_half = abs(half[0][ch] * mid[ch] + half[1][ch] - mid[1])
            err_off = abs(mid[ch] - mid[1])
            self.assertLess(err_half, err_off)
            self.assertGreater(err_half, abs(full[0][ch] * mid[ch] + full[1][ch] - mid[1]))

    def test_the_deviation_clamp_bounds_a_wild_axis(self):
        """A ref far off green must not drag the channel arbitrarily far."""
        limit = float(EXPOSURE_CONSTANTS["midtone_cast_max_offset"])
        gain, offset = neutral_axis_affine(_axis((0.5, 0.5, 5.0), (0.8, 0.8, -4.0)), 1.0)
        self.assertAlmostEqual(gain[2] * (0.5 + limit) + offset[2], 0.5, places=5)

    def test_the_gain_clamp_holds_when_green_barely_separates(self):
        gain_max = float(EXPOSURE_CONSTANTS["cast_affine_gain_limit"])
        gain, _ = neutral_axis_affine(_axis((0.30, 0.50, 0.70), (0.31, 0.501, 0.69)), 1.0)
        for ch in range(3):
            self.assertLessEqual(gain[ch], gain_max + 1e-6)
            self.assertGreaterEqual(gain[ch], 1.0 / gain_max - 1e-6)


def _slide_config(strength, normalize=False):
    cfg = DEFAULT_WORKSPACE_CONFIG
    return replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=normalize),
        exposure=replace(cfg.exposure, cast_removal_strength=strength),
    )


def _render(image, cfg):
    h, w = image.shape[:2]
    ctx = PipelineContext(
        original_size=(h, w),
        scale_factor=1.0,
        process_mode=cfg.process.process_mode,
        cam_xyz=None,
        camera_wb=None,
        wants_uv_grid=False,
    )
    norm = NormalizationProcessor(cfg.process, cfg.exposure.cast_removal_strength).process(image, ctx)
    return np.asarray(PhotometricProcessor(cfg.exposure, cfg.local, cfg.process).process(norm, ctx)), ctx


def _cast_slide(seed=5, cast=(1.0, 0.82, 0.62)):
    """A neutral wedge under a warm cast. The wedge spans the meter's three luma bands,
    which a narrower one does not — an empty shadow band returns no axis at all."""
    rng = np.random.default_rng(seed)
    v = np.geomspace(5e-4, 0.9, 64 * 64).astype(np.float32).reshape(64, 64)
    img = np.stack([v * cast[0], v * cast[1], v * cast[2]], axis=-1)
    return np.ascontiguousarray(img + rng.uniform(0, 1e-4, img.shape).astype(np.float32))


class TestSlideRender(unittest.TestCase):
    def test_off_is_the_untouched_capture(self):
        """The identity the whole transfer path rests on has to survive the new term."""
        img = _cast_slide()
        out, ctx = _render(img, _slide_config(0.0))
        gain = np.float32(2.0 ** float(TRANSFER_CONSTANTS["transfer_baseline_ev"]))
        expected = np.asarray(display_rendering(img * gain))
        rel = np.abs(out - expected) / np.maximum(expected, 1e-9)
        self.assertLess(float(rel.max()), 1e-4)
        # Nothing metered, so nothing to publish.
        self.assertNotIn("neutral_axis_refs", ctx.metrics)

    def test_on_meters_an_axis_and_neutralizes_the_cast(self):
        img = _cast_slide()
        off, _ = _render(img, _slide_config(0.0))
        on, ctx = _render(img, _slide_config(1.0))
        self.assertIsNotNone(ctx.metrics.get("neutral_axis_refs"))

        def spread(a):
            return float(np.mean(np.abs(a[:, :, 0] - a[:, :, 2])))

        self.assertLess(spread(on), spread(off))

    def test_normalize_on_also_meters_an_axis(self):
        """The print-curve slide path shares the negative's solve, so it needs the meter."""
        _, ctx = _render(_cast_slide(), _slide_config(1.0, normalize=True))
        self.assertIsNotNone(ctx.metrics.get("neutral_axis_refs"))
        # The P98 shadow tie stays negative-only.
        self.assertNotIn("shadow_log_refs", ctx.metrics)

    def test_bw_never_meters_an_axis(self):
        cfg = _slide_config(1.0, normalize=True)
        cfg = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
        _, ctx = _render(_cast_slide(), cfg)
        self.assertNotIn("neutral_axis_refs", ctx.metrics)


class TestSlideStartsOff(unittest.TestCase):
    def test_the_mirrored_default_matches_the_dataclass(self):
        self.assertEqual(_SHIPPED_CAST_STRENGTH, float(ExposureConfig.cast_removal_strength))

    def test_a_saved_slide_at_the_shipped_default_loads_off(self):
        data = migrate_flat_config({"process_mode": "Transparency", "cast_removal_strength": _SHIPPED_CAST_STRENGTH})
        self.assertEqual(data["cast_removal_strength"], 0.0)

    def test_a_saved_slide_with_a_chosen_value_is_left_alone(self):
        data = migrate_flat_config({"process_mode": "Transparency", "cast_removal_strength": 0.8})
        self.assertEqual(data["cast_removal_strength"], 0.8)

    def test_a_saved_negative_is_untouched(self):
        data = migrate_flat_config({"process_mode": "Color Negative", "cast_removal_strength": _SHIPPED_CAST_STRENGTH})
        self.assertEqual(data["cast_removal_strength"], _SHIPPED_CAST_STRENGTH)

    def test_a_legacy_slide_mode_name_migrates_too(self):
        data = migrate_flat_config({"process_mode": "E-6", "cast_removal_strength": _SHIPPED_CAST_STRENGTH})
        self.assertEqual(data["cast_removal_strength"], 0.0)

    def test_the_mode_switch_swaps_the_two_defaults(self):
        default = float(ExposureConfig.cast_removal_strength)
        self.assertEqual(cast_removal_for_mode(ProcessMode.E6, default), 0.0)
        self.assertEqual(cast_removal_for_mode(ProcessMode.C41, 0.0), default)

    def test_the_mode_switch_keeps_a_chosen_strength(self):
        self.assertEqual(cast_removal_for_mode(ProcessMode.E6, 0.8), 0.8)
        self.assertEqual(cast_removal_for_mode(ProcessMode.C41, 0.8), 0.8)


if __name__ == "__main__":
    unittest.main()
