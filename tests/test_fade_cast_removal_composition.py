"""Fade Restoration and Cast Removal on a transparency compose: Cast Removal's neutral-axis
meter reads the film after the fade matrix (features/transparency/processor.py), so it fits
the residual the fade leaves. No conflict guard exists between the two.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.rendering.engine import base_processor, exposure_processor


def _config(fade_ratio_g=1.0, fade_ratio_b=1.0, cast_strength=0.0):
    cfg = DEFAULT_WORKSPACE_CONFIG
    return replace(
        cfg,
        process=replace(
            cfg.process,
            process_mode=ProcessMode.E6,
            fade_ratio_g=fade_ratio_g,
            fade_ratio_b=fade_ratio_b,
        ),
        exposure=replace(cfg.exposure, cast_removal_strength=cast_strength),
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
    norm = base_processor(cfg).process(image, ctx)
    return np.asarray(exposure_processor(cfg).process(norm, ctx)), ctx


def _faded_neutral_wedge(ratio_g=1.0, ratio_b=1.0, seed=5):
    """A neutral wedge (spans the meter's three luma bands) with a known differential fade
    baked in via the forward model, so Cast Removal has a real per-channel residual to meet
    the fade correction partway on."""
    rng = np.random.default_rng(seed)
    t = np.geomspace(5e-4, 0.9, 64 * 64).astype(np.float64).reshape(64, 64)
    log_t = np.log10(t)
    concentration_log = np.stack([log_t, log_t + np.log10(ratio_g), log_t + np.log10(ratio_b)], axis=-1)
    img = np.power(10.0, concentration_log).astype(np.float32)
    return np.ascontiguousarray(img + rng.uniform(0, 1e-4, img.shape).astype(np.float32))


class TestFadeAndCastRemovalCompose(unittest.TestCase):
    def test_both_active_meters_an_axis_and_renders(self):
        """Neither feature disables the other: a non-identity fade and a non-zero Cast
        Removal strength together still meter a neutral axis and produce a finite render."""
        img = _faded_neutral_wedge(0.7, 1.2)
        cfg = _config(fade_ratio_g=0.7, fade_ratio_b=1.2, cast_strength=0.8)
        out, ctx = _render(img, cfg)
        self.assertIn("neutral_axis_refs", ctx.metrics)
        self.assertIsNotNone(ctx.metrics["neutral_axis_refs"])
        self.assertTrue(np.all(np.isfinite(out)))

    def test_cast_removal_solve_changes_when_fade_changes(self):
        """Cast Removal meters downstream of the fade matrix, so its own solved (gain,
        offset) depends on what fade did -- documenting the coupling, not a bug. Same
        capture, same Cast Removal strength, different fade ratios: different refs."""
        img = _faded_neutral_wedge(0.7, 1.2)
        cfg_a = _config(fade_ratio_g=0.7, fade_ratio_b=1.2, cast_strength=0.8)
        cfg_b = _config(fade_ratio_g=1.0, fade_ratio_b=1.0, cast_strength=0.8)
        _out_a, ctx_a = _render(img, cfg_a)
        _out_b, ctx_b = _render(img, cfg_b)
        refs_a, refs_b = ctx_a.metrics["neutral_axis_refs"], ctx_b.metrics["neutral_axis_refs"]
        self.assertIsNotNone(refs_a)
        self.assertIsNotNone(refs_b)
        self.assertNotEqual(refs_a[0], refs_b[0])  # midtone ref moved


if __name__ == "__main__":
    unittest.main()
