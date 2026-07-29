"""Density-space Vibrance/anti-vibrance and the per-pixel-modulated Dye Mute
toggle: identity/no-op, real effect in both directions, B&W inertness, and
the Option A equivalence for density_damping_spatial (per-pixel spread
replaces, rather than layers with, #667's uniform grade_saturation_damping)."""

import numpy as np
from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import grade_saturation_damping
from negpy.services.rendering.engine import DarkroomEngine
from negpy.services.rendering.image_processor import PipelineContext


def _render(overrides):
    engine = DarkroomEngine()
    rng = np.random.default_rng(0)
    img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
    settings = WorkspaceConfig.from_flat_dict({"paper_profile": "neutral", **overrides})
    return engine.process(img, settings, source_hash="density-vibrance-test")


class TestDensityVibranceRender:
    def test_identity_is_exact_noop(self):
        baseline = _render({})
        identity = _render({"density_vibrance": 0.0})
        np.testing.assert_array_equal(baseline, identity)

    def test_boost_changes_output(self):
        baseline = _render({})
        boosted = _render({"density_vibrance": 0.5})
        assert not np.array_equal(baseline, boosted)

    def test_compress_changes_output(self):
        baseline = _render({})
        compressed = _render({"density_vibrance": -0.3})
        assert not np.array_equal(baseline, compressed)

    def test_boost_and_compress_are_not_a_simple_sign_flip(self):
        """Boost targets muted pixels, compress targets vivid pixels -- different
        populations, so equal-magnitude boost/compress must not be mirror images."""
        boosted = _render({"density_vibrance": 0.4})
        compressed = _render({"density_vibrance": -0.4})
        assert not np.array_equal(boosted, compressed)

    def test_bw_mode_is_inert(self):
        bw_off = _render({"process_mode": "B&W", "density_vibrance": 0.0})
        bw_on = _render({"process_mode": "B&W", "density_vibrance": 0.6})
        np.testing.assert_allclose(bw_off, bw_on, atol=1e-6)


class TestDensityDampingSpatial:
    def test_off_by_default_matches_uniform_damping_bit_exact(self):
        """density_damping_spatial=False (default) must be byte-identical to
        manually pre-multiplying density_saturation by grade_saturation_damping
        -- #667's own uniform damping path, untouched. This is the actual
        proof that Option A only changes behavior when spatial is explicitly on."""
        engine = DarkroomEngine()
        rng = np.random.default_rng(2)
        img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
        base = WorkspaceConfig.from_flat_dict({"paper_profile": "kodak_endura", "grade": 90.0})

        with_damp = replace(base, exposure=replace(base.exposure, density_saturation=1.0, density_saturation_damping=0.3))
        ctx1 = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=with_damp.process.process_mode)
        out1 = engine.process(img, with_damp, source_hash="spatial-off-test", context=ctx1)
        slope_g = ctx1.metrics["print_slopes"][1]
        damp = grade_saturation_damping(slope_g, 0.3)

        manual = replace(base, exposure=replace(base.exposure, density_saturation=1.0 * damp, density_saturation_damping=0.0))
        ctx2 = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=manual.process.process_mode)
        out2 = engine.process(img, manual, source_hash="spatial-off-test", context=ctx2)

        np.testing.assert_array_equal(out1, out2)

    def test_spatial_on_changes_output_vs_uniform(self):
        uniform = _render({"paper_profile": "kodak_endura", "density_saturation_damping": 0.3, "density_damping_spatial": False})
        spatial = _render({"paper_profile": "kodak_endura", "density_saturation_damping": 0.3, "density_damping_spatial": True})
        assert not np.array_equal(uniform, spatial)

    def test_spatial_on_is_inert_at_zero_strength(self):
        baseline = _render({"density_damping_spatial": False})
        spatial_zero = _render({"density_damping_spatial": True, "density_saturation_damping": 0.0})
        np.testing.assert_array_equal(baseline, spatial_zero)

    def test_bw_mode_is_inert(self):
        bw_off = _render({"process_mode": "B&W", "density_damping_spatial": False})
        bw_on = _render({"process_mode": "B&W", "density_damping_spatial": True, "density_saturation_damping": 0.4})
        np.testing.assert_allclose(bw_off, bw_on, atol=1e-6)
