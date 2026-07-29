"""Dye Separation: the density-space matrix helpers (papers.py), the per-channel
resolve (logic.py), end-to-end render behavior (identity/no-op, real effect, B&W
inertness), and the migration off the retired keys it replaced."""

import logging

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import per_channel_dye_separation
from negpy.features.exposure.papers import (
    compose_density_matrices,
    resolve_dye_matrix,
    resolve_paper,
    resolve_saturation_matrix,
)
from negpy.services.rendering.engine import DarkroomEngine


class TestResolveSaturationMatrix:
    def test_identity_returns_none(self):
        assert resolve_saturation_matrix((1.0, 1.0, 1.0)) is None

    def test_full_desaturation_collapses_to_achromatic_mean(self):
        m = resolve_saturation_matrix((0.0, 0.0, 0.0))
        np.testing.assert_allclose(m, np.full((3, 3), 1.0 / 3.0))

    def test_rows_sum_to_one_even_with_per_channel_divergence(self):
        m = resolve_saturation_matrix((1.5, 0.7, 1.0))
        np.testing.assert_allclose(m.sum(axis=1), [1.0, 1.0, 1.0])

    def test_boost_strengthens_diagonal_weakens_off_diagonal(self):
        m = resolve_saturation_matrix((2.0, 2.0, 2.0))
        assert m[0, 0] > 1.0
        assert m[0, 1] < 0.0


class TestComposeDensityMatrices:
    def test_both_none_is_none(self):
        assert compose_density_matrices(None, None) is None

    def test_dye_only_returns_dye_unchanged(self):
        dye = np.array([[0.95, 0.04, 0.01], [0.08, 0.88, 0.04], [0.04, 0.14, 0.82]])
        result = compose_density_matrices(dye, None)
        np.testing.assert_allclose(result, dye)

    def test_sat_only_returns_sat_unchanged(self):
        sat = resolve_saturation_matrix((1.5, 1.0, 1.0))
        result = compose_density_matrices(None, sat)
        np.testing.assert_allclose(result, sat)

    def test_composition_is_sat_outermost_dye_innermost(self):
        """sat @ dye, not dye @ sat -- dye_matrix keeps acting on the print
        curve's real density unchanged; saturation layers on top as a final
        creative step. Uses a real (non-commuting) paper matrix so the two
        orders actually produce different results, proving the order is
        real and tested, not just documented."""
        paper = resolve_paper("kodak_endura")
        dye = resolve_dye_matrix(paper)
        sat = resolve_saturation_matrix((1.5, 0.8, 1.0))
        composed = compose_density_matrices(dye, sat)
        np.testing.assert_allclose(composed, sat @ dye)
        assert not np.allclose(composed, dye @ sat)


class TestPerChannelDyeSeparation:
    def test_global_only(self):
        assert per_channel_dye_separation(1.4, (0.0, 0.0, 0.0)) == (1.4, 1.4, 1.4)

    def test_trims_add_to_global(self):
        assert per_channel_dye_separation(1.0, (0.2, -0.1, 0.0)) == (1.2, 0.9, 1.0)

    def test_clamped_to_matrix_coefficient_range(self):
        assert per_channel_dye_separation(1.0, (-5.0, 5.0, 0.0)) == (0.0, 3.0, 1.0)


class TestDyeSeparationRender:
    def _render(self, overrides):
        engine = DarkroomEngine()
        rng = np.random.default_rng(0)
        img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
        settings = WorkspaceConfig.from_flat_dict({"paper_profile": "neutral", **overrides})
        return engine.process(img, settings, source_hash="dye-separation-test")

    def test_identity_is_exact_noop(self):
        baseline = self._render({})
        identity = self._render({"dye_separation": 1.0})
        np.testing.assert_array_equal(baseline, identity)

    def test_identity_is_exact_noop_with_real_dye_matrix(self):
        """Same as test_identity_is_exact_noop, but with a paper that has a real
        (non-identity) dye_matrix -- at dye_separation=1.0, compose_density_matrices
        multiplies by an identity sat matrix, which is exact in floating point, so this
        must stay bit-for-bit identical too, not just approximately close."""
        engine = DarkroomEngine()
        rng = np.random.default_rng(0)
        img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
        baseline = WorkspaceConfig.from_flat_dict({"paper_profile": "kodak_endura"})
        identity = WorkspaceConfig.from_flat_dict({"paper_profile": "kodak_endura", "dye_separation": 1.0})
        r1 = engine.process(img, baseline, source_hash="dye-separation-endura-test")
        r2 = engine.process(img, identity, source_hash="dye-separation-endura-test")
        np.testing.assert_array_equal(r1, r2)

    def test_boost_changes_output(self):
        baseline = self._render({})
        boosted = self._render({"dye_separation": 1.6})
        assert not np.array_equal(baseline, boosted)

    def test_per_channel_trim_changes_output(self):
        global_only = self._render({"dye_separation": 1.2})
        with_trim = self._render({"dye_separation": 1.2, "dye_separation_trim_red": 0.3})
        assert not np.array_equal(global_only, with_trim)

    def test_bw_mode_is_inert(self):
        bw_off = self._render({"process_mode": "B&W", "dye_separation": 1.0})
        bw_on = self._render({"process_mode": "B&W", "dye_separation": 1.8})
        np.testing.assert_allclose(bw_off, bw_on, atol=1e-6)


class TestDyeSeparationMigration:
    """Print Saturation (density_saturation) absorbed the per-pixel Dye Separation
    beside it and took over its name. Saves carry both keys, so the old per-pixel
    value must never survive as the new one -- its ±0.5 scale would read as a heavy
    desaturation under the 0.5-1.5 semantics."""

    def test_legacy_print_saturation_key_is_renamed(self):
        config = WorkspaceConfig.from_flat_dict({"density_saturation": 1.3})
        assert config.exposure.dye_separation == 1.3

    def test_legacy_print_saturation_trims_are_renamed(self):
        config = WorkspaceConfig.from_flat_dict({"density_saturation_trim_red": 0.3, "density_saturation_trim_blue": -0.2})
        assert config.exposure.dye_separation_trim_red == 0.3
        assert config.exposure.dye_separation_trim_blue == -0.2

    def test_print_saturation_wins_over_the_retired_per_pixel_value(self):
        config = WorkspaceConfig.from_flat_dict({"density_saturation": 1.3, "dye_separation": 0.4})
        assert config.exposure.dye_separation == 1.3

    def test_retired_per_pixel_value_alone_falls_back_to_the_default(self, caplog):
        """A granular preset can carry the old per-pixel key on its own, with no
        density_saturation to date it. Below the new slider floor it can only be the
        retired control (0.0 was its off), so it drops instead of being adopted."""
        for legacy in (0.0, -0.3, 0.2):
            with caplog.at_level(logging.WARNING):
                config = WorkspaceConfig.from_flat_dict({"dye_separation": legacy})
            assert caplog.text == ""
            assert config.exposure.dye_separation == 1.0

    def test_value_inside_the_new_range_is_kept(self):
        """The live format wins the overlap: 0.4 is reachable on the new slider, so a
        lone key at that value is read as the frame-wide control, not discarded."""
        config = WorkspaceConfig.from_flat_dict({"dye_separation": 0.4})
        assert config.exposure.dye_separation == 0.4

    def test_retired_density_vibrance_key_drops_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = WorkspaceConfig.from_flat_dict({"density_vibrance": -0.25})
        assert caplog.text == ""
        assert config.exposure.dye_separation == 1.0

    def test_retired_dye_mute_keys_drop_without_warning(self, caplog):
        """Dropped, not carried over. Dye Mute's strength drove a grade-coupled
        frame-wide damp, so the same number means something else here."""
        with caplog.at_level(logging.WARNING):
            config = WorkspaceConfig.from_flat_dict(
                {"chroma_damping": 0.5, "density_saturation_damping": 0.3, "density_damping_spatial": True}
            )
        assert caplog.text == ""
        assert config.exposure.dye_separation == 1.0
        assert not hasattr(config.lab, "chroma_damping")

    def test_retired_lab_vibrance_key_drops_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = WorkspaceConfig.from_flat_dict({"vibrance": 1.4})
        assert caplog.text == ""
        assert not hasattr(config.lab, "vibrance")


class TestLabSaturationIsGradeIndependent:
    """The Lab stage must render identically whatever slope the print curve
    ran at. Saturation damping lives in density space, not here."""

    def _run(self, img, slopes):
        from negpy.domain.interfaces import PipelineContext
        from negpy.features.lab.models import LabConfig
        from negpy.features.lab.processor import PhotoLabProcessor

        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2])
        if slopes is not None:
            ctx.metrics["print_slopes"] = slopes
        return PhotoLabProcessor(LabConfig(sharpen=0.0, saturation=1.4)).process(img, ctx)

    def _color_image(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.9, (16, 16, 3)).astype(np.float32)

    def test_slope_does_not_change_lab_saturation(self):
        img = self._color_image()
        soft = self._run(img, (2.0, 2.0, 2.0))
        hard = self._run(img, (8.0, 8.0, 8.0))
        np.testing.assert_array_equal(soft, hard)

    def test_missing_slopes_matches_present_slopes(self):
        img = self._color_image()
        np.testing.assert_array_equal(self._run(img, None), self._run(img, (8.0, 8.0, 8.0)))
