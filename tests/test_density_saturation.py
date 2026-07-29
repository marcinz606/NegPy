"""Density-space Saturation: the matrix helpers (papers.py), the per-channel
resolve (logic.py), and end-to-end render behavior (identity/no-op, real
effect, B&W inertness)."""

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import per_channel_density_saturation
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


class TestPerChannelDensitySaturation:
    def test_global_only(self):
        assert per_channel_density_saturation(1.4, (0.0, 0.0, 0.0)) == (1.4, 1.4, 1.4)

    def test_trims_add_to_global(self):
        assert per_channel_density_saturation(1.0, (0.2, -0.1, 0.0)) == (1.2, 0.9, 1.0)

    def test_clamped_to_matrix_coefficient_range(self):
        assert per_channel_density_saturation(1.0, (-5.0, 5.0, 0.0)) == (0.0, 3.0, 1.0)


class TestDensitySaturationRender:
    def _render(self, overrides):
        engine = DarkroomEngine()
        rng = np.random.default_rng(0)
        img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
        settings = WorkspaceConfig.from_flat_dict({"paper_profile": "neutral", **overrides})
        return engine.process(img, settings, source_hash="density-sat-test")

    def test_identity_is_exact_noop(self):
        baseline = self._render({})
        identity = self._render({"density_saturation": 1.0})
        np.testing.assert_array_equal(baseline, identity)

    def test_identity_is_exact_noop_with_real_dye_matrix(self):
        """Same as test_identity_is_exact_noop, but with a paper that has a real
        (non-identity) dye_matrix -- at density_saturation=1.0, compose_density_matrices
        multiplies by an identity sat matrix, which is exact in floating point, so this
        must stay bit-for-bit identical too, not just approximately close."""
        engine = DarkroomEngine()
        rng = np.random.default_rng(0)
        img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
        baseline = WorkspaceConfig.from_flat_dict({"paper_profile": "kodak_endura"})
        identity = WorkspaceConfig.from_flat_dict({"paper_profile": "kodak_endura", "density_saturation": 1.0})
        r1 = engine.process(img, baseline, source_hash="density-sat-endura-test")
        r2 = engine.process(img, identity, source_hash="density-sat-endura-test")
        np.testing.assert_array_equal(r1, r2)

    def test_nonzero_saturation_changes_output(self):
        baseline = self._render({})
        boosted = self._render({"density_saturation": 1.6})
        assert not np.array_equal(baseline, boosted)

    def test_per_channel_trim_changes_output(self):
        global_only = self._render({"density_saturation": 1.2})
        with_trim = self._render({"density_saturation": 1.2, "density_saturation_trim_red": 0.3})
        assert not np.array_equal(global_only, with_trim)

    def test_bw_mode_is_inert(self):
        bw_off = self._render({"process_mode": "B&W", "density_saturation": 1.0})
        bw_on = self._render({"process_mode": "B&W", "density_saturation": 1.8})
        np.testing.assert_allclose(bw_off, bw_on, atol=1e-6)
