"""Dye Separation: the density-space matrix helpers (papers.py), the per-channel
resolve (logic.py), end-to-end render behavior (identity/no-op, real effect, B&W
inertness), the Separation Damping law that tapers it per pixel, and the
migration off the retired keys it replaced."""

import logging

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import per_channel_dye_separation, separation_damping_gain
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.papers import (
    compose_density_matrices,
    resolve_dye_matrix,
    resolve_paper,
    resolve_saturation_matrix,
)
from negpy.services.rendering.engine import DarkroomEngine
from negpy.features.process.models import ProcessMode


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
        bw_off = self._render({"process_mode": ProcessMode.BW, "dye_separation": 1.0})
        bw_on = self._render({"process_mode": ProcessMode.BW, "dye_separation": 1.8})
        np.testing.assert_allclose(bw_off, bw_on, atol=1e-6)


class TestSeparationDampingGain:
    """The per-pixel k law. REF mirrors separation_damping_ref_spread."""

    REF = float(EXPOSURE_CONSTANTS["separation_damping_ref_spread"])

    def test_damping_zero_is_the_flat_k(self):
        for k in (0.5, 1.0, 1.4):
            for c in (0.0, 0.3, 1.5):
                assert separation_damping_gain(k, 0.0, c, self.REF) == k

    def test_identity_k_is_inert_at_any_chroma(self):
        for tau in (0.0, 0.5, 1.0):
            for c in (0.0, 0.2, self.REF, 2.0):
                assert separation_damping_gain(1.0, tau, c, self.REF) == 1.0

    def test_reference_spread_is_the_fixed_point(self):
        for k in (0.6, 1.4):
            assert separation_damping_gain(k, 1.0, self.REF, self.REF) == pytest.approx(1.0)

    def test_sign_of_the_change_differs_between_populations(self):
        """The whole point, and what #683 killed the previous attempt for: a
        frame-wide matrix scales every pixel's chroma the same way, so no setting
        of it can lift muted colour while pulling vivid colour down. This must."""
        muted = separation_damping_gain(1.4, 1.0, 0.05, self.REF)
        vivid = separation_damping_gain(1.4, 1.0, 3.0 * self.REF, self.REF)
        assert muted > 1.25
        assert vivid < 1.0
        # ... and mirrored below 1.0: muted collapses, vivid is pushed out.
        assert separation_damping_gain(0.7, 1.0, 0.05, self.REF) < 0.8
        assert separation_damping_gain(0.7, 1.0, 3.0 * self.REF, self.REF) > 1.0

    def test_chroma_transfer_is_monotone(self):
        """Non-monotone chroma means two pixels swap which reads as more
        saturated -- banding on the smooth gradients this control gets used on.
        Covers the full clamped k domain of per_channel_dye_separation."""
        cs = np.linspace(0.0, 2.24, 2000)
        for k in (0.0, 0.1, 0.5, 1.5, 1.9, 3.0):
            for tau in (0.5, 1.0):
                out = np.array([c * separation_damping_gain(k, tau, c, self.REF) for c in cs])
                assert np.diff(out).min() >= -1e-12, f"non-monotone at k={k}, damping={tau}"

    def test_zero_k_stays_a_full_collapse(self):
        assert separation_damping_gain(0.0, 1.0, 1.5, self.REF) == 0.0

    def test_gain_is_bounded_by_the_matrix_coefficient_range(self):
        for c in (0.0, 0.5, 2.24):
            assert 0.0 <= separation_damping_gain(0.02, 1.0, c, self.REF) <= 3.0


class TestSeparationDampingRender:
    def _render(self, overrides, seed=0):
        engine = DarkroomEngine()
        rng = np.random.default_rng(seed)
        img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
        settings = WorkspaceConfig.from_flat_dict({"paper_profile": "neutral", **overrides})
        return engine.process(img, settings, source_hash="separation-damping-test")

    def test_damping_zero_is_exact_noop(self):
        flat = self._render({"dye_separation": 1.4})
        explicit = self._render({"dye_separation": 1.4, "separation_damping": 0.0})
        np.testing.assert_array_equal(flat, explicit)

    def test_damping_zero_is_exact_noop_with_real_dye_matrix(self):
        """At damping 0 the composed sat @ dye path must still run untouched --
        the branch that moves the sat matrix into the kernel must not fire."""
        flat = self._render({"paper_profile": "kodak_endura", "dye_separation": 1.4})
        explicit = self._render({"paper_profile": "kodak_endura", "dye_separation": 1.4, "separation_damping": 0.0})
        np.testing.assert_array_equal(flat, explicit)

    def test_inert_without_a_separation_push(self):
        """It redistributes Dye Separation's push and has no effect of its own,
        so at dye_separation 1.0 it must not touch a single pixel."""
        baseline = self._render({})
        np.testing.assert_array_equal(baseline, self._render({"separation_damping": 1.0}))

    def test_damping_changes_output(self):
        flat = self._render({"dye_separation": 1.4})
        damped = self._render({"dye_separation": 1.4, "separation_damping": 1.0})
        assert not np.array_equal(flat, damped)

    def test_neutral_ramp_is_preserved(self):
        """c = 0 on a neutral, so the gain multiplies a zero deviation whatever k
        and damping say -- greys must stay exactly grey."""
        engine = DarkroomEngine()
        ramp = np.repeat(np.linspace(0.02, 0.98, 64, dtype=np.float32)[:, None], 3, axis=1)[None, :, :]
        for k in (0.6, 1.0, 1.5):
            for tau in (0.0, 0.5, 1.0):
                settings = WorkspaceConfig.from_flat_dict({"paper_profile": "neutral", "dye_separation": k, "separation_damping": tau})
                out = np.asarray(engine.process(ramp, settings, source_hash=f"sep-damp-ramp-{k}-{tau}"))
                assert float(np.max(np.abs(out - out.mean(axis=2, keepdims=True)))) < 1e-6

    def test_muted_and_vivid_populations_move_opposite_ways(self):
        """End-to-end form of test_sign_of_the_change_differs_between_populations,
        measured in the domain the law acts on: print-density chroma, recovered by
        inverting the output (paper_black skips BPC so the inversion is clean).
        The flat law scales every population by the same k -- that is what made the
        previous attempt read as a weaker Dye Separation (#683). This must scale
        the two populations in opposite directions."""
        engine = DarkroomEngine()
        ref = float(EXPOSURE_CONSTANTS["separation_damping_ref_spread"])
        oetf = 563.0 / 256.0

        h, w = 24, 48
        luma = np.repeat(np.linspace(0.05, 0.9, h, dtype=np.float32)[:, None], w, axis=1)
        amp = np.linspace(0.0, 0.45, w, dtype=np.float32)[None, :]
        img = np.stack([luma + amp, luma, luma - amp], axis=-1).clip(0.01, 0.99).astype(np.float32)

        def density_chroma(overrides, tag):
            settings = WorkspaceConfig.from_flat_dict({"paper_profile": "neutral", "paper_black": True, **overrides})
            out = np.asarray(engine.process(img, settings, source_hash=f"sep-damp-{tag}"), dtype=np.float64)
            d = -np.log10(np.clip(out, 1e-6, 1.0) ** oetf)
            r, g, b = d[..., 0], d[..., 1], d[..., 2]
            return np.sqrt(((r - g) ** 2 + (g - b) ** 2 + (r - b) ** 2) / 3.0)

        base = density_chroma({}, "base")
        flat = density_chroma({"dye_separation": 1.4}, "flat")
        damped = density_chroma({"dye_separation": 1.4, "separation_damping": 1.0}, "damped")

        muted = base < 0.4 * ref
        vivid = base > 2.0 * ref
        assert muted.sum() > 20 and vivid.sum() > 20, "fixture must span both populations"

        # The flat law pushes both populations the same way (it does not reach a
        # full 1.4 on the vivid one: k that high drives density above base
        # negative there and the output clamp eats the rest).
        assert flat[muted].mean() / base[muted].mean() == pytest.approx(1.4, abs=0.03)
        assert flat[vivid].mean() / base[vivid].mean() > 1.2
        # The damped law splits them, and the sign of the change flips.
        assert damped[muted].mean() / base[muted].mean() > 1.15
        assert damped[vivid].mean() / base[vivid].mean() < 1.0


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
