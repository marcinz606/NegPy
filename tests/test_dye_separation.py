"""Dye Separation renders, plus the retired Dye Mute keys it replaced."""

import logging

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.services.rendering.engine import DarkroomEngine


def _render(overrides):
    engine = DarkroomEngine()
    rng = np.random.default_rng(0)
    img = rng.uniform(0.05, 0.9, (32, 32, 3)).astype(np.float32)
    settings = WorkspaceConfig.from_flat_dict({"paper_profile": "neutral", **overrides})
    return engine.process(img, settings, source_hash="dye-separation-test")


class TestDyeSeparationRender:
    def test_identity_is_exact_noop(self):
        baseline = _render({})
        identity = _render({"dye_separation": 0.0})
        np.testing.assert_array_equal(baseline, identity)

    def test_positive_changes_output(self):
        baseline = _render({})
        separated = _render({"dye_separation": 0.5})
        assert not np.array_equal(baseline, separated)

    def test_negative_changes_output(self):
        baseline = _render({})
        clumped = _render({"dye_separation": -0.3})
        assert not np.array_equal(baseline, clumped)

    def test_directions_are_not_a_simple_sign_flip(self):
        """Each sign acts on a different set of pixels, so equal magnitudes
        must not come out as mirror images."""
        separated = _render({"dye_separation": 0.4})
        clumped = _render({"dye_separation": -0.4})
        assert not np.array_equal(separated, clumped)

    def test_positive_gain_is_a_vibrance_curve(self):
        """Guards the two constants against each other, on the kernel's own formula:
        muted pixels must take the larger relative gain, saturated ones must stay
        protected, and the result must stay monotone in spread or two pixels swap
        which one reads as more saturated."""
        scale = EXPOSURE_CONSTANTS["dye_separation_spread_scale"]
        gain = EXPOSURE_CONSTANTS["dye_separation_gain"]
        spread = np.linspace(1e-4, 4.0, 20000)
        s = 2.0 / (1.0 + np.exp(-spread / scale)) - 1.0
        k = 1.0 + gain * (1.0 - s)

        assert k[0] > 3.0  # a fully muted pixel gets the full gain
        assert np.interp(0.05, spread, k) > np.interp(0.5, spread, k)
        assert np.interp(0.5, spread, k) < 1.15  # saturated pixels stay protected
        assert np.all(np.diff(spread * k) > 0)

    def test_bw_mode_is_inert(self):
        bw_off = _render({"process_mode": "B&W", "dye_separation": 0.0})
        bw_on = _render({"process_mode": "B&W", "dye_separation": 0.6})
        np.testing.assert_allclose(bw_off, bw_on, atol=1e-6)


class TestDyeSeparationMigration:
    def test_legacy_density_vibrance_key_is_renamed(self):
        config = WorkspaceConfig.from_flat_dict({"density_vibrance": -0.25})
        assert config.exposure.dye_separation == -0.25

    def test_retired_dye_mute_keys_drop_without_warning(self, caplog):
        """Dropped, not carried over. Dye Mute's strength drove a grade-coupled
        frame-wide damp, so the same number means something else here."""
        with caplog.at_level(logging.WARNING):
            config = WorkspaceConfig.from_flat_dict(
                {"chroma_damping": 0.5, "density_saturation_damping": 0.3, "density_damping_spatial": True}
            )
        assert caplog.text == ""
        assert config.exposure.dye_separation == 0.0
        assert not hasattr(config.lab, "chroma_damping")

    def test_retired_lab_vibrance_key_drops_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = WorkspaceConfig.from_flat_dict({"vibrance": 1.4})
        assert caplog.text == ""
        assert not hasattr(config.lab, "vibrance")
