"""
Midtone neutrality of Cast Removal (the per-channel two-point gray balance).

Regression for the systematic green cast: green-dominant scene content inflates
green's normalization span, so the neutral midtone reads green. The legacy
shadow-only tie left it uncorrected; the two-point balance (low-chroma midtone +
shadow refs) neutralizes it. A pure neutral negative must stay neutral.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import per_channel_curve_params
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from negpy.kernel.image.logic import rgb_to_lab_working

_H, _W = 600, 400


def _negative(green_log: float | None = -0.22) -> np.ndarray:
    """
    Synthetic C-41 negative: a neutral gray ramp down the rows (per-channel dye
    gammas + orange mask) plus, when green_log is set, a green-dominant content
    block in the right columns that inflates green's bright (ceil) percentile.
    """
    E = np.linspace(0.0, 1.0, _H, dtype=np.float32)
    gamma = (0.66, 0.71, 0.68)  # small green-steepest dye gamma mismatch
    mask = (0.0, -0.12, -0.22)  # orange mask, red = reference
    base = -0.2
    log = np.empty((_H, _W, 3), np.float32)
    for ch in range(3):
        log[:, :, ch] = (base + mask[ch] - gamma[ch] * E)[:, None]
    if green_log is not None:
        gx = slice(int(0.82 * _W), _W)
        log[:, gx, 1] = green_log  # bright green -> inflates ceil_G
        log[:, gx, 0] = -0.50
        log[:, gx, 2] = -0.62
    return (10.0**log).astype(np.float32)


def _render(img: np.ndarray, cast_removal: bool, mode: str = "C41") -> np.ndarray:
    cfg = WorkspaceConfig()
    process = replace(cfg.process, analysis_buffer=0.0)
    ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=mode)
    norm = NormalizationProcessor(process).process(img, ctx)
    exp = replace(cfg.exposure, cast_removal=cast_removal)
    return PhotometricProcessor(exp).process(norm, ctx)


def _neutral_ab(out: np.ndarray) -> tuple[float, float]:
    """Mean (a*, b*) of the neutral-ramp midtone region (excludes the content block)."""
    patch = out[int(0.45 * _H) : int(0.55 * _H), 0 : int(0.78 * _W)].reshape(-1, 3)
    lab = rgb_to_lab_working(patch)
    return float(lab[:, 1].mean()), float(lab[:, 2].mean())


class TestMidtoneNeutral(unittest.TestCase):
    def test_green_dominant_midtone_neutralized(self):
        img = _negative(green_log=-0.22)
        a_off, _ = _neutral_ab(_render(img, cast_removal=False))
        a_on, b_on = _neutral_ab(_render(img, cast_removal=True))
        # Without the fix the neutral midtone is clearly green (a* << 0).
        self.assertLess(a_off, -8.0, f"fixture not green enough (a*={a_off:.1f})")
        # The two-point balance pulls it back to neutral on the a* (green/magenta) axis.
        self.assertLess(abs(a_on), 3.0, f"midtone still cast (a*={a_on:.1f})")
        self.assertLess(abs(b_on), 4.0, f"midtone b* off (b*={b_on:.1f})")
        self.assertLess(abs(a_on), abs(a_off) * 0.4)

    def test_pure_neutral_unchanged(self):
        # No content block: endpoint equalization is already exact, so Cast Removal
        # must be a near-no-op (no regression / no invented tint).
        img = _negative(green_log=None)
        off = _render(img, cast_removal=False)
        on = _render(img, cast_removal=True)
        a_on, b_on = _neutral_ab(on)
        self.assertLess(abs(a_on), 1.0)
        self.assertLess(abs(b_on), 1.0)
        self.assertTrue(np.allclose(on, off, atol=5e-3))

    def test_e6_noop(self):
        # E-6 measures no neutral axis -> falls back to the single shared curve.
        img = _negative(green_log=-0.22)
        off = _render(img, cast_removal=False, mode="E6")
        on = _render(img, cast_removal=True, mode="E6")
        self.assertTrue(np.allclose(on, off, atol=1e-6))


class TestTwoPointSolve(unittest.TestCase):
    """Direct check of the two-point solve: R/B render neutrals exactly as green."""

    def test_neutral_axis_makes_channels_match(self):
        # Green is the low (cast) channel at both tones; R==B higher.
        mid = (0.52, 0.44, 0.52)
        shadow = (0.84, 0.74, 0.84)
        slopes, pivots = per_channel_curve_params(
            grade=115.0,
            density=1.0,
            auto_normalize_contrast=True,
            cast_removal=True,
            lum_range=1.5,
            shadow_refs_norm=None,
            textural_range=0.75,
            neutral_axis_norm=(mid, shadow),
        )

        def core(ch, x):
            return slopes[ch] * (x - pivots[ch])

        # Each channel's neutral midtone/shadow maps to the SAME core as green's.
        for ch, label in ((0, "R"), (2, "B")):
            self.assertAlmostEqual(core(ch, mid[ch]), core(1, mid[1]), places=4, msg=label)
            self.assertAlmostEqual(core(ch, shadow[ch]), core(1, shadow[1]), places=4, msg=label)


if __name__ == "__main__":
    unittest.main()
