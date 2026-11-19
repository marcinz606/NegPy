"""Per-layer grade trims (crossover correction): a ΔISO-R per channel rotates
that layer's slope about the anchor, so casts that differ between shadows and
highlights can be fixed while midtones stay neutral."""

import dataclasses
import unittest

import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
    _grade_trim_mult,
    apply_characteristic_curve,
    per_channel_curve_params,
    per_channel_midtone_gamma,
    grade_coupled_shape,
    per_channel_toe_shoulder,
    print_curve,
    print_knees,
    KNEE_SPAN_TOE,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.papers import PaperProfile

_ANCHOR = EXPOSURE_CONSTANTS["assumed_anchor"]
_TARGET = EXPOSURE_CONSTANTS["reference_print_density"]


def _densities(value: float, slopes, pivots, curvatures) -> np.ndarray:
    """Per-channel print density for a uniform input patch."""
    img = np.full((2, 2, 3), value, dtype=np.float32)
    out = apply_characteristic_curve(
        img,
        (pivots[0], slopes[0]),
        (pivots[1], slopes[1]),
        (pivots[2], slopes[2]),
        curvatures=curvatures,
    )
    t = np.clip(np.asarray(out)[0, 0, :], 1e-12, None)
    return -np.log10(t)


def _regimes(grade_trims):
    """The two cast-removal regimes of per_channel_curve_params."""
    base = per_channel_curve_params(115.0, 1.0, False, 0.0, 1.3, None, grade_trims=grade_trims)
    neutral_axis = per_channel_curve_params(
        115.0,
        1.0,
        False,
        1.0,
        1.3,
        None,
        neutral_axis_norm=((_ANCHOR,) * 3, (0.9,) * 3, (0.2,) * 3),
        grade_trims=grade_trims,
    )
    return {"base": base, "neutral_axis": neutral_axis}


class TestGradeTrimMult(unittest.TestCase):
    def test_pure_iso_r_ratio(self):
        c = EXPOSURE_CONSTANTS
        self.assertAlmostEqual(_grade_trim_mult(115.0, 30.0, c), 115.0 / 145.0)
        self.assertAlmostEqual(_grade_trim_mult(115.0, -30.0, c), 115.0 / 85.0)
        self.assertAlmostEqual(_grade_trim_mult(115.0, 0.0, c), 1.0)

    def test_clamped_to_ladder(self):
        c = EXPOSURE_CONSTANTS
        self.assertAlmostEqual(_grade_trim_mult(170.0, 30.0, c), 170.0 / 180.0)
        self.assertAlmostEqual(_grade_trim_mult(60.0, -30.0, c), 60.0 / 50.0)


class TestCrossoverTrims(unittest.TestCase):
    def test_zero_trims_identity(self):
        for regime, params in _regimes((0.0, 0.0, 0.0)).items():
            untrimmed = _regimes((0.0, 0.0, 0.0))[regime]
            self.assertEqual(params, untrimmed, regime)

    def test_anchor_stays_neutral_all_regimes(self):
        """Trims rotate each layer about the anchor: the reference tone must
        print at the same density on all three channels, at the target."""
        for regime, (slopes, pivots, curvs) in _regimes((20.0, 0.0, -15.0)).items():
            d = _densities(_ANCHOR, slopes, pivots, curvs)
            for ch in range(3):
                self.assertAlmostEqual(d[ch], _TARGET, places=3, msg=f"{regime} ch={ch}")

    def test_crossover_signature(self):
        """Softer red layer (+ISO-R): red prints lighter than green in shadows,
        darker in highlights — the crossover shape filtration cannot produce."""
        slopes, pivots, curvs = _regimes((30.0, 0.0, 0.0))["base"]
        d_sh = _densities(0.9, slopes, pivots, curvs)
        d_hi = _densities(0.15, slopes, pivots, curvs)
        self.assertLess(d_sh[0], d_sh[1] - 0.01, "red not lighter in shadows")
        self.assertGreater(d_hi[0], d_hi[1] + 0.01, "red not darker in highlights")

    def test_monotonic_at_extreme_trims(self):
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        slopes, pivots, curvs = _regimes((30.0, 0.0, -30.0))["base"]
        for toe, shoulder in ((-1.0, -1.0), (1.0, 1.0)):
            out = apply_characteristic_curve(
                ramp,
                (pivots[0], slopes[0]),
                (pivots[1], slopes[1]),
                (pivots[2], slopes[2]),
                toe=toe,
                shoulder=shoulder,
                curvatures=curvs,
            )
            for ch in range(3):
                self.assertTrue(
                    np.all(np.diff(out[0, :, ch]) <= 1e-5),
                    f"tone reversal ch={ch} toe={toe} shoulder={shoulder}",
                )


class TestKneeTrims(unittest.TestCase):
    """Per-layer toe/shoulder trims: endpoint crossover — one layer's knee moves,
    the other layers and the opposite end stay put."""

    _PARAMS = (0.79, 5.375)

    def _render(self, value, params=None, **kwargs):
        img = np.full((2, 2, 3), value, dtype=np.float32)
        p = self._PARAMS if params is None else params
        out = apply_characteristic_curve(img, p, p, p, **kwargs)
        return np.asarray(out)[0, 0, :]

    def test_toe_trim_acts_on_channel_shadows_only(self):
        deep = (0.5, 5.375)  # pivot pushes a val=1 pixel deep into paper black
        base = self._render(1.0, params=deep)
        trimmed = self._render(1.0, params=deep, toe_trims=(1.0, 0.0, 0.0))
        self.assertGreater(trimmed[0], base[0] + 0.004, "red toe trim did not lift red shadows")
        np.testing.assert_allclose(trimmed[1:], base[1:], atol=1e-7, err_msg="toe trim leaked into other channels")
        hi_base = self._render(0.0)
        hi_trim = self._render(0.0, toe_trims=(1.0, 0.0, 0.0))
        self.assertAlmostEqual(hi_trim[0], hi_base[0], delta=0.015, msg="toe trim leaked into highlights")

    def test_monotonic_at_extreme_knee_trims(self):
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        toe_trims = (1.0, -1.0, 0.0)
        out = apply_characteristic_curve(ramp, self._PARAMS, self._PARAMS, self._PARAMS, toe_trims=toe_trims)
        for ch in range(3):
            self.assertTrue(np.all(np.diff(out[0, :, ch]) <= 1e-5), f"tone reversal ch={ch}")

    def test_chart_matches_kernel_per_channel(self):
        """The chart's per-channel traces (one CharacteristicCurve per layer) must
        match the kernel with knee trims applied."""
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        pivot, slope = self._PARAMS
        toe_trims = (0.6, 0.0, -0.4)
        res_kernel = apply_characteristic_curve(ramp, self._PARAMS, self._PARAMS, self._PARAMS, toe_trims=toe_trims)
        toe3, sh3 = per_channel_toe_shoulder(0.0, 0.0, toe_trims)
        for ch in range(3):
            curve = CharacteristicCurve(contrast=slope, pivot=pivot, toe=toe3[ch], shoulder=sh3[ch])
            density = np.asarray(curve(ramp[0, :, ch].astype(np.float32)))
            expected = np.clip(10.0 ** (-density), 0.0, 1.0)
            np.testing.assert_allclose(res_kernel[0, :, ch], expected, atol=1e-4)


class TestPrintKnees(unittest.TestCase):
    """Toe and Shoulder set the knee sharpness only: the paper's black and white stay, the
    reference tone holds, and a layer trim reshapes its own layer."""

    _SLOPE = 5.375

    def _curve(self, **sliders):
        exposure = dataclasses.replace(ExposureConfig(), **sliders)
        return print_curve(exposure, self._SLOPE, 0.79)

    def _density(self, x, **sliders):
        return float(np.asarray(self._curve(**sliders)(np.array([x], dtype=np.float32)))[0])

    def test_neutral_sliders_keep_the_papers_own_knees(self):
        toe3, sh3, ts3, ss3 = print_knees(self._SLOPE, 0.0, 0.0)
        self.assertEqual(ts3, (1.0, 1.0, 1.0))
        self.assertEqual(ss3, (1.0, 1.0, 1.0))
        self.assertEqual((toe3[1], sh3[1]), grade_coupled_shape(self._SLOPE, 0.0, 0.0))

    def test_toe_reshapes_the_shadows_and_keeps_paper_black(self):
        deep, black = 1.05, 3.0
        self.assertLess(self._density(deep, toe=1.0), self._density(deep) - 0.05)
        self.assertGreater(self._density(deep, toe=-1.0), self._density(deep))
        for toe in (-1.0, 1.0):
            self.assertAlmostEqual(self._density(black, toe=toe), self._density(black), places=2)

    def test_shoulder_reshapes_the_highlights_and_keeps_paper_white(self):
        light, white = 0.62, -2.0
        self.assertGreater(self._density(light, shoulder=1.0), self._density(light) + 0.03)
        self.assertLess(self._density(light, shoulder=-1.0), self._density(light))
        for shoulder in (-1.0, 1.0):
            self.assertAlmostEqual(self._density(white, shoulder=shoulder), self._density(white), places=2)

    def test_reference_tone_holds(self):
        ramp = np.linspace(0.0, 1.5, 4096, dtype=np.float32)
        for base in ({}, {"midtone_gamma": 0.5}, {"midtone_gamma": -0.5}, {"paper_profile": "kodak_portra_endura", "midtone_gamma": 0.5}):
            x_ref = float(np.interp(_TARGET, np.asarray(self._curve(**base)(ramp)), ramp))
            for sliders in ({"toe": 1.0}, {"toe": -1.0}, {"shoulder": 1.0}, {"shoulder": -1.0}):
                self.assertAlmostEqual(self._density(x_ref, **base, **sliders), _TARGET, delta=0.005, msg=str((base, sliders)))

    def test_each_knee_leaves_the_other_side_of_the_reference_tone(self):
        ramp = np.linspace(0.0, 1.5, 4096, dtype=np.float32)
        for base in ({}, {"midtone_gamma": 0.5}, {"paper_profile": "kodak_portra_endura"}, {"paper_profile": "frontier"}):
            paper_curve = np.asarray(self._curve(**base)(ramp))
            x_ref = float(np.interp(_TARGET, paper_curve, ramp))
            for slider, side in (("shoulder", ramp >= x_ref), ("toe", ramp <= x_ref)):
                for value in (-1.0, 1.0):
                    moved = np.asarray(self._curve(**base, **{slider: value})(ramp))
                    np.testing.assert_allclose(moved[side], paper_curve[side], atol=1e-5, err_msg=str((base, slider, value)))
                    self.assertGreater(float(np.max(np.abs(moved - paper_curve))), 0.05)

    def test_trim_acts_on_its_layer_only(self):
        _, _, base_tw, base_sw = print_knees(self._SLOPE, 0.2, -0.2)
        _, _, tw3, sw3 = print_knees(self._SLOPE, 0.2, -0.2, (0.5, 0.0, 0.0), (0.0, 0.0, -0.5))
        self.assertGreater(tw3[0], base_tw[0])
        self.assertEqual(tw3[1:], base_tw[1:])
        self.assertLess(sw3[2], base_sw[2])
        self.assertEqual(sw3[:2], base_sw[:2])

    def test_a_layer_reaches_its_own_maximum_density(self):
        paper = PaperProfile(label="t", d_max_drop=(0.0, 0.2, 0.4))
        toe3, _, tw3, sw3 = print_knees(self._SLOPE, 0.0, 0.0, paper=paper)
        black = np.full((2, 2, 3), 3.0, dtype=np.float32)
        params = (0.79, self._SLOPE)
        out = apply_characteristic_curve(
            black,
            params,
            params,
            params,
            toe=toe3[1],
            toe_trims=(toe3[0] - toe3[1], 0.0, toe3[2] - toe3[1]),
            paper=paper,
            knee_stretch=(tw3, sw3),
        )
        density = -np.log10(np.asarray(out)[0, 0, :])
        np.testing.assert_allclose(density[0] - density, [0.0, 0.2, 0.4], atol=1e-3)

    def test_slider_and_trim_clamp_to_the_slider_domain(self):
        _, _, tw3, _ = print_knees(self._SLOPE, 0.8, 0.0, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(tw3[0], KNEE_SPAN_TOE)


class TestSnapTrims(unittest.TestCase):
    """Per-layer Snap trims: midtone crossover — one layer's midtone gamma moves,
    the other layers stay put and the reference tone (v_star) is preserved."""

    _PARAMS = (0.79, 5.375)

    def _render(self, value, **kwargs):
        img = np.full((2, 2, 3), value, dtype=np.float32)
        out = apply_characteristic_curve(img, self._PARAMS, self._PARAMS, self._PARAMS, **kwargs)
        return np.asarray(out)[0, 0, :]

    def test_snap_trim_acts_on_channel_only(self):
        base = self._render(0.95)
        trimmed = self._render(0.95, snap_trims=(0.4, 0.0, 0.0))
        self.assertGreater(abs(trimmed[0] - base[0]), 0.003, "red snap trim did not move red midtones")
        np.testing.assert_allclose(trimmed[1:], base[1:], atol=1e-7, err_msg="snap trim leaked into other channels")

    def test_snap_trim_preserves_reference_tone(self):
        pivot, slope = self._PARAMS
        v_star = CharacteristicCurve(contrast=slope, pivot=pivot).v_star
        val = pivot + v_star / slope  # input that lands exactly on the S-curve centre
        base = self._render(val)
        trimmed = self._render(val, snap_trims=(0.5, -0.5, 0.3))
        np.testing.assert_allclose(trimmed, base, atol=1e-3, err_msg="snap trim moved the reference tone")

    def test_monotonic_at_extreme_snap_trims(self):
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        for mg, trims in ((-0.35, (-0.5, 0.0, 0.5)), (0.65, (0.5, -0.5, 0.0))):
            out = apply_characteristic_curve(ramp, self._PARAMS, self._PARAMS, self._PARAMS, midtone_gamma=mg, snap_trims=trims)
            for ch in range(3):
                self.assertTrue(
                    np.all(np.diff(out[0, :, ch]) <= 1e-5),
                    f"tone reversal ch={ch} mg={mg} trims={trims}",
                )

    def test_chart_matches_kernel_per_channel(self):
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        pivot, slope = self._PARAMS
        snap_trims = (0.3, 0.0, -0.2)
        res_kernel = apply_characteristic_curve(ramp, self._PARAMS, self._PARAMS, self._PARAMS, snap_trims=snap_trims)
        mg3 = per_channel_midtone_gamma(None, 0.0, snap_trims)
        for ch in range(3):
            curve = CharacteristicCurve(contrast=slope, pivot=pivot, midtone_gamma=mg3[ch])
            density = np.asarray(curve(ramp[0, :, ch].astype(np.float32)))
            expected = np.clip(10.0 ** (-density), 0.0, 1.0)
            np.testing.assert_allclose(res_kernel[0, :, ch], expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
