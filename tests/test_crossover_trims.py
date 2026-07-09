"""Per-layer grade trims (crossover correction): a ΔISO-R per channel rotates
that layer's slope about the anchor, so casts that differ between shadows and
highlights can be fixed while midtones stay neutral."""

import unittest

import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
    _grade_trim_mult,
    apply_characteristic_curve,
    per_channel_curve_params,
    per_channel_toe_shoulder,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS

_ANCHOR = EXPOSURE_CONSTANTS["assumed_anchor"]
_TARGET = EXPOSURE_CONSTANTS["anchor_target_density"]


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
    """The three cast-removal regimes of per_channel_curve_params."""
    base = per_channel_curve_params(115.0, 1.0, False, 0.0, 1.3, None, None, grade_trims=grade_trims)
    shadow_tie = per_channel_curve_params(115.0, 1.0, False, 1.0, 1.3, (0.88, 0.9, 0.92), None, grade_trims=grade_trims)
    neutral_axis = per_channel_curve_params(
        115.0,
        1.0,
        False,
        1.0,
        1.3,
        None,
        None,
        neutral_axis_norm=((_ANCHOR,) * 3, (0.9,) * 3, (0.2,) * 3),
        grade_trims=grade_trims,
    )
    return {"base": base, "shadow_tie": shadow_tie, "neutral_axis": neutral_axis}


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

    def test_shoulder_trim_acts_on_channel_highlights_only(self):
        base = self._render(0.0)
        trimmed = self._render(0.0, shoulder_trims=(1.0, 0.0, 0.0))
        self.assertLess(trimmed[0], base[0] - 0.01, "red shoulder trim did not compress red highlights")
        np.testing.assert_allclose(trimmed[1:], base[1:], atol=1e-7, err_msg="shoulder trim leaked into other channels")
        sh_base = self._render(1.0)
        sh_trim = self._render(1.0, shoulder_trims=(1.0, 0.0, 0.0))
        self.assertAlmostEqual(sh_trim[0], sh_base[0], delta=0.015, msg="shoulder trim leaked into shadows")

    def test_negative_toe_trim_tints_true_black(self):
        """With True Black on, a negative per-layer toe clips only that layer to 0."""
        params = (0.5, 5.375)  # deep into paper black
        img = np.full((2, 2, 3), 1.0, dtype=np.float32)
        out = np.asarray(apply_characteristic_curve(img, params, params, params, bpc=True, toe_trims=(-1.0, 0.0, 0.0)))[0, 0, :]
        self.assertEqual(out[0], 0.0, "red layer did not clip to exact black")
        self.assertGreater(out[1], 0.0)
        self.assertGreater(out[2], 0.0)

    def test_monotonic_at_extreme_knee_trims(self):
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        for toe_trims, sh_trims in (((1.0, -1.0, 0.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 0.0), (1.0, -1.0, 0.5))):
            out = apply_characteristic_curve(ramp, self._PARAMS, self._PARAMS, self._PARAMS, toe_trims=toe_trims, shoulder_trims=sh_trims)
            for ch in range(3):
                self.assertTrue(
                    np.all(np.diff(out[0, :, ch]) <= 1e-5),
                    f"tone reversal ch={ch} toe={toe_trims} shoulder={sh_trims}",
                )

    def test_chart_matches_kernel_per_channel(self):
        """The chart's per-channel traces (one CharacteristicCurve per layer) must
        match the kernel with knee trims applied."""
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        pivot, slope = self._PARAMS
        toe_trims, sh_trims = (0.6, 0.0, -0.4), (-0.3, 0.2, 0.0)
        res_kernel = apply_characteristic_curve(
            ramp, self._PARAMS, self._PARAMS, self._PARAMS, toe_trims=toe_trims, shoulder_trims=sh_trims
        )
        toe3, sh3 = per_channel_toe_shoulder(0.0, 0.0, toe_trims, sh_trims)
        for ch in range(3):
            curve = CharacteristicCurve(contrast=slope, pivot=pivot, toe=toe3[ch], shoulder=sh3[ch])
            density = np.asarray(curve(ramp[0, :, ch].astype(np.float32)))
            expected = np.clip(10.0 ** (-density), 0.0, 1.0)
            np.testing.assert_allclose(res_kernel[0, :, ch], expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
