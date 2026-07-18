"""Black point compensation (the Paper Black toggle, off): map the paper's
physical Dmax to display black (ICC relative-colorimetric soft-proof style), so
deep shadows can reach exactly 0 while a lifted toe and paper white survive."""

import unittest

import numpy as np

from negpy.features.exposure.logic import CharacteristicCurve, apply_characteristic_curve

_PARAMS = (0.5, 5.375)  # pivot pushes a val=1 pixel deep into paper black
_SHADOW = np.full((2, 2, 3), 1.0, dtype=np.float32)
_HIGHLIGHT = np.full((2, 2, 3), 0.0, dtype=np.float32)


def _render(img, **kwargs) -> float:
    return float(apply_characteristic_curve(img, _PARAMS, _PARAMS, _PARAMS, **kwargs)[0, 0, 0])


class TestBlackPointCompensation(unittest.TestCase):
    def test_bpc_collapses_shadow_floor(self):
        """The absolute render floors at 10^-d_max (~0.005); BPC collapses that
        floor by an order of magnitude (the curve reaches d_max asymptotically)."""
        base = _render(_SHADOW)
        compensated = _render(_SHADOW, bpc=True)
        self.assertGreater(base, 0.004)
        self.assertLess(compensated, 0.001)

    def test_negative_toe_reaches_exact_zero(self):
        """Negative toe raises the BPC clip point into the shadows — the deepest
        scene shadow prints at exactly 0 (the pure black the toe promises)."""
        self.assertGreater(_render(_SHADOW, toe=-1.0), 0.0)  # without BPC: never zero
        self.assertEqual(_render(_SHADOW, toe=-1.0, bpc=True), 0.0)

    def test_paper_white_preserved(self):
        base = _render(_HIGHLIGHT)
        compensated = _render(_HIGHLIGHT, bpc=True)
        self.assertAlmostEqual(compensated, base, delta=0.006)

    def test_toe_lift_survives(self):
        """BPC references the physical Dmax, not d_max_eff — a lifted toe stays lifted."""
        lifted = _render(_SHADOW, toe=1.0, bpc=True)
        self.assertGreater(lifted, 0.004)

    def test_chart_matches_kernel(self):
        ramp = np.linspace(-0.2, 1.2, 256, dtype=np.float32).reshape(1, 256, 1).repeat(3, axis=2)
        pivot, slope = _PARAMS
        for toe in (0.0, -0.6, -1.0):
            res_kernel = apply_characteristic_curve(ramp, _PARAMS, _PARAMS, _PARAMS, toe=toe, bpc=True)
            curve = CharacteristicCurve(contrast=slope, pivot=pivot, toe=toe, bpc=True)
            density = np.asarray(curve(ramp[0, :, 0].astype(np.float32)))
            expected = np.clip(10.0 ** (-density), 0.0, 1.0)
            np.testing.assert_allclose(res_kernel[0, :, 0], expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
