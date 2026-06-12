import itertools
import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import LogisticSigmoid, _expit, apply_characteristic_curve
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.processor import PhotometricProcessor


def _ramp_image(n: int = 256) -> np.ndarray:
    ramp = np.linspace(-0.2, 1.2, n, dtype=np.float32).reshape(1, n, 1)
    return np.repeat(ramp, 3, axis=2)


def _srgb_oetf(t: np.ndarray) -> np.ndarray:
    return np.where(t <= 0.0031308, 12.92 * t, 1.055 * np.power(t, 1.0 / 2.4) - 0.055)


class TestToeShoulderSmoothness(unittest.TestCase):
    def test_monotonic_over_full_slider_range(self):
        """
        The transfer must be non-increasing (input = density, output = brightness)
        for every toe/shoulder/width/pivot combination — no tone reversal.
        """
        img = _ramp_image()
        for toe, shoulder, width, pivot in itertools.product((-1.0, 1.0), (-1.0, 1.0), (0.1, 2.5, 5.0), (0.5, 0.79, 0.95)):
            params = (pivot, 5.375)
            res = apply_characteristic_curve(
                img,
                params,
                params,
                params,
                toe=toe,
                toe_width=width,
                shoulder=shoulder,
                shoulder_width=width,
            )
            row = res[0, :, 0]
            self.assertTrue(
                np.all(np.diff(row) <= 1e-5),
                f"tone reversal at toe={toe} shoulder={shoulder} width={width} pivot={pivot}",
            )

    def test_toe_has_useful_strength(self):
        """
        Full toe must lift deep shadows clearly at default grade/width —
        guards against the input-axis formulation whose lift was capped by
        the narrow shadow zone above the pivot. (Paper-black baseline is
        ~0.14 sRGB at d_max 2.3, so the lift is relative, not from zero.)
        """
        img = np.full((4, 4, 3), 1.0, dtype=np.float32)
        # Pivot chosen so the pixel prints as a deep shadow (~1.8 D) on the
        # Richards curve — the zone the toe lever exists to control.
        params = (0.64, 5.375)
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        lifted = float(apply_characteristic_curve(img, params, params, params, toe=1.0, toe_width=2.5)[0, 0, 0])
        self.assertGreater(lifted, base + 0.08)

    def test_toe_leaves_highlights_invariant(self):
        """
        Density-domain toe is anchored at D = 0: bright print tones (low
        density) must not move, at any width — guards against the
        pivot-density anchoring that shifted all highlights by a constant.
        """
        img = np.full((4, 4, 3), 0.0, dtype=np.float32)  # far highlight side
        params = (0.79, 5.375)
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        for toe, width in itertools.product((-1.0, 1.0), (0.1, 2.5, 5.0)):
            res = float(apply_characteristic_curve(img, params, params, params, toe=toe, toe_width=width)[0, 0, 0])
            self.assertAlmostEqual(res, base, places=3, msg=f"toe={toe} width={width}")

    def test_identity_when_disabled(self):
        """toe = shoulder = 0 must reproduce the plain sigmoid curve exactly."""
        img = _ramp_image()
        pivot, slope = 0.79, 5.375
        params = (pivot, slope)
        res = apply_characteristic_curve(img, params, params, params)

        density = (
            EXPOSURE_CONSTANTS["curve_asymptote"]
            * _expit(slope * (img[0, :, 0].astype(np.float64) - pivot)) ** EXPOSURE_CONSTANTS["paper_toe_nu"]
        )
        beta = EXPOSURE_CONSTANTS["dmax_shoulder"]
        density -= np.logaddexp(0.0, beta * (density - EXPOSURE_CONSTANTS["d_max"])) / beta
        expected = np.clip(_srgb_oetf(10.0 ** (-density)), 0.0, 1.0)
        np.testing.assert_allclose(res[0, :, 0], expected, atol=1e-5)

    def test_chart_matches_kernel(self):
        """LogisticSigmoid (H&D chart) must produce the same curve as the pipeline kernel."""
        img = _ramp_image()
        pivot, slope = 0.79, 5.375
        toe, toe_width, shoulder, shoulder_width = 0.6, 3.0, -0.4, 1.5
        params = (pivot, slope)
        res_kernel = apply_characteristic_curve(
            img, params, params, params, toe=toe, toe_width=toe_width, shoulder=shoulder, shoulder_width=shoulder_width
        )

        curve = LogisticSigmoid(contrast=slope, pivot=pivot, toe=toe, toe_width=toe_width, shoulder=shoulder, shoulder_width=shoulder_width)
        density = curve(img[0, :, 0].astype(np.float32))
        expected = np.clip(_srgb_oetf(10.0 ** (-density)), 0.0, 1.0)
        np.testing.assert_allclose(res_kernel[0, :, 0], expected, atol=1e-5)


class TestBWLuminanceBeforeCurve(unittest.TestCase):
    def test_bw_output_channels_identical(self):
        config = replace(WorkspaceConfig().exposure, toe=0.5, shoulder=0.3)
        ctx = PipelineContext(scale_factor=1.0, original_size=(8, 8), process_mode="B&W")
        rng = np.random.default_rng(0)
        img = rng.uniform(0.0, 1.0, (8, 8, 3)).astype(np.float32)

        res = PhotometricProcessor(config).process(img, ctx)
        np.testing.assert_array_almost_equal(res[..., 0], res[..., 1], decimal=6)
        np.testing.assert_array_almost_equal(res[..., 0], res[..., 2], decimal=6)


if __name__ == "__main__":
    unittest.main()
