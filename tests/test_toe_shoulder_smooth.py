import itertools
import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import CharacteristicCurve, apply_characteristic_curve, print_knees
from negpy.features.exposure.processor import PhotometricProcessor
from negpy.features.process.models import ProcessMode


def _ramp_image(n: int = 256) -> np.ndarray:
    ramp = np.linspace(-0.2, 1.2, n, dtype=np.float32).reshape(1, n, 1)
    return np.repeat(ramp, 3, axis=2)


_SLOPE = 5.375


def _knees(toe: float = 0.0, shoulder: float = 0.0):
    return print_knees(_SLOPE, toe, shoulder)[2:4]


class TestToeShoulderSmoothness(unittest.TestCase):
    def test_monotonic_over_full_slider_range(self):
        img = _ramp_image()
        for toe, shoulder, pivot in itertools.product((-1.0, 1.0), (-1.0, 1.0), (0.5, 0.79, 0.95)):
            params = (pivot, _SLOPE)
            res = apply_characteristic_curve(img, params, params, params, knee_stretch=_knees(toe, shoulder))
            row = res[0, :, 0]
            self.assertTrue(np.all(np.diff(row) <= 1e-5), f"tone reversal at toe={toe} shoulder={shoulder} pivot={pivot}")

    def test_toe_has_useful_strength(self):
        img = np.full((4, 4, 3), 1.1, dtype=np.float32)
        params = (0.79, _SLOPE)
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        soft = float(apply_characteristic_curve(img, params, params, params, knee_stretch=_knees(toe=1.0))[0, 0, 0])
        self.assertGreater(soft, base * 1.1)

    def test_toe_leaves_highlights_invariant(self):
        img = np.full((4, 4, 3), 0.0, dtype=np.float32)  # far highlight side
        params = (0.79, _SLOPE)
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        for toe in (-1.0, 1.0):
            res = float(apply_characteristic_curve(img, params, params, params, knee_stretch=_knees(toe=toe))[0, 0, 0])
            self.assertAlmostEqual(res, base, delta=1e-6, msg=f"toe={toe}")

    def test_shoulder_leaves_shadows_invariant(self):
        img = np.full((4, 4, 3), 1.0, dtype=np.float32)  # deep shadow side
        params = (0.79, _SLOPE)
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        for shoulder in (-1.0, 1.0):
            res = float(apply_characteristic_curve(img, params, params, params, knee_stretch=_knees(shoulder=shoulder))[0, 0, 0])
            self.assertAlmostEqual(res, base, delta=1e-6, msg=f"shoulder={shoulder}")

    def test_chart_matches_kernel(self):
        """CharacteristicCurve (H&D chart) must match the pipeline kernel."""
        img = _ramp_image()
        pivot = 0.79
        for toe, shoulder in ((0.0, 0.0), (0.6, -0.4), (-1.0, 1.0)):
            params = (pivot, _SLOPE)
            tw3, sw3 = _knees(toe, shoulder)
            res_kernel = apply_characteristic_curve(img, params, params, params, knee_stretch=(tw3, sw3))
            curve = CharacteristicCurve(contrast=_SLOPE, pivot=pivot, toe_stretch=tw3[0], shoulder_stretch=sw3[0])
            density = np.asarray(curve(img[0, :, 0].astype(np.float32)))
            expected = np.clip(10.0 ** (-density), 0.0, 1.0)
            np.testing.assert_allclose(res_kernel[0, :, 0], expected, atol=1e-4)


class TestBWLuminanceBeforeCurve(unittest.TestCase):
    def test_bw_output_channels_identical(self):
        config = replace(WorkspaceConfig().exposure, toe=0.5, shoulder=0.3)
        ctx = PipelineContext(scale_factor=1.0, original_size=(8, 8), process_mode=ProcessMode.BW)
        rng = np.random.default_rng(0)
        img = rng.uniform(0.0, 1.0, (8, 8, 3)).astype(np.float32)

        res = PhotometricProcessor(config).process(img, ctx)
        np.testing.assert_array_almost_equal(res[..., 0], res[..., 1], decimal=6)
        np.testing.assert_array_almost_equal(res[..., 0], res[..., 2], decimal=6)


if __name__ == "__main__":
    unittest.main()
