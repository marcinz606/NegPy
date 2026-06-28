import itertools
import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import CharacteristicCurve, apply_characteristic_curve
from negpy.features.exposure.processor import PhotometricProcessor


def _ramp_image(n: int = 256) -> np.ndarray:
    ramp = np.linspace(-0.2, 1.2, n, dtype=np.float32).reshape(1, n, 1)
    return np.repeat(ramp, 3, axis=2)


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
        """Full toe must visibly lift the deep shadows (blacks raised toward grey)."""
        img = np.full((4, 4, 3), 1.0, dtype=np.float32)  # deepest measured shadow
        params = (0.5, 5.375)  # pivot pushes the pixel deep into paper black
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        lifted = float(apply_characteristic_curve(img, params, params, params, toe=1.0, toe_width=2.5)[0, 0, 0])
        # Linear: paper black ~0.005, toe lift ~0.0048 (was 0.03 in sRGB).
        self.assertGreater(lifted, base + 0.004)

    def test_toe_leaves_highlights_invariant(self):
        """Toe shapes the shadow end: bright highlights stay put at any width."""
        img = np.full((4, 4, 3), 0.0, dtype=np.float32)  # far highlight side
        params = (0.79, 5.375)
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        for toe, width in itertools.product((-1.0, 1.0), (0.1, 2.5, 5.0)):
            res = float(apply_characteristic_curve(img, params, params, params, toe=toe, toe_width=width)[0, 0, 0])
            self.assertAlmostEqual(res, base, delta=0.015, msg=f"toe={toe} width={width}")

    def test_shoulder_leaves_shadows_invariant(self):
        """Shoulder shapes the highlight end: deep shadows stay put (sharp/default
        width; a very gentle shoulder is a global curvature control and may bleed)."""
        img = np.full((4, 4, 3), 1.0, dtype=np.float32)  # deep shadow side
        params = (0.79, 5.375)
        base = float(apply_characteristic_curve(img, params, params, params)[0, 0, 0])
        for shoulder, width in itertools.product((-1.0, 1.0), (0.1, 2.5)):
            res = float(apply_characteristic_curve(img, params, params, params, shoulder=shoulder, shoulder_width=width)[0, 0, 0])
            self.assertAlmostEqual(res, base, delta=0.015, msg=f"shoulder={shoulder} width={width}")

    def test_chart_matches_kernel(self):
        """CharacteristicCurve (H&D chart) must match the pipeline kernel."""
        img = _ramp_image()
        pivot, slope = 0.79, 5.375
        for toe, toe_width, shoulder, shoulder_width in ((0.0, 2.5, 0.0, 2.5), (0.6, 3.0, -0.4, 1.5)):
            params = (pivot, slope)
            res_kernel = apply_characteristic_curve(
                img, params, params, params, toe=toe, toe_width=toe_width, shoulder=shoulder, shoulder_width=shoulder_width
            )
            curve = CharacteristicCurve(
                contrast=slope, pivot=pivot, toe=toe, toe_width=toe_width, shoulder=shoulder, shoulder_width=shoulder_width
            )
            density = np.asarray(curve(img[0, :, 0].astype(np.float32)))
            # Kernel outputs linear reflectance (transmittance) now.
            expected = np.clip(10.0 ** (-density), 0.0, 1.0)
            np.testing.assert_allclose(res_kernel[0, :, 0], expected, atol=1e-4)


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
