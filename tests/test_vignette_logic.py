"""CPU unit tests for the Lab luminance vignette (features/vignette).

The vignette is a scalar gain on every channel, so it moves luma not color.
These tests check the gain shape (centre untouched, edges moved) and that the
per-channel colour is preserved: a luminance vignette, not a tint.
"""

import unittest

import numpy as np

from negpy.features.vignette.logic import apply_luminance_vignette


def _img(h: int, w: int, r: float = 0.5, g: float = 0.4, b: float = 0.3) -> np.ndarray:
    """A flat, non-neutral RGB field, so a tint and a luminance gain differ."""
    return np.full((h, w, 3), (r, g, b), dtype=np.float32)


class TestVignetteLogic(unittest.TestCase):
    def test_zero_strength_is_noop(self):
        img = _img(32, 32)
        out = apply_luminance_vignette(img, strength=0.0, feather=0.5, roundness=0.0)
        self.assertTrue(np.array_equal(out, img))

    def test_centre_is_untouched(self):
        img = _img(32, 32)
        out = apply_luminance_vignette(img, strength=-0.6, feather=0.5, roundness=0.0)
        # The central block sits inside the falloff, where the gain is 1.
        centre = out[15:17, 15:17, :]
        self.assertTrue(np.allclose(centre, img[15:17, 15:17, :], atol=1e-6))

    def test_negative_strength_darkens_edges(self):
        img = _img(32, 32)
        out = apply_luminance_vignette(img, strength=-0.6, feather=0.5, roundness=0.0)
        self.assertLess(float(out[0, 0, 0]), float(img[0, 0, 0]))

    def test_positive_strength_lightens_edges(self):
        img = _img(32, 32)
        out = apply_luminance_vignette(img, strength=0.6, feather=0.5, roundness=0.0)
        self.assertGreater(float(out[0, 0, 0]), float(img[0, 0, 0]))

    def test_gain_is_clamped_to_one(self):
        img = _img(32, 32, 0.9, 0.9, 0.9)
        out = apply_luminance_vignette(img, strength=2.0, feather=0.5, roundness=0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-6)

    def test_luminance_not_tint(self):
        img = _img(32, 32, 0.5, 0.4, 0.3)
        out = apply_luminance_vignette(img, strength=-0.6, feather=0.5, roundness=0.5)
        # A scalar gain keeps the per-channel ratios: the edge is darker, not re-coloured.
        self.assertTrue(np.allclose(out[0, 0] / out[0, 0, 0], img[0, 0] / img[0, 0, 0], atol=1e-4))
        self.assertLess(float(out[0, 0, 0]), float(img[0, 0, 0]))

    def test_roundness_morphs_circle_to_ellipse(self):
        # A wide image: the circle and ellipse falloff differ off the centre line.
        img = _img(24, 48)
        circle = apply_luminance_vignette(img, strength=-0.6, feather=0.5, roundness=0.0)
        ellipse = apply_luminance_vignette(img, strength=-0.6, feather=0.5, roundness=1.0)
        self.assertGreater(float(np.max(np.abs(circle - ellipse))), 1e-3)


if __name__ == "__main__":
    unittest.main()
