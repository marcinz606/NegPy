import unittest

import numpy as np

from negpy.features.exposure.normalization import analyze_log_exposure_bounds


def _offset_image() -> np.ndarray:
    # Gradient base with per-channel linear gain -> per-channel log offset (a colour
    # cast), identical span per channel.
    vals = np.linspace(0.02, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    return np.stack([vals, vals * 0.7, vals * 0.85], axis=-1)


def _mono_image() -> np.ndarray:
    vals = np.linspace(0.02, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    return np.stack([vals, vals, vals], axis=-1)


class TestColorLumaSplit(unittest.TestCase):
    def setUp(self):
        self.img = _offset_image()

    def test_luma_drives_mean_center_and_span(self):
        """Overall (mean) floor/ceil and span come purely from the luma sampling —
        the colour clip only redistributes per-channel deviation, never the mean."""
        p_luma = 0.6
        a = analyze_log_exposure_bounds(self.img, percentile_clip=p_luma, color_clip=5.0)
        b = analyze_log_exposure_bounds(self.img, percentile_clip=p_luma, color_clip=0.5)

        self.assertAlmostEqual(sum(a.floors) / 3.0, sum(b.floors) / 3.0, places=5)
        self.assertAlmostEqual(sum(a.ceils) / 3.0, sum(b.ceils) / 3.0, places=5)
        self.assertAlmostEqual((sum(a.ceils) - sum(a.floors)) / 3.0, (sum(b.ceils) - sum(b.floors)) / 3.0, places=5)

    def test_colour_clip_preserves_luma_span(self):
        """Changing the colour clip must not change the mean span (highlights)."""
        a = analyze_log_exposure_bounds(self.img, percentile_clip=0.0, color_clip=10.0)
        b = analyze_log_exposure_bounds(self.img, percentile_clip=0.0, color_clip=0.01)
        span_a = (sum(a.ceils) - sum(a.floors)) / 3.0
        span_b = (sum(b.ceils) - sum(b.floors)) / 3.0
        self.assertAlmostEqual(span_a, span_b, places=5)

    def test_colour_pass_injects_per_channel_cast(self):
        """The colour pass carries the per-channel cast into the bounds: the more
        attenuated channels (lower linear gain) get a lower floor."""
        bounds = analyze_log_exposure_bounds(self.img, percentile_clip=0.0, color_clip=5.0)
        # Channel gains were 1.0, 0.7, 0.85 -> log floors ordered r > b > g.
        self.assertLess(bounds.floors[1], bounds.floors[2])
        self.assertLess(bounds.floors[2], bounds.floors[0])

    def test_mono_image_has_no_cast(self):
        """Identical channels -> zero colour deviation -> all channels share the luma
        sampling regardless of colour clip."""
        mono = _mono_image()
        a = analyze_log_exposure_bounds(mono, percentile_clip=0.3, color_clip=10.0)
        b = analyze_log_exposure_bounds(mono, percentile_clip=0.3, color_clip=0.01)
        for ch in range(3):
            self.assertAlmostEqual(a.floors[ch], b.floors[ch], places=6)
            self.assertAlmostEqual(a.ceils[ch], b.ceils[ch], places=6)
            self.assertAlmostEqual(a.floors[ch], a.floors[0], places=6)


if __name__ == "__main__":
    unittest.main()
