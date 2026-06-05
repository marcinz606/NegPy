import unittest

import numpy as np

from negpy.features.exposure.normalization import analyze_log_exposure_bounds
from negpy.features.process.models import ProcessMode


def _gradient_image() -> np.ndarray:
    # Spread of linear values so percentiles (min/max in log space) are well-defined.
    vals = np.linspace(0.01, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    return np.stack([vals, vals, vals], axis=-1)


class TestDRangeMargin(unittest.TestCase):
    def setUp(self):
        self.img = _gradient_image()

    def test_zero_clip_samples_true_extremes(self):
        """clip == 0 must map the true log min/max (gentlest percentile stretch)."""
        bounds = analyze_log_exposure_bounds(self.img, percentile_clip=0.0)
        log = np.log10(np.clip(self.img, 1e-6, 1.0))
        for ch in range(3):
            self.assertAlmostEqual(bounds.floors[ch], float(log[:, :, ch].min()), places=4)
            self.assertAlmostEqual(bounds.ceils[ch], float(log[:, :, ch].max()), places=4)

    def test_negative_clip_expands_outward_c41(self):
        """Negative clip pushes bounds beyond the extremes by exactly the margin."""
        margin = 0.5
        base = analyze_log_exposure_bounds(self.img, percentile_clip=0.0)
        ext = analyze_log_exposure_bounds(self.img, percentile_clip=-margin)
        for ch in range(3):
            self.assertAlmostEqual(ext.floors[ch], base.floors[ch] - margin, places=5)
            self.assertAlmostEqual(ext.ceils[ch], base.ceils[ch] + margin, places=5)

    def test_positive_clip_unchanged(self):
        """Positive path is untouched: still pulls bounds inward from the extremes."""
        base = analyze_log_exposure_bounds(self.img, percentile_clip=0.0)
        clipped = analyze_log_exposure_bounds(self.img, percentile_clip=1.0)
        for ch in range(3):
            self.assertGreater(clipped.floors[ch], base.floors[ch])
            self.assertLess(clipped.ceils[ch], base.ceils[ch])

    def test_negative_clip_expands_outward_e6(self):
        """E6 maps f > c; outward expansion must grow |delta|, not shrink it."""
        margin = 0.5
        base = analyze_log_exposure_bounds(self.img, process_mode=ProcessMode.E6, percentile_clip=0.0)
        ext = analyze_log_exposure_bounds(self.img, process_mode=ProcessMode.E6, percentile_clip=-margin)
        for ch in range(3):
            self.assertAlmostEqual(ext.floors[ch], base.floors[ch] + margin, places=5)
            self.assertAlmostEqual(ext.ceils[ch], base.ceils[ch] - margin, places=5)


if __name__ == "__main__":
    unittest.main()
