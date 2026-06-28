import unittest

import numpy as np

from negpy.features.exposure.normalization import analyze_log_exposure_bounds
from negpy.features.process.models import ProcessMode


def _gradient_image(h: int, w: int, lo: float = 0.01, hi: float = 0.5) -> np.ndarray:
    """Smooth linear-space gradient negative, identical value distribution at any size."""
    col = np.linspace(lo, hi, h, dtype=np.float32)[:, None]
    img = np.repeat(col, w, axis=1)
    return np.stack([img] * 3, axis=-1)


class TestRobustBounds(unittest.TestCase):
    def test_isolated_speculars_do_not_move_bounds(self):
        """
        Scattered single-pixel extremes (speculars, dust) must not shift the
        floors/ceils, even when their total population exceeds the baseline
        clip fraction — the spatial prefilter rejects them.
        """
        img = _gradient_image(2048, 1536)
        clean = analyze_log_exposure_bounds(img)

        rng = np.random.default_rng(42)
        dirty = img.copy()
        n = int(img.shape[0] * img.shape[1] * 0.005)  # 0.5% of pixels
        ys = rng.integers(0, img.shape[0], n)
        xs = rng.integers(0, img.shape[1], n)
        dirty[ys, xs, :] = 1.0  # blown speculars (thin side)
        dirty[ys[: n // 2], (xs[: n // 2] + 7) % img.shape[1], :] = 1e-5  # dust (dense side)

        robust = analyze_log_exposure_bounds(dirty)
        for ch in range(3):
            self.assertAlmostEqual(clean.floors[ch], robust.floors[ch], delta=0.02)
            self.assertAlmostEqual(clean.ceils[ch], robust.ceils[ch], delta=0.02)

    def test_block_median_grid_parallel_is_bit_exact(self):
        """The threaded block-median (row strips) must equal the single-pass reshape-median."""
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS
        from negpy.features.exposure.normalization import _block_median_grid

        rng = np.random.default_rng(7)
        img = rng.random((2048, 3000, 3), dtype=np.float32)  # >2M px -> parallel path

        h, w = img.shape[:2]
        b = int(np.ceil(max(h, w) / int(EXPOSURE_CONSTANTS["analysis_grid"])))
        hb, wb = (h // b) * b, (w // b) * b
        ref = np.median(img[:hb, :wb].reshape(hb // b, b, wb // b, b, 3), axis=(1, 3))

        np.testing.assert_array_equal(_block_median_grid(img), ref)

    def test_resolution_invariance(self):
        """Same scene at different resolutions must yield near-identical bounds."""
        small = analyze_log_exposure_bounds(_gradient_image(1024, 768))
        large = analyze_log_exposure_bounds(_gradient_image(2048, 1536))
        for ch in range(3):
            self.assertAlmostEqual(small.floors[ch], large.floors[ch], delta=0.01)
            self.assertAlmostEqual(small.ceils[ch], large.ceils[ch], delta=0.01)

    def test_baseline_clip_rejects_small_coherent_outliers(self):
        """
        A coherent extreme region below the baseline clip fraction (survives the
        median prefilter) must still be excluded at percentile_clip = 0.
        """
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS

        img = _gradient_image(1024, 1024)
        clean = analyze_log_exposure_bounds(img)

        base_fraction = float(EXPOSURE_CONSTANTS["base_drange_clip"]) / 100.0
        side = max(2, int((img.shape[0] * img.shape[1] * base_fraction * 0.5) ** 0.5))
        dirty = img.copy()
        dirty[:side, :side, :] = 1.0  # coherent region at half the baseline fraction
        robust = analyze_log_exposure_bounds(dirty)
        for ch in range(3):
            self.assertAlmostEqual(clean.ceils[ch], robust.ceils[ch], delta=0.02)

    def test_large_regions_still_register(self):
        """A real highlight area (well above the baseline fraction) must move the ceil."""
        img = _gradient_image(1024, 1024)
        clean = analyze_log_exposure_bounds(img)

        lit = img.copy()
        lit[:200, :200, :] = 0.9  # ~3.8% of frame
        bounds = analyze_log_exposure_bounds(lit)
        for ch in range(3):
            self.assertGreater(bounds.ceils[ch], clean.ceils[ch] + 0.1)

    def test_margin_mode_expands_outward(self):
        """Negative percentile_clip still produces outward headroom on robust bounds."""
        img = _gradient_image(1024, 768)
        base = analyze_log_exposure_bounds(img, percentile_clip=0.0)
        wide = analyze_log_exposure_bounds(img, percentile_clip=-0.1)
        for ch in range(3):
            self.assertAlmostEqual(wide.floors[ch], base.floors[ch] - 0.1, delta=0.02)
            self.assertAlmostEqual(wide.ceils[ch], base.ceils[ch] + 0.1, delta=0.02)

    def test_e6_reversed_bounds(self):
        """E6 keeps reversed mapping (floors on the thin side) with robust analysis."""
        img = _gradient_image(1024, 768)
        bounds = analyze_log_exposure_bounds(img, process_mode=ProcessMode.E6, e6_normalize=True)
        for ch in range(3):
            self.assertGreater(bounds.floors[ch], bounds.ceils[ch])


if __name__ == "__main__":
    unittest.main()
