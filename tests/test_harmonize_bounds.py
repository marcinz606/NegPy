import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds,
    harmonize_bounds,
)
from negpy.features.exposure.processor import NormalizationProcessor
from negpy.features.process.models import ProcessMode


class TestHarmonizeBounds(unittest.TestCase):
    def test_c41_shared_green_range(self):
        bounds = LogNegativeBounds(floors=(-2.0, -1.5, -1.0), ceils=(-0.1, -0.3, -0.5))
        hb = harmonize_bounds(bounds)

        r_green = bounds.ceils[1] - bounds.floors[1]
        for ch in range(3):
            self.assertAlmostEqual(hb.ceils[ch] - hb.floors[ch], r_green, places=6)
            self.assertEqual(hb.ceils[ch], bounds.ceils[ch])

        self.assertAlmostEqual(hb.floors[1], bounds.floors[1], places=6)

    def test_e6_reversed_orientation_preserved(self):
        bounds = LogNegativeBounds(floors=(-0.2, -0.3, -0.4), ceils=(-2.5, -2.2, -2.0))
        hb = harmonize_bounds(bounds)

        r_green = bounds.ceils[1] - bounds.floors[1]
        self.assertLess(r_green, 0.0)
        for ch in range(3):
            self.assertAlmostEqual(hb.ceils[ch] - hb.floors[ch], r_green, places=6)
            self.assertGreater(hb.floors[ch], hb.ceils[ch])

    def test_e6_fixed_range_is_noop(self):
        img = np.random.default_rng(0).uniform(0.01, 0.9, (32, 32, 3)).astype(np.float32)
        bounds = analyze_log_exposure_bounds(img, process_mode=ProcessMode.E6, e6_normalize=False)
        hb = harmonize_bounds(bounds)

        for ch in range(3):
            self.assertAlmostEqual(hb.floors[ch], bounds.floors[ch], places=5)
            self.assertAlmostEqual(hb.ceils[ch], bounds.ceils[ch], places=5)

    def test_idempotent(self):
        bounds = LogNegativeBounds(floors=(-2.0, -1.5, -1.0), ceils=(-0.1, -0.3, -0.5))
        once = harmonize_bounds(bounds)
        twice = harmonize_bounds(once)

        for ch in range(3):
            self.assertAlmostEqual(twice.floors[ch], once.floors[ch], places=6)
            self.assertAlmostEqual(twice.ceils[ch], once.ceils[ch], places=6)


class TestNormalizationSharedScale(unittest.TestCase):
    def setUp(self):
        self.config = WorkspaceConfig()
        self.context = PipelineContext(scale_factor=1.0, original_size=(100, 100), process_mode="C41")

    def _processor(self, floors, ceils, **kwargs):
        process = replace(self.config.process, local_floors=floors, local_ceils=ceils, **kwargs)
        return NormalizationProcessor(process)

    def test_equal_density_steps_normalize_equally(self):
        """
        With channel-distinct bounds, a density step of k * R_green from each
        channel's ceil must land on the same normalized value (shared scale).
        """
        floors = (-2.0, -1.5, -1.0)
        ceils = (-0.1, -0.3, -0.5)
        r_green = ceils[1] - floors[1]

        k = 0.4
        img_val = tuple(10.0 ** (ceils[ch] - k * r_green) for ch in range(3))
        img = np.empty((4, 4, 3), dtype=np.float32)
        img[..., 0], img[..., 1], img[..., 2] = img_val

        res = self._processor(floors, ceils).process(img, self.context)

        for ch in range(3):
            self.assertAlmostEqual(float(res[0, 0, ch]), 1.0 - k, places=4)

    def test_unclamped_out_of_bounds(self):
        """
        Densities beyond the bounds must pass through unclamped (> 1.0),
        leaving rolloff to the characteristic curve.
        """
        floors = (-1.0, -1.0, -1.0)
        ceils = (-0.2, -0.2, -0.2)

        img = np.full((4, 4, 3), 10.0**-0.1, dtype=np.float32)
        res = self._processor(floors, ceils).process(img, self.context)
        self.assertGreater(float(res[0, 0, 0]), 1.0)

        img_low = np.full((4, 4, 3), 10.0**-1.5, dtype=np.float32)
        res_low = self._processor(floors, ceils).process(img_low, self.context)
        self.assertLess(float(res_low[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
