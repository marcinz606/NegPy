import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.processor import NormalizationProcessor


class TestNormalizationUnclamped(unittest.TestCase):
    def setUp(self):
        self.config = WorkspaceConfig()
        self.context = PipelineContext(scale_factor=1.0, original_size=(100, 100), process_mode="C41")

    def test_unclamped_out_of_bounds(self):
        """
        Densities beyond the bounds must pass through unclamped,
        leaving rolloff to the characteristic curve.
        """
        floors = (-1.0, -1.0, -1.0)
        ceils = (-0.2, -0.2, -0.2)
        process = replace(self.config.process, local_floors=floors, local_ceils=ceils)

        img = np.full((4, 4, 3), 10.0**-0.1, dtype=np.float32)
        res = NormalizationProcessor(process).process(img, self.context)
        self.assertGreater(float(res[0, 0, 0]), 1.0)

        img_low = np.full((4, 4, 3), 10.0**-1.5, dtype=np.float32)
        res_low = NormalizationProcessor(process).process(img_low, self.context)
        self.assertLess(float(res_low[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
