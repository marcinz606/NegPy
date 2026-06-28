import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.processor import PhotometricProcessor


class TestPaperDmin(unittest.TestCase):
    def setUp(self):
        self.config = WorkspaceConfig().exposure

    def _run(self, value: float, paper_dmin: bool) -> float:
        ctx = PipelineContext(scale_factor=1.0, original_size=(8, 8), process_mode="C41")
        img = np.full((8, 8, 3), value, dtype=np.float32)
        res = PhotometricProcessor(replace(self.config, paper_dmin=paper_dmin)).process(img, ctx)
        return float(res[0, 0, 0])

    def test_whites_capped_at_paper_base(self):
        # Floor on: nothing prints brighter than 10^-d_min (linear reflectance).
        ceiling = 10.0 ** -EXPOSURE_CONSTANTS["d_min"]
        self.assertLessEqual(self._run(0.0, paper_dmin=True), ceiling + 1e-6)
        self.assertLess(self._run(0.0, paper_dmin=True), self._run(0.0, paper_dmin=False))

    def test_off_keeps_pure_white_reachable(self):
        # Thin negative approaches white; linear (0.91 sRGB ≈ 0.807 linear).
        self.assertGreater(self._run(0.0, paper_dmin=False), 0.80)

    def test_shadows_unaffected_direction(self):
        # Deep blacks stay governed by d_max; the floor barely moves them.
        delta = abs(self._run(1.0, paper_dmin=True) - self._run(1.0, paper_dmin=False))
        self.assertLess(delta, 0.02)

    def test_serialization_roundtrip(self):
        config = WorkspaceConfig()
        config = replace(config, exposure=replace(config.exposure, paper_dmin=True))
        restored = WorkspaceConfig.from_flat_dict(config.to_dict())
        self.assertTrue(restored.exposure.paper_dmin)


if __name__ == "__main__":
    unittest.main()
