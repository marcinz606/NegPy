import unittest

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.finish.models import FinishConfig
from negpy.features.finish.processor import FinishProcessor


class TestFinishProcessor(unittest.TestCase):
    def _gradient_image(self) -> np.ndarray:
        return np.full((100, 100, 3), 0.5, dtype=np.float32)

    def _context(self) -> PipelineContext:
        return PipelineContext(original_size=(100, 100), scale_factor=1.0, process_mode="C41")

    def test_noop_when_strength_zero(self) -> None:
        """Processor returns image unchanged when vignette strength is 0."""
        img = self._gradient_image()
        config = FinishConfig(vignette_strength=0.0, vignette_size=0.5)
        processor = FinishProcessor(config)
        ctx = self._context()
        res = processor.process(img, ctx)
        np.testing.assert_array_equal(res, img)

    def test_applies_effect_when_nonzero(self) -> None:
        """Processor darkens corners when strength is negative."""
        img = self._gradient_image()
        config = FinishConfig(vignette_strength=-1.0, vignette_size=0.5)
        processor = FinishProcessor(config)
        ctx = self._context()
        res = processor.process(img, ctx)
        # Corner should be darker than center
        self.assertLess(float(res[0, 0].mean()), float(res[50, 50].mean()))

    def test_preserves_image_type(self) -> None:
        """Output is float32 in [0, 1]."""
        img = self._gradient_image()
        config = FinishConfig(vignette_strength=0.5, vignette_size=0.5)
        processor = FinishProcessor(config)
        ctx = self._context()
        res = processor.process(img, ctx)
        self.assertEqual(res.dtype, np.float32)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
