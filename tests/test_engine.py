import unittest
import numpy as np
from src.application.engine import DarkroomEngine
from src.core.models import WorkspaceConfig


class TestDarkroomEngine(unittest.TestCase):
    def test_pipeline_execution(self):
        """Verify that the engine can process an image from start to finish."""
        engine = DarkroomEngine()
        img = np.random.rand(100, 100, 3).astype(np.float32)
        settings = WorkspaceConfig.from_flat_dict(
            {"autocrop": False, "autocrop_offset": 0}
        )

        res = engine.process(img, settings)

        # Output should be 100x100 since autocrop is off and offset is 0
        self.assertEqual(res.shape, (100, 100, 3))
        self.assertLessEqual(np.max(res), 1.0)
        self.assertGreaterEqual(np.min(res), 0.0)

    def test_pipeline_with_crop(self):
        """Verify that engine correctly handles cropping."""
        engine = DarkroomEngine()
        img = np.random.rand(200, 200, 3).astype(np.float32)
        settings = WorkspaceConfig.from_flat_dict(
            {"autocrop": False, "autocrop_offset": 10}
        )

        res = engine.process(img, settings)

        self.assertLess(res.shape[0], 200)
        self.assertLess(res.shape[1], 200)


if __name__ == "__main__":
    unittest.main()
