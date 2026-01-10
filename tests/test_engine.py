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

        res = engine.process(img, settings, source_hash="dummy")

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

        res = engine.process(img, settings, source_hash="dummy")

        self.assertLess(res.shape[0], 200)
        self.assertLess(res.shape[1], 200)

    def test_engine_caching(self):
        """Verify that the engine populates and uses cache."""
        engine = DarkroomEngine()
        img = np.random.rand(100, 100, 3).astype(np.float32)
        settings = WorkspaceConfig()

        # First run - populate cache
        res1 = engine.process(img, settings, source_hash="file1")
        assert engine.cache.base is not None
        assert engine.cache.exposure is not None
        base_id = id(engine.cache.base.data)

        # Second run with same config - should use cache
        res2 = engine.process(img, settings, source_hash="file1")
        # In modern numpy, the id might change if array is copied, but here
        # we strictly return the cached object.
        assert id(engine.cache.base.data) == base_id
        assert np.array_equal(res1, res2)

        # Change source - should clear and re-populate with NEW data
        # We use a different image to ensure the content also changes
        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        res3 = engine.process(img2, settings, source_hash="file2")
        assert engine.cache.source_hash == "file2"
        # The content must be different because input image changed
        assert not np.array_equal(res1, res3)


if __name__ == "__main__":
    unittest.main()
