import unittest
import numpy as np
from src.services.rendering.gpu_engine import GPUEngine
from src.domain.models import WorkspaceConfig
from src.infrastructure.gpu.device import GPUDevice


class TestGPUEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = GPUDevice.get()
        if cls.gpu.is_available:
            cls.engine = GPUEngine()
        else:
            cls.engine = None

    def setUp(self):
        if self.engine is None:
            self.skipTest("GPU not available")

    def test_gpu_process_smoke(self):
        """Basic GPU processing smoke test."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        settings = WorkspaceConfig()

        res, metrics = self.engine.process(img, settings)

        self.assertEqual(res.ndim, 3)
        self.assertEqual(res.shape[2], 3)
        self.assertIn("active_roi", metrics)
        self.assertIn("histogram_raw", metrics)
        self.assertEqual(metrics["histogram_raw"].shape, (4, 256))

    def test_gpu_process_to_texture(self):
        """Verify process_to_texture returns a GPUTexture."""
        from src.infrastructure.gpu.resources import GPUTexture

        img = np.random.rand(64, 64, 3).astype(np.float32)
        settings = WorkspaceConfig()

        tex, metrics = self.engine.process_to_texture(img, settings)

        self.assertIsInstance(tex, GPUTexture)
        self.assertEqual(tex.width, metrics["base_positive"].width)

    def test_gpu_engine_cleanup(self):
        """Verify cleanup releases resources."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        settings = WorkspaceConfig()

        # Run once to populate cache
        self.engine.process_to_texture(img, settings)
        self.assertTrue(len(self.engine._tex_cache) > 0)

        self.engine.cleanup()
        self.assertEqual(len(self.engine._tex_cache), 0)

    def test_gpu_tiled_processing(self):
        """Verify tiled processing for large images."""
        # Force tiled path by using an image that exceeds 12M pixels or just a bit large
        # For tests, we'll keep it reasonable but enough to trigger logic if we lowered threshold
        # Or we can just call _process_tiled directly if it was public, but it's internal.
        # Let's use an image large enough.
        # The threshold is 12,000,000 pixels.
        # 4000 * 3001 = 12,003,000
        h, w = 3001, 4000
        img = np.random.rand(h, w, 3).astype(np.float32)
        settings = WorkspaceConfig()

        res, metrics = self.engine.process(img, settings)

        # Check if result matches expected aspect ratio or similar
        self.assertIsNotNone(res)
        self.assertTrue(res.shape[0] > 0)

    def test_gpu_engine_destroy_all(self):
        """Verify destroy_all clears persistent resources."""
        self.engine._init_resources()
        self.assertTrue(len(self._engine_buffers_count()) > 0)

        self.engine.destroy_all()
        self.assertEqual(len(self._engine_buffers_count()), 0)
        self.assertEqual(len(self.engine._pipelines), 0)

    def _engine_buffers_count(self):
        return self.engine._buffers


if __name__ == "__main__":
    unittest.main()
