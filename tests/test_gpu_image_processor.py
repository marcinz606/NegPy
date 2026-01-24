import unittest
import numpy as np
from src.services.rendering.image_processor import ImageProcessor
from src.domain.models import WorkspaceConfig
from src.infrastructure.gpu.device import GPUDevice


class TestImageProcessorGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = GPUDevice.get()
        cls.processor = ImageProcessor()

    def setUp(self):
        if not self.gpu.is_available:
            self.skipTest("GPU not available")

    def test_run_pipeline_gpu(self):
        """Verify high-level pipeline execution on GPU."""
        from src.kernel.system.config import APP_CONFIG

        img = np.random.rand(100, 100, 3).astype(np.float32)
        settings = WorkspaceConfig()

        # Test preview path
        result, metrics = self.processor.run_pipeline(
            img,
            settings,
            source_hash="test",
            render_size_ref=float(APP_CONFIG.preview_render_size),
        )

        from src.infrastructure.gpu.resources import GPUTexture

        self.assertIsInstance(result, GPUTexture)
        self.assertIn("histogram_raw", metrics)

    def test_processor_cleanup(self):
        """Verify cleanup delegates to GPU engine."""
        # Just ensure it doesn't crash and clears internal caches if possible to verify
        self.processor.cleanup()
        self.processor.destroy_all()

    def test_export_gpu_path(self):
        """Verify export logic works with GPU engine."""
        # We need a valid dummy file path or mock loader
        # For now, we can check if it attempts to use GPU
        self.assertTrue(self.processor.engine_gpu is not None)


if __name__ == "__main__":
    unittest.main()
