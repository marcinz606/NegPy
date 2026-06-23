"""GPU/CPU parity for the asymmetric H&D print curve.

The curve math lives in two places — the CPU kernel (logic.py) and the WGSL
shader (exposure.wgsl). They must agree, or GPU previews drift from CPU exports.
"""

import unittest

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuCurveParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu):
        result, _ = processor.run_pipeline(
            img, settings, "parity-src", render_size_ref=float(max(img.shape[:2])), prefer_gpu=prefer_gpu, readback_metrics=False
        )
        if hasattr(result, "readback"):
            arr = np.asarray(result.readback())[:, :, :3]
        else:
            arr = np.asarray(result)[:, :, :3]
        return arr.astype(np.float64)

    def test_cpu_gpu_match_default(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(0)
        # Synthetic linear negative: a smooth field so per-frame metrics are stable.
        h, w = 64, 64
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        settings = WorkspaceConfig()
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")


if __name__ == "__main__":
    unittest.main()
