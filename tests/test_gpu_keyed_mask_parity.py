"""GPU/CPU parity for tone-limited masks: the key weight, the per-mask shape planes and
the ISO-R sum of limited and unlimited grades live in both the CPU kernel and
exposure.wgsl."""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskKey, MaskShape
from negpy.infrastructure.gpu.device import GPUDevice


def _frame() -> np.ndarray:
    rng = np.random.default_rng(3)
    grad = np.linspace(0.05, 0.9, 64, dtype=np.float32)
    img = np.repeat(grad[None, :], 64, axis=0)
    img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
    return np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))


def _masks() -> tuple:
    return (
        LocalMask(vertices=((0.5, 0.0), (0.5, 0.8)), stops=1.5, shape=MaskShape.GRADIENT, key=MaskKey.HIGHLIGHTS, key_zone=5.0),
        LocalMask(
            vertices=((0.0, 0.3), (1.0, 0.3), (1.0, 1.0), (0.0, 1.0)),
            stops=-0.5,
            grade=-25.0,
            feather=0.02,
            key=MaskKey.SHADOWS,
            key_zone=4.0,
            key_softness=2.0,
        ),
        LocalMask(vertices=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)), stops=0.3, grade=-20.0),
    )


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuKeyedMaskParity(unittest.TestCase):
    def _render(self, processor, settings, prefer_gpu):
        result, _ = processor.run_pipeline(
            _frame(), settings, "keyed-parity", render_size_ref=64.0, prefer_gpu=prefer_gpu, readback_metrics=False
        )
        arr = np.asarray(result.readback() if hasattr(result, "readback") else result)[:, :, :3]
        return arr.astype(np.float64)

    def test_cpu_gpu_match_with_limited_and_unlimited_masks(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        base = WorkspaceConfig()
        settings = replace(base, local=LocalAdjustmentsConfig(masks=_masks()))
        cpu = self._render(processor, settings, prefer_gpu=False)
        gpu = self._render(processor, settings, prefer_gpu=True)
        plain = self._render(processor, base, prefer_gpu=True)

        self.assertGreater(float(np.mean(np.abs(gpu - plain))), 0.02, "the masks did nothing on the GPU")
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")


if __name__ == "__main__":
    unittest.main()
