"""GPU/CPU parity for the Lab luminance vignette.

The vignette math lives in two places -- the CPU kernel (features/vignette/logic.py
via negpy_core) and the WGSL shader (features/vignette/shaders/vignette.wgsl). They
must agree, or GPU previews drift from CPU exports.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuVignetteParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu):
        result, _ = processor.run_pipeline(
            img, settings, "parity-src", render_size_ref=float(max(img.shape[:2])), prefer_gpu=prefer_gpu, readback_metrics=False
        )
        if hasattr(result, "readback"):
            arr = np.asarray(result.readback())[:, :, :3]
        else:
            arr = np.asarray(result)[:, :, :3]
        return arr.astype(np.float64)

    def _make_img(self, rng, h=64, w=64):
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        return np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

    def _assert_parity(self, processor, img, settings):
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)
        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

    def _assert_engaged(self, processor, img, settings, base):
        # Both engines ignoring the vignette would pass parity vacuously.
        for label, gpu_path in (("cpu", False), ("gpu", True)):
            engaged = float(
                np.mean(
                    np.abs(
                        self._render(processor, base, img, prefer_gpu=gpu_path)
                        - self._render(processor, settings, img, prefer_gpu=gpu_path)
                    )
                )
            )
            self.assertGreater(engaged, 0.002, f"{label} vignette barely engaged ({engaged:.5f})")

    def test_cpu_gpu_match_darken(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(0)
        img = self._make_img(rng)
        s = WorkspaceConfig()
        settings = replace(s, vignette=replace(s.vignette, strength=-0.6, feather=0.5, roundness=0.0))
        self._assert_parity(processor, img, settings)
        self._assert_engaged(processor, img, settings, s)

    def test_cpu_gpu_match_lighten(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(1)
        img = self._make_img(rng)
        s = WorkspaceConfig()
        settings = replace(s, vignette=replace(s.vignette, strength=0.6, feather=0.5, roundness=0.0))
        self._assert_parity(processor, img, settings)
        self._assert_engaged(processor, img, settings, s)

    def test_cpu_gpu_match_ellipse(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(2)
        img = self._make_img(rng, h=48, w=96)
        s = WorkspaceConfig()
        settings = replace(s, vignette=replace(s.vignette, strength=-0.5, feather=0.7, roundness=1.0))
        self._assert_parity(processor, img, settings)
        self._assert_engaged(processor, img, settings, s)

    def test_cpu_gpu_match_hard_edge(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(3)
        img = self._make_img(rng)
        s = WorkspaceConfig()
        settings = replace(s, vignette=replace(s.vignette, strength=-0.8, feather=0.0, roundness=0.5))
        self._assert_parity(processor, img, settings)


if __name__ == "__main__":
    unittest.main()
