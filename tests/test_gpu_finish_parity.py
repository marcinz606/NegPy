"""GPU/CPU parity for the finish stage.

The finish math lives in two places — negpy/features/finish/logic.py and
finish.wgsl. They must agree, or GPU previews drift from CPU exports.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


def _test_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    h, w = 64, 64
    grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
    img = np.repeat(grad[None, :], h, axis=0)
    img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
    return np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuFinishParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu, size_ref=None):
        result, _ = processor.run_pipeline(
            img,
            settings,
            "parity-src",
            render_size_ref=size_ref or float(max(img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
        )
        if hasattr(result, "readback"):
            arr = np.asarray(result.readback())[:, :, :3]
        else:
            arr = np.asarray(result)[:, :, :3]
        return arr.astype(np.float64)

    def _assert_parity(self, settings):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        img = _test_image()
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

    def test_burn_radial(self):
        settings = WorkspaceConfig()
        settings = replace(settings, finish=replace(settings.finish, vignette_stops=1.5, vignette_size=0.6))
        self._assert_parity(settings)

    def test_burn_rectangular(self):
        settings = WorkspaceConfig()
        settings = replace(
            settings,
            finish=replace(settings.finish, vignette_stops=1.0, vignette_size=0.5, vignette_roundness=1.0),
        )
        self._assert_parity(settings)

    def test_filed_carrier(self):
        settings = WorkspaceConfig()
        settings = replace(
            settings,
            finish=replace(settings.finish, carrier_width=3.0, carrier_rough=1.0),
        )
        self._assert_parity(settings)

    def test_layout_mat_bottom_weight(self):
        """GPU layout pass: bottom-weighted mat matches CPU dims."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        settings = WorkspaceConfig()
        settings = replace(settings, finish=replace(settings.finish, border_size=1.0, border_bottom_weight=2.0))
        img = _test_image()
        gpu = self._render(processor, settings, img, prefer_gpu=True, size_ref=512.0)

        pw, ph, cw, ch, ox, oy, dpi = processor.engine_gpu._calculate_layout_dims(settings, img.shape[1], img.shape[0], 512.0)
        self.assertEqual(gpu.shape[:2], (ph, pw))
        # Bottom border (weight 2) is thicker than top
        self.assertGreater(ph - oy - ch, oy)
        # Mat corner is paper white
        np.testing.assert_allclose(gpu[0, 0], [1.0, 1.0, 1.0], atol=1e-3)

    def test_dodge_mixed_roundness(self):
        settings = WorkspaceConfig()
        settings = replace(
            settings,
            finish=replace(settings.finish, vignette_stops=-1.0, vignette_size=0.8, vignette_roundness=0.5),
        )
        self._assert_parity(settings)


if __name__ == "__main__":
    unittest.main()
