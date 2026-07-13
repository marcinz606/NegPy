"""GPU/CPU parity for the asymmetric H&D print curve.

The curve math lives in two places — the CPU kernel (logic.py) and the WGSL
shader (exposure.wgsl). They must agree, or GPU previews drift from CPU exports.
"""

import unittest
from dataclasses import replace

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

    def test_cpu_gpu_match_trims_true_black(self):
        """Crossover trims fold CPU-side, True Black + midtone gamma ride the
        uniforms — the WGSL mirror must track all of them."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(1)
        h, w = 64, 64
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        settings = WorkspaceConfig()
        settings = replace(
            settings,
            # WP/BP trims bake into the GPU normalization floors/ceils — parity
            # guards that pack against the CPU per-channel offset path.
            process=replace(
                settings.process,
                white_point_trim_red=0.08,
                black_point_trim_blue=-0.06,
            ),
            exposure=replace(
                settings.exposure,
                grade_trim_red=25.0,
                grade_trim_blue=-20.0,
                toe_trim_red=0.5,
                toe_trim_blue=-0.4,
                shoulder_trim_green=0.3,
                true_black=True,
                midtone_gamma=0.3,
                midtone_gamma_trim_red=0.4,
                midtone_gamma_trim_blue=-0.3,
                toe_width_trim_red=1.5,
                shoulder_width_trim_blue=-1.0,
                shadow_density=0.4,
                highlight_density=-0.15,
                shadow_grade=-20.0,
                highlight_grade=15.0,
                shadow_grade_trim_red=12.0,
                shadow_grade_trim_blue=-8.0,
                highlight_grade_trim_green=-10.0,
                toe=-0.6,
                paper_profile="fuji_crystal",
            ),
        )
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

    def test_cpu_gpu_match_bw_trims(self):
        """B&W end-to-end parity under per-channel trims + a chroma lab op.
        The tight tolerance catches any CPU-only or GPU-only B&W grading step
        (e.g. the removed CPU auto black point, which failed this at mad~0.03)."""
        from negpy.features.process.models import ProcessMode
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(2)
        h, w = 64, 64
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        settings = WorkspaceConfig()
        settings = replace(
            settings,
            process=replace(settings.process, process_mode=ProcessMode.BW),
            exposure=replace(
                settings.exposure,
                toe_trim_red=0.5,
                midtone_gamma_trim_green=0.4,
                grade_trim_blue=-20.0,
            ),
            lab=replace(settings.lab, saturation=1.8),
        )
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")


if __name__ == "__main__":
    unittest.main()
