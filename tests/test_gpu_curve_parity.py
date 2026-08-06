"""GPU/CPU parity for the asymmetric H&D print curve.

The curve math lives in two places — the CPU kernel (logic.py) and the WGSL
shader (exposure.wgsl). They must agree, or GPU previews drift from CPU exports.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


def _extreme_trims_settings() -> WorkspaceConfig:
    """Hard per-channel grade + WP/BP trims + Paper Black. Exercises the crossover
    trims, the print-curve uniforms, and — with Dye Separation off — the C-41 Cast
    Removal neutral-axis, all of which must track between the CPU and WGSL paths."""
    s = WorkspaceConfig()
    return replace(
        s,
        # WP/BP trims bake into the GPU normalization floors/ceils — parity
        # guards that pack against the CPU per-channel offset path.
        process=replace(s.process, white_point_trim_red=0.08, black_point_trim_blue=-0.06),
        exposure=replace(
            s.exposure,
            grade_trim_red=25.0,
            grade_trim_blue=-20.0,
            toe_trim_red=0.5,
            toe_trim_blue=-0.4,
            shoulder_trim_green=0.3,
            paper_black=False,
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

    def test_cpu_gpu_match_trims_paper_black(self):
        """Crossover trims fold CPU-side, Paper Black + midtone gamma ride the
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

        settings = _extreme_trims_settings()
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

    def test_cpu_gpu_match_dye_separation(self):
        """Dye Separation composes into the same dye_mix slot as the paper's
        real crosstalk -- uses Kodak Endura (a real, non-identity dye matrix) so
        this exercises the actual sat @ dye composition, not just the trivial
        no-paper/identity case."""
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

        s = WorkspaceConfig()
        settings = replace(
            s,
            exposure=replace(
                s.exposure,
                paper_profile="kodak_endura",
                dye_separation=1.5,
                dye_separation_trim_red=0.3,
                dye_separation_trim_blue=-0.2,
            ),
        )
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

    def test_cpu_gpu_match_separation_damping(self):
        """Separation Damping's per-pixel k is written twice -- the CPU kernel and
        WGSL -- so the two can drift, and it moves the sat matrix out of the
        dye_mix slot on both paths. Both directions: above 1.0 the damping lifts
        muted colour and pulls vivid colour down, below 1.0 the reverse."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(3)
        h, w = 64, 64
        # The shared fixture is near-neutral (channels x1.0/0.95/0.9), so the
        # damping barely engages on it and parity would pass vacuously. Sweep
        # chroma down the frame instead: neutral at the top, strongly coloured at
        # the bottom, which is the axis this operator keys off.
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        luma = np.repeat(grad[None, :], h, axis=0)
        chroma = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        img = np.stack([luma, luma * (1.0 - 0.55 * chroma), luma * (1.0 - 0.75 * chroma)], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        s = WorkspaceConfig()
        for sep, trim_r in ((1.5, 0.3), (0.6, -0.2)):
            with self.subTest(dye_separation=sep):
                settings = replace(
                    s,
                    exposure=replace(
                        s.exposure,
                        paper_profile="kodak_endura",
                        dye_separation=sep,
                        dye_separation_trim_red=trim_r,
                        separation_damping=1.0,
                    ),
                )
                cpu = self._render(processor, settings, img, prefer_gpu=False)
                gpu = self._render(processor, settings, img, prefer_gpu=True)

                self.assertEqual(cpu.shape, gpu.shape)
                mad = float(np.mean(np.abs(cpu - gpu)))
                mx = float(np.max(np.abs(cpu - gpu)))
                self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
                self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

                # Guard the parity assertions above against passing vacuously:
                # both engines agreeing to do nothing is not parity.
                flat = replace(settings, exposure=replace(settings.exposure, separation_damping=0.0))
                for label, gpu_path in (("cpu", False), ("gpu", True)):
                    engaged = float(
                        np.mean(np.abs(self._render(processor, flat, img, prefer_gpu=gpu_path) - (cpu if label == "cpu" else gpu)))
                    )
                    self.assertGreater(engaged, 0.002, f"{label} damping barely engaged ({engaged:.5f})")

    def test_cpu_gpu_match_trims_no_dye_separation(self):
        """Dye Separation below 1.0 scales chroma toward neutral, which masks
        CPU/GPU chroma disagreements. With it at identity (its default), the Cast
        Removal neutral-axis must be measured against the same (pre-trim) bounds on
        both paths — otherwise the WP/BP trims shift the CPU's cast strength and the
        paths drift (~0.057)."""
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

        settings = _extreme_trims_settings()
        self.assertEqual(settings.exposure.dye_separation, 1.0)
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

    def test_cpu_gpu_match_clahe(self):
        """CLAHE at full strength (issue #524 regression). 128x128 keeps each
        8x8 tile at >=256 samples so a handful of upstream f32/f64 bin flips
        can't move a tile CDF anywhere near the tolerance gate."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(3)
        h, w = 128, 128
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        settings = WorkspaceConfig()
        settings = replace(settings, lab=replace(settings.lab, clahe_strength=1.0))
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

    def test_cpu_gpu_match_local_grade(self):
        """Dodge/burn EV and the local grade share one uploaded texture (.r and .g)
        on the GPU and two kernel arguments on the CPU — a mask that changes both
        catches a swapped plane or a missing multiply in either path."""
        from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(5)
        h, w = 64, 64
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        s = WorkspaceConfig()
        square = ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85))
        masks = (
            PolygonMask(vertices=square, stops=0.8, feather=0.03, grade=-35.0),
            PolygonMask(vertices=((0.0, 0.6), (0.5, 0.6), (0.5, 1.0), (0.0, 1.0)), stops=0.0, feather=0.0, grade=30.0),
        )
        settings = replace(s, local=LocalAdjustmentsConfig(masks=masks))
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

        # Both paths ignoring the grade plane would pass parity vacuously.
        ev_only = replace(settings, local=LocalAdjustmentsConfig(masks=tuple(replace(m, grade=0.0) for m in masks)))
        for label, gpu_path in (("cpu", False), ("gpu", True)):
            engaged = float(np.mean(np.abs(self._render(processor, ev_only, img, prefer_gpu=gpu_path) - (cpu if label == "cpu" else gpu))))
            self.assertGreater(engaged, 0.002, f"{label} local grade barely engaged ({engaged:.5f})")


if __name__ == "__main__":
    unittest.main()
