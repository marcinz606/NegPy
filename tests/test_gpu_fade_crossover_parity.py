"""GPU/CPU parity for Fade Restoration's crossover table: the CPU's `apply_crossover` and the table in
normalization.wgsl, the GPU's own measurement of a missing table, and a tiled export."""

import unittest
from dataclasses import replace

import numpy as np

from fade_crossover_scene import CROSSOVER, build_scene, estimate, scan_image, survival_changes
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode, mode_aware_exposure_reset
from negpy.infrastructure.gpu.device import GPUDevice


def _settings(changes: dict) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(
        base,
        process=replace(base.process, process_mode=ProcessMode.E6, **changes),
        exposure=mode_aware_exposure_reset(ProcessMode.E6, base.exposure),
        export=replace(base.export, export_resolution_mode="original"),
    )


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuFadeCrossoverParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = scan_image(CROSSOVER, build_scene(192, 144))
        cls.est = estimate(cls.img)

    def setUp(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        self.processor = ImageProcessor()
        if self.processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

    def _render(self, settings, prefer_gpu):
        result, metrics = self.processor.run_pipeline(
            self.img, settings, "crossover-parity", render_size_ref=float(max(self.img.shape[:2])), prefer_gpu=prefer_gpu
        )
        arr = np.asarray(result.readback() if hasattr(result, "readback") else result)[:, :, :3]
        return arr.astype(np.float64), metrics

    def test_the_stored_table_renders_alike(self):
        settings = _settings(survival_changes(self.est, 1.0))
        cpu, _ = self._render(settings, False)
        gpu, _ = self._render(settings, True)
        off_cpu, _ = self._render(_settings(survival_changes(self.est, 0.0)), False)
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.05, "the table must move the render")
        self.assertLess(float(np.abs(cpu - gpu).max()), 0.01)
        self.assertLess(float(np.abs(cpu - gpu).mean()), 0.002)

    def test_a_density_shift_renders_alike(self):
        changes = dict(survival_changes(self.est, 1.0), fade_density_shift=-0.3)
        settings = _settings(changes)
        cpu, _ = self._render(settings, False)
        gpu, _ = self._render(settings, True)
        plain, _ = self._render(_settings(survival_changes(self.est, 1.0)), False)
        self.assertGreater(float(np.abs(cpu - plain).max()), 0.05, "the shift must move the render")
        self.assertLess(float(np.abs(cpu - gpu).max()), 0.01)
        self.assertLess(float(np.abs(cpu - gpu).mean()), 0.002)

    def test_half_strength_renders_alike(self):
        settings = _settings(survival_changes(self.est, 0.5))
        cpu, _ = self._render(settings, False)
        gpu, _ = self._render(settings, True)
        self.assertLess(float(np.abs(cpu - gpu).max()), 0.01)

    def test_both_engines_measure_a_missing_table_alike(self):
        changes = survival_changes(self.est, 1.0)
        settings = _settings(dict(changes, fade_crossover_table=None))
        cpu, cpu_metrics = self._render(settings, False)
        gpu, gpu_metrics = self._render(settings, True)
        cpu_table, cpu_space = cpu_metrics["fade_crossover"]
        gpu_table, gpu_space = gpu_metrics["fade_crossover"]
        self.assertEqual(cpu_space, changes["fade_crossover_space"])
        self.assertEqual(gpu_space, cpu_space)
        self.assertEqual(len(gpu_table), len(cpu_table))
        self.assertLess(float(np.abs(np.asarray(gpu_table) - np.asarray(cpu_table)).max()), 0.02)
        self.assertLess(float(np.abs(cpu - gpu).mean()), 0.004)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestTiledFadeCrossover(unittest.TestCase):
    def setUp(self):
        from negpy.services.rendering.gpu_engine import GPUEngine

        self.engine = GPUEngine()
        # Wider than TILE_SIZE, so the export spans more than one tile.
        self.img = scan_image(CROSSOVER, build_scene(2400, 300))
        self.est = estimate(self.img)

    def tearDown(self):
        self.engine.destroy_all()

    def test_a_tiled_export_takes_the_reference_pass_table(self):
        """The reference pass measures the missing table, and every tile applies it."""
        settings = _settings(dict(survival_changes(self.est, 1.0), fade_crossover_table=None))
        tiled, metrics = self.engine._process_tiled(self.img, settings, scale_factor=1.0)
        tex, _ = self.engine.process_to_texture(self.img, settings, scale_factor=1.0, apply_layout=False)
        direct = self.engine._readback_downsampled(tex)
        off, _ = self.engine._process_tiled(self.img, _settings(survival_changes(self.est, 0.0)), scale_factor=1.0)
        self.assertEqual(tiled.shape, direct.shape)
        self.assertGreater(float(np.abs(tiled - off).mean()), 0.01, "the table must reach the tiles")
        self.assertLess(float(np.abs(tiled - direct).mean()), 0.002)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuFollowsCrossover(unittest.TestCase):
    def test_one_gpu_engine_follows_a_strength_change(self):
        from negpy.services.rendering.gpu_engine import GPUEngine

        img = scan_image(CROSSOVER, build_scene(192, 144))
        est = estimate(img)
        engine = GPUEngine()
        try:
            out = []
            for strength in (0.0, 1.0):
                tex, _ = engine.process_to_texture(img, _settings(survival_changes(est, strength)), scale_factor=1.0, apply_layout=False)
                out.append(engine._readback_downsampled(tex))
        finally:
            engine.destroy_all()
        self.assertGreater(float(np.abs(out[1] - out[0]).max()), 0.05)
