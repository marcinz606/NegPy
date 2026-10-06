"""Shadow Reach on a slide reads the metered anchor whatever Auto Density says, so the GPU
must meter it for Auto Grade alone. Preview renders read metrics back and always metered
it; an export does not, and printed the slide without Shadow Reach."""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice


def _flat_slide(h: int, w: int) -> np.ndarray:
    """A low-contrast positive with texture everywhere: its dark tail never reaches the
    Shadow Reach density on its own, so the reach floor sets the contrast."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tone = 0.18 + 0.12 * (xx / w) + 0.04 * np.sin(xx / 23.0) * np.cos(yy / 17.0) + 0.03 * np.sin((xx + yy) / 41.0)
    return np.clip(np.stack([tone, tone * 0.95, tone * 0.9], axis=-1), 1e-4, 1.0).astype(np.float32)


def _slide(**exposure) -> WorkspaceConfig:
    s = WorkspaceConfig()
    s = replace(s, process=replace(s.process, process_mode=ProcessMode.E6), export=replace(s.export, export_resolution_mode="original"))
    return replace(s, exposure=replace(s.exposure, auto_exposure=False, auto_normalize_contrast=True, **exposure))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestSlideShadowReachOnExport(unittest.TestCase):
    def setUp(self):
        from negpy.services.rendering.gpu_engine import GPUEngine

        self.engines = []
        self.make = lambda: self.engines.append(GPUEngine()) or self.engines[-1]

    def tearDown(self):
        for engine in self.engines:
            engine.destroy_all()

    def _render(self, img, settings, readback):
        engine = self.make()
        tex, _ = engine.process_to_texture(img, settings, scale_factor=1.0, apply_layout=False, readback_metrics=readback)
        return engine._readback_downsampled(tex)

    def test_shadow_reach_applies_with_and_without_metric_readback(self):
        img = _flat_slide(256, 384)
        settings = _slide()
        preview = self._render(img, settings, readback=True)
        export = self._render(img, settings, readback=False)
        no_grade = self._render(img, replace(settings, exposure=replace(settings.exposure, auto_normalize_contrast=False)), readback=True)
        self.assertGreater(float(np.abs(preview - no_grade).mean()), 0.005, "Auto Grade must move this frame")
        np.testing.assert_allclose(export, preview, atol=1e-4)

    def test_tiled_export_applies_shadow_reach(self):
        img = _flat_slide(300, 2400)
        settings = _slide()
        engine = self.make()
        tiled, _ = engine._process_tiled(img, settings, scale_factor=1.0)
        direct = self._render(img, settings, readback=True)
        self.assertEqual(tiled.shape, direct.shape)
        self.assertLess(float(np.abs(tiled - direct).mean()), 0.0005)
