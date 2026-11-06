"""The preview layout pass must never resample content above its own resolution.

The canvas quotes zoom against the buffer the pipeline was handed
(``render_long_edge``), so a layout that upscales makes 1:1 read closer than one
scan pixel per device pixel.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import AspectRatio, ExportConfig, ExportResolutionMode, WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.export.print import PrintService
from negpy.services.rendering.gpu_engine import GPUEngine


def _dims(settings: WorkspaceConfig, cw: int, ch: int, size_ref):
    engine = GPUEngine.__new__(GPUEngine)
    return engine._calculate_layout_dims(settings, cw, ch, size_ref)


class TestPreviewLayoutScale(unittest.TestCase):
    def test_half_size_decode_is_not_upscaled(self):
        """A 3008 px preview of a 6016 px scan stays 3008 px under a 4000 px ref."""
        _, _, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 3008, 2008, 4000.0)
        self.assertEqual((cw, ch), (3008, 2008))

    def test_crop_is_not_upscaled(self):
        _, _, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 800, 533, 1600.0)
        self.assertEqual((cw, ch), (800, 533))

    def test_larger_content_still_scales_down_to_ref(self):
        _, _, cw, _, _, _, _ = _dims(WorkspaceConfig(), 6016, 4016, 1600.0)
        self.assertEqual(cw, 1600)

    def test_border_keeps_its_share_of_the_frame(self):
        settings = replace(WorkspaceConfig(), finish=replace(WorkspaceConfig().finish, border_size=1.0))
        pw, _, cw, _, _, _, _ = _dims(settings, 3008, 2008, 4000.0)
        clamped, _, clamped_cw, _, _, _, _ = _dims(settings, 3008, 2008, 3008.0)
        self.assertEqual(cw, 3008)
        self.assertAlmostEqual((pw - cw) / cw, (clamped - clamped_cw) / clamped_cw, places=2)

    def test_export_path_keeps_its_own_resolution(self):
        """No size_ref: the export math owns the paper, and may upscale to a print size."""
        _, _, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 800, 533, None)
        self.assertEqual((cw, ch), (800, 533))

    def test_unscaled_content_keeps_every_row(self):
        """1355 / (1355 / 908) is 907.999... in float; the layout keeps the 908th row."""
        _, ph, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 1355, 908, 1355.0)
        self.assertEqual((cw, ch, ph), (1355, 908, 908))

    def test_unscaled_portrait_content_keeps_every_column(self):
        pw, _, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 908, 1355, 1355.0)
        self.assertEqual((cw, ch, pw), (908, 1355, 908))


class TestLayoutIntegerAspect(unittest.TestCase):
    """Every layout formula scales a side by the image's own w:h without a float round trip."""

    SIZES = [(1355, 908), (500, 390), (1792, 1216), (1353, 911)]

    def test_fit_content_returns_the_image_at_its_own_size(self):
        for w, h in self.SIZES:
            self.assertEqual(PrintService.fit_content(w, h, w, h + 50), (w, h))
            self.assertEqual(PrintService.fit_content(h, w, h + 50, w), (h, w))

    def test_original_paper_matches_the_image(self):
        for w, h in self.SIZES:
            self.assertEqual(PrintService.paper_dims_from_long_edge(w, AspectRatio.ORIGINAL, w, h), (w, h))
            self.assertEqual(PrintService.paper_dims_from_long_edge(w, AspectRatio.ORIGINAL, h, w), (h, w))

    def test_apply_layout_keeps_an_unscaled_image_whole(self):
        export = replace(ExportConfig(), export_resolution_mode=ExportResolutionMode.TARGET_PX, export_target_long_edge_px=1355)
        out, rect = PrintService.apply_layout(np.zeros((908, 1355, 3), np.float32), export)
        self.assertEqual(out.shape[:2], (908, 1355))
        self.assertEqual(rect[2:], (1355, 908))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestPreviewSizeParity(unittest.TestCase):
    def test_cpu_and_gpu_render_the_same_size(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        img = np.random.default_rng(1).uniform(0.05, 0.9, (908, 1355, 3)).astype(np.float32)
        shapes = []
        for gpu in (False, True):
            result, _ = processor.run_pipeline(img, WorkspaceConfig(), "row-gap", render_size_ref=1355.0, prefer_gpu=gpu)
            arr = result.readback() if hasattr(result, "readback") else result
            shapes.append(np.asarray(arr).shape[:2])
        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(shapes[1], (908, 1355))


if __name__ == "__main__":
    unittest.main()
