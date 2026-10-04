import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


def _cropped_and_warped_settings() -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(
        s,
        geometry=replace(s.geometry, crop_rect=(0.15, 0.1, 0.8, 0.9), fine_rotation=1.5, converge_v=3.0),
    )


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestCropPreviewFullParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu, crop_preview_full):
        result, metrics = processor.run_pipeline(
            img,
            settings,
            "parity-src",
            render_size_ref=float(max(img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
            crop_preview_full=crop_preview_full,
        )
        arr = np.asarray(result.readback())[:, :, :3] if hasattr(result, "readback") else np.asarray(result)[:, :, :3]
        return arr.astype(np.float64), metrics

    def _img(self):
        rng = np.random.default_rng(0)
        h, w = 96, 128
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        return np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

    def test_full_frame_matches_cpu_at_the_crop_tools_own_tolerance(self):
        """Same numerical gap the cropped (already-shipped) GPU path has against
        the CPU engine -- full_frame introduces nothing beyond that."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        settings = _cropped_and_warped_settings()
        img = self._img()

        cropped_cpu, _ = self._render(processor, settings, img, prefer_gpu=False, crop_preview_full=False)
        cropped_gpu, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)
        cropped_tolerance = float(np.max(np.abs(cropped_cpu - cropped_gpu)))

        full_cpu, cpu_metrics = self._render(processor, settings, img, prefer_gpu=False, crop_preview_full=True)
        full_gpu, gpu_metrics = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=True)

        self.assertEqual(full_cpu.shape, img.shape)  # the whole frame, not the crop
        self.assertEqual(full_cpu.shape, full_gpu.shape)
        self.assertLessEqual(float(np.max(np.abs(full_cpu - full_gpu))), cropped_tolerance + 1e-9)
        # The overlay must still track the real crop, not the frame the render widened to.
        self.assertEqual(cpu_metrics["active_roi"], gpu_metrics["active_roi"])
        self.assertIsNotNone(cpu_metrics["active_roi"])

    def test_full_frame_off_still_crops_on_gpu(self):
        """full_frame is opt-in: nothing here reaches for it unasked."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        settings = _cropped_and_warped_settings()
        img = self._img()

        cropped_gpu, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)
        self.assertNotEqual(cropped_gpu.shape, img.shape)

    def test_toggling_full_frame_is_picked_up_with_no_other_setting_change(self):
        """Entering/leaving the crop tool alone must resize the render -- this is the
        one case a bare WorkspaceConfig diff cannot see, since full_frame is a render
        parameter, not a config field."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        settings = _cropped_and_warped_settings()
        img = self._img()

        cropped, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)
        full, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=True)
        back_to_cropped, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)

        self.assertNotEqual(cropped.shape, full.shape)
        self.assertEqual(cropped.shape, back_to_cropped.shape)

    def test_full_frame_drops_border_and_carrier(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        base = _cropped_and_warped_settings()
        framed = replace(base, finish=replace(base.finish, border_size=1.0, carrier_width=2.0))
        img = self._img()

        bordered, _ = self._render(processor, framed, img, prefer_gpu=True, crop_preview_full=False)
        cropped, _ = self._render(processor, base, img, prefer_gpu=True, crop_preview_full=False)
        self.assertGreater(bordered.shape[0], cropped.shape[0])

        plain_cpu, _ = self._render(processor, base, img, prefer_gpu=False, crop_preview_full=True)
        plain_gpu, _ = self._render(processor, base, img, prefer_gpu=True, crop_preview_full=True)
        tolerance = float(np.max(np.abs(plain_cpu - plain_gpu)))
        for prefer_gpu, plain in ((False, plain_cpu), (True, plain_gpu)):
            full, metrics = self._render(processor, framed, img, prefer_gpu=prefer_gpu, crop_preview_full=True)
            self.assertEqual(full.shape, img.shape)
            self.assertIn(metrics.get("content_rect"), (None, (0, 0, img.shape[1], img.shape[0])))
            self.assertLessEqual(float(np.max(np.abs(full - plain))), tolerance + 1e-9)

    def test_cached_late_stages_keep_the_finish_pass(self):
        from negpy.features.geometry.models import AspectRatio
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        base = _cropped_and_warped_settings()
        vignetted = replace(base, finish=replace(base.finish, vignette_stops=1.5))
        img = self._img()

        fresh, _ = self._render(processor, vignetted, img, prefer_gpu=True, crop_preview_full=True)
        cached, _ = self._render(processor, vignetted, img, prefer_gpu=True, crop_preview_full=True)
        self.assertLessEqual(float(np.max(np.abs(cached - fresh))), 1e-6)

        # An export-only change re-runs layout over the cached finish texture.
        bordered = replace(vignetted, finish=replace(vignetted.finish, border_size=0.5))
        self._render(processor, bordered, img, prefer_gpu=True, crop_preview_full=False)
        relayout = replace(bordered, export=replace(bordered.export, paper_aspect_ratio=AspectRatio.R_1_1))
        cached_path, _ = self._render(processor, relayout, img, prefer_gpu=True, crop_preview_full=False)
        fresh_engine = ImageProcessor()
        if fresh_engine.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        fresh_path, _ = self._render(fresh_engine, relayout, img, prefer_gpu=True, crop_preview_full=False)
        self.assertEqual(cached_path.shape, fresh_path.shape)
        self.assertLessEqual(float(np.max(np.abs(cached_path - fresh_path))), 1e-6)
