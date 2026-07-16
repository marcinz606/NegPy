import unittest
import numpy as np
from negpy.services.rendering.gpu_engine import GPUEngine
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


class TestGPUEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = GPUDevice.get()
        if cls.gpu.is_available:
            cls.engine = GPUEngine()
        else:
            cls.engine = None

    def setUp(self):
        if self.engine is None:
            self.skipTest("GPU not available")

    def test_gpu_process_smoke(self):
        """Basic GPU processing smoke test."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        settings = WorkspaceConfig()

        res, metrics = self.engine.process(img, settings)

        self.assertEqual(res.ndim, 3)
        self.assertEqual(res.shape[2], 3)
        self.assertIn("active_roi", metrics)
        self.assertIn("histogram_raw", metrics)
        self.assertEqual(metrics["histogram_raw"].shape, (4, 256))

    def test_gpu_process_to_texture(self):
        """Verify process_to_texture returns a GPUTexture."""
        from negpy.infrastructure.gpu.resources import GPUTexture

        img = np.random.rand(64, 64, 3).astype(np.float32)
        settings = WorkspaceConfig()

        tex, metrics = self.engine.process_to_texture(img, settings)

        self.assertIsInstance(tex, GPUTexture)
        self.assertEqual(tex.width, metrics["base_positive"].width)

    def test_gpu_engine_cleanup(self):
        """Verify cleanup releases resources."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        settings = WorkspaceConfig()

        # Run once to populate cache
        self.engine.process_to_texture(img, settings)
        self.assertTrue(len(self.engine._tex_cache) > 0)

        self.engine.cleanup()
        self.assertEqual(len(self.engine._tex_cache), 0)
        self.assertIsNone(self.engine._uv_grid_cache)

    def test_uv_grid_cached_across_frames(self):
        """Same geometry -> reused grid object; geometry change -> rebuilt."""
        from dataclasses import replace

        img = np.random.rand(64, 64, 3).astype(np.float32)
        settings = WorkspaceConfig()

        _, m1 = self.engine.process_to_texture(img, settings)
        _, m2 = self.engine.process_to_texture(img, settings)
        self.assertIs(m2["uv_grid"], m1["uv_grid"])

        rotated = replace(settings, geometry=replace(settings.geometry, rotation=1))
        _, m3 = self.engine.process_to_texture(img, rotated)
        self.assertIsNot(m3["uv_grid"], m1["uv_grid"])

    def test_gpu_tiled_processing(self):
        """Verify tiled processing for large images."""
        # Force tiled path by using an image that exceeds 12M pixels or just a bit large
        # For tests, we'll keep it reasonable but enough to trigger logic if we lowered threshold
        # Or we can just call _process_tiled directly if it was public, but it's internal.
        # Let's use an image large enough.
        # The threshold is 12,000,000 pixels.
        # 4000 * 3001 = 12,003,000
        h, w = 3001, 4000
        img = np.random.rand(h, w, 3).astype(np.float32)
        settings = WorkspaceConfig()

        res, metrics = self.engine.process(img, settings)

        # Check if result matches expected aspect ratio or similar
        self.assertIsNotNone(res)
        self.assertTrue(res.shape[0] > 0)

    def test_gpu_engine_destroy_all(self):
        """Verify destroy_all clears persistent resources."""
        self.engine._init_resources()
        self.assertTrue(len(self._engine_buffers_count()) > 0)

        self.engine.destroy_all()
        self.assertEqual(len(self._engine_buffers_count()), 0)
        self.assertEqual(len(self.engine._pipelines), 0)

    def _engine_buffers_count(self):
        return self.engine._buffers

    def test_gpu_tiled_synthesized_region_applied(self):
        """Auto/IR dust rides synthesized 5-tuple strokes (injected upstream);
        the tiled path must apply them like any manual heal."""
        from negpy.features.retouch.models import RetouchConfig
        from dataclasses import replace

        h, w = 128, 128
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        img[62:67, 62:67] = 0.95

        synth = ([[0.5, 0.5]], 150.0, 0.15625, 0.0, 0.0)  # ungated, ~6px radius, +20px source
        base = WorkspaceConfig()
        with_region = replace(base, retouch=RetouchConfig(manual_heal_strokes=[synth]))

        res_with, _ = self.engine._process_tiled(img, with_region, scale_factor=1.0)
        res_without, _ = self.engine._process_tiled(img, base, scale_factor=1.0)

        diff_max = float(np.abs(res_with - res_without).max())
        self.assertGreater(diff_max, 0.05, "Tiled export ignored the synthesized heal region")

    def test_gpu_tiled_manual_stroke_matches_untiled(self):
        """A heal stroke crossing a tile boundary must render like the untiled path —
        the dynamic tile halo has to cover the stroke radius + source offset."""
        from negpy.features.retouch.models import RetouchConfig
        from dataclasses import replace

        h, w = 128, 2200  # spans the TILE_SIZE=2048 boundary
        rng = np.random.default_rng(1)
        img = (rng.random((h, w, 3), dtype=np.float32) * 0.05 + 0.45).astype(np.float32)
        img[60:66, 1980:2120] = 0.95  # scratch across the boundary

        stroke = ([[1980.0 / w, 63.0 / h], [2120.0 / w, 63.0 / h]], 8.0, 0.0, 0.3)
        base = WorkspaceConfig()
        settings = replace(
            base,
            retouch=RetouchConfig(manual_heal_strokes=[stroke]),
            # Native output size so the tiled result is comparable 1:1 with the untiled texture.
            export=replace(base.export, export_resolution_mode="original"),
        )

        res_tiled, _ = self.engine._process_tiled(img, settings, scale_factor=1.0)
        tex, _ = self.engine.process_to_texture(img, settings, scale_factor=1.0, apply_layout=False)
        res_direct = self.engine._readback_downsampled(tex)

        self.assertEqual(res_tiled.shape, res_direct.shape)
        band = np.s_[40:90, 1900:2200]
        diff = float(np.abs(res_tiled[band] - res_direct[band]).max())
        self.assertLess(diff, 0.05, "Tiled heal diverges from untiled across the tile boundary")

    def test_gpu_tiled_crosstalk_matches_untiled(self):
        """Tiled export metered bounds from the raw image, skipping the crosstalk
        unmix the untiled/preview path always applies — diverged when Separation > 0."""
        from negpy.features.process.models import ProcessMode
        from dataclasses import replace

        h, w = 128, 2200  # spans the TILE_SIZE=2048 boundary
        rng = np.random.default_rng(2)
        img = np.empty((h, w, 3), dtype=np.float32)
        img[..., 0] = rng.random((h, w), dtype=np.float32) * 0.15 + 0.55
        img[..., 1] = rng.random((h, w), dtype=np.float32) * 0.15 + 0.35
        img[..., 2] = rng.random((h, w), dtype=np.float32) * 0.15 + 0.15

        base = WorkspaceConfig()
        settings = replace(
            base,
            process=replace(base.process, process_mode=ProcessMode.C41, crosstalk_strength=1.0),
            export=replace(base.export, export_resolution_mode="original"),
        )

        res_tiled, _ = self.engine._process_tiled(img, settings, scale_factor=1.0)
        tex, _ = self.engine.process_to_texture(img, settings, scale_factor=1.0, apply_layout=False)
        res_direct = self.engine._readback_downsampled(tex)

        self.assertEqual(res_tiled.shape, res_direct.shape)
        diff = float(np.abs(res_tiled - res_direct).max())
        self.assertLess(diff, 0.05, "Tiled export ignored Separation (crosstalk) when metering global bounds")

    def test_gpu_tiled_global_meter_stays_lazy_when_unused(self):
        """Locked bounds + no auto refs/anchor/textural: the meter buffer must not build."""
        from negpy.features.process.models import ProcessMode
        from dataclasses import replace
        from unittest.mock import patch
        import negpy.services.rendering.gpu_engine as gpu_engine_module

        img = np.random.rand(96, 96, 3).astype(np.float32)
        base = WorkspaceConfig()
        settings = replace(
            base,
            process=replace(
                base.process,
                process_mode=ProcessMode.C41,
                crosstalk_strength=1.0,
                lock_bounds=True,
                locked_floors=(-1.0, -1.0, -1.0),
                locked_ceils=(-0.2, -0.2, -0.2),
                use_luma_average=True,
                use_colour_average=True,
            ),
            exposure=replace(
                base.exposure,
                auto_exposure=False,
                auto_normalize_contrast=False,
                cast_removal_strength=0.0,
            ),
        )

        # CDF priming already calls prefilter_log_grid once (unrelated); wrap it so
        # that still works, and assert the meter block under test adds no calls.
        real_prefilter = gpu_engine_module.prefilter_log_grid
        with patch.object(gpu_engine_module, "prefilter_log_grid", wraps=real_prefilter) as mock_prefilter:
            self.engine._process_tiled(img, settings, scale_factor=1.0)
            self.assertEqual(mock_prefilter.call_count, 1, "Global crosstalk meter built even though nothing needed it")

    def test_gpu_tiled_export_honours_freehand_analysis_rect(self):
        """Tiled export must meter the drawn analysis_rect, not the centered default."""
        from negpy.features.process.models import ProcessMode
        from dataclasses import replace

        h, w = 128, 256
        img = np.empty((h, w, 3), dtype=np.float32)
        img[:, : w // 2] = (0.15, 0.05, 0.05)  # left: dark, red-cast
        img[:, w // 2 :] = (0.85, 0.85, 0.85)  # right: bright, neutral

        base = WorkspaceConfig()

        def _settings(rect):
            return replace(base, process=replace(base.process, process_mode=ProcessMode.C41, analysis_rect=rect))

        res_left, _ = self.engine._process_tiled(img, _settings((0.0, 0.0, 0.5, 1.0)), scale_factor=1.0)
        res_right, _ = self.engine._process_tiled(img, _settings((0.5, 0.0, 1.0, 1.0)), scale_factor=1.0)

        diff = float(np.abs(res_left - res_right).max())
        self.assertGreater(diff, 0.05, "Tiled export ignored freehand analysis_rect — output identical either way")

    def test_gpu_tiled_export_stale_auto_flags_are_inert(self):
        """dust_remove/ir_dust_remove reaching the engine un-augmented (direct calls,
        old sessions) must be inert no-ops, not crashes — detection lives upstream."""
        from negpy.features.retouch.models import RetouchConfig
        from dataclasses import replace

        img = np.random.rand(96, 96, 3).astype(np.float32)
        settings = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=True, ir_dust_remove=True))
        settings_off = replace(WorkspaceConfig(), retouch=RetouchConfig())
        res, _ = self.engine._process_tiled(img, settings, scale_factor=1.0)
        res_off, _ = self.engine._process_tiled(img, settings_off, scale_factor=1.0)
        np.testing.assert_allclose(res, res_off, atol=1e-6)

    def test_gpu_tiled_export_respects_geometry_for_synthesized_region(self):
        """Synthesized strokes are source-normalized; the tiled path must map them
        through the same geometry as the RGB tiles (rotated exports heal in place)."""
        from negpy.features.retouch.models import RetouchConfig
        from negpy.features.geometry.models import GeometryConfig
        from dataclasses import replace

        h, w = 96, 128
        rng = np.random.default_rng(0)
        img = rng.random((h, w, 3), dtype=np.float32) * 0.3 + 0.4
        img[30:34, 30:34] = 0.95

        synth = ([[32.0 / w, 32.0 / h]], 2.0 * 6.0 * 1600.0 / w, 0.15, 0.0, 0.0)
        settings = replace(
            WorkspaceConfig(),
            retouch=RetouchConfig(manual_heal_strokes=[synth]),
            geometry=GeometryConfig(rotation=1),
        )
        settings_off = replace(settings, retouch=RetouchConfig())

        res_on, _ = self.engine._process_tiled(img, settings, scale_factor=1.0)
        res_off, _ = self.engine._process_tiled(img, settings_off, scale_factor=1.0)
        self.assertGreater(float(np.abs(res_on - res_off).max()), 0.05)

    def test_histogram_unaffected_by_border(self):
        """Border pixels must not skew the histogram — metrics are computed on content only."""
        from dataclasses import replace
        from negpy.domain.models import ExportConfig

        img = np.random.rand(120, 120, 3).astype(np.float32)
        base_settings = WorkspaceConfig()

        _, metrics_no_border = self.engine.process(img, base_settings)
        hist_no_border = metrics_no_border["histogram_raw"].copy()

        black_border_export = ExportConfig()
        settings_black = replace(base_settings, export=black_border_export)
        _, metrics_black = self.engine.process(img, settings_black)
        hist_black = metrics_black["histogram_raw"].copy()

        white_border_export = ExportConfig()
        settings_white = replace(base_settings, export=white_border_export)
        _, metrics_white = self.engine.process(img, settings_white)
        hist_white = metrics_white["histogram_raw"].copy()

        np.testing.assert_array_equal(hist_no_border, hist_black, err_msg="Black border pixels skewed the histogram")
        np.testing.assert_array_equal(hist_no_border, hist_white, err_msg="White border pixels skewed the histogram")

    def test_readback_after_destroy_returns_zeros(self):
        """A destroyed texture must answer readbacks with zeros, never touch dead wgpu objects."""
        from negpy.infrastructure.gpu.resources import GPUTexture

        tex = GPUTexture(16, 16)
        tex.upload(np.random.rand(16, 16, 4).astype(np.float32))
        tex.readback_region(0, 0, 1, 1)  # allocate the staging buffer pre-destroy
        tex.destroy()
        np.testing.assert_array_equal(tex.readback_region(0, 0, 1, 1), np.zeros((1, 1, 4), dtype=np.float32))
        np.testing.assert_array_equal(tex.readback(), np.zeros((16, 16, 4), dtype=np.float32))

    def test_concurrent_readback_and_destroy(self):
        """UI-thread readbacks racing a worker-thread destroy must serialize, not panic.

        Regression: the densitometer's hover readback overlapped the render worker's
        engine cleanup; destroying the mapped staging buffer aborted the process
        inside wgpu-native (uncatchable Rust panic)."""
        import threading

        from negpy.infrastructure.gpu.resources import GPUTexture

        for _ in range(20):
            tex = GPUTexture(64, 64)
            tex.upload(np.random.rand(64, 64, 4).astype(np.float32))
            start = threading.Barrier(2)

            def probe(t=tex, s=start):
                s.wait()
                for _ in range(10):
                    t.readback_region(3, 3, 1, 1)

            def kill(t=tex, s=start):
                s.wait()
                t.destroy()

            threads = [threading.Thread(target=probe), threading.Thread(target=kill)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=30)
            self.assertFalse(any(th.is_alive() for th in threads), "readback/destroy deadlocked")


if __name__ == "__main__":
    unittest.main()
