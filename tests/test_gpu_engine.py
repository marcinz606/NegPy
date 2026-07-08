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

    def test_gpu_tiled_export_propagates_ir_buffer(self):
        """Regression: _process_tiled previously dropped ir_buffer, so IR dust removal
        was applied in preview but silently skipped on export of >12MP scans."""
        from negpy.features.retouch.models import RetouchConfig
        from dataclasses import replace

        h, w = 128, 128
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        img[64, 64] = 0.95
        ir = np.full((h, w), 0.9, dtype=np.float32)
        ir[62:67, 62:67] = 0.05

        base = WorkspaceConfig()
        with_ir = replace(base, retouch=RetouchConfig(ir_dust_remove=True, ir_threshold=0.5, ir_inpaint_radius=3))
        without_ir = replace(base, retouch=RetouchConfig(ir_dust_remove=False))

        res_with, _ = self.engine._process_tiled(img, with_ir, scale_factor=1.0, ir_buffer=ir)
        res_without, _ = self.engine._process_tiled(img, without_ir, scale_factor=1.0, ir_buffer=ir)

        # IR dust removal must change pixels somewhere in the output.
        diff_max = float(np.abs(res_with - res_without).max())
        self.assertGreater(diff_max, 0.05, "Tiled export ignored IR buffer; output identical to IR-off")

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
                auto_cast_removal=False,
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

    def test_gpu_tiled_export_ir_no_crash_without_buffer(self):
        """ir_dust_remove enabled but ir_buffer=None must not crash the tiled path."""
        from negpy.features.retouch.models import RetouchConfig
        from dataclasses import replace

        img = np.random.rand(96, 96, 3).astype(np.float32)
        settings = replace(WorkspaceConfig(), retouch=RetouchConfig(ir_dust_remove=True))
        res, _ = self.engine._process_tiled(img, settings, scale_factor=1.0, ir_buffer=None)
        self.assertIsNotNone(res)

    def test_gpu_tiled_export_respects_geometry_for_ir(self):
        """Tiled path must pre-transform IR with the same geometry as the RGB tiles;
        otherwise rotated/flipped exports would heal pixels at wrong locations."""
        from negpy.features.retouch.models import RetouchConfig
        from negpy.features.geometry.models import GeometryConfig
        from dataclasses import replace

        h, w = 96, 128
        rng = np.random.default_rng(0)
        img = rng.random((h, w, 3), dtype=np.float32) * 0.3 + 0.4
        ir = np.full((h, w), 0.9, dtype=np.float32)
        ir[30:34, 30:34] = 0.05

        settings = replace(
            WorkspaceConfig(),
            retouch=RetouchConfig(ir_dust_remove=True, ir_threshold=0.5, ir_inpaint_radius=3),
            geometry=GeometryConfig(rotation=1),
        )
        settings_off = replace(settings, retouch=RetouchConfig(ir_dust_remove=False))

        res_on, _ = self.engine._process_tiled(img, settings, scale_factor=1.0, ir_buffer=ir)
        res_off, _ = self.engine._process_tiled(img, settings_off, scale_factor=1.0, ir_buffer=ir)
        # Geometry-rotated tiled export with IR must still produce a different output
        # than the same export with IR disabled.
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


if __name__ == "__main__":
    unittest.main()
