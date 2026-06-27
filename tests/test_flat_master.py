import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import (
    AspectRatio,
    ColorSpace,
    ExportConfig,
    ExportFormat,
    ExportResolutionMode,
    WorkspaceConfig,
    flat_export_config,
    flat_master_config,
)
from negpy.features.exposure.logic import flat_curve_params
from negpy.features.exposure.models import RenderIntent
from negpy.services.rendering.engine import DarkroomEngine
from negpy.services.rendering.image_processor import ImageProcessor


def _ramp_image(n: int = 64) -> np.ndarray:
    """A 3-channel normalized-log ramp from below 0 to above 1 (unclamped)."""
    x = np.linspace(-0.3, 1.3, n, dtype=np.float32)
    return np.stack([x, x, x], axis=-1)[None, :, :]


def _ramp_in_range(lo: float, hi: float, n: int = 64) -> np.ndarray:
    """A 3-channel normalized-log ramp over a valid operating sub-range."""
    x = np.linspace(lo, hi, n, dtype=np.float32)
    return np.stack([x, x, x], axis=-1)[None, :, :]


class TestFlatCurveParams(unittest.TestCase):
    def test_returns_low_contrast_neutral_curve(self):
        gain, lift = flat_curve_params()
        # Log master: gain < 1 keeps it flat; lift is the shadow code.
        self.assertTrue(0.0 < gain < 1.0)
        self.assertTrue(0.0 < lift < 1.0)


class TestFlatPhotometric(unittest.TestCase):
    def _render(self, exposure_overrides=None, image=None) -> np.ndarray:
        from negpy.domain.interfaces import PipelineContext
        from negpy.features.exposure.models import ExposureConfig
        from negpy.features.exposure.processor import PhotometricProcessor
        from negpy.features.process.models import ProcessMode

        cfg = ExposureConfig(render_intent=RenderIntent.FLAT)
        if exposure_overrides:
            cfg = replace(cfg, **exposure_overrides)
        img = _ramp_image() if image is None else image
        ctx = PipelineContext(scale_factor=1.0, original_size=(1, img.shape[1]), process_mode=ProcessMode.C41)
        # Provide metrics a print render would consume so we prove flat ignores them.
        ctx.metrics["metered_anchor"] = 0.9
        return PhotometricProcessor(cfg).process(img, ctx)

    def test_monotonic_decreasing(self):
        out = self._render()[0, :, 1]
        diffs = np.diff(out)
        self.assertTrue(np.all(diffs <= 1e-6), "flat positive must be monotonic (decreasing) in scene density")

    def test_no_clipping_with_headroom(self):
        # Over the valid operating range the log code stays well off both endpoints.
        out = self._render(image=_ramp_in_range(0.0, 1.0))
        self.assertGreater(float(out.min()), 0.02, "flat master should keep shadows off the black point")
        self.assertLess(float(out.max()), 0.98, "flat master should keep highlights off the white point")

    def test_low_contrast(self):
        out = self._render(image=_ramp_in_range(0.0, 1.0))[0, :, 1]
        self.assertLess(float(out.max() - out.min()), 0.97)

    def test_flat_is_log_linear(self):
        # Regression guard: output IS the log code value — gamma-less and linear in
        # the log signal: code == lift + gain*(1 - val). No 10^-D, no sRGB.
        gain, lift = flat_curve_params()
        val = np.linspace(0.05, 0.95, 40, dtype=np.float32)
        img = np.stack([val, val, val], axis=-1)[None, :, :]
        out = self._render(image=img)[0, :, 1]
        np.testing.assert_allclose(out, lift + gain * (1.0 - val), atol=1e-4)

    def test_ignores_auto_and_creative_print_decisions(self):
        base = self._render()
        # None of these print decisions may affect the flat render.
        varied = self._render(
            {
                "auto_exposure": True,
                "auto_normalize_contrast": True,
                "cast_removal": True,
                "grade": 50.0,
                "density": 1.8,
                "toe": 0.9,
                "shoulder": 0.9,
                "paper_dmin": True,
                "surround": True,
                "flare": True,
            }
        )
        np.testing.assert_allclose(base, varied, atol=1e-6)


class TestFlatEngineSkipsStages(unittest.TestCase):
    def test_creative_stages_bypassed(self):
        engine = DarkroomEngine()
        img = np.random.rand(48, 48, 3).astype(np.float32)

        flat = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, render_intent=RenderIntent.FLAT))
        res_default = engine.process(img, flat, source_hash="flat_a")

        # Crank every creative stage; flat intent must ignore them all.
        loud = replace(
            flat,
            lab=replace(flat.lab, sharpen=1.0, saturation=2.0, clahe_strength=1.0, glow_amount=1.0, halation_strength=1.0),
            toning=replace(flat.toning, sepia_strength=1.0, selenium_strength=1.0),
            finish=replace(flat.finish, vignette_strength=0.9),
        )
        res_loud = engine.process(img, loud, source_hash="flat_b")

        np.testing.assert_allclose(res_default, res_loud, atol=1e-6)

    def test_flat_differs_from_print(self):
        engine = DarkroomEngine()
        img = np.random.rand(48, 48, 3).astype(np.float32)
        print_cfg = WorkspaceConfig()
        flat_cfg = flat_master_config(print_cfg)

        res_print = engine.process(img, print_cfg, source_hash="p")
        res_flat = engine.process(img, flat_cfg, source_hash="f")
        self.assertFalse(np.allclose(res_print, res_flat, atol=1e-3))


class TestFlatRollConsistency(unittest.TestCase):
    """A flat master must render an identical patch across frames when the roll
    shares one locked normalization baseline (req #8)."""

    def _images_with_shared_patch(self):
        a = np.full((32, 32, 3), 0.05, dtype=np.float32)
        b = np.full((32, 32, 3), 0.80, dtype=np.float32)
        a[0:8, 0:8] = 0.2
        b[0:8, 0:8] = 0.2  # identical mid patch, very different surroundings
        return a, b

    def _flat_cfg(self, *, locked: bool):
        base = flat_master_config(WorkspaceConfig())
        proc = replace(
            base.process,
            use_roll_average=locked,
            locked_floors=(-2.0, -2.0, -2.0) if locked else (0.0, 0.0, 0.0),
            locked_ceils=(-0.1, -0.1, -0.1) if locked else (0.0, 0.0, 0.0),
        )
        return replace(base, process=proc)

    def test_locked_bounds_give_identical_patch(self):
        a, b = self._images_with_shared_patch()
        cfg = self._flat_cfg(locked=True)
        ra = DarkroomEngine().process(a, cfg, source_hash="ra")
        rb = DarkroomEngine().process(b, cfg, source_hash="rb")
        np.testing.assert_allclose(ra[0:8, 0:8], rb[0:8, 0:8], atol=1e-6)

    def test_per_frame_bounds_drift(self):
        # Proves the locking matters: without it, the same patch renders differently.
        a, b = self._images_with_shared_patch()
        cfg = self._flat_cfg(locked=False)
        ra = DarkroomEngine().process(a, cfg, source_hash="ra2")
        rb = DarkroomEngine().process(b, cfg, source_hash="rb2")
        self.assertFalse(np.allclose(ra[0:8, 0:8], rb[0:8, 0:8], atol=1e-3))


class TestFlatConfigHelpers(unittest.TestCase):
    def test_flat_master_config_disables_automation(self):
        cfg = flat_master_config(WorkspaceConfig())
        self.assertEqual(cfg.exposure.render_intent, RenderIntent.FLAT)
        self.assertFalse(cfg.exposure.auto_exposure)
        self.assertFalse(cfg.exposure.auto_normalize_contrast)
        self.assertFalse(cfg.exposure.cast_removal)
        self.assertFalse(cfg.exposure.surround)
        self.assertFalse(cfg.exposure.flare)
        self.assertFalse(cfg.exposure.paper_dmin)
        self.assertEqual(cfg.exposure.toe, 0.0)
        self.assertEqual(cfg.exposure.shoulder, 0.0)

    def test_flat_master_config_preserves_framing_and_white_balance(self):
        src = WorkspaceConfig(exposure=replace(WorkspaceConfig().exposure, wb_cyan=0.1, wb_yellow=-0.2))
        cfg = flat_master_config(src)
        self.assertEqual(cfg.exposure.wb_cyan, 0.1)
        self.assertEqual(cfg.exposure.wb_yellow, -0.2)
        self.assertEqual(cfg.geometry, src.geometry)
        self.assertEqual(cfg.process, src.process)

    def test_flat_export_config_tiff_wide_gamut_full_res(self):
        out = flat_export_config(ExportConfig(), fmt=ExportFormat.TIFF)
        self.assertEqual(out.export_fmt, ExportFormat.TIFF)
        self.assertEqual(out.export_color_space, ColorSpace.PROPHOTO.value)
        self.assertEqual(out.export_resolution_mode, ExportResolutionMode.ORIGINAL.value)
        self.assertEqual(out.paper_aspect_ratio, AspectRatio.ORIGINAL)

    def test_flat_export_config_dng(self):
        out = flat_export_config(ExportConfig(), fmt=ExportFormat.DNG)
        self.assertEqual(out.export_fmt, ExportFormat.DNG)


class TestRenderIntentSerialization(unittest.TestCase):
    def test_round_trips_through_flat_dict(self):
        cfg = flat_master_config(WorkspaceConfig())
        restored = WorkspaceConfig.from_flat_dict(cfg.to_dict())
        self.assertEqual(restored.exposure.render_intent, RenderIntent.FLAT)

    def test_legacy_config_defaults_to_print(self):
        # An old workspace dict without render_intent must default to PRINT.
        data = WorkspaceConfig().to_dict()
        data.pop("render_intent", None)
        restored = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(restored.exposure.render_intent, RenderIntent.PRINT)


class TestImageProcessorFlatRouting(unittest.TestCase):
    def test_is_flat_detection(self):
        flat = flat_master_config(WorkspaceConfig())
        self.assertTrue(ImageProcessor._is_flat(flat))
        self.assertFalse(ImageProcessor._is_flat(WorkspaceConfig()))


class TestFlatWideGamut(unittest.TestCase):
    """Camera-native → ProPhoto matrix for the flat master (req #6)."""

    # A representative camera matrix (rawpy rgb_xyz_matrix == LibRaw cam_xyz, XYZ→cam),
    # 4×3 with the conventional zero last row for an RGB sensor (from a Nikon NEF).
    NIKON = [
        [0.8828, -0.2406, -0.0694],
        [-0.4874, 1.2603, 0.2541],
        [-0.066, 0.1509, 0.7587],
        [0.0, 0.0, 0.0],
    ]

    def test_matrix_is_3x3_and_neutral_preserving(self):
        from negpy.infrastructure.display.camera_color import camera_to_prophoto_matrix

        m = camera_to_prophoto_matrix(self.NIKON)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.shape, (3, 3))
        # Row-normalised → a neutral camera value maps to a neutral ProPhoto value.
        np.testing.assert_allclose(m.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-5)

    def test_apply_keeps_neutral_neutral(self):
        from negpy.infrastructure.display.camera_color import apply_camera_to_prophoto, camera_to_prophoto_matrix

        m = camera_to_prophoto_matrix(self.NIKON)
        assert m is not None
        gray = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = apply_camera_to_prophoto(gray, m)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_apply_clamps_negatives(self):
        from negpy.infrastructure.display.camera_color import apply_camera_to_prophoto, camera_to_prophoto_matrix

        m = camera_to_prophoto_matrix(self.NIKON)
        assert m is not None
        sat = np.zeros((1, 1, 3), dtype=np.float32)
        sat[0, 0] = [1.0, 0.0, 0.0]
        out = apply_camera_to_prophoto(sat, m)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_degenerate_inputs_return_none(self):
        from negpy.infrastructure.display.camera_color import camera_to_prophoto_matrix

        self.assertIsNone(camera_to_prophoto_matrix(None))
        self.assertIsNone(camera_to_prophoto_matrix(np.zeros((4, 3))))  # singular
        self.assertIsNone(camera_to_prophoto_matrix(np.zeros((2, 2))))  # wrong shape


class TestFlatDngEncode(unittest.TestCase):
    def test_dng_bytes_roundtrip(self):
        import io

        import tifffile

        rgb = (np.random.rand(16, 16, 3) * 65535).astype(np.uint16)
        data = ImageProcessor._encode_dng_bytes(rgb)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

        readback = tifffile.imread(io.BytesIO(data))
        np.testing.assert_array_equal(readback, rgb)


if __name__ == "__main__":
    unittest.main()
