"""
E-6 with Normalize off must render the capture, not a print of it.

The contract these tests pin down:
  - at default settings nothing shapes the capture: the scene stage is an exact identity,
    and what reaches the display is that capture through the standard baseline + filmic
    rendering, so a slide opens looking the way a raw converter shows it (a bare
    linear-to-gamma encode does not — it is ~1.5 EV dark with no highlight roll-off);
  - the render stays exposure-dependent, so a bracketed set renders as a bracket
    (this is what measured bounds destroy, and why the window here is fixed);
  - every control that stays visible still moves the image, monotonically;
  - the GPU shader agrees with the CPU.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.features.exposure.normalization import LogNegativeBounds, normalize_log_image
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from negpy.features.exposure.transfer import (
    TRANSFER_CONSTANTS,
    TRANSFER_DENSITY_RANGE,
    apply_transfer_curve,
    display_rendering,
    is_transparency_transfer,
    transfer_bounds,
    transfer_curve_params,
    transfer_widths,
)
from negpy.features.process.capture_color import apply_camera_matrix, camera_to_working_matrix
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

# A real camera's XYZ->cam matrix (Nikon Z6/Z7-class), so the colour maths is exercised
# against a non-identity transform rather than a contrived one.
CAM_XYZ = [
    [0.6988, -0.1384, -0.0714],
    [-0.5631, 1.3410, 0.2447],
    [-0.1485, 0.2204, 0.7318],
]


def _e6_config(normalize=False, **exposure_overrides):
    cfg = DEFAULT_WORKSPACE_CONFIG
    process = replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=normalize)
    exposure = replace(cfg.exposure, **exposure_overrides) if exposure_overrides else cfg.exposure
    return replace(cfg, process=process, exposure=exposure)


CAMERA_WB = [1.9375, 1.0, 1.43359375]


def _run_stages(image, cfg, cam_xyz=CAM_XYZ, camera_wb=None):
    """Base + exposure stages only — the two the transfer path replaces."""
    h, w = image.shape[:2]
    ctx = PipelineContext(
        original_size=(h, w),
        scale_factor=1.0,
        process_mode=cfg.process.process_mode,
        cam_xyz=cam_xyz,
        camera_wb=camera_wb,
        wants_uv_grid=False,
    )
    norm = NormalizationProcessor(cfg.process).process(image, ctx)
    return np.asarray(PhotometricProcessor(cfg.exposure, cfg.local, cfg.process).process(norm, ctx)), ctx


def _rendered(scene_linear):
    """The standard rendering the transfer path applies on top of the untouched scene."""
    gain = 2.0 ** float(TRANSFER_CONSTANTS["transfer_baseline_ev"])
    return np.asarray(display_rendering(np.asarray(scene_linear, dtype=np.float32) * np.float32(gain)))


def _ramp(lo=1e-4, hi=0.6, n=512):
    v = np.geomspace(lo, hi, n).astype(np.float32)
    return np.stack([v, v, v], axis=-1)[None, :, :]


class TestModeSelection(unittest.TestCase):
    def test_only_e6_with_normalize_off_takes_the_transfer_path(self):
        self.assertTrue(is_transparency_transfer(ProcessMode.E6, False))
        self.assertFalse(is_transparency_transfer(ProcessMode.E6, True))
        self.assertFalse(is_transparency_transfer(ProcessMode.C41, False))
        self.assertFalse(is_transparency_transfer(ProcessMode.BW, False))

    def test_flat_intent_still_wins(self):
        """FLAT is an explicit export master; it must not be hijacked by the transfer."""
        from negpy.features.exposure.models import RenderIntent

        self.assertFalse(is_transparency_transfer(ProcessMode.E6, False, RenderIntent.FLAT))


class TestIdentityAtDefaults(unittest.TestCase):
    """Defaults must not shape the capture; the display rendering on top is fixed."""

    def test_grade_reference_matches_the_shipped_default(self):
        # If these drift apart the default render silently stops being identity, which is
        # the one property this feature exists to provide.
        self.assertAlmostEqual(
            float(TRANSFER_CONSTANTS["transfer_grade_ref"]),
            float(DEFAULT_WORKSPACE_CONFIG.exposure.grade),
            places=6,
        )

    def test_knee_width_reference_matches_the_shipped_widths(self):
        self.assertAlmostEqual(float(TRANSFER_CONSTANTS["transfer_width_ref"]), DEFAULT_WORKSPACE_CONFIG.exposure.toe_width)
        self.assertAlmostEqual(float(TRANSFER_CONSTANTS["transfer_width_ref"]), DEFAULT_WORKSPACE_CONFIG.exposure.shoulder_width)

    def test_curve_inverts_the_normalization_exactly(self):
        img = _ramp()
        norm = normalize_log_image(np.log10(np.clip(img, 1e-6, None)).astype(np.float32), LogNegativeBounds(*transfer_bounds()))
        cfg = DEFAULT_WORKSPACE_CONFIG
        offset, contrast, toe3, sh3 = transfer_curve_params(cfg.exposure)
        tw3, sw3 = transfer_widths(cfg.exposure)
        out = np.asarray(apply_transfer_curve(norm, offset, contrast, toe3, sh3, (0.0, 0.0, 0.0), tw3, sw3))
        expected = _rendered(img)
        rel = np.abs(out - expected) / np.maximum(expected, 1e-9)
        self.assertLess(float(rel.max()), 1e-4)

    def test_the_scene_stage_alone_is_an_exact_inverse(self):
        """Isolates the identity the controls deviate from, without the display rendering
        sitting on top of it."""
        img = _ramp()
        norm = normalize_log_image(np.log10(np.clip(img, 1e-6, None)).astype(np.float32), LogNegativeBounds(*transfer_bounds()))
        scene = np.power(10.0, -(norm.astype(np.float64) * TRANSFER_DENSITY_RANGE))
        rel = np.abs(scene - img) / np.maximum(img, 1e-9)
        self.assertLess(float(rel.max()), 1e-5)

    def test_display_rendering_is_monotonic_and_reaches_toward_white(self):
        x = np.geomspace(1e-5, 64.0, 4000).astype(np.float32)
        y = np.asarray(display_rendering(x))
        self.assertTrue(bool(np.all(np.diff(y) >= -1e-7)))
        self.assertAlmostEqual(float(display_rendering(np.array([0.0], np.float32))[0]), 0.0, places=6)
        # Highlights roll off to display white instead of stopping at the sensor's ceiling.
        self.assertGreater(float(y[-1]), 0.95)

    def test_pipeline_returns_the_capture_through_the_camera_matrix(self):
        # Near-neutral with moderate chroma: a plausible camera signal, so the matrix
        # stays in gamut and identity can be asserted exactly. Uniform-random RGB is not
        # a camera signal — a chunk of it lands outside the working space (see below).
        rng = np.random.default_rng(7)
        grey = (rng.random((24, 32, 1)) * 0.45 + 0.01).astype(np.float32)
        img = np.clip(grey * (1.0 + 0.15 * (rng.random((24, 32, 3)) - 0.5)), 1e-4, None).astype(np.float32)

        out, _ = _run_stages(img, _e6_config())

        scene = apply_camera_matrix(img, camera_to_working_matrix(CAM_XYZ))
        self.assertGreaterEqual(float(scene.min()), 0.0, "test fixture drifted out of gamut")
        expected = _rendered(scene)
        rel = np.abs(out - expected) / np.maximum(expected, 1e-6)
        self.assertLess(float(rel.max()), 1e-4)

    def test_out_of_gamut_colours_clamp_instead_of_producing_nan(self):
        """The matrix can send a saturated capture negative; the log stage must floor it."""
        rng = np.random.default_rng(3)
        img = (rng.random((16, 16, 3)) * 0.45 + 0.005).astype(np.float32)

        out, _ = _run_stages(img, _e6_config())

        self.assertTrue(bool(np.all(np.isfinite(out))))
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_absent_camera_matrix_passes_the_buffer_through(self):
        """A scanner TIFF carries no matrix; it is already in the working space."""
        rng = np.random.default_rng(8)
        img = (rng.random((16, 16, 3)) * 0.4 + 0.01).astype(np.float32)

        out, _ = _run_stages(img, _e6_config(), cam_xyz=None)

        self.assertLess(float(np.abs(out - _rendered(img)).max()), 1e-5)

    def test_degenerate_camera_matrix_is_rejected_not_applied(self):
        self.assertIsNone(camera_to_working_matrix(None))
        self.assertIsNone(camera_to_working_matrix([[0.0] * 3] * 3))
        self.assertIsNone(camera_to_working_matrix([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))


class TestExposureFaithfulness(unittest.TestCase):
    def test_bounds_are_fixed_and_content_independent(self):
        dark, bright = _ramp(hi=0.05), _ramp(hi=0.6)
        _, ctx_dark = _run_stages(dark, _e6_config())
        _, ctx_bright = _run_stages(bright, _e6_config())
        self.assertEqual(ctx_dark.metrics["final_bounds"].floors, ctx_bright.metrics["final_bounds"].floors)
        self.assertEqual(ctx_dark.metrics["final_bounds"].ceils, ctx_bright.metrics["final_bounds"].ceils)

    def _bracket_means(self, normalize):
        base = _ramp(hi=0.4)
        return [float(_run_stages((base * s).astype(np.float32), _e6_config(normalize=normalize))[0].mean()) for s in (0.25, 0.5, 1.0, 2.0)]

    def test_a_bracket_renders_as_a_bracket(self):
        """Measured bounds make exposures of one scene converge. A transparency must not.

        Asserted against the competing behaviour rather than a fixed ratio: the display
        rendering compresses, so a 2x scene change is deliberately less than 2x on screen
        (every tone curve does this, Lightroom's included). What must hold is that each
        exposure stays clearly, monotonically apart.
        """
        transfer = self._bracket_means(normalize=False)
        converged = self._bracket_means(normalize=True)

        self.assertEqual(transfer, sorted(transfer))
        for lo, hi in zip(transfer, transfer[1:]):
            self.assertGreater(hi / max(lo, 1e-9), 1.35)
        # An 8x scene range must survive as a wide output range, not collapse to one render.
        transfer_spread = transfer[-1] / max(transfer[0], 1e-9)
        converged_spread = converged[-1] / max(converged[0], 1e-9)
        self.assertGreater(transfer_spread, 4.0)
        self.assertGreater(transfer_spread, 4.0 * converged_spread)

    def test_normalize_on_still_converges(self):
        """The old behaviour has to survive untouched on the other side of the toggle."""
        means = self._bracket_means(normalize=True)
        self.assertLess(max(means) / max(min(means), 1e-9), 1.5)


class TestControlsStayLive(unittest.TestCase):
    """Hidden paper controls are fine; visible ones that do nothing are not."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.img = (rng.random((16, 24, 3)) * 0.35 + 0.02).astype(np.float32)
        self.base, _ = _run_stages(self.img, _e6_config())

    def _rendered(self, **overrides):
        return _run_stages(self.img, _e6_config(**overrides))[0]

    def test_density_moves_exposure_and_higher_is_darker(self):
        lighter = self._rendered(density=0.5)
        darker = self._rendered(density=1.5)
        self.assertGreater(float(lighter.mean()), float(self.base.mean()))
        self.assertLess(float(darker.mean()), float(self.base.mean()))

    def test_grade_moves_contrast(self):
        harder = self._rendered(grade=70.0)
        softer = self._rendered(grade=160.0)
        self.assertGreater(float(harder.std()), float(self.base.std()))
        self.assertLess(float(softer.std()), float(self.base.std()))

    def test_toe_and_shoulder_move_their_own_end_only(self):
        # Measured on a wide ramp and in relative terms: an absolute delta in linear
        # light is dominated by the highlights whatever the curve does, so it cannot
        # tell the two knees apart.
        ramp = _ramp(lo=1e-3, hi=0.6)
        base = _run_stages(ramp, _e6_config())[0][0, :, 1]
        values = ramp[0, :, 1]
        shadows, highs = values < np.percentile(values, 10), values > np.percentile(values, 90)

        def rel_shift(**overrides):
            out = _run_stages(ramp, _e6_config(**overrides))[0][0, :, 1]
            r = np.abs(out - base) / np.maximum(base, 1e-9)
            return float(r[shadows].mean()), float(r[highs].mean())

        toe_shadow, toe_high = rel_shift(toe=0.8)
        self.assertGreater(toe_shadow, toe_high)

        sh_shadow, sh_high = rel_shift(shoulder=0.8)
        self.assertGreater(sh_high, sh_shadow)

    def test_knee_widths_are_wired_to_the_width_sliders(self):
        narrow = self._rendered(toe=0.8, toe_width=0.5)
        wide = self._rendered(toe=0.8, toe_width=5.0)
        self.assertGreater(float(np.abs(wide - narrow).max()), 1e-4)

    def test_white_balance_still_shifts_channels(self):
        warmed = self._rendered(wb_cyan=0.5)
        delta = (warmed - self.base).reshape(-1, 3).mean(axis=0)
        self.assertGreater(abs(float(delta[0])), 1e-4)

    def test_curve_stays_monotonic_under_extreme_settings(self):
        for overrides in (
            {"toe": 1.0, "shoulder": 1.0},
            {"toe": -1.0, "shoulder": -1.0},
            {"grade": 50.0, "density": 0.0},
            {"grade": 180.0, "density": 2.0},
        ):
            with self.subTest(**overrides):
                out = _run_stages(_ramp(), _e6_config(**overrides))[0][0, :, 1]
                self.assertTrue(bool(np.all(np.diff(out) >= -1e-6)), f"non-monotonic for {overrides}")


class TestCaptureTogglesAreInert(unittest.TestCase):
    """Linear RAW and Narrowband are both sticky global settings that would otherwise
    change this render invisibly. Hiding them is only safe because they do nothing here."""

    def _wb_applied(self):
        rng = np.random.default_rng(23)
        grey = (rng.random((16, 16, 1)) * 0.4 + 0.01).astype(np.float32)
        return np.clip(grey * (1.0 + 0.1 * (rng.random((16, 16, 3)) - 0.5)), 1e-4, None).astype(np.float32)

    def test_linear_raw_renders_the_same_as_a_camera_wb_decode(self):
        """Linear RAW decodes with user_wb=[1,1,1,1]; the row-normalized matrix assumes a
        balanced signal, so the multipliers have to be folded back in."""
        balanced = self._wb_applied()
        # Undo the decode-side white balance to synthesize the Linear RAW buffer.
        wb = np.asarray(CAMERA_WB, dtype=np.float32) / CAMERA_WB[1]
        unbalanced = (balanced / wb).astype(np.float32)

        cfg_wb = _e6_config()
        cfg_linear = replace(cfg_wb, process=replace(cfg_wb.process, linear_raw=True))

        with_wb, _ = _run_stages(balanced, cfg_wb)
        linear, _ = _run_stages(unbalanced, cfg_linear, camera_wb=CAMERA_WB)

        rel = np.abs(linear - with_wb) / np.maximum(with_wb, 1e-6)
        self.assertLess(float(rel.max()), 1e-4)

    def test_without_the_fold_linear_raw_would_cast_badly(self):
        """Guards the guard: if the fold silently stopped happening, the test above has to
        be capable of failing."""
        balanced = self._wb_applied()
        wb = np.asarray(CAMERA_WB, dtype=np.float32) / CAMERA_WB[1]
        unbalanced = (balanced / wb).astype(np.float32)

        # camera_wb=None is the un-folded path.
        uncorrected, _ = _run_stages(unbalanced, _e6_config(), camera_wb=None)
        with_wb, _ = _run_stages(balanced, _e6_config())

        ratio_ref = float(with_wb[..., 0].mean() / with_wb[..., 1].mean())
        ratio_bad = float(uncorrected[..., 0].mean() / uncorrected[..., 1].mean())
        self.assertGreater(abs(ratio_ref - ratio_bad), 0.2)

    def test_camera_wb_fold_is_green_normalised(self):
        """Only channel ratios may change; overall exposure must not."""
        plain = camera_to_working_matrix(CAM_XYZ)
        folded = camera_to_working_matrix(CAM_XYZ, CAMERA_WB)
        self.assertIsNotNone(folded)
        # Green column scales by wb_g/wb_g = 1, so the green response is untouched.
        self.assertTrue(np.allclose(plain[:, 1], folded[:, 1], atol=1e-6))

    def test_degenerate_camera_wb_is_ignored(self):
        base = camera_to_working_matrix(CAM_XYZ)
        for bad in ([0.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0], [float("nan"), 1.0, 1.0]):
            with self.subTest(bad=bad):
                self.assertTrue(np.allclose(camera_to_working_matrix(CAM_XYZ, bad), base, atol=1e-6))


class TestAutomaticGradingIsOff(unittest.TestCase):
    def test_auto_density_and_auto_grade_do_not_change_the_render(self):
        """They meter the frame to pick a look, which is what this path exists to avoid."""
        rng = np.random.default_rng(13)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        on, _ = _run_stages(img, _e6_config(auto_exposure=True, auto_normalize_contrast=True))
        off, _ = _run_stages(img, _e6_config(auto_exposure=False, auto_normalize_contrast=False))
        self.assertLess(float(np.abs(on - off).max()), 1e-6)

    def test_crosstalk_unmix_is_not_applied(self):
        """It models negative-film dye crosstalk and defaults to 0.5 — it would tint the
        pass-through, so the transfer path must ignore it."""
        rng = np.random.default_rng(17)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        cfg = _e6_config()
        strong = replace(cfg, process=replace(cfg.process, crosstalk_strength=1.0))
        self.assertLess(float(np.abs(_run_stages(img, cfg)[0] - _run_stages(img, strong)[0]).max()), 1e-6)


class TestNormalizationContract(unittest.TestCase):
    def test_transfer_bounds_span_the_declared_range(self):
        floors, ceils = transfer_bounds()
        for f, c in zip(floors, ceils):
            self.assertAlmostEqual(f, 0.0)
            self.assertAlmostEqual(f - c, TRANSFER_DENSITY_RANGE)

    def test_metrics_downstream_panels_read_are_published(self):
        out, ctx = _run_stages(_ramp(), _e6_config())
        for key in ("final_bounds", "log_bounds", "normalized_log", "histogram_density", "norm_density_range"):
            self.assertIn(key, ctx.metrics)

    def test_default_process_config_keeps_the_print_path(self):
        """PhotometricProcessor's process_config defaults to a print; a missing argument
        must never silently route an existing caller into the transfer."""
        self.assertTrue(ProcessConfig().e6_normalize)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuTransferParity(unittest.TestCase):
    """The transfer curve lives twice — transfer.py and transfer.wgsl. They must agree,
    or the preview drifts from the export."""

    def _render(self, processor, settings, img, prefer_gpu, cam_xyz=CAM_XYZ):
        result, _ = processor.run_pipeline(
            img,
            settings,
            f"transfer-parity-{prefer_gpu}",
            render_size_ref=float(max(img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
            cam_xyz=cam_xyz,
        )
        arr = np.asarray(result.readback()) if hasattr(result, "readback") else np.asarray(result)
        return arr[:, :, :3].astype(np.float64)

    def _both(self, settings, cam_xyz=CAM_XYZ):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        # The shipped autocrop_offset insets the CPU render by a pixel, which would
        # compare two different framings rather than two curve implementations.
        settings = replace(settings, geometry=replace(settings.geometry, autocrop_offset=0))

        rng = np.random.default_rng(2)
        h, w = 64, 64
        grad = np.linspace(0.02, 0.5, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.005, img.shape).astype(np.float32))

        cpu = self._render(processor, settings, img, prefer_gpu=False, cam_xyz=cam_xyz)
        gpu = self._render(processor, settings, img, prefer_gpu=True, cam_xyz=cam_xyz)
        self.assertEqual(cpu.shape, gpu.shape)
        return cpu, gpu

    def _assert_parity(self, cpu, gpu):
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")

    def test_defaults_match(self):
        self._assert_parity(*self._both(_e6_config()))

    def test_no_camera_matrix_matches(self):
        self._assert_parity(*self._both(_e6_config(), cam_xyz=None))

    def test_moved_controls_match(self):
        """Every live control at once, including the per-channel trims that the CPU
        folds and the shader reads from its own uniform lanes."""
        settings = _e6_config(
            density=1.4,
            grade=75.0,
            toe=0.6,
            shoulder=-0.5,
            toe_width=4.0,
            shoulder_width=1.2,
            toe_trim_red=0.3,
            shoulder_trim_blue=-0.25,
            toe_width_trim_green=1.0,
            shoulder_width_trim_red=-0.5,
            wb_cyan=0.3,
            wb_yellow=-0.2,
        )
        self._assert_parity(*self._both(settings))


if __name__ == "__main__":
    unittest.main()
