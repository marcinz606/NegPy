"""
A slide must render the capture, not a print of it.

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

import itertools
import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.features.exposure.normalization import LogNegativeBounds, normalize_log_image
from negpy.services.rendering.engine import base_processor, exposure_processor
from negpy.features.transparency.logic import (
    TRANSFER_CONSTANTS,
    TRANSFER_DENSITY_RANGE,
    apply_transfer_curve,
    display_rendering,
    transfer_assumed_anchor,
    transfer_auto_terms,
    transfer_bounds,
    transfer_curve_params,
    transfer_highlight_hold_density,
    transfer_shadow_reach_density,
    transfer_widths,
)
from negpy.features.process.capture_color import apply_camera_matrix, camera_to_working_matrix
from negpy.features.process.models import ProcessConfig, ProcessMode, auto_meter_for_mode, cast_removal_for_mode
from negpy.features.process.path import RenderPath, render_path
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

# A real camera's XYZ->cam matrix (Nikon Z6/Z7-class), so the color maths is exercised
# against a non-identity transform rather than a contrived one.
CAM_XYZ = [
    [0.6988, -0.1384, -0.0714],
    [-0.5631, 1.3410, 0.2447],
    [-0.1485, 0.2204, 0.7318],
]


def _e6_config(positive_source=False, **exposure_overrides):
    cfg = DEFAULT_WORKSPACE_CONFIG
    process = replace(cfg.process, process_mode=ProcessMode.E6, positive_source=positive_source)
    # The values a slide actually starts at — the same rewrite every route into E-6
    # applies. Without it these read the negative's defaults, which no slide ever carries.
    overrides = {
        "cast_removal_strength": cast_removal_for_mode(ProcessMode.E6, cfg.exposure.cast_removal_strength),
        "auto_exposure": auto_meter_for_mode(ProcessMode.E6, cfg.exposure.auto_exposure),
        "auto_normalize_contrast": auto_meter_for_mode(ProcessMode.E6, cfg.exposure.auto_normalize_contrast),
        **exposure_overrides,
    }
    exposure = replace(cfg.exposure, **overrides)
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
    norm = base_processor(cfg).process(image, ctx)
    return np.asarray(exposure_processor(cfg).process(norm, ctx)), ctx


def _rendered(scene_linear):
    """The standard rendering the transfer path applies on top of the untouched scene."""
    gain = 2.0 ** float(TRANSFER_CONSTANTS["transfer_baseline_ev"])
    return np.asarray(display_rendering(np.asarray(scene_linear, dtype=np.float32) * np.float32(gain)))


def _ramp(lo=1e-4, hi=0.6, n=512):
    v = np.geomspace(lo, hi, n).astype(np.float32)
    return np.stack([v, v, v], axis=-1)[None, :, :]


class TestModeSelection(unittest.TestCase):
    def test_e6_takes_the_transfer_path(self):
        def path(mode):
            return render_path(ProcessConfig(process_mode=mode))

        self.assertIs(path(ProcessMode.E6), RenderPath.TRANSFER)
        self.assertIs(path(ProcessMode.C41), RenderPath.PRINT)
        self.assertIs(path(ProcessMode.BW), RenderPath.PRINT)

    def test_positive_source_is_slide_only(self):
        """A file already positivized before NegPy saw it has nothing left to meter or
        invert, and only Slide can carry that: a config built in any other mode drops
        the flag, so the negative path is never reached with it set."""
        for mode in (ProcessMode.C41, ProcessMode.BW):
            self.assertFalse(ProcessConfig(process_mode=mode, positive_source=True).positive_source)
        self.assertTrue(ProcessConfig(process_mode=ProcessMode.E6, positive_source=True).positive_source)
        self.assertIs(render_path(ProcessConfig(process_mode=ProcessMode.E6, positive_source=True)), RenderPath.POSITIVE)

    def test_flat_render_of_a_raw_slide_folds_camera_wb_like_its_print(self):
        """A Flat slide decodes like its print, without white balance, so the base stage
        folds the as-shot multipliers back in."""
        from negpy.features.exposure.models import RenderIntent
        from negpy.services.rendering.engine import DarkroomEngine

        cfg = _e6_config(render_intent=RenderIntent.FLAT)
        img = np.ascontiguousarray(np.broadcast_to(_ramp(1e-3, 0.5, 64), (8, 64, 3)).astype(np.float32))

        def render(camera_wb):
            ctx = PipelineContext(
                original_size=img.shape[:2],
                scale_factor=1.0,
                process_mode=cfg.process.process_mode,
                cam_xyz=CAM_XYZ,
                camera_wb=camera_wb,
                wants_uv_grid=False,
                cache_stages=False,
            )
            return np.asarray(DarkroomEngine().process(img.copy(), cfg, "flat-wb", ctx))

        self.assertGreater(float(np.abs(render(None) - render(CAMERA_WB)).max()), 1e-3)


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

    def test_out_of_gamut_colors_clamp_instead_of_producing_nan(self):
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


class TestPositiveSourceSkipsDisplayRendering(unittest.TestCase):
    """A Positive frame is already a finished image, not a raw capture below the
    sensor's white level, so the baseline gain and filmic display_rendering — both
    otherwise unconditional on this path — must not run."""

    def test_output_is_the_bare_scene_not_the_rendered_one(self):
        img = _ramp()
        norm = normalize_log_image(np.log10(np.clip(img, 1e-6, None)).astype(np.float32), LogNegativeBounds(*transfer_bounds()))
        cfg = _e6_config(positive_source=True)
        offset, contrast, toe3, sh3 = transfer_curve_params(cfg.exposure)
        tw3, sw3 = transfer_widths(cfg.exposure)
        out = np.asarray(apply_transfer_curve(norm, offset, contrast, toe3, sh3, (0.0, 0.0, 0.0), tw3, sw3, positive_source=True))
        rel = np.abs(out - img) / np.maximum(img, 1e-9)
        self.assertLess(float(rel.max()), 1e-4)

        # Guard the guard: without the flag, the same inputs still take the baseline +
        # filmic path (this is TestIdentityAtDefaults' own assertion, restated so a
        # regression here fails loudly next to the fix it is guarding).
        rendered = np.asarray(apply_transfer_curve(norm, offset, contrast, toe3, sh3, (0.0, 0.0, 0.0), tw3, sw3, positive_source=False))
        self.assertGreater(float(np.abs(rendered - img).max()), 0.01)

    def test_pipeline_level_matches_a_bare_ODT_free_encode(self):
        """Auto Density/Auto Grade are metering, so isolating the ODT/gain skip means
        turning them off; TestAutoOnPositiveSource covers what they do left on."""
        rng = np.random.default_rng(29)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        cfg = _e6_config(positive_source=True, auto_exposure=False, auto_normalize_contrast=False)
        out, _ = _run_stages(img, cfg, cam_xyz=None)
        rel = np.abs(np.asarray(out) - img) / np.maximum(img, 1e-9)
        self.assertLess(float(rel.max()), 1e-4)

    def test_default_is_off_and_unaffected_frames_keep_the_camera_render(self):
        self.assertFalse(ProcessConfig().positive_source)

    def test_a_negative_mode_frame_cannot_be_positive(self):
        """Positive is Slide-only: a C-41 or B&W config drops the flag, so the negative
        path renders identically whether it was asked for or not."""
        rng = np.random.default_rng(5)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        for mode in (ProcessMode.C41, ProcessMode.BW):
            with self.subTest(process_mode=mode):
                cfg = DEFAULT_WORKSPACE_CONFIG
                process = replace(cfg.process, process_mode=mode)
                exposure = replace(cfg.exposure, cast_removal_strength=cast_removal_for_mode(mode, cfg.exposure.cast_removal_strength))
                base_cfg = replace(cfg, process=process, exposure=exposure)
                on = replace(base_cfg, process=replace(base_cfg.process, positive_source=True))
                self.assertFalse(on.process.positive_source)
                out_on, _ = _run_stages(img, on)
                out_off, _ = _run_stages(img, base_cfg)
                self.assertLess(float(np.abs(np.asarray(out_on) - np.asarray(out_off)).max()), 1e-6)


class TestExposureFaithfulness(unittest.TestCase):
    def test_bounds_are_fixed_and_content_independent(self):
        dark, bright = _ramp(hi=0.05), _ramp(hi=0.6)
        _, ctx_dark = _run_stages(dark, _e6_config())
        _, ctx_bright = _run_stages(bright, _e6_config())
        self.assertEqual(ctx_dark.metrics["final_bounds"].floors, ctx_bright.metrics["final_bounds"].floors)
        self.assertEqual(ctx_dark.metrics["final_bounds"].ceils, ctx_bright.metrics["final_bounds"].ceils)

    def _bracket_means(self):
        base = _ramp(hi=0.4)
        return [float(_run_stages((base * s).astype(np.float32), _e6_config())[0].mean()) for s in (0.25, 0.5, 1.0, 2.0)]

    def test_a_bracket_renders_as_a_bracket(self):
        """Measured bounds make exposures of one scene converge. A transparency must not.

        The display rendering compresses, so a 2x scene change is deliberately less than 2x
        on screen (every tone curve does this, Lightroom's included). What must hold is that
        each exposure stays clearly, monotonically apart.
        """
        transfer = self._bracket_means()

        self.assertEqual(transfer, sorted(transfer))
        for lo, hi in zip(transfer, transfer[1:]):
            self.assertGreater(hi / max(lo, 1e-9), 1.35)
        # An 8x scene range must survive as a wide output range, not collapse to one render.
        self.assertGreater(transfer[-1] / max(transfer[0], 1e-9), 4.0)


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

    def test_zone_density_opens_shadows_without_moving_the_highlights(self):
        """Shadows/Highlights Density are the transfer path's only mid-sparing controls —
        Toe and Grade both drag the whole scale with them. Negative shadow_density lifts
        (the Density convention: positive adds density, so it darkens)."""
        ramp = _ramp(lo=1e-3, hi=0.6)
        base = _run_stages(ramp, _e6_config())[0][0, :, 1]
        values = ramp[0, :, 1]
        shadows, highs = values < np.percentile(values, 10), values > np.percentile(values, 90)

        lifted = _run_stages(ramp, _e6_config(shadow_density=-0.6))[0][0, :, 1]
        rel = np.abs(lifted - base) / np.maximum(base, 1e-9)
        self.assertGreater(float(rel[shadows].mean()), 10.0 * float(rel[highs].mean()))
        self.assertGreater(float(lifted[shadows].mean()), float(base[shadows].mean()))

        pulled = _run_stages(ramp, _e6_config(highlight_density=0.4))[0][0, :, 1]
        rel_h = np.abs(pulled - base) / np.maximum(base, 1e-9)
        self.assertGreater(float(rel_h[highs].mean()), float(rel_h[shadows].mean()))
        self.assertLess(float(pulled[highs].mean()), float(base[highs].mean()))

    def test_zone_density_matches_the_print_by_tonal_position_not_raw_density(self):
        """Regression: the centres were copied from the print path as raw density numbers.
        The two curves do not share a density scale — a print runs d_min..d_max, this runs
        0..TRANSFER_DENSITY_RANGE — so 1.50 sat 64% of the way to black on a print but 50%
        here, and the Shadows slider reached into the midtones on a slide. What has to
        match is the *position on the scale*, not the number."""
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS as C
        from negpy.features.transparency.logic import TRANSFER_DENSITY_RANGE, zone_geometry

        sh_c, hi_c, k = zone_geometry()
        d_min, span = float(C["d_min"]), float(C["d_max"]) - float(C["d_min"])
        anchor = float(C["anchor_target_density"])
        for got, print_density in (
            (sh_c, anchor + float(C["zone_density_shadow_offset"])),
            (hi_c, anchor + float(C["zone_density_highlight_offset"])),
        ):
            self.assertAlmostEqual(got / TRANSFER_DENSITY_RANGE, (print_density - d_min) / span, places=6)
        # Sharpness scales with the range so the transition spans the same share of the scale.
        self.assertAlmostEqual(k * TRANSFER_DENSITY_RANGE / span, float(C["zone_density_sharpness"]), places=6)
        # And the shadow centre must sit below the halfway point, or it is a midtone control.
        self.assertGreater(sh_c / TRANSFER_DENSITY_RANGE, 0.55)

    def test_zone_density_leaves_the_midtones_alone(self):
        """The property the mis-centring broke: a shadow lift must not move mid-grey."""
        ramp = _ramp(lo=1e-4, hi=0.9)
        base = _run_stages(ramp, _e6_config())[0][0, :, 1]
        lifted = _run_stages(ramp, _e6_config(shadow_density=-0.6))[0][0, :, 1]
        mid = (base > 0.40) & (base < 0.75)
        deep = base < 0.06
        self.assertTrue(mid.any() and deep.any())
        self.assertLess(float(np.abs(lifted - base)[mid].max()), 0.03, "a shadow lift moved the midtones")
        # Mid-sparing means the shadows move by more, *relative to where they started* —
        # in display terms the deep end sits near zero, so an absolute comparison against
        # the midtones is meaningless.
        deep_rel = float(((lifted - base)[deep] / np.maximum(base[deep], 1e-4)).mean())
        mid_rel = float((np.abs(lifted - base)[mid] / np.maximum(base[mid], 1e-4)).mean())
        self.assertGreater(deep_rel, 0.15, "a shadow lift did nothing to the shadows")
        self.assertGreater(deep_rel / max(mid_rel, 1e-6), 5.0, "the lift was not mid-sparing")

    def test_knee_widths_are_wired_to_the_width_sliders(self):
        narrow = self._rendered(toe=0.8, toe_width=0.5)
        wide = self._rendered(toe=0.8, toe_width=5.0)
        self.assertGreater(float(np.abs(wide - narrow).max()), 1e-4)

    def test_white_balance_still_shifts_channels(self):
        warmed = self._rendered(wb_cyan=0.5)
        delta = (warmed - self.base).reshape(-1, 3).mean(axis=0)
        self.assertGreater(abs(float(delta[0])), 1e-4)

    def test_shadow_and_highlight_white_balance_still_shift_channels(self):
        """Regression for issue #1077: Shadows/Highlights WB did nothing on Slides
        because the transfer path never read shadow_cyan/highlight_cyan at all."""
        shadow = self._rendered(shadow_cyan=0.5)
        delta_shadow = (shadow - self.base).reshape(-1, 3).mean(axis=0)
        self.assertGreater(abs(float(delta_shadow[0])), 1e-4)

        highlight = self._rendered(highlight_cyan=0.5)
        delta_highlight = (highlight - self.base).reshape(-1, 3).mean(axis=0)
        self.assertGreater(abs(float(delta_highlight[0])), 1e-4)

    def test_shadow_and_highlight_wb_favor_their_own_end(self):
        ramp = _ramp(lo=1e-3, hi=0.6)
        base = _run_stages(ramp, _e6_config())[0][0, :, 0]
        values = ramp[0, :, 1]
        shadows, highs = values < np.percentile(values, 10), values > np.percentile(values, 90)

        def rel_shift(**overrides):
            out = _run_stages(ramp, _e6_config(**overrides))[0][0, :, 0]
            r = np.abs(out - base) / np.maximum(base, 1e-9)
            return float(r[shadows].mean()), float(r[highs].mean())

        sh_shadow, sh_high = rel_shift(shadow_cyan=0.5)
        self.assertGreater(sh_shadow, sh_high)

        hi_shadow, hi_high = rel_shift(highlight_cyan=0.5)
        self.assertGreater(hi_high, hi_shadow)

    def test_shadow_highlight_wb_geometry_matches_the_print_by_tonal_position(self):
        """Same contract as zone_geometry: the centre carries across by fraction of
        span, not by the print's raw density number."""
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS as C
        from negpy.features.transparency.logic import TRANSFER_DENSITY_RANGE, wb_split_geometry

        centre, k = wb_split_geometry()
        d_min, span = float(C["d_min"]), float(C["d_max"]) - float(C["d_min"])
        anchor = float(C["anchor_target_density"])
        self.assertAlmostEqual(centre / TRANSFER_DENSITY_RANGE, (anchor - d_min) / span, places=6)
        self.assertAlmostEqual(k * TRANSFER_DENSITY_RANGE / span, 3.0, places=6)

    def test_dye_separation_still_pushes_color_with_no_paper_to_compose_into(self):
        """This path has no paper matrix, so Dye Separation must apply straight to
        density instead of silently doing nothing off the print path."""
        base_chroma = float(np.abs(self.base[..., 0] - self.base[..., 2]).mean())

        grey = self._rendered(dye_separation=0.0)
        self.assertLess(float(np.abs(grey[..., 0] - grey[..., 2]).mean()), base_chroma * 0.1)

        boosted = self._rendered(dye_separation=2.0)
        self.assertGreater(float(np.abs(boosted[..., 0] - boosted[..., 2]).mean()), base_chroma)

    def test_dye_separation_trims_are_wired_per_channel(self):
        """Mirrors the print path: a trim on one channel must move only that channel,
        since an untrimmed channel keeps k = dye_separation exactly (pivot + 1.0 * e is
        the original density)."""
        trimmed = self._rendered(dye_separation_trim_red=0.6)
        self.assertGreater(float(np.abs(trimmed[..., 0] - self.base[..., 0]).mean()), 1e-4)
        np.testing.assert_allclose(trimmed[..., 1], self.base[..., 1], atol=1e-6)
        np.testing.assert_allclose(trimmed[..., 2], self.base[..., 2], atol=1e-6)

    def test_separation_damping_still_tapers_the_push_with_no_paper_here_either(self):
        """Separation Damping has no per-layer trim of its own on this path, but the
        chroma taper itself needs no paper and must still run over each channel's k."""
        flat = self._rendered(dye_separation=1.4)
        damped = self._rendered(dye_separation=1.4, separation_damping=1.0)
        self.assertGreater(float(np.abs(flat - damped).max()), 1e-4)

        # Inert without a separation push, same as on the print path.
        self.assertTrue(np.array_equal(self.base, self._rendered(separation_damping=1.0)))

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


class TestDyeSeparationReference(unittest.TestCase):
    """Dye Separation's reference density: the mean of the channels, each softly capped
    where it is far denser than the pixel's lightest channel and near the dense end."""

    # Edits that run ahead of Dye Separation and move a channel's density away from its
    # capture density: none; brighter with more contrast; darker with toe and shoulder.
    EDITS = ((0.0, 1.0, 0.0), (-0.7, 1.35, 0.0), (0.5, 0.8, 0.4))

    @staticmethod
    def _density(d, separation, trims=(0.0, 0.0, 0.0), damping=0.0, edit=(0.0, 1.0, 0.0)):
        """Capture densities in, densities out. A positive source skips the display
        rendering, so the output is 10**-D exactly while it stays inside (0, 1)."""
        exposure, contrast, knee = edit
        n = np.asarray(d, dtype=np.float32) / TRANSFER_DENSITY_RANGE
        out = apply_transfer_curve(
            n,
            exposure,
            contrast,
            (knee,) * 3,
            (knee,) * 3,
            positive_source=True,
            separation=separation,
            separation_trims=trims,
            damping=damping,
        )
        return -np.log10(out)

    def test_neutrals_hold_at_every_density_and_trim(self):
        grey = np.repeat(np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :, None], 3, axis=2)
        for exposure in (-1.0, 0.0, 1.0):  # also pushes grays past both ends of the window

            def curve(k, trims=(0.0, 0.0, 0.0)):
                return apply_transfer_curve(grey, exposure, 1.0, (0.0,) * 3, (0.0,) * 3, separation=k, separation_trims=trims)

            flat = curve(1.0)
            for k, trims in ((0.5, (0.0, 0.0, 0.0)), (1.5, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.4, -0.4))):
                np.testing.assert_allclose(curve(k, trims), flat, atol=1e-5, err_msg=f"separation {k}, trims {trims} moved a neutral")

    def test_ordinary_colors_keep_the_channel_mean(self):
        """Light, dark and brighter-than-white colors keep the print path's
        M(k) = diag(k) + (1-k)J."""
        k = 1.5
        for d in (
            [0.3, 1.5, 1.5],
            [0.4, 0.5, 1.6],
            [1.4, 0.6, 1.3],
            [0.55, 0.75, 0.9],
            [2.0, 1.4, 1.9],
            [2.2, 2.6, 2.9],
            [-0.3, 1.5, 1.2],
        ):
            d = np.asarray(d, dtype=np.float32)
            expected = d.mean() + k * (d - d.mean())
            got = self._density([[d]], k)[0, 0]
            inside = expected > 0.0  # a channel pushed past white clips to 1.0
            np.testing.assert_allclose(got[inside], expected[inside], atol=2e-3, err_msg=str(d))

    def test_a_channel_near_the_dense_end_does_not_steer_the_others(self):
        """An out-of-gamut blue sky: red sits at the capture's dense end, where it is mostly
        noise. Moving it there must not move green or blue, with or without damping, and
        whatever the edits ahead of Dye Separation did to red's density."""
        for edit in self.EDITS:
            for damping in (0.0, 1.0):
                for lo, hi in ((2.8, 3.0), (3.0, 3.6)):
                    a = self._density([[[lo, 1.53, 0.88]]], 1.5, damping=damping, edit=edit)[0, 0]
                    b = self._density([[[hi, 1.53, 0.88]]], 1.5, damping=damping, edit=edit)[0, 0]
                    np.testing.assert_allclose(b[1:], a[1:], atol=5e-3, err_msg=f"red {lo}->{hi}, damping {damping}, edit {edit}")

    def test_a_smooth_gradient_never_reverses(self):
        """Along a smooth one-channel ramp, every output moves one way only. A reference
        density that can fall as a channel darkens folds the gradient into bands."""
        sweep = np.linspace(0.0, 3.6, 721, dtype=np.float32)
        fixed = (0.3, 0.9, 1.5, 2.1, 2.7, 3.2)
        for (k, trims), edit in itertools.product(((0.5, (0.0, 0.0, 0.0)), (1.5, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.4, -0.4))), self.EDITS):
            for ch in range(3):
                others = [c for c in range(3) if c != ch]
                for a in fixed:
                    for b in fixed:
                        d = np.empty((1, sweep.size, 3), dtype=np.float32)
                        d[0, :, ch], d[0, :, others[0]], d[0, :, others[1]] = sweep, a, b
                        steps = np.diff(self._density(d, k, trims, edit=edit)[0], axis=0)
                        for out in range(3):
                            s = np.sign(steps[np.abs(steps[:, out]) > 1e-6, out])
                            self.assertFalse(
                                np.any(s[1:] * s[:-1] < 0), f"k {k}, trims {trims}, edit {edit}, ramp on {ch}, output {out}, at {a}/{b}"
                            )


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


class TestWhiteBlackPointOnTheTransferPath(unittest.TestCase):
    """White/Black Point deviate the fixed window the same way they deviate a measured
    one: additive, and inert at zero (TransparencyBaseProcessor)."""

    def test_zero_offsets_leave_the_fixed_window_untouched(self):
        floors, ceils = transfer_bounds()
        _, ctx = _run_stages(_ramp(), _e6_config())
        self.assertEqual(ctx.metrics["final_bounds"].floors, floors)
        self.assertEqual(ctx.metrics["final_bounds"].ceils, ceils)

    def test_white_point_deviates_the_window_and_the_render(self):
        cfg = _e6_config()
        moved = replace(cfg, process=replace(cfg.process, white_point_offset=0.15))
        base, ctx_base = _run_stages(_ramp(), cfg)
        out, ctx_moved = _run_stages(_ramp(), moved)
        self.assertNotEqual(ctx_moved.metrics["final_bounds"].floors, ctx_base.metrics["final_bounds"].floors)
        self.assertGreater(float(np.abs(out - base).max()), 1e-3)

    def test_black_point_deviates_the_window_and_the_render(self):
        cfg = _e6_config()
        moved = replace(cfg, process=replace(cfg.process, black_point_offset=0.15))
        base, ctx_base = _run_stages(_ramp(), cfg)
        out, ctx_moved = _run_stages(_ramp(), moved)
        self.assertNotEqual(ctx_moved.metrics["final_bounds"].ceils, ctx_base.metrics["final_bounds"].ceils)
        self.assertGreater(float(np.abs(out - base).max()), 1e-3)

    def test_applies_on_a_positive_source_frame_too(self):
        """Positive frames take the transfer path on every mode, not only a slide."""
        cfg = _e6_config(positive_source=True)
        moved = replace(cfg, process=replace(cfg.process, white_point_offset=0.15))
        base, _ = _run_stages(_ramp(), cfg)
        out, _ = _run_stages(_ramp(), moved)
        self.assertGreater(float(np.abs(out - base).max()), 1e-3)

    def test_cast_removals_neutral_axis_ignores_the_creative_offset(self):
        """Cast Removal meters gray balance against the pre-trim window, so a White/Black
        Point nudge must not move what it measures."""
        rng = np.random.default_rng(41)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        cfg = _e6_config(cast_removal_strength=1.0)
        moved = replace(cfg, process=replace(cfg.process, white_point_offset=0.15, black_point_offset=-0.1))

        _, ctx_base = _run_stages(img, cfg)
        _, ctx_moved = _run_stages(img, moved)

        self.assertEqual(ctx_base.metrics["neutral_axis_refs"], ctx_moved.metrics["neutral_axis_refs"])


class TestAutomaticGradingOnARawSlide(unittest.TestCase):
    def test_auto_density_and_auto_grade_start_off(self):
        """A slide was exposed deliberately, so a switch into Slide turns both off."""
        from negpy.features.process.models import with_process_mode

        slide = with_process_mode(DEFAULT_WORKSPACE_CONFIG, ProcessMode.E6)
        self.assertFalse(slide.exposure.auto_exposure)
        self.assertFalse(slide.exposure.auto_normalize_contrast)

    def test_auto_density_and_auto_grade_move_the_render_when_on(self):
        rng = np.random.default_rng(13)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        on, _ = _run_stages(img, _e6_config(auto_exposure=True, auto_normalize_contrast=True))
        off, _ = _run_stages(img, _e6_config())
        self.assertGreater(float(np.abs(on - off).max()), 1e-4)

    def test_crosstalk_unmix_is_not_applied(self):
        """It models negative-film dye crosstalk and defaults to 0.5 — it would tint the
        pass-through, so the transfer path must ignore it."""
        rng = np.random.default_rng(17)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        cfg = _e6_config()
        strong = replace(cfg, process=replace(cfg.process, crosstalk_strength=1.0))
        self.assertLess(float(np.abs(_run_stages(img, cfg)[0] - _run_stages(img, strong)[0]).max()), 1e-6)


class TestAutoDensityGradeOnAPositiveFrame(unittest.TestCase):
    """Auto Density/Auto Grade meter a Positive frame exactly as they would a negative,
    restated on this curve by transfer_auto_terms."""

    def _cfg(self, **overrides):
        return _e6_config(positive_source=True, **{"auto_exposure": True, "auto_normalize_contrast": True, **overrides})

    def test_the_toggles_move_the_render(self):
        rng = np.random.default_rng(19)
        img = (rng.random((16, 16, 3)) * 0.3 + 0.02).astype(np.float32)
        pos, _ = _run_stages(img, self._cfg())
        off, _ = _run_stages(img, self._cfg(auto_exposure=False, auto_normalize_contrast=False))
        self.assertGreater(float(np.abs(pos - off).max()), 1e-4)

    def test_auto_density_pulls_a_dark_frame_brighter(self):
        """The metered anchor is placed at the same mid-grey pivot Grade rotates about,
        so an under-exposed frame's own midtone gets pulled toward it."""
        rng = np.random.default_rng(23)
        dark = (rng.random((24, 24, 3)) * 0.02 + 0.002).astype(np.float32)
        on, _ = _run_stages(dark, self._cfg())
        off, _ = _run_stages(dark, self._cfg(auto_exposure=False))
        self.assertGreater(float(on.mean()), float(off.mean()))

    def test_auto_grade_expands_contrast_on_a_flat_frame(self):
        """A compressed textural range should print harder, not flat -- Auto Grade's
        whole point on a negative, restated here via effective_grade_range/grade_to_slope."""
        grad = np.linspace(0.03, 0.3, 32, dtype=np.float32)
        base = np.repeat(grad[None, :], 32, axis=0)
        flat = np.ascontiguousarray((np.stack([base, base * 0.95, base * 0.9], axis=-1) * 0.2 + 0.15).astype(np.float32))
        on, _ = _run_stages(flat, self._cfg())
        off, _ = _run_stages(flat, self._cfg(auto_normalize_contrast=False))
        self.assertGreater(float(on.std()), 2.0 * float(off.std()))

    def test_metrics_used_by_transfer_auto_terms_are_published(self):
        _, ctx = _run_stages(_ramp(), self._cfg())
        for key in ("metered_anchor", "textural_range", "shadow_point", "highlight_point"):
            self.assertIn(key, ctx.metrics)
            self.assertIsNotNone(ctx.metrics[key])

    def test_a_raw_slide_publishes_the_same_metrics(self):
        _, ctx = _run_stages(_ramp(), _e6_config())
        for key in ("metered_anchor", "textural_range", "shadow_point", "highlight_point"):
            self.assertIsNotNone(ctx.metrics.get(key))


class TestAutoMeterForMode(unittest.TestCase):
    """auto_meter_for_mode: the rewrite with_process_mode applies to Auto Density/Auto
    Grade on a Film Mode switch."""

    def test_untouched_negative_default_turns_off_entering_slide(self):
        self.assertFalse(auto_meter_for_mode(ProcessMode.E6, current=True))

    def test_already_off_stays_off_entering_slide(self):
        self.assertFalse(auto_meter_for_mode(ProcessMode.E6, current=False))

    def test_untouched_slide_default_turns_on_leaving_slide(self):
        for mode in (ProcessMode.C41, ProcessMode.BW):
            with self.subTest(mode=mode):
                self.assertTrue(auto_meter_for_mode(mode, current=False))

    def test_already_on_stays_on_leaving_slide(self):
        self.assertTrue(auto_meter_for_mode(ProcessMode.C41, current=True))


class TestTransferAutoTerms(unittest.TestCase):
    """Unit-level coverage of transfer_auto_terms itself: exact enough to pin the
    solves that a pipeline-level render can only show indirectly."""

    def _exp(self, **overrides):
        return replace(DEFAULT_WORKSPACE_CONFIG.exposure, **overrides)

    def test_none_inputs_leave_everything_manual(self):
        """A None meter must win over the toggles being on."""
        exp = self._exp(auto_exposure=True, auto_normalize_contrast=True)
        offset, contrast, hl_auto = transfer_auto_terms(exp, 0.3, 1.4, None, None, None, None)
        self.assertAlmostEqual(offset, 0.3)
        self.assertAlmostEqual(contrast, 1.4)
        self.assertEqual(hl_auto, 0.0)

    def test_toggles_off_leave_everything_manual_even_with_real_metrics(self):
        exp = self._exp(auto_exposure=False, auto_normalize_contrast=False)
        offset, contrast, hl_auto = transfer_auto_terms(exp, 0.3, 1.4, 0.6, 0.5, 0.7, 0.1)
        self.assertAlmostEqual(offset, 0.3)
        self.assertAlmostEqual(contrast, 1.4)
        self.assertEqual(hl_auto, 0.0)

    def test_auto_density_places_the_metered_anchor_at_the_contrast_pivot(self):
        exp = self._exp(auto_exposure=True, auto_normalize_contrast=False)
        anchor = 0.5
        offset, contrast, _ = transfer_auto_terms(exp, 0.0, 1.0, None, anchor, None, None)
        pivot = float(TRANSFER_CONSTANTS["transfer_contrast_pivot"])
        d_anchor_final = anchor * TRANSFER_DENSITY_RANGE - offset
        self.assertAlmostEqual(d_anchor_final, pivot, places=6)
        self.assertAlmostEqual(contrast, 1.0)

    def test_shadow_reach_raises_contrast_to_hit_its_target(self):
        exp = self._exp(auto_exposure=True, auto_normalize_contrast=True)
        # A flat, low-textural-range frame so Auto Grade alone barely moves contrast,
        # with a shadow point far below the anchor so reaching the target needs a push.
        offset, contrast, _ = transfer_auto_terms(exp, 0.0, 1.0, 2.5, 0.25, 0.9, 0.1)
        pivot = float(TRANSFER_CONSTANTS["transfer_contrast_pivot"])
        d_shadow_final = pivot + (0.9 * TRANSFER_DENSITY_RANGE - offset - pivot) * contrast
        self.assertGreaterEqual(d_shadow_final, transfer_shadow_reach_density() - 1e-6)

    def test_shadow_reach_never_lowers_the_contrast_auto_grade_already_picked(self):
        exp = self._exp(auto_exposure=True, auto_normalize_contrast=True)
        _, base_contrast, _ = transfer_auto_terms(exp, 0.0, 1.0, 2.5, 0.25, None, None)
        _, with_reach, _ = transfer_auto_terms(exp, 0.0, 1.0, 2.5, 0.25, 0.9, 0.1)
        self.assertGreaterEqual(with_reach, base_contrast - 1e-9)

    def test_shadow_reach_is_a_noop_without_span_between_anchor_and_shadow_point(self):
        exp = self._exp(auto_exposure=True, auto_normalize_contrast=True)
        _, base_contrast, _ = transfer_auto_terms(exp, 0.0, 1.0, 2.5, 0.25, None, None)
        _, degenerate, _ = transfer_auto_terms(exp, 0.0, 1.0, 2.5, 0.25, 0.25, 0.1)
        self.assertAlmostEqual(degenerate, base_contrast)

    def test_highlight_hold_burns_only_when_the_highlight_is_too_bright(self):
        exp = self._exp(auto_exposure=False, auto_normalize_contrast=True)
        target = transfer_highlight_hold_density()
        too_bright = target / TRANSFER_DENSITY_RANGE * 0.3  # well under target
        already_holds = min(1.0, (target * 3.0) / TRANSFER_DENSITY_RANGE)  # comfortably over target
        _, _, hl_bright = transfer_auto_terms(exp, 0.0, 1.0, None, None, None, too_bright)
        _, _, hl_holds = transfer_auto_terms(exp, 0.0, 1.0, None, None, None, already_holds)
        self.assertGreater(hl_bright, 0.0)
        self.assertEqual(hl_holds, 0.0)

    def test_highlight_hold_is_capped(self):
        exp = self._exp(auto_exposure=False, auto_normalize_contrast=True)
        _, _, hl_auto = transfer_auto_terms(exp, 0.0, 1.0, None, None, None, 1e-6)
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS

        self.assertLessEqual(hl_auto, float(EXPOSURE_CONSTANTS["highlight_hold_max"]) + 1e-9)

    def test_derived_constants_sit_inside_the_fixed_window(self):
        for value in (transfer_shadow_reach_density(), transfer_highlight_hold_density()):
            self.assertGreater(value, 0.0)
            self.assertLess(value, TRANSFER_DENSITY_RANGE)
        # Shadow reach targets deep density, highlight hold targets shallow.
        self.assertGreater(transfer_shadow_reach_density(), transfer_highlight_hold_density())

    def test_assumed_anchor_matches_the_contrast_pivot_fraction(self):
        self.assertAlmostEqual(transfer_assumed_anchor(), float(TRANSFER_CONSTANTS["transfer_contrast_pivot"]) / TRANSFER_DENSITY_RANGE)


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
        """The C-41 default process_mode keeps a bare ProcessConfig on the print path."""
        conf = ProcessConfig()
        self.assertEqual(conf.process_mode, ProcessMode.C41)
        self.assertIs(render_path(conf), RenderPath.PRINT)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuTransferParity(unittest.TestCase):
    """The transfer curve lives twice — transparency/logic.py and transfer.wgsl. They must agree,
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

    def _both(self, settings, cam_xyz=CAM_XYZ, img=None):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        # The shipped autocrop_offset insets the CPU render by a pixel, which would
        # compare two different framings rather than two curve implementations.
        settings = replace(settings, geometry=replace(settings.geometry, autocrop_offset=0))

        if img is None:
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

    def test_active_crosstalk_matches(self):
        """Regression: the shader applied the unmix on the print branch only, so an E-6
        matrix moved the CPU render and did nothing at all on the GPU — which is the
        engine the app actually uses. The parity tests missed it because every other case
        runs with crosstalk gated off, where both engines agree trivially."""
        from negpy.features.process.models import ProcessMode as _PM

        settings = _e6_config()
        active = replace(
            settings,
            process=replace(
                settings.process,
                crosstalk_strength=1.0,
                crosstalk_process=_PM.E6,
                crosstalk_matrix=(1.0, -0.05, -0.002, -0.29, 1.0, -0.05, -0.09, -0.19, 1.0),
            ),
        )
        cpu, gpu = self._both(active)
        self._assert_parity(cpu, gpu)

        # Guard the guard: the matrix must actually be doing something, or this passes
        # for the wrong reason.
        off_cpu, _ = self._both(settings)
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01)

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
            shadow_density=-0.5,
            highlight_density=0.3,
            dye_separation=1.3,
            dye_separation_trim_red=0.2,
            dye_separation_trim_blue=-0.15,
        )
        self._assert_parity(*self._both(settings))

    def test_dye_separation_matches(self):
        """Dye Separation carries no paper matrix on this path — each channel scales its
        own deviation from a reference density instead of a matmul — and CPU/GPU must
        apply that same per-channel k."""
        settings = _e6_config()
        active = _e6_config(dye_separation=1.6)
        cpu, gpu = self._both(active)
        self._assert_parity(cpu, gpu)

        off_cpu, off_gpu = self._both(settings)
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01, "dye separation inert on the CPU")
        self.assertGreater(float(np.abs(gpu - off_gpu).max()), 0.01, "dye separation inert on the GPU")

    def test_dye_separation_reference_matches_past_the_window(self):
        """A saturated blue whose red runs into the dense end, where the channel cap
        engages, with and without damping."""
        red = np.logspace(-1.5, -4.0, 64, dtype=np.float32)
        img = np.stack([np.repeat(red[None, :], 16, axis=0), np.full((16, 64), 0.03), np.full((16, 64), 0.13)], axis=-1)
        img = np.ascontiguousarray(img.astype(np.float32))
        for damping in (0.0, 1.0):
            settings = _e6_config(dye_separation=1.5, dye_separation_trim_green=0.3, separation_damping=damping, density=0.4)
            self._assert_parity(*self._both(settings, cam_xyz=None, img=img))

    def test_dye_separation_trims_match(self):
        """The per-channel trims (same fields the print path's per-layer view edits)
        must reach the shader's separation.xyz lanes the same way the CPU folds them."""
        settings = _e6_config()
        active = _e6_config(dye_separation_trim_red=0.4, dye_separation_trim_green=-0.3)
        cpu, gpu = self._both(active)
        self._assert_parity(cpu, gpu)

        off_cpu, off_gpu = self._both(settings)
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01, "trims inert on the CPU")
        self.assertGreater(float(np.abs(gpu - off_gpu).max()), 0.01, "trims inert on the GPU")

    def test_separation_damping_matches(self):
        """Separation Damping tapers each channel's own k by the same shared chroma on
        this path too, and CPU/GPU must taper it by the same law."""
        flat = _e6_config(dye_separation=1.4)
        damped = _e6_config(dye_separation=1.4, separation_damping=1.0)
        cpu, gpu = self._both(damped)
        self._assert_parity(cpu, gpu)

        flat_cpu, flat_gpu = self._both(flat)
        self.assertGreater(float(np.abs(cpu - flat_cpu).max()), 0.01, "damping inert on the CPU")
        self.assertGreater(float(np.abs(gpu - flat_gpu).max()), 0.01, "damping inert on the GPU")

    def test_separation_damping_with_trims_matches(self):
        """Damping combined with an asymmetric per-channel k (not just a shared one) is
        the path that used to collapse to a single scalar before every channel got its
        own trim — CPU and GPU must still agree once the trims split the k apart."""
        settings = _e6_config(dye_separation=1.4, dye_separation_trim_red=0.5, separation_damping=0.8)
        self._assert_parity(*self._both(settings))

    def test_cast_removal_matches(self):
        """Cast Removal reaches this curve as a per-channel affine the shader mirrors in
        its own uniform lanes. The wedge spans the meter's three luma bands, which the
        gradient the other parity cases use does not — without that there is no axis and
        this would pass for the wrong reason."""
        rng = np.random.default_rng(11)
        v = np.geomspace(5e-4, 0.9, 64 * 64).astype(np.float32).reshape(64, 64)
        img = np.stack([v, v * 0.82, v * 0.62], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 1e-4, img.shape).astype(np.float32))

        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        def _pair(strength):
            settings = _e6_config(cast_removal_strength=strength)
            settings = replace(settings, geometry=replace(settings.geometry, autocrop_offset=0))
            return (
                self._render(processor, settings, img, prefer_gpu=False),
                self._render(processor, settings, img, prefer_gpu=True),
            )

        cpu_on, gpu_on = _pair(1.0)
        self._assert_parity(cpu_on, gpu_on)

        # Guard the guard: the solve must actually be moving the render.
        cpu_off, _ = _pair(0.0)
        self.assertGreater(float(np.abs(cpu_on - cpu_off).max()), 0.01)

    def test_zone_black_taper_matches(self):
        """The taper rides a uniform lane the shader did not have. Asserted against a
        render whose deepest tones actually reach it, and guarded both ways: a shader that
        ignored the lane would still pass a bare parity check, which is exactly how the
        crosstalk unmix stayed broken on the GPU."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        # Deep enough to reach the bottom of the density window, where the taper lives —
        # _both's own gradient stops around density 1.7 and would never engage it.
        h, w = 64, 64
        grad = np.logspace(np.log10(0.4), np.log10(3e-4), w, dtype=np.float32)
        img = np.ascontiguousarray(np.stack([np.repeat(grad[None, :], h, 0)] * 3, axis=-1))
        lifted = replace(_e6_config(shadow_density=-0.8), geometry=replace(_e6_config().geometry, autocrop_offset=0))

        def both(tag):
            # A fresh processor per variant: the engine caches on the source hash, which a
            # patched module constant does not change, so a shared one would hand back the
            # previous render and the guard below would pass on a stale buffer.
            proc = ImageProcessor()
            return (
                self._render(proc, lifted, img, prefer_gpu=False, cam_xyz=CAM_XYZ),
                self._render(proc, lifted, img, prefer_gpu=True, cam_xyz=CAM_XYZ),
            )

        cpu, gpu = both("on")
        self._assert_parity(cpu, gpu)

        # The taper must actually be doing something at the black end on BOTH engines, or
        # parity here is vacuous. Compared against the same lift with the taper spanning
        # nothing, which is the un-tapered behaviour.
        # Patched in both namespaces: gpu_engine binds the constant at import, so patching
        # only the source module would leave the shader packing the real value and the GPU
        # half of this guard would silently pass on an unchanged render.
        from unittest.mock import patch

        with (
            patch("negpy.features.transparency.logic.ZONE_BLACK_TAPER", 1e-6),
            patch("negpy.services.rendering.gpu_engine.ZONE_BLACK_TAPER", 1e-6),
        ):
            flat_cpu, flat_gpu = both("off")
        self.assertGreater(float(np.abs(cpu - flat_cpu).max()), 0.01, "taper inert on the CPU")
        self.assertGreater(float(np.abs(gpu - flat_gpu).max()), 0.01, "taper inert on the GPU")

    def test_positive_source_matches(self):
        """The gain and display_rendering skip must agree bit-for-bit-ish on both engines,
        or Positive would look different in the live preview than in an export."""
        settings = _e6_config(positive_source=True, density=1.4, toe=0.5)
        cpu, gpu = self._both(settings)
        self._assert_parity(cpu, gpu)

        # Guard the guard: the flag must actually change the render on both engines.
        off_cpu, off_gpu = self._both(_e6_config(density=1.4, toe=0.5))
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01, "positive_source inert on the CPU")
        self.assertGreater(float(np.abs(gpu - off_gpu).max()), 0.01, "positive_source inert on the GPU")

    def test_auto_density_and_grade_match_on_a_positive_frame(self):
        """Auto Density/Auto Grade meter working-space, camera-matrix-applied grids on
        both engines (transfer_assumed_anchor, cam_prefiltered) -- the two places CPU
        and GPU build that grid independently and could drift apart."""
        settings = _e6_config(positive_source=True, auto_exposure=True, auto_normalize_contrast=True)
        cpu, gpu = self._both(settings)
        self._assert_parity(cpu, gpu)

        # Guard the guard: the toggles must actually be moving the render on both
        # engines, or parity here would pass for the wrong reason.
        off_cpu, off_gpu = self._both(_e6_config(positive_source=True))
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01, "auto density/grade inert on the CPU")
        self.assertGreater(float(np.abs(gpu - off_gpu).max()), 0.01, "auto density/grade inert on the GPU")

    def test_white_black_point_matches(self):
        """White/Black Point deviate the fixed window on both engines, the same
        technique the measured path already uses -- both bake the offset into the
        floors/ceils they upload, so the shader needs no lanes of its own."""
        settings = _e6_config()
        moved = replace(settings, process=replace(settings.process, white_point_offset=0.12, black_point_offset=-0.08))
        cpu, gpu = self._both(moved)
        self._assert_parity(cpu, gpu)

        off_cpu, off_gpu = self._both(settings)
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01, "white/black point inert on the CPU")
        self.assertGreater(float(np.abs(gpu - off_gpu).max()), 0.01, "white/black point inert on the GPU")

    def test_zone_density_matches(self):
        """Zone Density rides a uniform lane the transfer shader did not have. Asserted on
        its own, and against an inert render, so parity cannot pass by both engines
        ignoring it — which is exactly how the crosstalk unmix stayed broken on the GPU."""
        settings = _e6_config()
        active = _e6_config(shadow_density=-0.7, highlight_density=0.4)
        cpu, gpu = self._both(active)
        self._assert_parity(cpu, gpu)

        off_cpu, off_gpu = self._both(settings)
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01, "zone density inert on the CPU")
        self.assertGreater(float(np.abs(gpu - off_gpu).max()), 0.01, "zone density inert on the GPU")

    def test_shadow_highlight_wb_matches(self):
        """Shadows/Highlights WB rides uniform lanes the transfer shader did not have
        (issue #1077: the sliders had no effect on Slides)."""
        settings = _e6_config()
        active = _e6_config(shadow_cyan=0.5, highlight_yellow=-0.4)
        cpu, gpu = self._both(active)
        self._assert_parity(cpu, gpu)

        off_cpu, off_gpu = self._both(settings)
        self.assertGreater(float(np.abs(cpu - off_cpu).max()), 0.01, "shadow/highlight WB inert on the CPU")
        self.assertGreater(float(np.abs(gpu - off_gpu).max()), 0.01, "shadow/highlight WB inert on the GPU")


if __name__ == "__main__":
    unittest.main()


class TestCrosstalkIsModeAware(unittest.TestCase):
    """A crosstalk matrix describes one dye set. Every bundled profile is a color
    negative stock, so without a mode gate a slide silently gets a negative's
    correction — and the render disagrees with a UI that already hides it for B&W."""

    def _img(self):
        rng = np.random.default_rng(31)
        grad = np.linspace(0.03, 0.5, 48, dtype=np.float32)
        img = np.repeat(grad[None, :], 48, axis=0)
        return np.ascontiguousarray(np.stack([img, img * 0.7, img * 0.45], axis=-1) + rng.uniform(0, 0.01, (48, 48, 3)).astype(np.float32))

    def _delta(self, mode, profile_process):
        from negpy.domain.interfaces import PipelineContext

        img = self._img()
        out = []
        for strength in (0.0, 1.0):
            cfg = DEFAULT_WORKSPACE_CONFIG
            proc = replace(
                cfg.process,
                process_mode=mode,
                crosstalk_strength=strength,
                crosstalk_process=profile_process,
            )
            ctx = PipelineContext(original_size=img.shape[:2], scale_factor=1.0, process_mode=mode, cam_xyz=CAM_XYZ, wants_uv_grid=False)
            out.append(np.asarray(base_processor(replace(cfg, process=proc)).process(img.copy(), ctx)))
        return float(np.abs(out[0] - out[1]).max())

    def test_a_c41_profile_does_nothing_to_e6(self):
        self.assertEqual(self._delta(ProcessMode.E6, ProcessMode.C41), 0.0)

    def test_a_c41_profile_still_works_on_c41(self):
        self.assertGreater(self._delta(ProcessMode.C41, ProcessMode.C41), 1e-4)

    def test_an_e6_profile_applies_on_a_slide(self):
        """The transfer path honours crosstalk rather than hard-skipping it: a
        rig-calibrated matrix is a capture correction, like Hue Trim."""
        self.assertGreater(self._delta(ProcessMode.E6, ProcessMode.E6), 1e-4)

    def test_legacy_configs_without_the_field_stay_c41(self):
        from negpy.features.process.models import ProcessConfig

        self.assertEqual(str(ProcessConfig().crosstalk_process), str(ProcessMode.C41))


def test_normalization_shader_reads_the_transfer_decision_it_is_given():
    """The WGSL must not re-derive render_path: it had no positive_source term,
    so a Positive frame took the print branch on the GPU and the transfer branch on
    the CPU."""
    from pathlib import Path

    import negpy

    src = (Path(negpy.__file__).parent / "features/exposure/shaders/normalization.wgsl").read_text()

    assert "transfer_flag" in src
    assert "params.mode == 2u" not in src
