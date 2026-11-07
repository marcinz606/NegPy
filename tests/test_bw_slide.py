"""B&W slide: a Slide with B&W on renders one gray density, from B&W reversal film or a color slide."""

import unittest
from types import SimpleNamespace
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import BwFilter
from negpy.features.metadata.gear_models import FilmColorType
from negpy.features.process.models import ProcessConfig, ProcessMode, with_film_fields, with_process_mode, with_slide_bw
from negpy.features.process.path import RenderPath, is_mono, render_path
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from tests.test_transparency_transfer import _e6_config, _run_stages


def _bw_slide(source_color=False, **exposure):
    cfg = _e6_config(**exposure)
    return replace(cfg, process=replace(cfg.process, slide_bw=True, source_color=source_color))


def _color_scene():
    rng = np.random.default_rng(7)
    return np.ascontiguousarray(rng.uniform(0.02, 0.5, (24, 24, 3)).astype(np.float32))


class TestModel(unittest.TestCase):
    def test_slide_only(self):
        for mode in (ProcessMode.C41, ProcessMode.BW):
            self.assertFalse(ProcessConfig(process_mode=mode, slide_bw=True).slide_bw)
        self.assertTrue(ProcessConfig(process_mode=ProcessMode.E6, slide_bw=True).slide_bw)

    def test_leaving_slide_drops_it(self):
        cfg = _bw_slide()
        self.assertFalse(with_process_mode(cfg, ProcessMode.C41).process.slide_bw)

    def test_setter_is_a_no_op_off_slide(self):
        self.assertIs(with_slide_bw(DEFAULT_WORKSPACE_CONFIG, True), DEFAULT_WORKSPACE_CONFIG)
        self.assertTrue(with_slide_bw(_e6_config(), True).process.slide_bw)

    def test_film_fields_apply_it(self):
        cfg = with_film_fields(DEFAULT_WORKSPACE_CONFIG, {"process_mode": ProcessMode.E6, "slide_bw": True})
        self.assertTrue(cfg.process.slide_bw)

    def test_round_trips_the_flat_dict(self):
        cfg = _bw_slide()
        self.assertTrue(WorkspaceConfig.from_flat_dict(cfg.to_dict()).process.slide_bw)

    def test_is_mono_and_keeps_the_slide_path(self):
        cfg = _bw_slide()
        self.assertTrue(is_mono(cfg.process))
        self.assertIs(render_path(cfg.process), RenderPath.TRANSFER)
        self.assertFalse(is_mono(_e6_config().process))


class TestRender(unittest.TestCase):
    def test_renders_gray(self):
        out, _ = _run_stages(_color_scene(), _bw_slide())
        np.testing.assert_array_equal(out[..., 0], out[..., 1])
        np.testing.assert_array_equal(out[..., 1], out[..., 2])

    def test_color_controls_do_nothing(self):
        plain, _ = _run_stages(_color_scene(), _bw_slide())
        moved, _ = _run_stages(
            _color_scene(),
            _bw_slide(wb_cyan=0.3, shadow_magenta=0.4, dye_separation=1.8, cast_removal_strength=1.0),
        )
        np.testing.assert_array_equal(plain, moved)

    def test_filter_acts_on_a_color_slide_only(self):
        img = _color_scene()
        red, _ = _run_stages(img, _bw_slide(source_color=True, bw_filter=BwFilter.RED_25))
        none, _ = _run_stages(img, _bw_slide(source_color=True))
        self.assertGreater(float(np.abs(red - none).max()), 1e-3)
        mono_red, _ = _run_stages(img, _bw_slide(bw_filter=BwFilter.RED_25))
        mono_none, _ = _run_stages(img, _bw_slide())
        np.testing.assert_array_equal(mono_red, mono_none)

    def test_toning_acts(self):
        from negpy.domain.interfaces import PipelineContext
        from negpy.services.rendering.engine import print_stages

        cfg = _bw_slide()
        toned = replace(cfg, toning=replace(cfg.toning, sepia_strength=1.0))
        img = np.full((4, 4, 3), 0.4, dtype=np.float32)
        ctx = PipelineContext(original_size=(4, 4), scale_factor=1.0, process_mode=ProcessMode.E6)
        self.assertGreater(float(np.abs(print_stages(toned, img, ctx) - img).max()), 1e-3)
        color = replace(toned, process=replace(toned.process, slide_bw=False))
        np.testing.assert_array_equal(print_stages(color, img, ctx), img)

    def test_exports_one_channel(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        buf = np.full((4, 4, 3), 0.5, dtype=np.float32)
        processor = ImageProcessor.__new__(ImageProcessor)
        self.assertEqual(len(processor.buffer_to_pil(buf, _bw_slide()).getbands()), 1)
        self.assertEqual(len(processor.buffer_to_pil(buf, _e6_config()).getbands()), 3)

    def test_contact_sheet_type(self):
        from negpy.services.export.contact_sheet_roll import film_color_type

        frame = SimpleNamespace(config=_bw_slide())
        self.assertEqual(film_color_type(frame), FilmColorType.BW_SLIDE)  # type: ignore[arg-type]
