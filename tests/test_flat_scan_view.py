"""Flat Scan is a view that follows the edit: renders, the memo, the thumbnail and the saved edit."""

import gc
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

import fade_crossover_scene as scene
from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig, flat_master_config
from negpy.features.exposure.models import RenderIntent
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.features.process.models import ProcessMode
from negpy.services.export.launch_recipe import LaunchRecipe
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.services.rendering.engine import DarkroomEngine


class FlatScanViewTest(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock(spec=DesktopSessionManager)
        self.session.state = AppState()
        self.session.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as rw,
            patch("negpy.desktop.controller.PreviewManager") as pm,
        ):
            rw.return_value = MagicMock()
            pm.return_value = MagicMock(spec=PreviewManager)
            pm.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.session)
        self.state = self.controller.state
        self.state.preview_raw = np.empty((8, 8, 3), dtype=np.float32)
        self.state.current_file_hash = "frame"
        self.tasks: list = []
        self.controller.render_requested.connect(self.tasks.append)

    def tearDown(self):
        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.prefetch_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _edit(self, **exposure) -> None:
        self.state.config = replace(self.state.config, exposure=replace(self.state.config.exposure, **exposure))

    def _render(self, **kwargs):
        self.controller._is_rendering = False
        self.controller.request_render(**kwargs)
        return self.tasks[-1]

    def _finish(self, task) -> None:
        """Stand in for the worker completing `task`, with the fields it stamps on the metrics."""
        self.controller._is_rendering = False
        self.controller._on_render_finished(
            None,
            {
                "base_positive": np.zeros((2, 2, 3), np.float32),
                "source_hash": task.source_hash,
                "memo_key": task.memo_key,
                "interactive": task.interactive,
                "config_override": task.config_override,
                "ephemeral": task.ephemeral,
                "compare": task.compare,
            },
        )

    def _flat_on(self) -> None:
        self.controller.toggle_flat_peek(True)
        self.assertTrue(self.state.flat_peek)

    def test_an_edit_keeps_flat_scan(self):
        self._flat_on()
        seen: list = []
        self.controller.flat_peek_changed.connect(seen.append)
        self._edit(wb_cyan=0.12)
        task = self._render()
        self.assertTrue(self.state.flat_peek)
        self.assertEqual(seen, [])
        self.assertTrue(task.config_override)
        self.assertEqual(task.config.exposure.render_intent, RenderIntent.FLAT)
        self.assertEqual(task.config.exposure.wb_cyan, 0.12)
        self.assertEqual(task.config, flat_master_config(self.state.config))

    def test_turning_flat_scan_on_and_off_renders_the_full_preview(self):
        self._flat_on()
        self.assertFalse(self.tasks[-1].interactive)
        self.assertEqual(self.tasks[-1].config.exposure.render_intent, RenderIntent.FLAT)
        self.controller._is_rendering = False
        self.controller.toggle_flat_peek(False)
        self.assertFalse(self.tasks[-1].config_override)
        self.assertEqual(self.tasks[-1].config.exposure.render_intent, RenderIntent.PRINT)

    def test_a_drag_renders_the_proxy_and_the_release_the_full_preview(self):
        self.state.preview_proxy = np.empty((4, 4, 3), dtype=np.float32)
        self._flat_on()
        drag = self._render(readback_metrics=False)
        self.assertTrue(drag.interactive)
        self.assertIs(drag.buffer, self.state.preview_proxy)
        self.assertFalse(drag.readback_metrics)
        release = self._render()
        self.assertFalse(release.interactive)
        self.assertIs(release.buffer, self.state.preview_raw)
        self.assertTrue(release.readback_metrics)
        self.assertEqual(release.config.exposure.render_intent, RenderIntent.FLAT)

    def test_flat_scan_render_is_not_memoized_or_thumbnailed(self):
        self._flat_on()
        self._edit(wb_magenta=0.05)
        task = self._render()
        self.assertEqual(task.memo_key, "")
        self.assertFalse(task.wants_thumbnail)
        with patch.object(self.controller, "_update_thumbnail_from_state") as thumb:
            self._finish(task)
        thumb.assert_not_called()
        self.assertEqual(len(self.controller._render_memo._entries), 0)

    def test_going_back_to_a_frame_after_flat_scan_paints_the_print(self):
        self._flat_on()
        self._edit(wb_yellow=0.07)
        self._finish(self._render())
        self.controller._is_rendering = False
        self.controller.toggle_flat_peek(False)
        task = self.tasks[-1]
        self.assertEqual(task.memo_key, self.controller._render_memo_key())
        self._finish(task)
        entries = list(self.controller._render_memo._entries.values())
        self.assertEqual(len(entries), 1)
        self.assertIsNotNone(self.controller._render_memo.get("frame", self.controller._render_memo_key()))
        self.assertEqual(self.state.config.exposure.render_intent, RenderIntent.PRINT)

    def test_flat_scan_never_changes_the_saved_edit(self):
        self._edit(wb_cyan=0.1)
        edit = self.state.config
        self._flat_on()
        self._finish(self._render())
        self.assertIs(self.state.config, edit)
        bounds = LogNegativeBounds(floors=(-0.1, -0.2, -0.3), ceils=(-2.0, -2.1, -2.2))
        self.controller._on_metrics_updated({"log_bounds_base": bounds, "source_hash": "frame", "config_override": True})
        written = self.session.update_config.call_args.args[0]
        self.assertEqual(written.exposure, edit.exposure)
        self.assertEqual(written.process, replace(edit.process, local_floors=bounds.floors, local_ceils=bounds.ceils))

    def test_compare_capture_ephemeral_and_explicit_overrides_are_not_substituted(self):
        self._flat_on()
        baseline = replace(self.state.config, exposure=replace(self.state.config.exposure, density=1.7))
        compare = self._render(readback_metrics=False, config_override=baseline, compare_capture=True)
        self.assertIs(compare.config, baseline)
        ephemeral = self._render(ephemeral=True)
        self.assertFalse(ephemeral.config_override)
        self.assertIs(ephemeral.config, self.state.config)
        other = self._render(readback_metrics=False, config_override=baseline)
        self.assertIs(other.config, baseline)

    def test_rerender_active_view_renders_the_flat_master(self):
        self._flat_on()
        self.controller._is_rendering = False
        self.controller.rerender_active_view()
        self.assertTrue(self.state.flat_peek)
        self.assertEqual(self.tasks[-1].config, flat_master_config(self.state.config))

    def _switch_frame(self, path: str = "next.dng") -> None:
        self.controller.preview_load_requested.disconnect(self.controller.preview_load_worker.process)
        self.controller.load_file(path)

    def test_flat_scan_renders_both_diptych_halves_flat(self):
        halves = (self.state.config, self.state.config)
        info = {"split_x": 0.5, "gutter_thickness": 0.0, "crop_rect": None}
        with patch.object(self.controller, "active_diptych", return_value=(info, halves)):
            self._flat_on()
            self.assertEqual([h.exposure.render_intent for h in self.tasks[-1].diptych], [RenderIntent.FLAT] * 2)
            self.controller._is_rendering = False
            self.controller.toggle_flat_peek(False)
            self.assertEqual([h.exposure.render_intent for h in self.tasks[-1].diptych], [RenderIntent.PRINT] * 2)

    def test_roll_average_ends_flat_scan(self):
        self._flat_on()
        with patch.object(self.controller, "_saved_baseline_fields", return_value={"locked_floors": (0.1, 0.1, 0.1)}):
            self.controller.toggle_average_peek(True)
        self.assertTrue(self.state.average_peek)
        self.assertFalse(self.state.flat_peek)

    def test_flat_scan_survives_a_frame_switch(self):
        self._flat_on()
        # The memo holds the next frame's print, which must not be painted as its flat master.
        memo_payload = {"base_positive": np.ones((2, 2, 3), np.float32), "content_rect": None}
        with patch.object(self.controller, "_file_hash_for_path", return_value="next"):
            self.controller._render_memo.store("next", self.controller._render_memo_key(), memo_payload)
            self._switch_frame()
        self.assertTrue(self.state.flat_peek)
        self.assertIsNot(self.state.last_metrics.get("base_positive"), memo_payload["base_positive"])
        # The decode lands and its first render is the flat master.
        self.state.preview_raw = np.empty((8, 8, 3), dtype=np.float32)
        self.assertEqual(self._render().config.exposure.render_intent, RenderIntent.FLAT)

    def test_a_print_recipe_frame_ends_flat_scan(self):
        self.state.flat_output = True
        self._flat_on()
        with patch.object(self.controller, "launch_recipe_for", return_value=(LaunchRecipe(intent="print"), None)):
            self._switch_frame()
        self.assertFalse(self.state.flat_peek)

    def test_flat_scan_survives_rotate_flip_and_fine_rotation(self):
        self._flat_on()
        self.state.config = replace(self.state.config, geometry=replace(self.state.config.geometry, rotation=1, flip_horizontal=True))
        self.controller._is_rendering = False
        self.controller.rerender_active_view()
        self.assertTrue(self.state.flat_peek)
        self.assertEqual(self.tasks[-1].config, flat_master_config(self.state.config))
        # A committed fine rotation, crop or Keystone ends the peeks, then renders.
        self.state.config = replace(self.state.config, geometry=replace(self.state.config.geometry, fine_rotation=0.8))
        self.controller.reset_all_peeks()
        self.assertTrue(self.state.flat_peek)
        self.assertEqual(self._render().config.geometry.fine_rotation, 0.8)
        self.assertEqual(self.tasks[-1].config.exposure.render_intent, RenderIntent.FLAT)

    def test_flat_scan_survives_undo_and_redo(self):
        self._flat_on()
        before = self.state.config
        self._edit(wb_cyan=0.2)
        self._render()
        # Undo restores a config and emits state_changed, which starts the render debounce.
        self.state.config = before
        self.controller._render_if_state_moved()
        self.assertTrue(self.controller._render_debounce.isActive())
        task = self._render()
        self.assertTrue(self.state.flat_peek)
        self.assertEqual(task.config, flat_master_config(before))

    def test_compare_ends_flat_scan(self):
        self._flat_on()
        self.controller.toggle_compare()
        self.assertFalse(self.state.flat_peek)
        self.assertTrue(self.state.compare_mode)

    def test_the_export_size_view_ends_flat_scan(self):
        self._flat_on()
        self.controller.toggle_export_size_view(True)
        self.assertFalse(self.state.flat_peek)

    def test_linear_intent_ends_flat_scan(self):
        self._flat_on()
        self.controller.set_linear_output(True)
        self.assertFalse(self.state.flat_peek)

    def test_closing_the_last_file_ends_flat_scan(self):
        self.session.session_emptied.connect.assert_any_call(self.controller._end_flat_scan)
        self._flat_on()
        self.controller._end_flat_scan()
        self.assertFalse(self.state.flat_peek)


def _persisted(config: WorkspaceConfig, img: np.ndarray) -> dict:
    """The metrics _on_metrics_updated writes into the edit, from one render of `config`."""
    ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=config.process.process_mode)
    DarkroomEngine().process(img, config, "persisted", ctx)
    bounds = ctx.metrics.get("log_bounds_base") or ctx.metrics.get("log_bounds")
    return {
        "floors": bounds.floors,
        "ceils": bounds.ceils,
        "fade_white": ctx.metrics.get("fade_white"),
        "fade_crossover": ctx.metrics.get("fade_crossover"),
    }


def test_a_slide_with_cast_removal_writes_the_same_bounds_in_flat_scan():
    """Flat Scan turns Cast Removal off, which changes the slide's base stage key."""
    img = scene.scan_image(scene.linear_fade(scene.CROSSOVER, (0.8, 0.9, 0.6)), scene.build_scene(192, 144))
    cfg = WorkspaceConfig()
    cfg = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6, fade_ratio_r=0.8, fade_white_anchor=True, fade_crossover=True),
        exposure=replace(cfg.exposure, cast_removal_strength=1.0),
    )
    print_side, flat_side = _persisted(cfg, img), _persisted(flat_master_config(cfg), img)
    assert repr(print_side) == repr(flat_side)
