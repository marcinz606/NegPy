import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import replace

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState, ToolMode
from negpy.desktop.workers.export import ExportTask, resolve_export_target_path
from negpy.domain.models import ExportConfig, ExportFormat, ExportPreset, ExportPresetOutputMode, WorkspaceConfig
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestAppController(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        # Patch GPU-touching classes before AppController.__init__ so no real GPU is created
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        # Stop all background threads before the controller is GC'd
        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_load_file_emits_zoom_reset(self):
        """Test that loading a file normally resets the zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng")

        mock_slot.assert_called_once_with(1.0)
        self.assertFalse(self.controller.state.hq_preview)

    def test_decode_failure_badges_file_and_success_clears_it(self):
        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a.dng", "path": "/tmp/a.dng", "hash": "h1"}]

        self.controller._on_preview_load_failed("/tmp/a.dng", "decode boom")
        self.assertEqual(state.uploaded_files[0]["decode_failed"], "decode boom")

        # A later successful load clears the badge even when the frame is no longer
        # the requested one (the handler prefix runs before the early return).
        self.controller._requested_file_path = "/tmp/other.dng"
        self.controller._on_preview_loaded("/tmp/a.dng", None, (0, 0), "", None, "")
        self.assertNotIn("decode_failed", state.uploaded_files[0])

    def test_clear_roll_baseline_resets_axes(self):
        state = self.mock_session_manager.state
        state.config = replace(
            state.config,
            process=replace(state.config.process, use_luma_average=True, use_colour_average=True, roll_name="PORTRA-04"),
        )

        self.controller.clear_roll_baseline()

        cfg = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(cfg.process.use_luma_average)
        self.assertFalse(cfg.process.use_colour_average)
        self.assertIsNone(cfg.process.roll_name)

    def test_thumbnail_miss_marks_file_unreadable(self):
        from PIL import Image

        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "bad.dng", "path": "/tmp/bad.dng", "hash": "h1"},
            {"name": "good.dng", "path": "/tmp/good.dng", "hash": "h2"},
        ]
        self.controller._thumb_requested = ["bad.dng", "good.dng"]

        self.controller._on_thumbnails_finished({"good.dng": Image.new("RGB", (4, 4))})

        self.assertIn("decode_failed", state.uploaded_files[0])
        self.assertNotIn("decode_failed", state.uploaded_files[1])

    def test_render_thumbnail_update_does_not_badge_other_frames(self):
        from PIL import Image

        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "a.dng", "path": "/tmp/a.dng", "hash": "h1"},
            {"name": "b.dng", "path": "/tmp/b.dng", "hash": "h2"},
        ]
        self.controller._thumb_requested = ["a.dng", "b.dng"]
        img = Image.new("RGB", (4, 4))
        self.controller._on_thumbnails_finished({"a.dng": img, "b.dng": img})
        self.assertNotIn("decode_failed", state.uploaded_files[0])

        # update_rendered() re-emits finished with a single-file dict after every
        # settled render — it must not badge the frames absent from that dict.
        self.controller._on_thumbnails_finished({"a.dng": img})
        self.assertNotIn("decode_failed", state.uploaded_files[1])

    def test_capture_worker_cancelled_is_forwarded(self):
        cancelled = MagicMock()
        self.controller.capture_cancelled.connect(cancelled)

        self.controller.capture_worker.cancelled.emit()

        cancelled.assert_called_once_with()

    def test_thumbnail_refreshes_on_config_changed_settle(self):
        """Filmstrip thumbnail is re-captured on every settled render whose config
        differs from the last capture (covers in-place edits and reset), but not on a
        repeat settle with the same config object."""
        from negpy.domain.models import WorkspaceConfig

        self.controller._update_thumbnail_from_state = MagicMock()
        self.controller._pending_render_task = None
        self.controller._thumb_config = None

        cfg = WorkspaceConfig()
        self.controller.state.config = cfg
        self.controller._on_render_finished(None, {})
        self.assertEqual(self.controller._update_thumbnail_from_state.call_count, 1)
        self.controller._update_thumbnail_from_state.assert_called_with(force_readback=True, persist=False)

        # Same config object -> no redundant refresh.
        self.controller._on_render_finished(None, {})
        self.assertEqual(self.controller._update_thumbnail_from_state.call_count, 1)

        self.controller.state.config = replace(cfg, exposure=replace(cfg.exposure, density=1.0))
        self.controller._on_render_finished(None, {})
        self.assertEqual(self.controller._update_thumbnail_from_state.call_count, 2)

    def test_thumbnail_not_refreshed_while_pending_or_ephemeral(self):
        """Don't capture a premature frame while a newer render is queued, nor the
        low-quality splash (ephemeral) render."""
        import numpy as np

        from negpy.domain.models import WorkspaceConfig
        from negpy.desktop.workers.render import RenderTask

        self.controller._update_thumbnail_from_state = MagicMock()
        self.controller._thumb_config = None
        self.controller.state.config = WorkspaceConfig()

        self.controller._pending_render_task = RenderTask(
            buffer=np.zeros((1, 1, 3), np.float32),
            config=WorkspaceConfig(),
            source_hash="x",
            preview_size=1.0,
        )
        self.controller._on_render_finished(None, {})
        self.controller._update_thumbnail_from_state.assert_not_called()

        self.controller._pending_render_task = None
        self.controller._on_render_finished(None, {"ephemeral": True})
        self.controller._update_thumbnail_from_state.assert_not_called()

    def test_proof_active_gated_by_toggle(self):
        """proof_active() is False unless the soft-proof toggle is on, even with an
        export color space set (which always resolves an output profile)."""
        self.controller.state.soft_proof_enabled = False
        self.assertFalse(self.controller.proof_active())
        self.controller.state.soft_proof_enabled = True
        # An export color space resolves an effective output profile → proof active.
        self.assertTrue(self.controller.proof_active())

    def test_load_file_preserve_zoom(self):
        """Test that load_file with preserve_zoom=True skips resetting zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng", preserve_zoom=True)

        mock_slot.assert_not_called()

    def test_toggle_hq_preview_preserves_zoom(self):
        """Test that toggling HQ mode persists via session and preserves zoom."""
        self.controller.state.current_file_path = "dummy.dng"

        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.toggle_hq_preview()

        # Persistence delegated to session
        self.mock_session_manager.set_hq_preview.assert_called_once_with(True)

        # Zoom should NOT be reset
        mock_slot.assert_not_called()

    def test_preview_loaded_updates_state_and_emits_signal(self):
        """Successful preview loads should publish dimensions before rendering starts."""
        mock_slot = MagicMock()
        self.controller.preview_loaded.connect(mock_slot)
        self.controller.request_render = MagicMock()
        self.controller._requested_file_path = "dummy.dng"

        raw = object()
        dims = (1234, 5678)

        self.controller._on_preview_loaded("dummy.dng", raw, dims, "", None, "")

        self.assertIs(self.controller.state.preview_raw, raw)
        self.assertEqual(self.controller.state.original_res, dims)
        self.assertEqual(self.controller.state.current_file_path, "dummy.dng")
        self.assertFalse(self.controller.state.has_ir)
        self.assertIsNone(self.controller.state.preview_ir)
        mock_slot.assert_called_once_with()
        self.controller.request_render.assert_called_once_with()

    def test_stale_preview_decode_is_dropped(self):
        """A decode that lands after the user switched files must not be applied —
        accepting it pairs the old buffer with the new file's hash and poisons the
        per-source analysis cache (green/red cast on the new file)."""
        self.controller.request_render = MagicMock()
        self.controller._requested_file_path = "current.dng"
        self.controller.state.preview_raw = None

        self.controller._on_preview_loaded("stale.dng", object(), (10, 20), "", None, "")

        self.assertIsNone(self.controller.state.preview_raw)
        self.controller.request_render.assert_not_called()

    def test_apply_auto_crop_enables_auto_crop_and_clears_manual_rect(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9), auto_crop_enabled=False)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.apply_auto_crop()

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertTrue(saved_config.geometry.auto_crop_enabled)
        self.assertIsNone(saved_config.geometry.manual_crop_rect)
        self.controller.request_render.assert_called_once_with()

    def test_reset_crop_disables_auto_crop_and_clears_manual_rect(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9), auto_crop_enabled=True)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.reset_crop()

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(saved_config.geometry.auto_crop_enabled)
        self.assertIsNone(saved_config.geometry.manual_crop_rect)
        self.controller.request_render.assert_called_once_with()

    def _export_task(self, path, overwrite=False):
        preset = ExportPreset(
            name="t",
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            output_path=os.path.dirname(path),
            overwrite=overwrite,
        )
        return ExportTask(
            file_info={"name": os.path.basename(path), "path": path, "hash": "h"},
            params=WorkspaceConfig(),
            export_settings=preset,
        )

    def _set_export_overwrite(self, value):
        cfg = self.controller.state.config
        self.controller.state.config = replace(cfg, export=replace(cfg.export, overwrite=value))

    def test_export_overwrite_pref_on_skips_prompt_and_overwrites(self):
        self._set_export_overwrite(True)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock()
            out = self.controller._resolve_export_conflicts([task])
            self.assertTrue(out[0].export_settings.overwrite)
            self.controller._prompt_overwrite_conflicts.assert_not_called()

    def test_export_conflict_overwrite_sets_flag_true(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(True, False))
            out = self.controller._resolve_export_conflicts([task])
            self.assertEqual(len(out), 1)
            self.assertTrue(out[0].export_settings.overwrite)

    def test_export_conflict_rename_sets_flag_false(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(False, False))
            out = self.controller._resolve_export_conflicts([task])
            self.assertFalse(out[0].export_settings.overwrite)

    def test_export_conflict_cancel_returns_none(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(None, False))
            self.assertIsNone(self.controller._resolve_export_conflicts([task]))

    def test_export_conflict_remember_persists_preference(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(True, True))
            self.controller._set_overwrite_preference = MagicMock()
            self.controller._resolve_export_conflicts([task])
            self.controller._set_overwrite_preference.assert_called_once_with(True)

    def test_export_no_conflict_passes_through_without_prompt(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))  # target not created
            self.controller._prompt_overwrite_conflicts = MagicMock()
            out = self.controller._resolve_export_conflicts([task])
            self.assertEqual(out, [task])
            self.controller._prompt_overwrite_conflicts.assert_not_called()

    def test_manual_crop_rect_changed_disables_auto_crop(self):
        geometry = replace(self.controller.state.config.geometry, auto_crop_enabled=True)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.2, 0.3, 0.8, 0.9, True)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(saved_config.geometry.auto_crop_enabled)
        self.assertEqual(saved_config.geometry.manual_crop_rect, (0.2, 0.3, 0.8, 0.9))
        self.controller.request_render.assert_called_once_with()

    def test_handle_crop_rect_changed_updates_rect(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.3, 0.25, 0.7, 0.55, True)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.geometry.manual_crop_rect, (0.3, 0.25, 0.7, 0.55))
        self.controller.request_render.assert_called_once_with()

    def test_handle_crop_rect_changed_noop_when_tool_inactive(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=None)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.NONE
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.1, 0.1, 0.5, 0.5, True)

        self.mock_session_manager.update_config.assert_not_called()
        self.controller.request_render.assert_not_called()

    def test_handle_crop_rect_changed_does_not_deactivate_tool(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.3, 0.25, 0.7, 0.55, True)

        self.assertEqual(self.controller.state.active_tool, ToolMode.CROP_MANUAL)

    def test_handle_crop_rect_changed_live_drag_does_not_persist(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.3, 0.25, 0.7, 0.55, False)

        self.assertEqual(self.mock_session_manager.update_config.call_args.kwargs.get("persist"), False)
        self.controller.request_render.assert_not_called()

    def test_handle_crop_rect_changed_defers_bounds_invalidation(self):
        """During drag the auto-exposure bounds are left untouched (only flagged dirty),
        so the base cache survives and the frame doesn't re-normalize each step."""
        process = replace(self.controller.state.config.process, local_floors=(0.1, 0.2, 0.3), lock_bounds=False)
        self.controller.state.config = replace(self.controller.state.config, process=process)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.2, 0.3, 0.8, 0.9, True)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.process.local_floors, (0.1, 0.2, 0.3))
        self.assertTrue(self.controller._crop_bounds_dirty)

    def test_leaving_crop_tool_invalidates_bounds_once(self):
        """Closing the crop tool with a pending change recomputes bounds a single time."""
        process = replace(
            self.controller.state.config.process,
            local_floors=(0.1, 0.2, 0.3),
            local_ceils=(0.4, 0.5, 0.6),
            lock_bounds=False,
        )
        self.controller.state.config = replace(self.controller.state.config, process=process)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller._crop_bounds_dirty = True
        self.controller.request_render = MagicMock()

        self.controller.set_active_tool(ToolMode.NONE)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.process.local_floors, (0.0, 0.0, 0.0))
        self.assertEqual(saved_config.process.local_ceils, (0.0, 0.0, 0.0))
        self.assertFalse(self.controller._crop_bounds_dirty)
        self.controller.request_render.assert_called_once()

    def test_apply_auto_crop_exits_manual_crop_tool(self):
        """Enabling autocrop while the manual crop tool is active deactivates the tool."""
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.apply_auto_crop()

        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)

    def _seed_two_masks(self):
        from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask

        verts = ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))
        masks = (
            PolygonMask(vertices=verts, strength=0.3, feather=0.02),
            PolygonMask(vertices=verts, strength=-0.3, feather=0.02),
        )
        self.controller.state.config = replace(self.controller.state.config, local=LocalAdjustmentsConfig(masks=masks))

    def test_set_local_mask_visible_toggles_hidden_set(self):
        self._seed_two_masks()
        self.controller.canvas = None  # tolerate no registered canvas
        self.controller.set_local_mask_visible(1, False)
        self.assertEqual(self.controller.state.local_hidden_masks, {1})
        self.controller.set_local_mask_visible(1, True)
        self.assertEqual(self.controller.state.local_hidden_masks, set())

    def test_delete_local_mask_confirmed_remaps_view_indices(self):
        self._seed_two_masks()
        self.controller.request_render = MagicMock()
        self.controller.state.local_selected_mask = 1
        self.controller.state.local_hidden_masks = {1}

        with patch("negpy.desktop.view.confirm.confirm_delete_mask", return_value=True):
            self.controller.delete_local_mask(0)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(len(saved_config.local.masks), 1)
        self.assertEqual(self.controller.state.local_selected_mask, 0)
        self.assertEqual(self.controller.state.local_hidden_masks, {0})

    def test_delete_local_mask_cancelled_is_noop(self):
        self._seed_two_masks()
        self.controller.request_render = MagicMock()
        self.mock_session_manager.update_config.reset_mock()

        with patch("negpy.desktop.view.confirm.confirm_delete_mask", return_value=False):
            self.controller.delete_local_mask(0)

        self.mock_session_manager.update_config.assert_not_called()

    def test_lasso_completion_adds_mask_and_exits_draw_mode(self):
        import numpy as np

        self.controller.state.active_tool = ToolMode.LOCAL_DRAW
        self.controller.state.last_metrics["uv_grid"] = np.zeros((2, 2, 2), dtype=np.float32)
        self.controller.request_render = MagicMock()

        self.controller.handle_lasso_completed([(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(len(saved_config.local.masks), 1)
        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)


class TestBatchExportFiltering(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp/out")
        self.controller._run_export_tasks = MagicMock()
        self.controller._confirm_bulk_export = MagicMock(return_value=True)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _captured_tasks(self):
        self.controller._run_export_tasks.assert_called_once()
        return self.controller._run_export_tasks.call_args.args[0]

    def test_export_all_with_no_filter(self):
        self.visible_indices = [0, 1, 2]
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2", "IMG_0002.cr2", "scan.tif"])

    def test_export_all_respects_filter(self):
        self.visible_indices = [0, 1]  # only IMG_*
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2", "IMG_0002.cr2"])

    def test_export_all_zero_matches_does_not_dispatch(self):
        self.visible_indices = []
        self.controller.request_batch_export()
        self.controller._run_export_tasks.assert_not_called()

    def test_export_all_preserves_display_order(self):
        self.visible_indices = [2, 0]  # reversed visible order from sort+filter
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["scan.tif", "IMG_0001.cr2"])

    def test_export_all_override_settings_applies_current_export_to_all(self):
        self.visible_indices = [0, 1]
        self.controller.state.config = replace(
            self.controller.state.config,
            export=replace(self.controller.state.config.export, export_path="/orig"),
        )
        self.controller.request_batch_export(override_settings=True)
        tasks = self._captured_tasks()
        for t in tasks:
            self.assertEqual(t.params.export.export_path, "/tmp/out")

    def test_export_all_saved_overrides_path_with_session_values(self):
        """all_saved scope uses session path/mode even when per-file configs are stale."""
        self.visible_indices = [0, 1]
        session_export = self.controller.state.config.export
        stale_export = replace(
            session_export,
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            export_path="/stale/default",
            output_subfolder="old_sub",
        )
        stale_config = replace(self.controller.state.config, export=stale_export)
        # Per-file config has stale ABSOLUTE; session has SAME_AS_SOURCE
        self.mock_session_manager.repo.load_file_settings.return_value = stale_config
        self.controller.request_batch_export(override_settings=False)
        tasks = self._captured_tasks()
        self.assertEqual(len(tasks), 2)
        for t in tasks:
            # output_mode and output_subfolder come from session config
            self.assertEqual(t.params.export.output_mode, session_export.output_mode)
            self.assertEqual(t.params.export.output_subfolder, session_export.output_subfolder)
            # export_path is validated by _ensure_valid_export_path (mocked to /tmp/out)
            self.assertEqual(t.params.export.export_path, "/tmp/out")
            # Format/quality from per-file config is preserved
            self.assertEqual(t.params.export.export_fmt, stale_export.export_fmt)


class TestPresetBatchExport(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]
        self.mock_session_manager.state.export_presets = [
            ExportPreset(name="JPEG", enabled=True, export_fmt=ExportFormat.JPEG),
            ExportPreset(name="TIFF", enabled=True, export_fmt=ExportFormat.TIFF),
        ]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller._validate_preset_paths = MagicMock(return_value=True)
        self.controller._run_export_tasks = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _captured_tasks(self):
        self.controller._run_export_tasks.assert_called_once()
        return self.controller._run_export_tasks.call_args.args[0]

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_respects_filter(self, mock_question):
        from PyQt6.QtWidgets import QMessageBox

        mock_question.return_value = QMessageBox.StandardButton.Yes
        self.visible_indices = [0, 1]
        self.controller.request_preset_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual(len(tasks), 4)
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2"] * 2 + ["IMG_0002.cr2"] * 2)

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_zero_visible_does_not_dispatch(self, mock_question):
        self.visible_indices = []
        self.controller.request_preset_batch_export()
        self.controller._run_export_tasks.assert_not_called()
        mock_question.assert_not_called()

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_cancel_does_not_dispatch(self, mock_question):
        from PyQt6.QtWidgets import QMessageBox

        mock_question.return_value = QMessageBox.StandardButton.Cancel
        self.controller.request_preset_batch_export()
        self.controller._run_export_tasks.assert_not_called()

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_confirmation_message(self, mock_question):
        from PyQt6.QtWidgets import QMessageBox

        mock_question.return_value = QMessageBox.StandardButton.Yes
        self.visible_indices = [0, 1, 2]
        self.controller.request_preset_batch_export()
        mock_question.assert_called_once()
        message = mock_question.call_args.args[2]
        self.assertIn("3 frames", message)
        self.assertIn("2 presets", message)
        self.assertIn("6 files", message)


class TestPresetExportSelected(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.state.current_file_path = "/tmp/IMG_0002.cr2"
        self.mock_session_manager.state.current_file_hash = "h2"

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]
        self.mock_session_manager.state.export_presets = [
            ExportPreset(name="JPEG", enabled=True, export_fmt=ExportFormat.JPEG),
            ExportPreset(name="TIFF", enabled=True, export_fmt=ExportFormat.TIFF),
        ]
        self.mock_session_manager.state.selected_indices = [2, 0]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller._validate_preset_paths = MagicMock(return_value=True)
        self.controller._run_export_tasks = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_preset_export_selected_confirms_and_uses_display_order(self):
        from PyQt6.QtWidgets import QMessageBox

        self.mock_session_manager.state.selected_indices = [2, 0]
        with patch("negpy.desktop.controller.QMessageBox.question") as mock_question:
            mock_question.return_value = QMessageBox.StandardButton.Yes
            self.controller.request_preset_export_selected()
            mock_question.assert_called_once()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual(len(tasks), 4)
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2"] * 2 + ["scan.tif"] * 2)

    def test_preset_export_single_selection_uses_preview_frame(self):
        self.mock_session_manager.state.selected_indices = [0]
        self.mock_session_manager.state.selected_file_idx = 2
        self.mock_session_manager.state.current_file_path = "/tmp/scan.tif"
        self.controller.request_preset_export_selected()
        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual(len(tasks), 2)
        self.assertEqual({t.file_info["name"] for t in tasks}, {"scan.tif"})

    def test_preset_export_selected_skips_excluded(self):
        self.mock_session_manager.state.uploaded_files[0]["excluded"] = True
        with patch("negpy.desktop.controller.QMessageBox.question") as mock_question:
            self.controller.request_preset_export_selected()
            mock_question.assert_not_called()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual([t.file_info["name"] for t in tasks], ["scan.tif"] * 2)

    def test_preset_export_current_frame_menu_unchanged(self):
        self.controller.request_preset_export()
        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual(len(tasks), 2)
        self.assertEqual({t.file_info["name"] for t in tasks}, {"IMG_0002.cr2"})

    def test_batch_export_default_skips_rejected(self):
        self.mock_session_manager.state.uploaded_files[1]["excluded"] = True
        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp")
        self.controller._confirm_bulk_export = MagicMock(return_value=True)

        self.controller.request_batch_export()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        names = [t.file_info["name"] for t in tasks]
        self.assertEqual(names, ["IMG_0001.cr2", "scan.tif"])

    def test_export_selected_skips_rejected(self):
        self.mock_session_manager.state.uploaded_files[0]["excluded"] = True
        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp")
        self.controller._confirm_bulk_export = MagicMock(return_value=True)

        self.controller.request_export_selected()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual([t.file_info["name"] for t in tasks], ["scan.tif"])

    def test_batch_normalization_records_history_for_other_files(self):
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

        pushed = {c.args[0] for c in self.mock_session_manager.push_external_history.call_args_list}
        # The active file (h2) records its step via update_config(persist=True) instead.
        self.assertEqual(pushed, {"h1", "h3"})
        self.mock_session_manager.update_config.assert_called()


class TestSessionRestore(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)
        self.controller.request_asset_discovery = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _mock_settings(self, files, active):
        def get(key, default=None):
            return {"session_files": files, "session_active_path": active}.get(key, default)

        self.mock_session_manager.repo.get_global_setting.side_effect = get

    def test_saved_session_paths_filters_missing(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dng") as tf:
            self._mock_settings([tf.name, "/does/not/exist.dng"], tf.name)
            self.assertEqual(self.controller.saved_session_paths(), [tf.name])
            self.assertFalse(os.path.exists("/does/not/exist.dng"))

    def test_restore_session_selects_active_and_discovers(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dng") as a, tempfile.NamedTemporaryFile(suffix=".dng") as b:
            self._mock_settings([a.name, b.name], b.name)
            self.controller.restore_session()
            self.assertEqual(self.controller._pending_scanned_file, b.name)
            self.controller.request_asset_discovery.assert_called_once_with([a.name, b.name], auto_open=True, restore_triplets={})

    def test_restore_session_no_saved_files_is_noop(self):
        self._mock_settings([], None)
        self.controller.restore_session()
        self.controller.request_asset_discovery.assert_not_called()


class TestRgbScanModeReload(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)
        self.controller.request_asset_discovery = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_toggle_with_no_files_only_saves_flag(self):
        self.controller.set_rgb_scan_mode(True)
        self.mock_session_manager.repo.save_global_setting.assert_called_once_with("rgbscan_mode", True)
        self.controller.request_asset_discovery.assert_not_called()

    def test_toggle_with_loaded_files_rediscovers_all_exposures(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "a (RGB)", "path": "/r1.dng", "hash": "h1", "green_path": "/g1.dng", "blue_path": "/b1.dng"},
            {"name": "c", "path": "/c.dng", "hash": "h2"},
        ]
        state.current_file_path = "/r1.dng"
        self.controller.set_rgb_scan_mode(False)
        self.controller.request_asset_discovery.assert_called_once_with(
            ["/r1.dng", "/g1.dng", "/b1.dng", "/c.dng"], replace_existing=True, reselect_path="/r1.dng"
        )

    def test_discovery_finished_replace_rebuilds_and_reselects(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "r", "path": "/r.dng", "hash": "h1"},
            {"name": "g", "path": "/g.dng", "hash": "h2"},
            {"name": "b", "path": "/b.dng", "hash": "h3"},
        ]

        def add_files(_paths, validated_info=None):
            state.uploaded_files.extend(validated_info or [])

        self.mock_session_manager.add_files.side_effect = add_files
        self.controller.generate_missing_thumbnails = MagicMock()
        self.controller._replace_after_discovery = True
        self.controller._reselect_after_discovery = "/g.dng"  # was viewing the green exposure

        merged = [{"name": "r (RGB)", "path": "/r.dng", "hash": "h1", "green_path": "/g.dng", "blue_path": "/b.dng"}]
        self.controller._on_discovery_finished(merged)

        self.assertEqual(state.uploaded_files, merged)
        self.mock_session_manager.select_file.assert_called_once_with(0)


class TestDiscoveryProgressPopup(unittest.TestCase):
    """Folder-load hashing drives the shared batch progress popup."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.get_global_setting.return_value = False

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_request_discovery_opens_popup(self):
        started = []
        self.controller.batch_started.connect(lambda title, ab: started.append((title, ab)))
        self.controller.request_asset_discovery(["/a.dng"])
        self.assertEqual(started, [("Hashing files", False)])

    def test_progress_feeds_popup(self):
        progress = []
        self.controller.batch_progress.connect(lambda c, t, n: progress.append((c, t, n)))
        self.controller._on_discovery_progress(2, 5, "x")
        self.assertEqual(progress, [(2, 5, "x")])

    def test_finished_closes_popup_before_thumbnails(self):
        order = []
        self.controller.batch_finished.connect(lambda: order.append("finished"))
        self.controller.generate_missing_thumbnails = MagicMock(side_effect=lambda: order.append("thumbs"))
        self.controller._replace_after_discovery = True
        self.controller._reselect_after_discovery = "/r.dng"
        self.mock_session_manager.add_files.side_effect = lambda _p, validated_info=None: None
        self.mock_session_manager.state.uploaded_files = [{"name": "r", "path": "/r.dng", "hash": "h1"}]

        self.controller._on_discovery_finished([{"name": "r", "path": "/r.dng", "hash": "h1"}])

        self.assertEqual(order, ["finished", "thumbs"])

    def test_back_to_back_capture_completions_are_discovered_in_order(self):
        self.controller.asset_discovery_requested.disconnect(self.controller.discovery_worker.process)
        tasks = []
        self.controller.asset_discovery_requested.connect(tasks.append)
        self.controller.generate_missing_thumbnails = MagicMock()
        state = self.mock_session_manager.state
        self.mock_session_manager.add_files.side_effect = lambda _paths, validated_info=None: state.uploaded_files.extend(
            validated_info or []
        )
        req = MagicMock()
        req.white_mode = False
        req.rgb_mode = True
        self.controller._last_capture_req = req

        first_paths = ["/roll/frame1_R.dng", "/roll/frame1_G.dng", "/roll/frame1_B.dng"]
        second_paths = ["/roll/frame2_R.dng", "/roll/frame2_G.dng", "/roll/frame2_B.dng"]
        self.controller._on_capture_finished(first_paths)
        self.controller._on_capture_finished(second_paths)

        self.assertEqual([task.paths for task in tasks], [first_paths])

        self.controller._on_discovery_finished([{"name": "frame1", "path": first_paths[0], "hash": "h1"}])
        self.assertEqual([task.paths for task in tasks], [first_paths, second_paths])
        self.assertIn(os.path.normcase(os.path.abspath(first_paths[0])), self.controller._pending_capture_imports)
        self.assertIn(os.path.normcase(os.path.abspath(second_paths[0])), self.controller._pending_capture_imports)

        self.controller._on_discovery_finished([{"name": "frame2", "path": second_paths[0], "hash": "h2"}])
        self.assertEqual([f["path"] for f in state.uploaded_files], [first_paths[0], second_paths[0]])
        self.mock_session_manager.select_file.assert_called_with(1)
        self.assertIn(os.path.normcase(os.path.abspath(first_paths[0])), self.controller._pending_capture_imports)
        self.assertNotIn(os.path.normcase(os.path.abspath(second_paths[0])), self.controller._pending_capture_imports)


class TestBatchAnalysisFiltering(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.emitted = []
        self.controller.normalization_requested.connect(self.emitted.append)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_analysis_respects_filter(self):
        self.visible_indices = [0, 1]  # only IMG_*
        with patch("negpy.desktop.controller.QMessageBox") as mock_box:
            mock_box.StandardButton.Yes = 1
            mock_box.question.return_value = 1
            self.controller.request_batch_normalization()
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual([f["name"] for f in self.emitted[0].files], ["IMG_0001.cr2", "IMG_0002.cr2"])

    def test_analysis_zero_matches_does_not_dispatch(self):
        self.visible_indices = []
        with patch("negpy.desktop.controller.QMessageBox") as mock_box:
            mock_box.StandardButton.Yes = 1
            mock_box.question.return_value = 1
            self.controller.request_batch_normalization()
        self.assertEqual(self.emitted, [])


class TestContactSheetOutputDir(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.asset_model = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.visible_files = [
            {"name": "a.cr2", "path": "/rolls/frame/a.cr2", "hash": "h1"},
            {"name": "b.cr2", "path": "/rolls/frame/b.cr2", "hash": "h2"},
        ]

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_custom_path_wins_over_export_destination(self):
        export = ExportConfig(
            contact_sheet_output_path="/custom/contact",
            output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/custom/contact")

    def test_empty_path_uses_source_folder_when_same_as_source(self):
        export = ExportConfig(
            contact_sheet_output_path="",
            output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/rolls/frame")

    def test_empty_path_uses_export_path_when_absolute(self):
        export = ExportConfig(
            contact_sheet_output_path="",
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            export_path="/home/user/NegPy/export",
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/home/user/NegPy/export")

    def test_whitespace_only_path_falls_back_to_export_rules(self):
        export = ExportConfig(
            contact_sheet_output_path="   ",
            output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/rolls/frame")


if __name__ == "__main__":
    unittest.main()
