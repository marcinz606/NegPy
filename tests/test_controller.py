import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import replace

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState, ToolMode
from negpy.domain.models import ExportConfig, ExportFormat, ExportPreset, ExportPresetOutputMode
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

    def test_local_overlay_visible_default_on(self):
        self.assertTrue(AppState().show_local_overlay)

    def test_set_local_overlay_visible_toggles_flag(self):
        self.controller.canvas = None  # tolerate no registered canvas
        self.controller.set_local_overlay_visible(False)
        self.assertFalse(self.controller.state.show_local_overlay)
        self.controller.set_local_overlay_visible(True)
        self.assertTrue(self.controller.state.show_local_overlay)

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

    def test_preset_export_selected_uses_display_order_without_confirmation(self):
        self.mock_session_manager.state.selected_indices = [2, 0]
        with patch("negpy.desktop.controller.QMessageBox.question") as mock_question:
            self.controller.request_preset_export_selected()
            mock_question.assert_not_called()

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
