import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import replace

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState, ToolMode
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


if __name__ == "__main__":
    unittest.main()
