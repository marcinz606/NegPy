import sys
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.view.canvas.overlay import export_loupe_center
from negpy.domain.models import ExportConfig, ExportResolutionMode, WorkspaceConfig
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestExportSizeView(unittest.TestCase):
    def setUp(self):
        session = MagicMock(spec=DesktopSessionManager)
        session.state = AppState()
        session.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(session)
        state = self.controller.state
        state.preview_raw = np.empty((8, 8, 3), dtype=np.float32)
        state.current_file_path = "/scans/frame.nef"
        state.current_file_hash = "h1"
        state.original_res = (6000, 4000)
        self.export_buffer = np.full((40, 60, 3), 0.5, dtype=np.float32)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.prefetch_load_thread,
            self.controller.scan_thread,
            self.controller._export_view_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _land(self, key=None):
        key = self.controller._export_view_task()[0] if key is None else key
        self.controller._on_export_view_done(key, (self.export_buffer, (2, 3, 56, 34)))

    def test_turning_it_on_starts_an_export_render_and_ends_the_peeks(self):
        self.controller.state.negative_peek = True
        self.controller.toggle_export_size_view(force=True)
        self.assertTrue(self.controller.state.export_size_view)
        self.assertFalse(self.controller.state.negative_peek)
        self.assertTrue(self.controller._export_view_timer.isActive())

    def test_the_landed_render_is_painted_as_a_proofed_print_in_export_pixels(self):
        self.controller.toggle_export_size_view(force=True)
        self._land()
        frame = self.controller.state.peek_frame
        self.assertIs(frame["base_positive"], self.export_buffer)
        self.assertEqual(frame["content_rect"], (2, 3, 56, 34))
        self.assertTrue(frame["proof"])
        self.assertTrue(frame["export_view"])
        # The zoom readout divides by the scan's long edge, so 100% is one export pixel.
        self.assertEqual(frame["render_long_edge"], 6000)

    def test_a_render_for_an_older_edit_is_dropped_and_rendered_again(self):
        self.controller.toggle_export_size_view(force=True)
        stale = self.controller._export_view_task()[0]
        self.controller.state.config = replace(self.controller.state.config, lab=replace(self.controller.state.config.lab, sharpen=0.9))
        self.controller._export_view_timer.stop()
        self._land(stale)
        self.assertIsNone(self.controller.state.export_view)
        self.assertTrue(self.controller._export_view_timer.isActive())

    def test_an_edit_drops_the_export_render_it_moved_past(self):
        self.controller.state.grain_focuser = True
        self._land()
        self.assertIsNotNone(self.controller.state.export_view)
        self.controller.state.config = replace(self.controller.state.config, lab=replace(self.controller.state.config.lab, sharpen=0.9))
        self.controller._schedule_export_view()
        self.assertIsNone(self.controller.state.export_view)

    def test_the_loupe_alone_asks_for_the_export_render(self):
        self.controller.toggle_grain_focuser(force=True)
        self.assertTrue(self.controller._export_view_timer.isActive())

    def test_nothing_asking_frees_the_export_render(self):
        self.controller.state.grain_focuser = True
        self._land()
        self.controller.toggle_grain_focuser(force=False)
        self.assertIsNone(self.controller.state.export_view)

    def test_a_peek_ends_export_size(self):
        self.controller.toggle_export_size_view(force=True)
        with patch.object(self.controller, "request_render"):
            self.controller.toggle_negative_peek(force=True)
        self.assertFalse(self.controller.state.export_size_view)

    def test_the_task_uses_the_export_panel_settings(self):
        export = replace(self.controller.state.config.export, export_resolution_mode=ExportResolutionMode.TARGET_PX.value)
        self.controller.state.config = replace(self.controller.state.config, export=export)
        _key, task = self.controller._export_view_task()
        self.assertEqual(task.export_settings.export_resolution_mode, ExportResolutionMode.TARGET_PX.value)


def test_export_loupe_center_maps_into_the_picture_inside_the_mat() -> None:
    assert export_loupe_center(0.5, 0.5, (10, 20, 100, 50), 120, 90) == (60.0, 45.0)
    assert export_loupe_center(0.0, 1.0, None, 120, 90) == (0.0, 90.0)


def test_render_export_view_returns_the_export_buffer_and_its_picture_rect() -> None:
    service = ImageProcessor(use_gpu=False)
    img = np.random.default_rng(1).random((80, 120, 3), dtype=np.float32)
    service._prepare_export_source = lambda *a, **k: (img, "sRGB", "token")
    params = WorkspaceConfig()
    params = replace(params, finish=replace(params.finish, border_size=0.5))
    export = ExportConfig(export_resolution_mode=ExportResolutionMode.TARGET_PX.value, export_target_long_edge_px=100)

    buffer, rect = service.render_export_view("/fake/shot.tif", params, export, "hash", prefer_gpu=False)
    exported, _ = service.render_export("/fake/shot.tif", params, export, "hash", prefer_gpu=False)

    np.testing.assert_array_equal(buffer, exported)
    assert rect is not None
    x, y, w, h = rect
    assert x > 0 and y > 0 and x + w < buffer.shape[1] and y + h < buffer.shape[0]
