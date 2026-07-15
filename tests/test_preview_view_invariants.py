"""
View contract: changing zoom must not trigger a full render (GPU pipeline) by itself.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import QCoreApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.services.rendering.preview_manager import PreviewManager

if not QCoreApplication.instance():
    _ = QCoreApplication(sys.argv)


class TestZoomDoesNotRequestRender(unittest.TestCase):
    """`zoom_requested` is wired to canvas.set_zoom only; must not call `request_render`."""

    def setUp(self) -> None:
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self) -> None:
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.prefetch_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_zoom_requested_does_not_call_request_render(self) -> None:
        mock_canvas = MagicMock()
        mock_canvas.set_zoom = MagicMock()
        with patch.object(self.controller, "request_render") as req:
            self.controller.register_canvas(mock_canvas)
            self.controller.zoom_requested.emit(2.0)

        req.assert_not_called()
        mock_canvas.set_zoom.assert_called_once_with(2.0)


if __name__ == "__main__":
    unittest.main()
