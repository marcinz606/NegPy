import unittest
from unittest.mock import patch

import numpy as np

from negpy.desktop.view.main_window import _display_buffer_for_canvas


class _FakeGPUTexture:
    def __init__(self, array: np.ndarray):
        self._array = array

    def readback(self) -> np.ndarray:
        return self._array


class TestDisplayBufferForCanvas(unittest.TestCase):
    def test_gpu_readback_drops_alpha_channel(self):
        rgba = np.zeros((4, 5, 4), dtype=np.float32)
        rgba[:, :, 0] = 0.25
        rgba[:, :, 1] = 0.5
        rgba[:, :, 2] = 0.75
        rgba[:, :, 3] = 1.0

        with patch("negpy.desktop.view.main_window.GPUTexture", _FakeGPUTexture):
            buffer = _display_buffer_for_canvas(_FakeGPUTexture(rgba))

        self.assertIsInstance(buffer, np.ndarray)
        self.assertEqual(buffer.shape, (4, 5, 3))
        np.testing.assert_allclose(buffer[:, :, 0], 0.25)
        np.testing.assert_allclose(buffer[:, :, 1], 0.5)
        np.testing.assert_allclose(buffer[:, :, 2], 0.75)

    def test_non_gpu_buffer_passes_through(self):
        array = np.zeros((2, 2, 3), dtype=np.float32)

        self.assertIs(_display_buffer_for_canvas(array), array)


class TestDropEvent(unittest.TestCase):
    def test_drop_auto_opens_like_add_dialogs(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from negpy.desktop.view.main_window import MainWindow

        stub = SimpleNamespace(controller=MagicMock())
        url = MagicMock()
        url.toLocalFile.return_value = "/tmp/scan.dng"
        event = MagicMock()
        event.mimeData.return_value.urls.return_value = [url]

        MainWindow.dropEvent(stub, event)

        stub.controller.request_asset_discovery.assert_called_once_with(["/tmp/scan.dng"], auto_open=True)


class TestSafeRollClose(unittest.TestCase):
    @staticmethod
    def _bare_window():
        from PyQt6.QtWidgets import QMainWindow

        from negpy.desktop.view.main_window import MainWindow

        window = MainWindow.__new__(MainWindow)
        QMainWindow.__init__(window)
        return window

    def test_active_roll_close_is_ignored_and_requests_safe_stop_once(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from negpy.desktop.view.main_window import MainWindow

        controller = SimpleNamespace(
            ls5000_roll_operation_active=MagicMock(return_value=True),
            request_ls5000_roll_stop=MagicMock(),
        )
        window = SimpleNamespace(
            controller=controller,
            _close_after_ls5000_roll=False,
            _show_ls5000_safe_close_notice=MagicMock(),
        )
        event = MagicMock()

        MainWindow.closeEvent(window, event)
        MainWindow.closeEvent(window, event)

        self.assertTrue(window._close_after_ls5000_roll)
        self.assertEqual(event.ignore.call_count, 2)
        controller.request_ls5000_roll_stop.assert_called_once_with()
        self.assertEqual(window._show_ls5000_safe_close_notice.call_count, 2)

    def test_terminal_roll_signal_closes_only_after_operation_is_released(self):
        from types import MethodType, SimpleNamespace
        from unittest.mock import MagicMock

        from PyQt6.QtWidgets import QApplication

        from negpy.desktop.view.main_window import MainWindow

        class Signal:
            def __init__(self):
                self.callbacks = []

            def connect(self, callback):
                self.callbacks.append(callback)

            def emit(self, *args):
                for callback in self.callbacks:
                    callback(*args)

        active = {"value": True}
        controller = SimpleNamespace(
            ls5000_roll_operation_active=lambda: active["value"],
            ls5000_roll_preview_ready=Signal(),
            ls5000_roll_finished=Signal(),
            ls5000_roll_error=Signal(),
            set_status=MagicMock(),
        )
        notice = MagicMock()
        window = SimpleNamespace(
            controller=controller,
            _close_after_ls5000_roll=True,
            _ls5000_safe_close_notice=notice,
            close=MagicMock(),
        )
        window._on_ls5000_roll_terminal = MethodType(
            MainWindow._on_ls5000_roll_terminal,
            window,
        )
        MainWindow._connect_ls5000_safe_close_signals(window)

        controller.ls5000_roll_preview_ready.emit("token", object())
        QApplication.processEvents()
        window.close.assert_not_called()

        active["value"] = False
        controller.ls5000_roll_finished.emit(object())
        QApplication.processEvents()

        notice.hide.assert_called_once_with()
        window.close.assert_called_once_with()

    def test_normal_close_still_persists_geometry_and_is_accepted(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from PyQt6.QtGui import QCloseEvent

        from negpy.desktop.view.main_window import MainWindow

        repo = MagicMock()
        window = self._bare_window()
        window.controller = SimpleNamespace(
            ls5000_roll_operation_active=lambda: False,
            session=SimpleNamespace(repo=repo),
        )
        window.setGeometry(12, 34, 800, 600)
        event = QCloseEvent()

        MainWindow.closeEvent(window, event)

        self.assertTrue(event.isAccepted())
        repo.save_global_setting.assert_called_once_with(
            "window_geometry",
            [12, 34, 800, 600],
        )
        window.deleteLater()

    def test_safe_close_notice_is_visible_and_nonblocking(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from PyQt6.QtWidgets import QApplication

        from negpy.desktop.view.main_window import MainWindow

        window = self._bare_window()
        window.controller = SimpleNamespace(set_status=MagicMock())
        window._ls5000_safe_close_notice = None

        MainWindow._show_ls5000_safe_close_notice(window)
        QApplication.processEvents()

        notice = window._ls5000_safe_close_notice
        self.assertIsNotNone(notice)
        self.assertFalse(notice.isModal())
        self.assertTrue(notice.isVisible())
        self.assertIn("Do not Force Quit", notice.informativeText())
        notice.close()
        window.deleteLater()


if __name__ == "__main__":
    unittest.main()
