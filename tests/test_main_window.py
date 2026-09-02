import unittest
from unittest.mock import patch

import numpy as np


class _FakeGPUTexture:
    """Counts readbacks, so a test can prove the GPU display path never triggers one."""

    def __init__(self, array: np.ndarray):
        self._array = array
        self.height, self.width = array.shape[:2]
        self.readbacks = 0

    def readback(self) -> np.ndarray:
        self.readbacks += 1
        return self._array


def _rgba() -> np.ndarray:
    rgba = np.zeros((4, 5, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.25
    rgba[:, :, 1] = 0.5
    rgba[:, :, 2] = 0.75
    rgba[:, :, 3] = 1.0
    return rgba


class TestCanvasBufferRouting(unittest.TestCase):
    """A GPU texture goes to the shader untouched; only the CPU overlay reads it back."""

    def _stub(self, gpu_enabled: bool):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        return SimpleNamespace(
            state=SimpleNamespace(gpu_enabled=gpu_enabled),
            gpu_widget=MagicMock(),
            overlay=MagicMock(),
            _raise_floating_widgets=MagicMock(),
            zoom_changed=MagicMock(),
            zoom_level=1.0,
            _last_buffer=None,
        )

    def test_gpu_path_does_not_read_back(self):
        from negpy.desktop.view.canvas.widget import ImageCanvas

        tex = _FakeGPUTexture(_rgba())
        stub = self._stub(True)
        with patch("negpy.desktop.view.canvas.widget.GPUTexture", _FakeGPUTexture):
            ImageCanvas.update_buffer(stub, tex, "Adobe RGB")

        self.assertEqual(tex.readbacks, 0, "the GPU display path must not copy pixels to the host")
        stub.gpu_widget.update_texture.assert_called_once_with(tex)
        self.assertIsNone(stub.overlay.update_buffer.call_args[0][0])

    def test_cpu_path_reads_back_and_drops_alpha(self):
        from negpy.desktop.view.canvas.widget import ImageCanvas

        tex = _FakeGPUTexture(_rgba())
        stub = self._stub(False)
        with patch("negpy.desktop.view.canvas.widget.GPUTexture", _FakeGPUTexture):
            ImageCanvas.update_buffer(stub, tex, "Adobe RGB")

        self.assertEqual(tex.readbacks, 1)
        buffer = stub.overlay.update_buffer.call_args[0][0]
        self.assertEqual(buffer.shape, (4, 5, 3))
        np.testing.assert_allclose(buffer[:, :, 0], 0.25)
        np.testing.assert_allclose(buffer[:, :, 1], 0.5)
        np.testing.assert_allclose(buffer[:, :, 2], 0.75)

    def test_ndarray_passes_through_untouched(self):
        from negpy.desktop.view.canvas.widget import ImageCanvas

        array = np.zeros((2, 2, 3), dtype=np.float32)
        stub = self._stub(True)
        with patch("negpy.desktop.view.canvas.widget.GPUTexture", _FakeGPUTexture):
            ImageCanvas.update_buffer(stub, array, "Adobe RGB")

        self.assertIs(stub.overlay.update_buffer.call_args[0][0], array)


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


class TestBatchStartedPopupSuppression(unittest.TestCase):
    """The popup is suppressed by sequence identity (did THIS batch come from the
    hot-folder poll), never by whether the Hot Folder toggle happens to be checked."""

    def _stub(self, hot_folder_sequence_active: bool, hot_folder_checked: bool = True):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        controller = MagicMock()
        controller.hot_folder_sequence_active = hot_folder_sequence_active
        return SimpleNamespace(
            controller=controller,
            progress_dialog=MagicMock(),
            session_panel=SimpleNamespace(
                file_browser=SimpleNamespace(hot_folder_btn=SimpleNamespace(isChecked=lambda: hot_folder_checked))
            ),
        )

    def test_manual_import_shows_popup_even_when_hot_folder_checked(self):
        from negpy.desktop.view.main_window import MainWindow

        stub = self._stub(hot_folder_sequence_active=False, hot_folder_checked=True)

        MainWindow._on_batch_started(stub, "Hashing files", False)

        stub.progress_dialog.start.assert_called_once_with("Hashing files", False)

    def test_hot_folder_sequence_silent_for_discovery_and_thumbnails(self):
        from negpy.desktop.view.main_window import MainWindow

        stub = self._stub(hot_folder_sequence_active=True, hot_folder_checked=True)

        MainWindow._on_batch_started(stub, "Hashing files", False)
        MainWindow._on_batch_started(stub, "Generating thumbnails", False)

        stub.progress_dialog.start.assert_not_called()

    def test_abortable_batch_always_shows_even_mid_hot_folder_sequence(self):
        from negpy.desktop.view.main_window import MainWindow

        stub = self._stub(hot_folder_sequence_active=True, hot_folder_checked=True)

        MainWindow._on_batch_started(stub, "Exporting", True)

        stub.progress_dialog.start.assert_called_once_with("Exporting", True)


if __name__ == "__main__":
    unittest.main()
