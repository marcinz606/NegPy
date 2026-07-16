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


if __name__ == "__main__":
    unittest.main()
