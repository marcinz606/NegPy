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
            state=SimpleNamespace(gpu_enabled=gpu_enabled, gpu_viewport_failed=""),
            gpu_widget=MagicMock(),
            overlay=MagicMock(),
            _raise_floating_widgets=MagicMock(),
            _sync_navigator=MagicMock(),
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

        stub.controller.request_asset_discovery.assert_called_once_with(["/tmp/scan.dng"], auto_open=True, user_add=True)


class TestBatchReportsInTheStrip(unittest.TestCase):
    """Hot Folder batches are quiet by sequence identity (did THIS batch come from the
    hot-folder poll), never by whether the Hot Folder toggle happens to be checked."""

    def _stub(self, hot_folder_sequence_active: bool):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from negpy.desktop.view.main_window import MainWindow

        controller = MagicMock()
        controller.hot_folder_sequence_active = hot_folder_sequence_active
        stub = SimpleNamespace(controller=controller, canvas=SimpleNamespace(hud=MagicMock()), _batch_message="")
        stub._show_batch_message = lambda text: MainWindow._show_batch_message(stub, text)
        return stub

    def test_manual_import_posts_its_title_even_when_hot_folder_checked(self):
        from negpy.desktop.view.main_window import MainWindow

        stub = self._stub(hot_folder_sequence_active=False)
        MainWindow._on_batch_started(stub, "Hashing files", False)
        stub.canvas.hud.showMessage.assert_called_once_with("Hashing files…", -1)
        stub.canvas.hud.set_abortable.assert_called_once_with(False)

    def test_hot_folder_sequence_keeps_the_bar_but_posts_no_message(self):
        from negpy.desktop.view.main_window import MainWindow

        stub = self._stub(hot_folder_sequence_active=True)
        MainWindow._on_batch_started(stub, "Hashing files", False)
        MainWindow._on_batch_progress(stub, 1, 3, "a.tif")
        stub.canvas.hud.showMessage.assert_not_called()
        stub.canvas.hud.set_progress.assert_called_once_with(1, 3)

    def test_abortable_batch_reports_each_item_and_offers_abort(self):
        from negpy.desktop.view.main_window import MainWindow

        stub = self._stub(hot_folder_sequence_active=True)
        MainWindow._on_batch_started(stub, "Exporting", True)
        MainWindow._on_batch_progress(stub, 2, 10, "frame_02.tif")
        stub.canvas.hud.set_abortable.assert_called_once_with(True)
        stub.canvas.hud.showMessage.assert_called_with("Exporting: frame_02.tif (2/10)", -1)

    def test_finishing_clears_only_the_batch_message(self):
        from negpy.desktop.view.main_window import MainWindow

        stub = self._stub(hot_folder_sequence_active=False)
        MainWindow._on_batch_started(stub, "Exporting", True)
        MainWindow._on_batch_progress(stub, 10, 10, "frame_10.tif")
        MainWindow._on_batch_finished(stub)
        stub.canvas.hud.clear_message.assert_called_once_with("Exporting: frame_10.tif (10/10)")
        stub.canvas.hud.hide_progress.assert_called_once()
        stub.canvas.hud.set_abortable.assert_called_with(False)


if __name__ == "__main__":
    unittest.main()


class TestScanSetupPicksScanSource(unittest.TestCase):
    """Picking Camera or Film Scanner on the Scanning Setup card switches the Scan tab to match."""

    def _card(self):
        from unittest.mock import MagicMock

        from negpy.desktop.session import AppState
        from negpy.desktop.view.sidebar.scan_setup import ScanSetupSidebar

        controller = MagicMock()
        controller.state = AppState()
        controller.session.repo.get_global_setting.return_value = None
        card = ScanSetupSidebar(controller)
        card.sync_ui()
        picks: list = []
        card.capture_picked.connect(picks.append)
        return card, picks

    def _follow(self, capture: str):
        from unittest.mock import MagicMock

        from negpy.desktop.view.sidebar.right_panel import RightPanel

        stub = MagicMock()
        RightPanel._follow_scan_setup(stub, capture)
        return stub

    def test_a_camera_pick_selects_camera(self):
        self._follow("camera").show_scan_source.assert_called_once_with("camera")

    def test_a_scanner_pick_selects_film_scanner(self):
        card, picks = self._card()
        card.capture_btn.setCurrentIndex(card.capture_btn.findData("scanner"))
        self.assertEqual(picks, ["scanner"])
        self._follow("scanner").show_scan_source.assert_called_once_with("film")

    def test_a_light_change_leaves_the_scan_tab_alone(self):
        card, picks = self._card()
        card.light_btn.setCurrentIndex(card.light_btn.findData("narrowband"))
        self.assertEqual(picks, [])
