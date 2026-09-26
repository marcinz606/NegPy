"""The overlay's host-side images (the before/after baseline, the test strip) use the canvas shader's display profile."""

from unittest.mock import MagicMock, patch

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.widget import ImageCanvas


def test_gpu_frame_gives_the_overlay_the_monitor_profile(qapp):
    state = AppState()
    gpu = MagicMock()
    gpu.is_available = True
    with (
        patch("negpy.desktop.view.canvas.widget.GPUDevice.get", return_value=gpu),
        patch("negpy.desktop.view.canvas.widget.GPUCanvasWidget.initialize_gpu"),
    ):
        canvas = ImageCanvas(state)
    state.gpu_enabled = True

    canvas.gpu_widget.set_display_transform = MagicMock()
    canvas.gpu_widget.update_texture = MagicMock()
    canvas.overlay.update_buffer = MagicMock()
    texture = MagicMock(width=4, height=3)
    monitor = b"monitor-icc"
    with patch("negpy.desktop.view.canvas.widget.GPUTexture", type(texture)):
        canvas.update_buffer(texture, "Adobe RGB", monitor_icc_bytes=monitor)

    canvas.gpu_widget.set_display_transform.assert_called_once_with("Adobe RGB", monitor, None)
    assert canvas.overlay.update_buffer.call_args.kwargs["monitor_icc_bytes"] == monitor
