"""A viewport that could not start its GPU surface must say so and route frames to the CPU path."""

from unittest.mock import MagicMock, patch

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.widget import ImageCanvas


def test_viewport_failure_is_recorded_and_frames_take_the_cpu_path(qapp):
    state = AppState()
    gpu = MagicMock()
    gpu.is_available = True
    with (
        patch("negpy.desktop.view.canvas.widget.GPUDevice.get", return_value=gpu),
        patch("negpy.desktop.view.canvas.widget.GPUCanvasWidget.initialize_gpu", side_effect=RuntimeError("no surface")),
    ):
        canvas = ImageCanvas(state)
    assert state.gpu_viewport_failed == "no surface"
    assert state.gpu_enabled, "the pipeline toggle is the user's; only the viewport failed"

    canvas.gpu_widget.update_texture = MagicMock()
    canvas.overlay.update_buffer = MagicMock()
    texture = MagicMock()
    texture.readback.return_value = MagicMock(ndim=2)
    with patch("negpy.desktop.view.canvas.widget.GPUTexture", type(texture)):
        canvas.update_buffer(texture, "sRGB")
    canvas.gpu_widget.update_texture.assert_not_called()
    canvas.overlay.update_buffer.assert_called_once()
