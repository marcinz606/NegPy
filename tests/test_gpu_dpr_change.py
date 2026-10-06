"""A device-pixel-ratio change refreshes the GPU surface.

A window dragged to a screen with another scale factor changes the surface size
in device pixels without a resize event, so the render widget's physical size
must be recomputed and the swapchain reconfigured through the resize debounce.
"""

from unittest.mock import MagicMock

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget


def _send_dpr_change(widget) -> None:
    QApplication.sendEvent(widget, QEvent(QEvent.Type.DevicePixelRatioChange))


def test_dpr_change_recomputes_the_physical_size(qapp):
    widget = GPUCanvasWidget()
    widget.resize(400, 300)
    subwidget = widget.canvas._subwidget
    subwidget.resize(400, 300)
    subwidget.devicePixelRatioF = lambda: 2.0

    _send_dpr_change(widget)

    assert subwidget.get_physical_size() == (800, 600)
    assert widget.resize_timer.isActive()


def test_the_debounce_reconfigures_a_live_context(qapp):
    widget = GPUCanvasWidget()
    widget.device = MagicMock()
    widget.context = MagicMock()
    widget._configure_context = MagicMock()

    widget._perform_resize()

    widget._configure_context.assert_called_once()


def test_dpr_change_without_a_device_is_harmless(qapp):
    widget = GPUCanvasWidget()

    _send_dpr_change(widget)
    widget._perform_resize()

    assert widget.device is None
