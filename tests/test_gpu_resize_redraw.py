"""A resize draws the frame at once, before Qt paints the last bitmap stretched to the new size."""

import inspect
import sys
from unittest.mock import MagicMock

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget


def _resize(widget) -> None:
    QApplication.sendEvent(widget, QResizeEvent(QSize(600, 300), QSize(400, 300)))


def test_a_resize_with_a_frame_draws_at_once(qapp):
    widget = GPUCanvasWidget()
    widget._draw_now = MagicMock()
    widget.current_texture_view = object()

    _resize(widget)

    widget._draw_now.assert_called_once()


def test_a_resize_without_a_frame_draws_nothing(qapp):
    widget = GPUCanvasWidget()
    widget._draw_now = MagicMock()

    _resize(widget)

    widget._draw_now.assert_not_called()


def test_a_resize_during_the_draw_does_not_draw_again(qapp):
    widget = GPUCanvasWidget()
    widget.current_texture_view = object()
    widget._draw_now = MagicMock(side_effect=lambda: _resize(widget))

    _resize(widget)

    widget._draw_now.assert_called_once()


def test_a_macos_bitmap_present_draws_without_force_draw(qapp, monkeypatch):
    # force_draw pumps the event loop on macOS, inside the resize.
    monkeypatch.setattr(sys, "platform", "darwin")
    widget = GPUCanvasWidget()
    widget.canvas.force_draw = MagicMock()
    widget.canvas._subwidget._draw_and_present = MagicMock()
    widget.presents_to_screen = lambda: False

    widget._draw_now()

    widget.canvas._subwidget._draw_and_present.assert_called_once_with(force_sync=True)
    widget.canvas.force_draw.assert_not_called()


def test_a_screen_present_uses_force_draw(qapp, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    widget = GPUCanvasWidget()
    widget.canvas.force_draw = MagicMock()
    widget.canvas._subwidget._draw_and_present = MagicMock()
    widget.presents_to_screen = lambda: True

    widget._draw_now()

    widget.canvas.force_draw.assert_called_once()
    widget.canvas._subwidget._draw_and_present.assert_not_called()


def test_other_platforms_use_force_draw(qapp, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    widget = GPUCanvasWidget()
    widget.canvas.force_draw = MagicMock()
    widget.canvas._subwidget._draw_and_present = MagicMock()
    widget.presents_to_screen = lambda: False

    widget._draw_now()

    widget.canvas.force_draw.assert_called_once()
    widget.canvas._subwidget._draw_and_present.assert_not_called()


def test_rendercanvas_keeps_the_private_draw_call(qapp):
    # _draw_now calls a private rendercanvas method; an upgrade that renames it fails here, not on every resize.
    params = inspect.signature(GPUCanvasWidget().canvas._subwidget._draw_and_present).parameters
    assert params["force_sync"].kind is inspect.Parameter.KEYWORD_ONLY
