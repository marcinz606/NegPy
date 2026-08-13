"""The overlay must not erase the frame the GPU canvas painted underneath it.

rendercanvas presents through a bitmap on Qt: the canvas blits its frame into the
shared backing store. The alpha hole the overlay punches on macOS/Windows was written
for a native surface underneath; over a bitmap present it wipes the frame, and the
canvas shows the bare window background (white on macOS, black on Windows).
"""

import sys

import pytest
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QWidget

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay

W, H = 60, 40
FRAME = QColor(20, 160, 90)


class _FakeGPUWidget:
    def __init__(self, visible: bool, to_screen: bool):
        self._visible, self._to_screen = visible, to_screen

    def isVisible(self) -> bool:
        return self._visible

    def presents_to_screen(self) -> bool:
        return self._to_screen


class _FakeCanvas(QWidget):
    """Stands in for ImageCanvas: owns the background color and the GPU widget."""

    def __init__(self, gpu):
        super().__init__()
        self._bg_color = QColor("#050505")
        self.gpu_widget = gpu


def _paint_over_frame(gpu) -> QImage:
    """Paint the overlay onto a surface that already holds a rendered frame."""
    canvas = _FakeCanvas(gpu)
    overlay = CanvasOverlay(AppState(), canvas)
    overlay.resize(W, H)

    surface = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
    surface.fill(FRAME)
    overlay.render(surface)
    return surface


@pytest.mark.skipif(sys.platform not in ("darwin", "win32"), reason="the alpha hole is macOS/Windows only")
def test_bitmap_present_frame_survives_the_overlay():
    surface = _paint_over_frame(_FakeGPUWidget(visible=True, to_screen=False))

    assert QColor(surface.pixel(W // 2, H // 2)) == FRAME


@pytest.mark.skipif(sys.platform not in ("darwin", "win32"), reason="the alpha hole is macOS/Windows only")
def test_screen_present_still_gets_its_hole():
    surface = _paint_over_frame(_FakeGPUWidget(visible=True, to_screen=True))

    assert QColor.fromRgba(surface.pixel(W // 2, H // 2)).alpha() == 0


def test_hidden_gpu_widget_gets_the_canvas_background():
    surface = _paint_over_frame(_FakeGPUWidget(visible=False, to_screen=False))

    assert QColor(surface.pixel(W // 2, H // 2)) == QColor("#050505")


def test_presents_to_screen_reads_the_canvas_present_method():
    from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget

    widget = GPUCanvasWidget.__new__(GPUCanvasWidget)

    class _Sub:
        _present_to_screen = False

    class _Canvas:
        _subwidget = _Sub()

    widget.canvas = _Canvas()
    assert widget.presents_to_screen() is False

    _Sub._present_to_screen = True
    assert widget.presents_to_screen() is True

    # Before get_context() the method is unresolved; no hole until it is known.
    _Sub._present_to_screen = None
    assert widget.presents_to_screen() is False
