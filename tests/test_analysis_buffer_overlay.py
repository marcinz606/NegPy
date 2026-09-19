from PyQt6.QtCore import QEventLoop, QTimer

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def _pump(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_overlay_stays_visible_while_slider_held_without_moving(qapp):
    overlay = CanvasOverlay(AppState())
    overlay.show_analysis_buffer(0.1)
    overlay.set_analysis_buffer_dragging(True)

    _pump(1200)  # past the 1s hide timer that used to fire regardless of the press
    assert overlay._buffer_overlay_visible


def test_overlay_hides_shortly_after_release(qapp):
    overlay = CanvasOverlay(AppState())
    overlay.show_analysis_buffer(0.1)
    overlay.set_analysis_buffer_dragging(True)
    overlay.set_analysis_buffer_dragging(False)

    assert overlay._buffer_overlay_visible
    _pump(1200)
    assert not overlay._buffer_overlay_visible
