from PyQt6.QtCore import QPointF

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import _LEAK_HANDLE_INSET_PX, CanvasOverlay


def _overlay(qapp) -> CanvasOverlay:
    overlay = CanvasOverlay(AppState())
    overlay.resize(400, 300)
    return overlay


def test_an_end_on_the_canvas_keeps_its_place(qapp):
    overlay = _overlay(qapp)
    end = QPointF(350.0, 150.0)
    assert overlay._leak_end_in_view(QPointF(200.0, 150.0), end) == end


def test_an_end_past_the_canvas_is_held_inside_it_on_the_line(qapp):
    overlay = _overlay(qapp)
    held = overlay._leak_end_in_view(QPointF(200.0, 150.0), QPointF(900.0, 500.0))
    assert held.x() == 400.0 - _LEAK_HANDLE_INSET_PX
    # Still on the line from the center through the end.
    assert abs((held.y() - 150.0) / (held.x() - 200.0) - 350.0 / 700.0) < 1e-9


def test_a_line_off_the_canvas_keeps_its_ends(qapp):
    overlay = _overlay(qapp)
    end = QPointF(900.0, 500.0)
    assert overlay._leak_end_in_view(QPointF(700.0, 400.0), end) == end
