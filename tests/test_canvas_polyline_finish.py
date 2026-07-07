from PyQt6.QtCore import QPointF, QRectF

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def _overlay_with_view() -> CanvasOverlay:
    overlay = CanvasOverlay(AppState())
    overlay._view_rect = QRectF(0, 0, 100, 100)
    return overlay


def test_enter_finishes_scratch_polyline() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.SCRATCH_PICK)
    overlay._scratch_pts = [QPointF(10, 10), QPointF(40, 40)]

    emitted = []
    overlay.scratch_completed.connect(emitted.append)
    overlay._finish_draw_if_active()

    assert len(emitted) == 1
    assert len(emitted[0]) == 2
    assert overlay._scratch_pts == []


def test_enter_finishes_lasso_polygon() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.LOCAL_DRAW)
    overlay._lasso_drawing = True
    overlay._lasso_pts = [QPointF(10, 10), QPointF(40, 10), QPointF(25, 40)]

    emitted = []
    overlay.lasso_completed.connect(emitted.append)
    overlay._finish_draw_if_active()

    assert len(emitted) == 1
    assert len(emitted[0]) == 3
    assert overlay._lasso_drawing is False


def test_enter_ignores_incomplete_lasso() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.LOCAL_DRAW)
    overlay._lasso_drawing = True
    overlay._lasso_pts = [QPointF(10, 10), QPointF(40, 10)]

    emitted = []
    overlay.lasso_completed.connect(emitted.append)
    overlay._finish_draw_if_active()

    # Two points can't close a polygon — keep drawing instead of wiping them.
    assert emitted == []
    assert overlay._lasso_drawing is True
    assert len(overlay._lasso_pts) == 2


def test_enter_noop_without_active_draw() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    emitted = []
    overlay.scratch_completed.connect(emitted.append)
    overlay.lasso_completed.connect(emitted.append)
    overlay._finish_draw_if_active()

    assert emitted == []
