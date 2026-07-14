from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


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


def test_inflight_points_track_view_rect_change() -> None:
    # Zoom/pan while drawing must keep placed points pinned to the image, not the screen.
    overlay = _overlay_with_view()  # old rect (0,0,100,100)
    overlay._lasso_pts = [QPointF(25, 25), QPointF(75, 50)]
    overlay._scratch_pts = [QPointF(50, 50)]
    old = QRectF(0, 0, 100, 100)
    overlay._view_rect = QRectF(50, 50, 200, 200)  # new zoomed/panned rect
    overlay._remap_inflight_points(old)
    # (25,25) sits at viewport-norm (0.25,0.25) -> 50 + 0.25*200 = 100
    assert overlay._lasso_pts[0] == QPointF(100, 100)
    assert overlay._lasso_pts[1] == QPointF(200, 150)
    assert overlay._scratch_pts[0] == QPointF(150, 150)


def test_enter_noop_without_active_draw() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    emitted = []
    overlay.scratch_completed.connect(emitted.append)
    overlay.lasso_completed.connect(emitted.append)
    overlay._finish_draw_if_active()

    assert emitted == []


def test_enter_confirms_crop() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.CROP_MANUAL)

    confirmed = []
    overlay.crop_confirmed.connect(lambda: confirmed.append(True))
    overlay._finish_draw_if_active()

    assert confirmed == [True]


def _overlay_with_parent() -> CanvasOverlay:
    """Overlay whose move/release paths (which consult parent()._is_panning) work."""
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    parent._is_panning = False
    overlay = CanvasOverlay(AppState(), parent)
    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay._test_parent = parent  # keep the parent alive for the overlay's lifetime
    return overlay


def test_heal_click_places_single_spot() -> None:
    overlay = _overlay_with_parent()
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    clicks: list = []
    strokes: list = []
    overlay.clicked.connect(lambda x, y: clicks.append((x, y)))
    overlay.scratch_completed.connect(strokes.append)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(30, 30)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(30, 30), Qt.MouseButton.NoButton))

    assert strokes == []
    assert len(clicks) == 1
    assert abs(clicks[0][0] - 0.3) < 1e-6 and abs(clicks[0][1] - 0.3) < 1e-6


def test_heal_drag_paints_continuous_stroke() -> None:
    overlay = _overlay_with_parent()
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    clicks: list = []
    strokes: list = []
    overlay.clicked.connect(lambda x, y: clicks.append((x, y)))
    overlay.scratch_completed.connect(strokes.append)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))
    for p in (QPointF(30, 30), QPointF(60, 60)):
        overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, p))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(90, 90), Qt.MouseButton.NoButton))

    assert clicks == []
    assert len(strokes) == 1
    assert len(strokes[0]) >= 3  # press + drag samples + release point
    assert overlay._heal_drag_pts == []


def test_heal_drag_outside_image_is_ignored() -> None:
    overlay = _overlay_with_parent()
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    clicks: list = []
    overlay.clicked.connect(lambda x, y: clicks.append((x, y)))
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(150, 150)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(150, 150), Qt.MouseButton.NoButton))

    assert clicks == []
    assert overlay._heal_drag_pts == []


def test_esc_ladder_first_clears_points_then_nothing() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.SCRATCH_PICK)
    overlay._scratch_pts = [QPointF(10, 10), QPointF(40, 40)]

    assert overlay.cancel_in_progress() is True
    assert overlay._scratch_pts == []
    # Second rung: nothing left in progress — caller exits the tool instead.
    assert overlay.cancel_in_progress() is False


def test_esc_ladder_clears_straighten_line() -> None:
    overlay = _overlay_with_view()
    overlay.set_tool_mode(ToolMode.STRAIGHTEN)
    overlay._straighten_p1 = QPointF(10, 10)
    overlay._straighten_p2 = QPointF(90, 12)

    assert overlay.cancel_in_progress() is True
    assert overlay._straighten_p1 is None


def test_context_cancel_two_stage() -> None:
    from unittest.mock import MagicMock

    from negpy.desktop.view.keyboard_shortcuts import _context_cancel

    controller, window = MagicMock(), MagicMock()
    window.canvas.overlay.cancel_in_progress.return_value = True
    _context_cancel(controller, window)
    controller.cancel_active_tool.assert_not_called()

    window.canvas.overlay.cancel_in_progress.return_value = False
    _context_cancel(controller, window)
    controller.cancel_active_tool.assert_called_once()


def _crop_overlay(rect=(0.2, 0.2, 0.8, 0.8)) -> CanvasOverlay:
    overlay = _overlay_with_parent()
    overlay.set_tool_mode(ToolMode.CROP_MANUAL)
    overlay._crop_rect_norm = rect
    return overlay


def test_stray_click_outside_crop_keeps_rect() -> None:
    overlay = _crop_overlay()
    emitted: list = []
    overlay.crop_rect_changed.connect(lambda *a: emitted.append(a))

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(5, 5)))
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(10, 8)))  # < slop
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(10, 8), Qt.MouseButton.NoButton))

    assert emitted == []
    assert overlay._crop_rect_norm == (0.2, 0.2, 0.8, 0.8)
    assert overlay._crop_drag_mode is None


def test_drag_outside_crop_redraws_past_slop() -> None:
    overlay = _crop_overlay()
    emitted: list = []
    overlay.crop_rect_changed.connect(lambda *a: emitted.append(a))

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(5, 5)))
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(60, 60)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(60, 60), Qt.MouseButton.NoButton))

    assert len(emitted) == 1
    x1, y1, x2, y2, _final = emitted[0]
    assert abs(x1 - 0.05) < 0.02 and abs(x2 - 0.6) < 0.02


def test_fresh_crop_draw_keeps_immediate_feel() -> None:
    overlay = _crop_overlay(rect=None)
    overlay._crop_rect_norm = None
    emitted: list = []
    overlay.crop_rect_changed.connect(lambda *a: emitted.append(a))

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 10)))
    assert overlay._crop_draw_armed is True
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(20, 20)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(20, 20), Qt.MouseButton.NoButton))

    assert len(emitted) == 1
