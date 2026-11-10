from dataclasses import replace
import math

import pytest
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPainter
from PyQt6.QtWidgets import QWidget

from negpy.desktop.session import AppState, ToolMode
import negpy.desktop.view.canvas.overlay as canvas_overlay
from negpy.desktop.view.canvas.overlay import CanvasOverlay

_DISPLAY = (200, 160)
_INITIAL_RECT = (0.2, 0.2, 0.8, 0.8)


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


def _crop_overlay() -> CanvasOverlay:
    parent = QWidget()
    parent._is_panning = False
    overlay = CanvasOverlay(AppState(), parent)
    overlay.setFixedSize(*_DISPLAY)
    overlay._test_parent = parent
    overlay._view_rect = QRectF(0, 0, *_DISPLAY)
    overlay._current_size = _DISPLAY
    overlay.set_tool_mode(ToolMode.CROP_MANUAL)
    overlay._crop_rect_norm = _INITIAL_RECT
    return overlay


@pytest.mark.parametrize(
    ("edge", "press", "drag", "cursor", "changed_index", "expected"),
    [
        ("left", QPointF(40, 80), QPointF(60, 80), Qt.CursorShape.SizeHorCursor, 0, 0.3),
        ("right", QPointF(160, 80), QPointF(140, 80), Qt.CursorShape.SizeHorCursor, 2, 0.7),
        ("top", QPointF(100, 32), QPointF(100, 48), Qt.CursorShape.SizeVerCursor, 1, 0.3),
        ("bottom", QPointF(100, 128), QPointF(100, 112), Qt.CursorShape.SizeVerCursor, 3, 0.7),
    ],
)
def test_edge_drag_resizes_only_its_axis(edge, press, drag, cursor, changed_index, expected):
    overlay = _crop_overlay()
    emitted = []
    overlay.crop_rect_changed.connect(lambda *args: emitted.append(args))

    overlay._update_crop_hover_cursor(press)
    assert overlay.cursor().shape() == cursor
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, press))
    assert overlay._crop_drag_mode == "edge"
    assert overlay._crop_edge_which == edge

    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, drag))

    rect = overlay._crop_rect_norm
    assert rect[changed_index] == pytest.approx(expected)
    assert all(rect[i] == _INITIAL_RECT[i] for i in range(4) if i != changed_index)
    assert emitted[-1][-1] is False


def test_edge_drag_past_opposite_edge_clamps_without_inverting():
    overlay = _crop_overlay()
    press = QPointF(40, 80)
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, press))
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(190, 80)))

    x1, _, x2, _ = overlay._crop_rect_norm
    assert x1 == pytest.approx(0.68)
    assert x1 < x2
    assert (x2 - x1) * _DISPLAY[0] == pytest.approx(24.0)


def test_locked_ratio_hides_edge_handles_and_does_not_start_edge_drag():
    overlay = _crop_overlay()
    config = overlay.state.config
    overlay.state.config = replace(
        config,
        geometry=replace(config.geometry, autocrop_ratio="4:3"),
    )
    edge_midpoint = QPointF(40, 80)

    assert overlay._crop_edge_handle_screen_points() is None
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, edge_midpoint))

    assert overlay._crop_drag_mode == "move"
    assert overlay._crop_drag_mode != "edge"


def test_edge_handle_stays_at_midpoint_when_visible_and_clear():
    overlay = _crop_overlay()
    overlay._crop_rect_norm = (0.2, 0.3, 0.8, 0.7)

    handles = overlay._crop_edge_handle_screen_points()

    assert handles["top"] == QPointF(100, 48)
    assert handles["bottom"] == QPointF(100, 112)
    assert handles["left"] == QPointF(40, 80)
    assert handles["right"] == QPointF(160, 80)


def test_edge_handle_slides_to_visible_part_for_drawing_hover_and_press(monkeypatch):
    overlay = _crop_overlay()
    overlay._view_rect = QRectF(-200, 0, 240, 160)
    overlay._crop_rect_norm = (0.0, 0.3, 1.0, 0.8)
    handle = overlay._crop_edge_handle_screen_points()["top"]
    drawn = {}
    monkeypatch.setattr(
        canvas_overlay,
        "draw_resize_handles",
        lambda painter, corners, edges=None, viewport=None: drawn.update(edges=edges),
    )
    monkeypatch.setattr(overlay, "_draw_rotation_handles", lambda painter, corners: None)
    image = QImage(*_DISPLAY, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        overlay._draw_crop_tool(painter)
    finally:
        painter.end()

    assert handle.x() == pytest.approx(10.0)
    assert handle.y() == pytest.approx(48.0)
    assert drawn["edges"]["top"] == handle

    overlay._update_crop_hover_cursor(handle)
    assert overlay.cursor().shape() == Qt.CursorShape.SizeVerCursor
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, handle))
    assert overlay._crop_drag_mode == "edge"
    assert overlay._crop_edge_which == "top"


@pytest.mark.parametrize(
    ("edge", "view_rect", "crop_rect"),
    [
        ("top", QRectF(-200, 0, 240, 160), (0.0, 0.3, 1.0, 0.6)),
        ("bottom", QRectF(-200, 0, 240, 160), (0.0, 0.3, 1.0, 0.6)),
        ("left", QRectF(0, -200, 200, 240), (0.3, 0.0, 0.6, 1.0)),
        ("right", QRectF(0, -200, 200, 240), (0.3, 0.0, 0.6, 1.0)),
    ],
)
def test_edge_handle_slides_to_visible_span_for_each_edge(edge, view_rect, crop_rect):
    overlay = _crop_overlay()
    overlay._view_rect = view_rect
    overlay._crop_rect_norm = crop_rect

    handle = overlay._crop_edge_handle_screen_points()[edge]

    assert (handle.x() if edge in ("top", "bottom") else handle.y()) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("edge", "view_rect", "crop_rect"),
    [
        ("top", QRectF(0, -200, 200, 400), (0.2, 0.1, 0.8, 0.2)),
        ("bottom", QRectF(0, -200, 200, 400), (0.2, 0.1, 0.8, 0.2)),
        ("left", QRectF(-200, 0, 400, 160), (0.1, 0.2, 0.2, 0.8)),
        ("right", QRectF(-200, 0, 400, 160), (0.1, 0.2, 0.2, 0.8)),
    ],
)
def test_edge_handle_is_hidden_when_its_edge_is_outside_the_viewport(edge, view_rect, crop_rect):
    overlay = _crop_overlay()
    overlay._view_rect = view_rect
    overlay._crop_rect_norm = crop_rect

    assert edge not in overlay._crop_edge_handle_screen_points()


def test_edge_handle_relaxes_viewport_gap_when_no_inset_position_is_clear():
    overlay = _crop_overlay()
    overlay._view_rect = QRectF(-100, 0, 200, 160)
    overlay._crop_rect_norm = (0.35, 0.5, 0.65, 0.8)

    handle = overlay._crop_edge_handle_screen_points()["top"]

    assert handle.x() == pytest.approx(0.0)


def test_edge_handle_avoids_rotation_target_nearest_to_midpoint():
    overlay = _crop_overlay()
    overlay._crop_rect_norm = (0.15, 0.0, 0.85, 0.8)
    handles = overlay._crop_edge_handle_screen_points()
    rotation = overlay._crop_rotation_handle_points()["top"]
    top = handles["top"]

    assert top.x() < 100
    assert top.x() == pytest.approx(100 - math.sqrt(21**2 - 13**2) - 1e-3, abs=1e-3)
    assert (top.x() - rotation.x()) ** 2 + (top.y() - rotation.y()) ** 2 > 21**2


def test_right_edge_handle_slides_around_rotation_target_instead_of_hiding():
    overlay = _crop_overlay()
    overlay._crop_rect_norm = (0.15, 0.0, 0.9, 0.8)

    handles = overlay._crop_edge_handle_screen_points()
    assert "right" in handles, (overlay._crop_corner_screen_points(), overlay._crop_rotation_handle_points(), handles)
    right = handles["right"]
    rotation = overlay._crop_rotation_handle_points()["right"]

    assert right.x() == pytest.approx(180)
    assert right.y() != pytest.approx(64)
    assert (right.x() - rotation.x()) ** 2 + (right.y() - rotation.y()) ** 2 > 21**2


def test_top_edge_handle_is_not_hidden_when_viewport_is_taller_than_wide():
    overlay = _crop_overlay()
    overlay.setFixedSize(160, 200)
    overlay._view_rect = QRectF(0, 0, 160, 200)
    overlay._crop_rect_norm = (0.2, 0.9, 0.8, 0.95)

    assert "top" in overlay._crop_edge_handle_screen_points()


@pytest.mark.parametrize(
    ("name", "point", "viewport"),
    [
        ("top", QPointF(100, 0), QRectF(0, 0, 200, 160)),
        ("bottom", QPointF(100, 160), QRectF(0, 0, 200, 160)),
        ("left", QPointF(0, 80), QRectF(0, 0, 200, 160)),
        ("right", QPointF(200, 80), QRectF(0, 0, 200, 160)),
    ],
)
def test_edge_handle_draws_fully_inside_viewport_at_border(name, point, viewport):
    class _Painter:
        rects = []

        def setPen(self, _pen):
            pass

        def setBrush(self, _brush):
            pass

        def drawRect(self, rect):
            self.rects.append(rect)

    painter = _Painter()
    canvas_overlay.draw_resize_handles(painter, {}, {name: point}, viewport)
    rect = painter.rects[0]

    assert viewport.contains(rect)


def test_overlapping_edge_hits_keep_top_first_priority():
    overlay = _crop_overlay()
    overlap = QPointF(100, 80)

    assert overlay._hit_test_crop_edge(overlap, {"top": overlap, "left": overlap}) == "top"


def test_edge_handle_hides_when_corners_leave_no_safe_position():
    overlay = _crop_overlay()
    overlay._crop_rect_norm = (0.2, 0.3, 0.3, 0.7)

    handles = overlay._crop_edge_handle_screen_points()

    assert "top" not in handles
    assert "bottom" not in handles


def test_edge_release_emits_final_commit_and_clears_edge_state():
    overlay = _crop_overlay()
    emitted = []
    overlay.crop_rect_changed.connect(lambda *args: emitted.append(args))
    press = QPointF(40, 80)
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, press))
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(60, 80)))
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(60, 80),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    overlay.mouseReleaseEvent(release)

    assert emitted[-1] == (*overlay._crop_rect_norm, True)
    assert overlay._crop_drag_mode is None
    assert overlay._crop_edge_which is None
