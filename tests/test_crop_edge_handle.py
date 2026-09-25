from dataclasses import replace

import pytest
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QWidget

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay

_DISPLAY = (200, 160)
_INITIAL_RECT = (0.2, 0.2, 0.8, 0.8)


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


def _crop_overlay() -> CanvasOverlay:
    parent = QWidget()
    parent._is_panning = False
    overlay = CanvasOverlay(AppState(), parent)
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

    assert overlay._crop_edge_midpoint_screen_points() is None
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, edge_midpoint))

    assert overlay._crop_drag_mode == "move"
    assert overlay._crop_drag_mode != "edge"


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