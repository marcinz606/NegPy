"""Canvas side of zone placement: a placed pin is a handle — grab it, drag it, and
its caption reads the tone under it while it moves."""

from typing import Optional

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QWidget

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay, zone_pin_caption
from negpy.features.exposure.placement import ZonePin


class _CanvasStub(QWidget):
    _is_panning = False


def _event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


def _press(pos: QPointF) -> QMouseEvent:
    return _event(QEvent.Type.MouseButtonPress, pos)


def _move(pos: QPointF) -> QMouseEvent:
    return _event(QEvent.Type.MouseMove, pos)


def _release(pos: QPointF) -> QMouseEvent:
    return _event(QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.NoButton)


def _pin(nx: float, ny: float, label: str = "V", target: float = 5.0) -> ZonePin:
    return ZonePin(nx=nx, ny=ny, val_rgb=(0.5, 0.5, 0.5), val_luma=0.5, target_zone=target, label=label)


def _overlay(pins: Optional[list] = None) -> CanvasOverlay:
    state = AppState()
    state.zone_pins = list(pins or [])
    canvas = _CanvasStub()
    overlay = CanvasOverlay(state, canvas)
    overlay._canvas_ref = canvas  # Qt deletes the overlay with its parent; keep it alive

    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay._current_size = (100, 100)
    overlay.set_tool_mode(ToolMode.ZONE_PLACE)
    return overlay


def test_pressing_a_pin_grabs_it_instead_of_placing_another() -> None:
    overlay = _overlay([_pin(0.5, 0.5)])
    placed: list = []
    overlay.clicked.connect(lambda nx, ny: placed.append((nx, ny)))

    overlay.mousePressEvent(_press(QPointF(52, 53)))

    assert placed == []
    assert overlay._pin_drag_index == 0


def test_pressing_away_from_a_pin_still_places_one() -> None:
    overlay = _overlay([_pin(0.5, 0.5)])
    placed: list = []
    overlay.clicked.connect(lambda nx, ny: placed.append((nx, ny)))

    overlay.mousePressEvent(_press(QPointF(10, 10)))

    assert placed == [(0.1, 0.1)]
    assert overlay._pin_drag_index is None


def test_dragging_reports_live_moves_and_a_final_one_on_release() -> None:
    overlay = _overlay([_pin(0.2, 0.2), _pin(0.5, 0.5)])
    moves: list = []
    overlay.zone_pin_moved.connect(lambda i, nx, ny, final: moves.append((i, round(nx, 3), round(ny, 3), final)))

    overlay.mousePressEvent(_press(QPointF(50, 50)))
    overlay.mouseMoveEvent(_move(QPointF(70, 30)))
    overlay.mouseReleaseEvent(_release(QPointF(80, 20)))

    assert moves == [(1, 0.7, 0.3, False), (1, 0.8, 0.2, True)]
    assert overlay._pin_drag_index is None


def test_a_drag_off_the_frame_clamps_to_the_content() -> None:
    overlay = _overlay([_pin(0.5, 0.5)])
    moves: list = []
    overlay.zone_pin_moved.connect(lambda i, nx, ny, final: moves.append((nx, ny)))

    overlay.mousePressEvent(_press(QPointF(50, 50)))
    overlay.mouseMoveEvent(_move(QPointF(400, -80)))

    assert moves == [(1.0, 0.0)]


def test_pins_are_only_grabbable_with_the_tool_up() -> None:
    overlay = _overlay([_pin(0.5, 0.5)])
    overlay.set_tool_mode(ToolMode.NONE)

    overlay.mousePressEvent(_press(QPointF(50, 50)))

    assert overlay._pin_drag_index is None


def test_putting_the_tool_down_mid_drag_ends_it() -> None:
    overlay = _overlay([_pin(0.5, 0.5)])
    overlay.mousePressEvent(_press(QPointF(50, 50)))

    overlay.set_tool_mode(ToolMode.NONE)

    assert overlay._pin_drag_index is None


def test_the_caption_shows_the_target_only_while_it_differs() -> None:
    assert zone_pin_caption(0, _pin(0.5, 0.5, label="IV⅓", target=6.0)) == "1 · IV⅓ → VI"
    assert zone_pin_caption(1, _pin(0.5, 0.5, label="V", target=5.0)) == "2 · V"
