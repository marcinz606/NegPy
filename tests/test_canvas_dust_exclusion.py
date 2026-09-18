from dataclasses import replace

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def _right_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.RightButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.RightButton, buttons, Qt.KeyboardModifier.NoModifier)


def _overlay(dust_remove: bool) -> CanvasOverlay:
    state = AppState()
    state.config = replace(state.config, retouch=replace(state.config.retouch, dust_remove=dust_remove))
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, 100, 100)
    return overlay


def test_right_drag_paints_an_exclusion() -> None:
    overlay = _overlay(dust_remove=True)
    emitted = []
    overlay.dust_exclusion_painted.connect(emitted.append)

    overlay.mousePressEvent(_right_event(QEvent.Type.MouseButtonPress, QPointF(20, 20)))
    for x in (40, 60, 80):
        overlay.mouseMoveEvent(_right_event(QEvent.Type.MouseMove, QPointF(x, 20)))
    overlay.mouseReleaseEvent(_right_event(QEvent.Type.MouseButtonRelease, QPointF(80, 20), Qt.MouseButton.NoButton))

    assert len(emitted) == 1
    assert len(emitted[0]) > 1, "the drag path, not a single point"
    assert overlay._exclude_drag_pts == []


def test_right_press_is_inert_while_optical_removal_is_off() -> None:
    overlay = _overlay(dust_remove=False)
    overlay.mousePressEvent(_right_event(QEvent.Type.MouseButtonPress, QPointF(20, 20)))
    # Nothing armed, so the press keeps falling through to the context menu.
    assert overlay._exclude_drag_pts == []
