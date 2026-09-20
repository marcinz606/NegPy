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


def test_right_click_excludes_when_the_toggle_is_on() -> None:
    overlay = _overlay(dust_remove=True)
    overlay.state.right_click_excludes = True
    emitted = []
    overlay.dust_exclusion_painted.connect(emitted.append)

    overlay.mousePressEvent(_right_event(QEvent.Type.MouseButtonPress, QPointF(30, 30)))
    overlay.mouseReleaseEvent(_right_event(QEvent.Type.MouseButtonRelease, QPointF(30, 30), Qt.MouseButton.NoButton))

    # No parent here, so reaching for the canvas menu would raise instead of excluding.
    assert len(emitted) == 1
    assert len(emitted[0]) == 1, "a click excludes one spot"


def test_right_click_keeps_the_heal_menu_while_a_tool_is_live() -> None:
    from negpy.desktop.session import ToolMode

    overlay = _overlay(dust_remove=True)
    overlay.state.right_click_excludes = True
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    emitted = []
    overlay.dust_exclusion_painted.connect(emitted.append)

    overlay.mousePressEvent(_right_event(QEvent.Type.MouseButtonPress, QPointF(30, 30)))
    try:
        overlay.mouseReleaseEvent(_right_event(QEvent.Type.MouseButtonRelease, QPointF(30, 30), Qt.MouseButton.NoButton))
    except AttributeError:
        pass  # parentless overlay: the menu call is the behaviour under test
    assert emitted == [], "the heal tool keeps right-click for its own menu"


def test_right_press_is_inert_while_optical_removal_is_off() -> None:
    overlay = _overlay(dust_remove=False)
    overlay.mousePressEvent(_right_event(QEvent.Type.MouseButtonPress, QPointF(20, 20)))
    # Nothing armed, so the press keeps falling through to the context menu.
    assert overlay._exclude_drag_pts == []


def test_the_amber_band_covers_what_the_mask_releases() -> None:
    """The overlay redraws the band in Qt while the mask rasterizes it in OpenCV. The band
    is the cut now, so a drift between the two would show the user film it did not release."""
    import numpy as np
    from PyQt6.QtGui import QImage, QPainter

    from negpy.features.retouch.logic import exclusion_cover
    from negpy.features.retouch.models import HEAL_SIZE_REF

    side, diameter = 200, 24.0
    points = [[0.25, 0.4], [0.5, 0.6], [0.75, 0.4]]
    size = diameter * HEAL_SIZE_REF / side
    cover = exclusion_cover([(points, size)], (side, side))

    overlay = _overlay(dust_remove=True)
    overlay._view_rect = QRectF(0, 0, side, side)
    image = QImage(side, side, QImage.Format.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(Qt.GlobalColor.white)
    overlay._fill_brush_band(painter, [QPointF(x * side, y * side) for x, y in points], diameter / 2.0)
    painter.end()

    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    drawn = np.frombuffer(ptr, dtype=np.uint8).reshape(side, image.bytesPerLine())[:, :side] > 0

    painted, released = drawn.sum(), cover.astype(bool).sum()
    assert painted and released, "both routes drew a band"
    overlap = (drawn & cover.astype(bool)).sum()
    assert overlap / max(painted, released) > 0.9, "the drawn band and the released one are the same band"
