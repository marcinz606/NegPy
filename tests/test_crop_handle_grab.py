"""A crop handle must start a drag wherever its grab radius reaches.

The handles sit on the crop rect, which can lie on — or, with a print border,
outside — the image content. The hover cursor hit-tests them geometrically, so
it offers a resize the press then has to honour.
"""

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QWidget

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay

_DISPLAY = (200, 160)


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


def _crop_overlay(content_rect=None) -> CanvasOverlay:
    parent = QWidget()
    parent._is_panning = False
    overlay = CanvasOverlay(AppState(), parent)
    overlay._test_parent = parent
    overlay._view_rect = QRectF(0, 0, *_DISPLAY)
    overlay._current_size = _DISPLAY
    overlay._content_rect = content_rect
    overlay.set_tool_mode(ToolMode.CROP_MANUAL)
    # Full-frame crop: every handle sits on the image boundary.
    overlay._crop_rect_norm = (0.0, 0.0, 1.0, 1.0)
    return overlay


def test_handle_on_the_frame_edge_starts_a_drag():
    """The grab radius straddles the boundary, so half of it lands outside the
    content rect — the press there used to be dropped without a word."""
    overlay = _crop_overlay()

    # 4 px outside the top-left corner: inside the 10 px grab radius.
    press = QPointF(-4, -4)
    overlay._update_crop_hover_cursor(press)
    offered_resize = overlay.cursor().shape() in (Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeBDiagCursor)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, press))

    assert offered_resize, "hover did not offer a resize; the premise of this test is gone"
    assert overlay._crop_drag_mode == "corner"
    assert overlay._crop_anchor_screen is not None


def test_the_drag_then_actually_moves_the_rect():
    overlay = _crop_overlay()
    emitted: list = []
    overlay.crop_rect_changed.connect(lambda *a: emitted.append(a))

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(-4, -4)))
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(40, 32)))

    assert emitted, "corner drag emitted no rect change"
    assert emitted[-1][:4] != (0.0, 0.0, 1.0, 1.0)


def test_handle_outside_a_bordered_content_rect_starts_a_drag():
    """With a print border the content rect is inset, so a full-frame crop's
    handles sit well outside it — not merely on its edge."""
    overlay = _crop_overlay(content_rect=(20, 16, 160, 128))

    press = QPointF(0, 0)  # the crop rect's own top-left, 20 px outside the content
    overlay._update_crop_hover_cursor(press)
    offered_resize = overlay.cursor().shape() in (Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeBDiagCursor)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, press))

    assert offered_resize
    assert overlay._crop_drag_mode == "corner"


def test_interior_press_still_moves_the_box():
    overlay = _crop_overlay()

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(100, 80)))

    assert overlay._crop_drag_mode == "move"


def test_press_outside_the_view_does_not_start_a_crop_drag():
    """Past the widget entirely — no handle in reach, nothing to start."""
    overlay = _crop_overlay()

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(-80, -80)))

    assert overlay._crop_drag_mode is None


def test_clicked_still_only_fires_over_the_image():
    """The picker signal stays gated on real image coords; only the crop drag
    is allowed to start from outside."""
    overlay = _crop_overlay(content_rect=(20, 16, 160, 128))
    clicks: list = []
    overlay.clicked.connect(lambda x, y: clicks.append((x, y)))

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(0, 0)))

    assert clicks == []
