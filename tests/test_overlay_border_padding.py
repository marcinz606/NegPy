"""Frame overlays measure the picture, not the border padding around it."""

from unittest.mock import MagicMock

from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay

_app = QApplication.instance() or QApplication([])


def _bordered_overlay() -> CanvasOverlay:
    overlay = CanvasOverlay(AppState())
    overlay._view_rect = QRectF(0, 0, 200, 100)
    overlay._current_size = (200, 100)
    overlay._content_rect = (20, 10, 160, 80)  # 20 px / 10 px of border each side
    return overlay


def _rects(painter: MagicMock) -> list[QRectF]:
    return [c.args[0] for c in painter.drawRect.call_args_list if c.args and isinstance(c.args[0], QRectF)]


def test_the_analysis_buffer_margin_sits_inside_the_picture() -> None:
    overlay = _bordered_overlay()
    overlay._buffer_overlay_visible = True
    overlay._buffer_overlay_ratio = 0.1
    painter = MagicMock()
    overlay._draw_ui(painter)
    inner = _rects(painter)[-1]
    assert (inner.x(), inner.y(), inner.width(), inner.height()) == (36.0, 18.0, 128.0, 64.0)


def test_the_crop_to_valid_wedge_grows_out_of_the_picture() -> None:
    overlay = _bordered_overlay()
    overlay._crop_preview_rect = (0.1, 0.1, 0.9, 0.9)
    overlay._crop_preview_visible = True
    painter = MagicMock()
    overlay._draw_ui(painter)
    top = _rects(painter)[0]
    assert top.bottom() == 10.0  # the wedge ends at the picture's top edge, not the border's


def test_a_zoom_mid_drag_keeps_the_exclusion_stroke_and_the_rotate_handle_on_the_image() -> None:
    from PyQt6.QtCore import QPointF

    overlay = CanvasOverlay(AppState())
    old = QRectF(0, 0, 200, 100)
    overlay._view_rect = QRectF(-100, -50, 400, 200)  # zoomed 2x about the center
    overlay._exclude_drag_pts = [QPointF(50, 25)]
    overlay._rotate_center = QPointF(100, 50)
    overlay._rotate_press = QPointF(150, 50)

    overlay._remap_inflight_points(old)

    assert overlay._exclude_drag_pts == [QPointF(0, 0)]
    assert overlay._rotate_center == QPointF(100, 50)
    assert overlay._rotate_press == QPointF(200, 50)
