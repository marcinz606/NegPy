from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QImage, QPainter

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay, grid_interior_fractions


def test_rotation_grid_toggles_with_timer() -> None:
    overlay = CanvasOverlay(AppState())
    assert overlay._rotation_grid_visible is False

    overlay.show_rotation_grid()
    assert overlay._rotation_grid_visible is True
    assert overlay._rotation_grid_timer.isActive()

    overlay._hide_rotation_grid()
    assert overlay._rotation_grid_visible is False


def test_grid_interior_fractions() -> None:
    assert grid_interior_fractions(3) == [1 / 3, 2 / 3]
    assert len(grid_interior_fractions(10)) == 9


def test_draw_rotation_grid_paints_without_error() -> None:
    overlay = CanvasOverlay(AppState())
    img = QImage(64, 64, QImage.Format.Format_ARGB32)
    painter = QPainter(img)
    try:
        overlay._draw_rotation_grid(painter, QRectF(0, 0, 64, 64))
    finally:
        painter.end()
