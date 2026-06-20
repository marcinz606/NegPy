from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def test_rotation_grid_toggles_with_timer() -> None:
    overlay = CanvasOverlay(AppState())
    assert overlay._rotation_grid_visible is False

    overlay.show_rotation_grid()
    assert overlay._rotation_grid_visible is True
    assert overlay._rotation_grid_timer.isActive()

    overlay._hide_rotation_grid()
    assert overlay._rotation_grid_visible is False
