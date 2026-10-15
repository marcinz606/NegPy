from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay

_app = QApplication.instance() or QApplication([])


def test_leaving_compare_drops_the_divider_cursor() -> None:
    overlay = CanvasOverlay(AppState())
    overlay.setCursor(Qt.CursorShape.SplitHCursor)
    overlay.state.compare_mode = False

    overlay.refresh_compare()

    assert overlay.cursor().shape() != Qt.CursorShape.SplitHCursor
