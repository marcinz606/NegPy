import sys

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow

from negpy.desktop.view.widgets.tutorial_overlay import TutorialOverlay, TutorialStep


def _steps() -> list[TutorialStep]:
    return [
        TutorialStep("Welcome", "First step", lambda _: None),
        TutorialStep("Finish", "Second step", lambda _: None),
    ]


def test_tutorial_overlay_next_button_is_clickable() -> None:
    win = QMainWindow()
    win.setGeometry(100, 100, 900, 700)
    win.show()

    overlay = TutorialOverlay(win)
    finished: list[bool] = []
    overlay.finished.connect(finished.append)

    overlay.start(_steps())
    QApplication.processEvents()

    QTest.mouseClick(overlay._next_btn, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert overlay._idx == 1

    QTest.mouseClick(overlay._next_btn, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert finished == [True]
    assert not overlay.isVisible()


def test_tutorial_overlay_uses_top_level_window_on_windows() -> None:
    win = QMainWindow()
    overlay = TutorialOverlay(win)

    if sys.platform == "win32":
        assert overlay.isWindow()
        assert overlay.windowFlags() & Qt.WindowType.Tool
        assert overlay.windowFlags() & Qt.WindowType.FramelessWindowHint
        assert overlay.windowModality() == Qt.WindowModality.WindowModal
    else:
        assert not overlay.isWindow()
