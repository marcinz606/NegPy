import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QPushButton

from negpy.desktop.main import _AppStyle


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_buttons_take_tab_focus_only(qapp):
    """Click focus would leave the sheet's focus border on the last button pressed."""
    previous = qapp.style().objectName() or "Fusion"
    qapp.setStyle(_AppStyle("Fusion"))
    try:
        assert QPushButton().focusPolicy() == Qt.FocusPolicy.TabFocus
    finally:
        qapp.setStyle(previous)
