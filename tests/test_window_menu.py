"""The macOS Window menu: the window commands, and never merged into the app menu."""

import pytest
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMainWindow, QWidget

from negpy.desktop.view.window_menu import MENU_KEYS, WindowMenu, listed_windows


@pytest.fixture
def window(qapp):
    win = QMainWindow()
    win.setWindowTitle("NegPy")
    yield win
    win.close()
    win.deleteLater()


def test_standard_items_and_keys(window):
    menu = WindowMenu(window)

    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts[:4] == ["Minimize", "Zoom", "Close", "Bring All to Front"]
    assert menu.act_minimize.shortcut() == QKeySequence(MENU_KEYS["minimize"])
    assert menu.act_close.shortcut() == QKeySequence(MENU_KEYS["close"])


def test_every_item_keeps_its_place_in_the_menu_bar(window):
    # Qt's macOS heuristic moves "about"/"settings"/"quit"-looking items into the app menu,
    # and window titles carry the open file's name.
    window.setWindowTitle("NegPy — about_settings_quit.tif")
    menu = WindowMenu(window)
    menu.refresh()

    roles = {a.menuRole() for a in menu.actions() if not a.isSeparator()}
    assert roles == {QAction.MenuRole.NoRole}


def test_zoom_toggles_maximized(window):
    window.show()
    menu = WindowMenu(window)

    menu.act_zoom.trigger()
    assert window.isMaximized()

    menu.act_zoom.trigger()
    assert not window.isMaximized()


def test_minimize_and_zoom_retire_in_full_screen(window):
    window.show()
    menu = WindowMenu(window)

    window.showFullScreen()
    menu.refresh()
    assert window.isFullScreen()
    assert not menu.act_minimize.isEnabled()
    assert not menu.act_zoom.isEnabled()

    window.showNormal()
    menu.refresh()
    assert not window.isFullScreen()
    assert menu.act_minimize.isEnabled()
    assert menu.act_zoom.isEnabled()


def test_window_list_names_visible_titled_windows(window, qapp):
    window.show()
    other = QWidget()
    other.setWindowTitle("Live View")
    other.show()
    untitled = QWidget()
    untitled.show()
    qapp.processEvents()

    try:
        menu = WindowMenu(window)
        menu.refresh()
        listed = [a.text() for a in menu.actions() if a.isCheckable()]

        assert "Live View" in listed
        assert "NegPy" in listed
        assert "" not in listed
        assert listed == sorted(listed)
        assert all(w.windowTitle() for w in listed_windows())
    finally:
        other.close()
        untitled.close()


def test_window_list_is_rebuilt_not_appended(window):
    window.show()
    menu = WindowMenu(window)

    menu.refresh()
    first = len([a for a in menu.actions() if a.isCheckable()])
    menu.refresh()
    menu.refresh()

    assert len([a for a in menu.actions() if a.isCheckable()]) == first
