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
    assert texts[:5] == ["Minimize", "Zoom", "Enter Full Screen", "Close", "Bring All to Front"]
    assert menu.act_minimize.shortcut() == QKeySequence(MENU_KEYS["minimize"])
    assert menu.act_full_screen.shortcut() == QKeySequence(MENU_KEYS["full_screen"])
    assert menu.act_close.shortcut() == QKeySequence(MENU_KEYS["close"])
    assert {"minimize", "full_screen", "close", "hide", "hide_others", "quit"} <= set(MENU_KEYS)


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
    assert menu.act_full_screen.text() == "Exit Full Screen"

    window.showNormal()
    menu.refresh()
    assert not window.isFullScreen()
    assert menu.act_minimize.isEnabled()
    assert menu.act_zoom.isEnabled()
    assert menu.act_full_screen.text() == "Enter Full Screen"


def test_full_screen_toggles_main_window_and_preserves_maximized_state(window, monkeypatch):
    window.show()
    window.showMaximized()
    menu = WindowMenu(window)

    # Even when a floating utility window has focus, ⌃⌘F toggles the main window.
    panel = QWidget()
    panel.setWindowTitle("Controls")
    panel.show()
    monkeypatch.setattr("negpy.desktop.view.window_menu.QApplication.activeWindow", lambda: panel)
    try:
        menu.act_full_screen.trigger()
        assert window.isFullScreen()
        assert not panel.isFullScreen()

        menu.act_full_screen.trigger()
        assert not window.isFullScreen()
        assert window.isMaximized()
    finally:
        panel.close()


def test_full_screen_ignores_trigger_while_modal_dialog_is_open(window, monkeypatch):
    window.show()
    menu = WindowMenu(window)
    modal = QWidget()
    monkeypatch.setattr("negpy.desktop.view.window_menu.QApplication.activeModalWidget", lambda: modal)
    monkeypatch.setattr("negpy.desktop.view.window_menu.QApplication.activeWindow", lambda: modal)

    menu.refresh()
    assert not menu.act_minimize.isEnabled()
    assert not menu.act_zoom.isEnabled()
    assert not menu.act_full_screen.isEnabled()
    menu.act_full_screen.trigger()
    menu.act_minimize.trigger()
    menu.act_zoom.trigger()
    assert not window.isFullScreen()
    assert not modal.isMaximized()
    assert not modal.isMinimized()

    # After the dropdown closes, shortcuts re-enable so key equivalents do not stay stuck off.
    menu._restore_shortcut_enabled()
    assert menu.act_full_screen.isEnabled()
    assert menu.act_minimize.isEnabled()


def test_minimize_and_zoom_disable_when_window_is_minimized(window, monkeypatch):
    window.show()
    window.showMinimized()
    monkeypatch.setattr("negpy.desktop.view.window_menu.QApplication.activeWindow", lambda: None)
    menu = WindowMenu(window)

    menu.refresh()
    assert not menu.act_minimize.isEnabled()
    assert not menu.act_zoom.isEnabled()
    assert not menu.act_full_screen.isEnabled()


def test_minimize_ignores_trigger_while_in_full_screen(window):
    window.show()
    menu = WindowMenu(window)
    window.showFullScreen()
    assert window.isFullScreen()

    # Key equivalent invoked directly without opening Window menu (act_minimize still enabled):
    menu.act_minimize.trigger()
    assert not window.isMinimized()
    assert window.isFullScreen()


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
