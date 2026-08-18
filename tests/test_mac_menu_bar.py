"""The macOS menu bar: built on darwin only, dispatching through the shortcut manager.

The window stub carries exactly the surface MacMenuBar reaches for, so a rename in
MainWindow shows up here rather than at runtime on someone's Mac.
"""

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMainWindow, QMenu

from negpy.desktop.view.mac_menu_bar import MacMenuBar, install_mac_menus
from negpy.desktop.view.shortcut_registry import REGISTRY, default_bindings, set_current_bindings
from negpy.desktop.view.window_menu import MENU_KEYS
from negpy.kernel.system.version import ISSUES_PAGE


class _FakeShortcuts:
    """Returns a handler only for real registry ids, so a typo in a menu item's action id
    raises while the bar is built."""

    def __init__(self):
        self.ran: list[str] = []

    def action_for(self, action_id: str):
        if action_id not in REGISTRY:
            return None
        return lambda: self.ran.append(action_id)

    def open_editor(self, parent=None) -> None:
        self.ran.append("open_editor")


@pytest.fixture
def window(qapp):
    win = QMainWindow()
    win.setWindowTitle("NegPy")
    win.shortcut_manager = _FakeShortcuts()
    win.show_tutorial = MagicMock()

    set_current_bindings(default_bindings())
    yield win
    win.close()
    win.deleteLater()


@pytest.fixture
def bar(window):
    return MacMenuBar(window)


def _items(menu: QMenu) -> list[QAction]:
    return [a for a in menu.actions() if not a.isSeparator()]


def _by_text(bar: MacMenuBar, text: str) -> QAction:
    for action in _items(bar.help_menu):
        if action.text() == text:
            return action
    raise AssertionError(f"no menu item {text!r}")


def test_not_installed_off_macos(window):
    with patch("negpy.desktop.view.mac_menu_bar.sys.platform", "win32"):
        assert install_mac_menus(window) is None


def test_menus_are_in_apple_order(window):
    with patch("negpy.desktop.view.mac_menu_bar.sys.platform", "darwin"):
        bar = install_mac_menus(window)

    assert [action.menu().title() for action in bar.actions()] == ["Window", "Help"]


def test_bar_has_no_parent(bar, window):
    # A bar owned by the main window is only synced while that window is active, so the menus
    # dropped out of the bar whenever the live view or a floating panel took focus.
    assert bar.parent() is None
    assert window.findChild(MacMenuBar) is None


def test_every_item_dispatches_through_the_shortcut_manager(bar, window):
    # Ids are validated while the bar is built (a fake handler is only returned for real
    # registry ids), so reaching here at all proves every id exists.
    assert {action_id for _, action_id in bar._keyed} <= set(REGISTRY)

    _by_text(bar, "Keyboard Shortcuts").trigger()
    assert window.shortcut_manager.ran == ["show_shortcuts"]


def test_non_registry_items_run_their_own_slot(bar, window):
    _by_text(bar, "Take the Tour").trigger()
    window.show_tutorial.assert_called_once()

    _by_text(bar, "Customize Shortcuts…").trigger()
    assert "open_editor" in window.shortcut_manager.ran


def test_only_command_combinations_become_key_equivalents(bar):
    # AppKit fires a menu key equivalent before Qt sees the event, so a bare "?" would trip
    # while typing into the search box and an Option combination would eat its character.
    assert _by_text(bar, "Keyboard Shortcuts").shortcut().isEmpty()  # bound to "?"
    assert _by_text(bar, "Analysis Panel Guide").shortcut().isEmpty()  # unbound
    assert bar.window_menu.act_minimize.shortcut() == QKeySequence(MENU_KEYS["minimize"])


def test_menu_keys_do_not_collide_with_the_registry(bar):
    # A native menu key equivalent outranks a QShortcut without firing activatedAmbiguously,
    # so a collision would kill a binding silently.
    reserved = list(MENU_KEYS.values())
    assert len({QKeySequence(k).toString() for k in reserved}) == len(reserved)

    taken = {QKeySequence(k) for k in reserved}
    clashes = [action_id for action_id, entry in REGISTRY.items() if entry.default_key and QKeySequence(entry.default_key) in taken]
    assert clashes == []


def test_a_rebind_reaches_the_menu(bar):
    set_current_bindings({**default_bindings(), "show_shortcuts": "Ctrl+Shift+K", "show_analysis_help": "F8"})
    bar.sync_shortcuts()

    assert _by_text(bar, "Keyboard Shortcuts").shortcut() == QKeySequence("Ctrl+Shift+K")
    # Rebound to a key the menu must not claim, so the item gives its key equivalent up.
    assert _by_text(bar, "Analysis Panel Guide").shortcut().isEmpty()


def test_no_view_menu_of_our_own(bar):
    # AppKit adds an Enter Full Screen item to any menu titled View, so a View menu of ours
    # holding the same item showed it twice.
    assert "View" not in [action.menu().title() for action in bar.actions()]


def test_report_an_issue_opens_the_tracker(bar):
    opened = []
    with patch("negpy.desktop.view.mac_menu_bar.QDesktopServices.openUrl", side_effect=lambda url: opened.append(url.toString())):
        _by_text(bar, "Report an Issue…").trigger()

    assert opened == [ISSUES_PAGE]


def test_nothing_is_merged_into_the_application_menu(bar):
    # Qt's macOS heuristic moves "about"/"settings"/"quit"-looking items into the app menu.
    menus = [bar.window_menu, bar.help_menu]
    roles = {a.menuRole() for menu in menus for a in _items(menu)}

    assert roles == {QAction.MenuRole.NoRole}
