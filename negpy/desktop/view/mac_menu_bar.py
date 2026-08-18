"""The macOS menu bar: Window and Help.

Only macOS gets one. Qt hands it to the global bar there, at no cost in window space;
Windows and Linux draw a QMenuBar inside the window, and NegPy has no use for a strip of
chrome above the canvas. Nothing here is new functionality — every item runs something a
button, a window control or a keyboard shortcut already runs.

This is the first stage: the standard macOS window commands, which do nothing in an app
with no menu bar, and Help. The File and View menus come later.

There is no View menu, and so no Enter Full Screen item. AppKit adds one of its own to any
menu titled View, and ours only ever appeared under it as the duplicate.

**Key equivalents are only ever ⌘ combinations.** AppKit dispatches a menu key equivalent
before Qt's shortcut machinery sees the event, so a bare-key equivalent ("?" for Keyboard
Shortcuts) would fire while the user types into the film-strip search box, and an Option
combination would eat the character it composes. Items bound to those keys are listed
without a key; their QShortcut still works. ``_menu_key`` is the one place that decides.

Items dispatch through ``ShortcutManager.action_for``, so a menu item and its shortcut can
never run different code, and ``sync_shortcuts`` re-reads the bindings after a rebind.
"""

import sys
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QKeySequence
from PyQt6.QtWidgets import QMenu, QMenuBar

from negpy.desktop.view.shortcut_registry import key_for
from negpy.desktop.view.window_menu import WindowMenu
from negpy.kernel.system.version import ISSUES_PAGE

if TYPE_CHECKING:
    from negpy.desktop.view.main_window import MainWindow


def _menu_key(key: str) -> QKeySequence:
    """A menu key equivalent for ``key``, or an empty one. ⌘ combinations only — see the
    module docstring."""
    return QKeySequence(key) if key and "Ctrl+" in key else QKeySequence()


class MacMenuBar(QMenuBar):
    """The application-wide menu bar.

    It has no parent, which is what makes it application-wide on macOS. A bar owned by the
    main window is only shown while that window is in front — Qt syncs the active window's
    bar, and the live view, the calibration window and a floating panel have none, so the
    menus blinked out of the bar whenever one of them took focus.
    """

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(None)
        self._window = window
        self._keyed: list[tuple[QAction, str]] = []

        # Apple's order: the app's own menus, then Window, Help.
        self.window_menu = WindowMenu(window, self)
        self.addMenu(self.window_menu)
        self.help_menu = self._build_help()

        self.sync_shortcuts()

    # -- construction ---------------------------------------------------------------

    def _menu(self, title: str) -> QMenu:
        menu = QMenu(title, self)
        self.addMenu(menu)
        return menu

    def _add(
        self,
        menu: QMenu,
        text: str,
        *,
        action_id: str = "",
        slot: Optional[Callable[[], None]] = None,
        key: str = "",
    ) -> QAction:
        if slot is None:
            slot = self._window.shortcut_manager.action_for(action_id)
            if slot is None:
                raise KeyError(f"No shortcut action for menu item {text!r} ({action_id!r})")

        action = QAction(text, self)
        # Window titles carry the open file's name, and Qt's macOS merge heuristic moves any
        # item whose text looks like "about", "settings" or "quit" into the application menu.
        # NoRole keeps every item where it was put, whatever a frame is called.
        action.setMenuRole(QAction.MenuRole.NoRole)
        action.triggered.connect(lambda _checked=False, run=slot: run())
        if action_id:
            self._keyed.append((action, action_id))
        elif key:
            action.setShortcut(_menu_key(key))
        menu.addAction(action)
        return action

    def _build_help(self) -> QMenu:
        menu = self._menu("Help")
        self._add(menu, "Take the Tour", slot=self._window.show_tutorial)
        self._add(menu, "Keyboard Shortcuts", action_id="show_shortcuts")
        self._add(menu, "Customize Shortcuts…", slot=self._open_shortcut_editor)
        self._add(menu, "Analysis Panel Guide", action_id="show_analysis_help")
        menu.addSeparator()
        # No registry entry: the item exists only in this bar, and a binding would advertise
        # it in the shortcut editor and the ? overlay on Windows and Linux, where nothing
        # can run it.
        self._add(menu, "Report an Issue…", slot=self._report_an_issue)
        self._add(menu, "Check for Updates…", action_id="check_for_updates")
        return menu

    # -- state ----------------------------------------------------------------------

    def sync_shortcuts(self) -> None:
        """Re-read every menu item's key from the live bindings. Called after a rebind, or
        the retired key would keep working from the menu."""
        for action, action_id in self._keyed:
            action.setShortcut(_menu_key(key_for(action_id)))

    # -- slots ----------------------------------------------------------------------

    def _open_shortcut_editor(self) -> None:
        self._window.shortcut_manager.open_editor(self._window)

    @staticmethod
    def _report_an_issue() -> None:
        QDesktopServices.openUrl(QUrl(ISSUES_PAGE))


def install_mac_menus(window: "MainWindow") -> Optional[MacMenuBar]:
    """Give ``window`` the macOS menu bar. Returns None on every other platform, which keeps
    its menu-bar-free layout. Call after the shortcut manager exists: the items dispatch
    through it.
    """
    if sys.platform != "darwin":
        return None
    return MacMenuBar(window)
