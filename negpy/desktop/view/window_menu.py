"""The macOS Window menu.

Without a menu bar, the key equivalents AppKit takes from the standard Window menu — Cmd+M,
Cmd+W — do nothing. That is why NegPy has a bar at all; this is one of the menus
``mac_menu_bar`` assembles.

These keys stay out of ``shortcut_registry``. They are platform window commands, not NegPy
actions, and a native menu key equivalent silently outranks a QShortcut — so a rebindable
copy could only ever disagree with the menu. ``MENU_KEYS`` is here so a test can prove no
registry default collides with one.

Full screen is not here. AppKit adds its own Enter Full Screen item to any menu titled
View, so an item of ours could only ever be the second one in the menu.

Qt maps the window states onto AppKit correctly, so nothing here needs Objective-C:
``showMaximized`` leaves the NSWindow ``isZoomed``.
"""

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QWidget

# Qt swaps Ctrl and Meta on macOS, so "Ctrl+M" is ⌘M.
MENU_KEYS: dict[str, str] = {
    "minimize": "Ctrl+M",
    "close": "Ctrl+W",
}

_EXCLUDED_WINDOW_TYPES = {
    Qt.WindowType.Popup,
    Qt.WindowType.ToolTip,
    Qt.WindowType.SplashScreen,
    Qt.WindowType.Desktop,
    Qt.WindowType.SubWindow,
}


def listed_windows() -> list[QWidget]:
    """The windows the menu lists, sorted by title so the list does not reorder itself
    between openings. Popups, tooltips and untitled windows are not windows a user can
    choose, so they are left out."""
    windows = [
        widget
        for widget in QApplication.topLevelWidgets()
        if widget.isWindow() and widget.isVisible() and widget.windowTitle() and widget.windowType() not in _EXCLUDED_WINDOW_TYPES
    ]
    return sorted(windows, key=lambda widget: widget.windowTitle())


class WindowMenu(QMenu):
    """Minimize / Zoom / Close / Bring All to Front, then the list of open windows.

    Qt parks a dummy menu on ``NSApp.windowsMenu`` to absorb AppKit's automatic window
    list, so that list cannot be had for free — this rebuilds its own on every opening.
    """

    def __init__(self, window: QMainWindow, parent: Optional[QWidget] = None) -> None:
        super().__init__("Window", parent if parent is not None else window)
        self._window = window
        self._dynamic: list[QAction] = []

        self.act_minimize = self._add("Minimize", MENU_KEYS["minimize"], self._minimize)
        self.act_zoom = self._add("Zoom", "", self._zoom)
        self.addSeparator()
        self.act_close = self._add("Close", MENU_KEYS["close"], self._close)
        self.addSeparator()
        self.act_bring_all = self._add("Bring All to Front", "", self._bring_all_to_front)
        self._list_separator = self.addSeparator()

        self.aboutToShow.connect(self.refresh)

    def _add(self, text: str, key: str, slot) -> QAction:
        action = QAction(text, self)
        # Window titles carry the open file's name, and Qt's macOS merge heuristic moves any
        # item whose text looks like "about", "settings" or "quit" into the application menu.
        # NoRole keeps every item where it was put, whatever a frame is called.
        action.setMenuRole(QAction.MenuRole.NoRole)
        if key:
            action.setShortcut(QKeySequence(key))
        action.triggered.connect(slot)
        self.addAction(action)
        return action

    def target(self) -> QWidget:
        """The window a command acts on: the front one, which may be the live view or a
        floating panel rather than the main window."""
        active = QApplication.activeWindow()
        return active if active is not None else self._window

    def _minimize(self) -> None:
        self.target().showMinimized()

    def _zoom(self) -> None:
        target = self.target()
        target.showNormal() if target.isMaximized() else target.showMaximized()

    def _close(self) -> None:
        self.target().close()

    def _bring_all_to_front(self) -> None:
        for widget in listed_windows():
            widget.raise_()
        self._window.raise_()
        self._window.activateWindow()

    def refresh(self) -> None:
        """Match the menu to the front window. macOS retires Minimize and Zoom in full
        screen, which the front window can enter from the green button or ⌃⌘F."""
        target = self.target()
        full_screen = target.isFullScreen()
        self.act_minimize.setEnabled(not full_screen)
        self.act_zoom.setEnabled(not full_screen)
        self._rebuild_window_list(target)

    def _rebuild_window_list(self, target: QWidget) -> None:
        for action in self._dynamic:
            self.removeAction(action)
        self._dynamic.clear()

        for widget in listed_windows():
            action = QAction(widget.windowTitle(), self)
            action.setMenuRole(QAction.MenuRole.NoRole)
            action.setCheckable(True)
            action.setChecked(widget is target)
            action.triggered.connect(lambda _checked, w=widget: self._raise(w))
            self.addAction(action)
            self._dynamic.append(action)

    @staticmethod
    def _raise(widget: QWidget) -> None:
        if widget.isMinimized():
            widget.showNormal()
        widget.raise_()
        widget.activateWindow()
