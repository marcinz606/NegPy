"""Keep a modeless window above the main window."""

import sys

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication, QWidget

_PANEL_PROPERTY = "negpy_floating_panel"
_key_guard: "_PanelKeyGuard | None" = None


def float_over_app(widget: QWidget, platform: str | None = None, *, claim_keys: bool = True) -> None:
    """Make a modeless window a macOS utility panel, so the main window cannot bury it.

    macOS gives a non-modal dialog no ordering over its parent, so this raises it to an
    NSPanel. Windows and Linux already keep an owned window above its owner, and the panel
    chrome there costs a taskbar button, so they are left alone.

    claim_keys=False leaves the main window's shortcuts live while the panel has focus, for a
    panel with no control that takes a key. Call before the first show(): setting flags on a
    visible window hides it.
    """
    if (sys.platform if platform is None else platform) != "darwin":
        return
    widget.setWindowFlags(widget.windowFlags() | Qt.WindowType.Tool)
    # A panel hides while another app is frontmost. An export outlives the app being
    # frontmost, so the window has to stay put.
    widget.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
    if claim_keys:
        widget.setProperty(_PANEL_PROPERTY, True)
        _install_key_guard()


class _PanelKeyGuard(QObject):
    """Keeps the main window's shortcuts out of a focused panel.

    Qt matches a Tool window's keys against its parent window's shortcuts as well, so Esc,
    the arrows and bare letters would run main-window actions instead of reaching the panel.
    Accepting the ShortcutOverride sends every key to the focused control as a key press, so
    a panel handles its own keys in keyPressEvent; a QShortcut on a panel never fires. A
    Ctrl or Cmd chord passes, so the menu bar and the app-wide commands keep their keys.
    """

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() != QEvent.Type.ShortcutOverride or not isinstance(obj, QWidget):
            return False
        if not obj.window().property(_PANEL_PROPERTY):
            return False
        if ev.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            return False
        ev.accept()
        return True


def _install_key_guard() -> None:
    global _key_guard
    app = QApplication.instance()
    if _key_guard is None and app is not None:
        _key_guard = _PanelKeyGuard()
        app.installEventFilter(_key_guard)
