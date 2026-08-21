"""Keep a modeless window above the main window."""

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget


def float_over_app(widget: QWidget, platform: str | None = None) -> None:
    """Make a modeless window a macOS utility panel, so the main window cannot bury it.

    macOS gives a non-modal dialog no ordering over its parent, so this raises it to an
    NSPanel. Windows and Linux already keep an owned window above its owner, and the panel
    chrome there costs a taskbar button, so they are left alone.

    Call before the first show(): setting flags on a visible window hides it.
    """
    if (sys.platform if platform is None else platform) != "darwin":
        return
    widget.setWindowFlags(widget.windowFlags() | Qt.WindowType.Tool)
    # A panel hides while another app is frontmost. An export outlives the app being
    # frontmost, so the window has to stay put.
    widget.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
