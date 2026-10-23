"""The macOS window background in the theme's base color. At a fractional device pixel ratio a partial
repaint leaves device pixels unpainted, and they show the NSWindow background as hairlines.
An alternate color theme must update this fix: the color is ``THEME.bg_dark``, set once per native window."""

import ctypes
import ctypes.util
import sys
from functools import cache
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.theme import THEME

# The native window id the background was set for; Qt makes a new one when it recreates the window.
_PAINTED_FOR = "_negpy_ns_background"


@cache
def _objc() -> Optional[ctypes.CDLL]:
    path = ctypes.util.find_library("objc")
    if not path:
        return None
    lib = ctypes.cdll.LoadLibrary(path)
    lib.objc_getClass.restype = ctypes.c_void_p
    lib.sel_registerName.restype = ctypes.c_void_p
    return lib


@cache
def _msg_send(restype, *argtypes):
    """objc_msgSend typed for one signature, which arm64 needs to pass doubles in the float registers."""
    lib = _objc()
    assert lib is not None
    address = ctypes.cast(lib.objc_msgSend, ctypes.c_void_p).value
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(address)


def _send(receiver: int, selector: str, *args, restype=ctypes.c_void_p, argtypes=()):
    lib = _objc()
    assert lib is not None
    return _msg_send(restype, *argtypes)(receiver, lib.sel_registerName(selector.encode()), *args)


def paint_window_background(widget: QWidget) -> None:
    """Give a top-level widget's NSWindow the theme's base color. A translucent window keeps the clear
    background Qt gives it."""
    if sys.platform != "darwin" or not widget.isWindow() or widget.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground):
        return
    lib = _objc()
    if lib is None:
        return
    native = int(widget.winId())
    if widget.property(_PAINTED_FOR) == native:
        return
    window = _send(native, "window")
    if not window:
        return
    color = QColor(THEME.bg_dark)
    ns_color = _send(
        lib.objc_getClass(b"NSColor"),
        "colorWithSRGBRed:green:blue:alpha:",
        color.redF(),
        color.greenF(),
        color.blueF(),
        1.0,
        argtypes=(ctypes.c_double,) * 4,
    )
    _send(window, "setBackgroundColor:", ns_color, restype=None, argtypes=(ctypes.c_void_p,))
    widget.setProperty(_PAINTED_FOR, native)
