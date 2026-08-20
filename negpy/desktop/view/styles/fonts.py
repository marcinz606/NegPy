"""UI and monospace font stacks resolved against the installed families.

Qt gives a family name it cannot find to the alias table, and populating that
table walks every installed font — a visible part of startup. Resolve each stack
to a name the system has, once, and give Qt only that. Resolution needs a live
QApplication, so it must not run at import time.
"""

from functools import lru_cache

from PyQt6.QtGui import QFontDatabase

UI_STACK = ("Inter", "Segoe UI", "Roboto", "Arial")
MONO_STACK = ("Consolas", "SF Mono", "Menlo", "DejaVu Sans Mono", "Courier New")


@lru_cache(maxsize=None)
def _resolve(stack: tuple[str, ...], fallback: QFontDatabase.SystemFont) -> str:
    installed = set(QFontDatabase.families())
    for family in stack:
        if family in installed:
            return family
    return QFontDatabase.systemFont(fallback).family()


def ui_font_family() -> str:
    return _resolve(UI_STACK, QFontDatabase.SystemFont.GeneralFont)


def mono_font_family() -> str:
    return _resolve(MONO_STACK, QFontDatabase.SystemFont.FixedFont)
