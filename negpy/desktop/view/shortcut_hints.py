"""A toast that names a control's keyboard shortcut after the mouse uses the control."""

import html
from typing import Callable

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QAbstractButton, QLabel, QWidget

from negpy.desktop.view.shortcut_registry import REGISTRY, key_for, keycap_cells, shortcut_ids_in

HINT_MS = 2500


class ShortcutHints(QObject):
    """An application event filter. A left-button release over a control whose tooltip came
    from tooltip_with_shortcut shows its keys, once per shortcut per session. A key press sends
    no mouse event, so a control used from the keyboard never shows a hint."""

    def __init__(self, state, show: Callable[[str], None], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._show = show
        self._shown: set[tuple[str, ...]] = set()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if (
            event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self._state.shortcut_hints
            and isinstance(obj, QWidget)
        ):
            self._hint(obj)
        return False

    def _hint(self, widget: QWidget) -> None:
        while widget is not None:
            ids = tuple(shortcut_ids_in(widget.toolTip()))
            if ids:
                keys = [key for key in (key_for(action_id) for action_id in ids) if key]
                # A release that bubbles to the parent finds the same ids again; the set stops it.
                if keys and ids not in self._shown:
                    self._shown.add(ids)
                    self._show(_hint_html(_control_name(widget, ids), keys))
                return
            if widget.isWindow():
                return
            widget = widget.parentWidget()


def _control_name(widget: QWidget, ids: tuple[str, ...]) -> str:
    """The name the control shows: a slider's label or a button's text, else the action's description."""
    for candidate in (widget, widget.parentWidget()):
        label = getattr(candidate, "label", None)
        if isinstance(label, QLabel) and label.text():
            return label.text()
    if isinstance(widget, (QAbstractButton, QLabel)) and widget.text().strip():
        return widget.text().strip().replace("&&", "&")
    return REGISTRY[ids[0]].description


def _hint_html(name: str, keys: list[str]) -> str:
    return (
        f'<table cellspacing="0" cellpadding="0"><tr><td style="padding-right:8px;">{html.escape(name)}</td>'
        f"{keycap_cells(keys, '/')}</tr></table>"
    )
