"""Search field that accepts text queries and live shortcut presses."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QLineEdit

from negpy.desktop.view.widgets.key_sequence_edit import (
    KnownBindingsProvider,
    format_binding_display,
    key_event_to_portable,
    should_capture_binding,
)


class ShortcutSearchLineEdit(QLineEdit):
    """Line edit that can be searched by name or by pressing a bound shortcut."""

    bindingCaptured = pyqtSignal(str)

    def __init__(self, known_bindings: KnownBindingsProvider, parent=None) -> None:
        super().__init__(parent)
        self._known_bindings = known_bindings

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape, Qt.Key.Key_Backspace):
            super().keyPressEvent(event)
            return

        known = self._known_bindings()
        portable = key_event_to_portable(event)
        allow_single_key = not self.text()
        if portable and should_capture_binding(event, known, portable=portable, allow_single_key=allow_single_key):
            text = format_binding_display(portable)
            self.setText(text)
            self.textEdited.emit(text)
            self.bindingCaptured.emit(portable)
            event.accept()
            return

        super().keyPressEvent(event)
