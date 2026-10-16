"""The list of shapes a panel draws on the canvas, one row each: Dodge & Burn's masks, Light Leak's
leaks. A row is a custom widget with flat icon buttons; clicking its background selects it."""

from typing import Sequence

import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QPushButton, QWidget

from negpy.desktop.view.styles.theme import THEME

ROW_HEIGHT = 30


class ShapeRow(QWidget):
    """One list row; clicking its background, not its buttons, selects the shape."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)


def row_icon_button(icon_name: str, checkable: bool) -> QPushButton:
    btn = QPushButton()
    btn.setCheckable(checkable)
    btn.setFlat(True)
    btn.setIcon(qta.icon(icon_name, color=THEME.text_primary))
    btn.setFixedSize(26, 22)
    btn.setStyleSheet("QPushButton {border: none; padding: 0px;}")
    return btn


class ShapeList(QListWidget):
    def __init__(self, tooltip: str) -> None:
        super().__init__()
        self.setToolTip(tooltip)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The row is a custom widget, so drop the app-wide item padding, margin and border that
        # would otherwise squeeze and clip it.
        self.setStyleSheet(
            f"QListView::item {{ border: none; margin: 0px; padding: 0px; }}QListView::item:selected {{ background-color: {THEME.bg_hover}; }}"
        )

    def set_rows(self, rows: Sequence[QWidget], selected: int) -> None:
        """Show *rows*, *selected* highlighted when it is one of them; hidden when empty."""
        self.blockSignals(True)
        self.clear()
        for row in rows:
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, ROW_HEIGHT))
            self.addItem(item)
            self.setItemWidget(item, row)
        if 0 <= selected < len(rows):
            self.setCurrentRow(selected)
        else:
            self.clearSelection()
        self.setVisible(bool(rows))
        if rows:
            self.setFixedHeight(ROW_HEIGHT * len(rows) + 2 * self.frameWidth())
        self.blockSignals(False)
