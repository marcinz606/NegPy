import qtawesome as qta
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QToolButton

from negpy.desktop.view.styles.templates import wrap_tooltip
from negpy.desktop.view.styles.theme import THEME


class SortButton(QToolButton):
    """A section toolbar's Sort dropdown: one order out of *orders*, then Ascending or
    Descending. The Library and the Film Strip each own one and sort independently."""

    order_selected = pyqtSignal(str)
    direction_selected = pyqtSignal(bool)  # True for descending

    def __init__(self, orders: tuple[tuple[str, str], ...], tooltip: str, parent=None):
        super().__init__(parent)
        self.setIcon(qta.icon("fa5s.sort", color=THEME.text_primary))
        self.setToolTip(wrap_tooltip(tooltip))
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self)
        self._orders: dict[str, QAction] = {}
        order_group = QActionGroup(self)
        order_group.setExclusive(True)
        for key, label in orders:
            action = menu.addAction(label)
            action.setCheckable(True)
            order_group.addAction(action)
            action.triggered.connect(lambda _checked=False, k=key: self.order_selected.emit(k))
            self._orders[key] = action
        menu.addSeparator()
        direction_group = QActionGroup(self)
        direction_group.setExclusive(True)
        self.ascending_action = menu.addAction("Ascending")
        self.descending_action = menu.addAction("Descending")
        for action, descending in ((self.ascending_action, False), (self.descending_action, True)):
            action.setCheckable(True)
            direction_group.addAction(action)
            action.triggered.connect(lambda _checked=False, d=descending: self.direction_selected.emit(d))
        self.setMenu(menu)

    def order_action(self, key: str) -> QAction:
        return self._orders[key]

    def show_order(self, order: str, descending: bool) -> None:
        """Tick *order* and the direction. Emits nothing."""
        for key, action in self._orders.items():
            action.setChecked(key == order)
        self.ascending_action.setChecked(not descending)
        self.descending_action.setChecked(descending)
