"""Custom Routing window for Pick Better: the round order, reorderable by drag and drop.

A single list shows the eleven tunable parameters in the order the rounds visit them (row 1 is
round 1). Dragging a row to a new place reorders the route, so the name on the row is what that
round proposes. A Randomize on/off sits below the list: when on, each run shuffles a copy of this
order, and the order set here is used only when Randomize is off, so turning it off restores it.
Apply commits the route and the Randomize choice; Cancel discards both.
"""

from negpy.kernel.system.i18n import tr

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import hint_label, labeled_toggle, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.features.pickbetter.models import PICK_PARAMS

_ID_ROLE = Qt.ItemDataRole.UserRole


class _ReorderList(QListWidget):
    """Drag-to-reorder list. QListWidget's internal move is a remove plus an insert, so
    rowsMoved never fires and the drop itself is the only reliable signal."""

    reordered = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self.reordered.emit()


class CustomRoutingDialog(QDialog):
    """The round order the Pick Better rounds visit, reorderable by drag and drop, with a
    Randomize on/off. `route` is the order set here (round 1 = first row); when Randomize is on
    each run shuffles a copy and this order is used only when Randomize is off."""

    def __init__(
        self,
        parent,
        param_order: tuple[int, ...],
        randomize: bool,
        *,
        repo=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Custom Routing"))
        self.resize(420, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        root.addWidget(self._build_list())
        self._randomize_toggle = labeled_toggle(
            "fa5s.random", tr("Randomize"), randomize, tr("Shuffle the order on each run; the order set here is used only when this is off")
        )
        root.addWidget(self._randomize_toggle)
        root.addWidget(hint_label(tr("When on, each run shuffles this order; the order you set here is used only when Randomize is off.")))
        root.addLayout(self._build_footer())

        self._fill_list(param_order)
        self._list.reordered.connect(self._on_reordered)
        remember_dialog_geometry(self, repo, "custom_routing")

    def _build_list(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_sm)
        self._list = _ReorderList()
        self._list.setObjectName("preset_list")
        layout.addWidget(self._list)
        return pane

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        self.apply_btn = QPushButton(tr("Apply"))
        self.apply_btn.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.apply_btn)
        pin_dialog_default(self.apply_btn, cancel, scope=self)
        return row

    def _fill_list(self, param_order: tuple[int, ...]) -> None:
        self._list.clear()
        for position, index in enumerate(param_order):
            item = QListWidgetItem(self._row_text(position, index))
            item.setData(_ID_ROLE, index)
            self._list.addItem(item)

    @staticmethod
    def _row_text(position: int, index: int) -> str:
        return f"{position + 1}. {PICK_PARAMS[index].name}"

    def _on_reordered(self) -> None:
        # The dropped list is the truth; renumber each row to its new round position.
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setText(self._row_text(i, item.data(_ID_ROLE)))

    def route(self) -> tuple[int, ...]:
        return tuple(self._list.item(i).data(_ID_ROLE) for i in range(self._list.count()))

    def randomize_on(self) -> bool:
        return self._randomize_toggle.isChecked()
