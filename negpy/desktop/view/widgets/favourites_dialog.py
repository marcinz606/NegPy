from negpy.kernel.system.i18n import tr
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import (
    dialog_pane_qss,
    hint_label,
    pane_header_qss,
    pin_dialog_default,
    section_subheader,
    tool_toggle,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry

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


class FavouritesDialog(QDialog):
    """Column-chooser for the Favourites panel: toggle sliders on the left, order them on the
    right. Takes plain (id, category, label) triples rather than widgets so it stays testable
    without a live controls panel."""

    def __init__(
        self,
        parent,
        choices: list[tuple[str, str, str]],
        selected: list[str],
        *,
        title: str = tr("Edit Favorites"),
        chosen_header: str = tr("FAVORITES"),
        hint: str = tr("Drag to reorder. A favorite is the same control as in its own card."),
        defaults: list[str] | None = None,
        repo=None,
    ):
        super().__init__(parent)
        self._choices = choices
        known = {slider_id for slider_id, _, _ in choices}
        self._chosen = [slider_id for slider_id in selected if slider_id in known]
        self._label_by_id = {slider_id: label for slider_id, _, label in choices}
        self._chosen_header = chosen_header
        self._defaults = [slider_id for slider_id in defaults if slider_id in known] if defaults is not None else None

        self.setWindowTitle(title)
        self.resize(620, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        panes = QHBoxLayout()
        panes.setSpacing(THEME.space_xl)
        panes.addWidget(self._build_available(), 1)
        panes.addWidget(self._build_chosen(), 1)
        root.addLayout(panes)
        root.addWidget(hint_label(hint))
        root.addLayout(self._build_footer())

        self._rebuild_chosen()
        remember_dialog_geometry(self, repo, "favourites")

    def _build_available(self) -> QWidget:
        pane = QWidget()
        pane.setObjectName("dialog_pane")
        pane.setStyleSheet(dialog_pane_qss())
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, THEME.space_xl, 0)
        layout.setSpacing(THEME.space_sm)

        title = QLabel(tr("AVAILABLE"))
        title.setStyleSheet(pane_header_qss())
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("available_body")
        body.setStyleSheet("QWidget#available_body { background: transparent; }")
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, THEME.space_md, 0)
        column.setSpacing(THEME.space_xs)
        self.available_toggles: dict[str, QPushButton] = {}
        current_category = None
        for slider_id, category, label in self._choices:
            if category != current_category:
                current_category = category
                column.addWidget(section_subheader(category.upper()))
            toggle = tool_toggle("", label, tr("Show {value} here", value=label), align_left=True)
            toggle.setChecked(slider_id in self._chosen)
            toggle.toggled.connect(lambda checked, sid=slider_id: self._on_item_toggled(sid, checked))
            self.available_toggles[slider_id] = toggle
            column.addWidget(toggle)
        column.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll)
        return pane

    def _build_chosen(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_sm)

        title = QLabel(self._chosen_header)
        title.setStyleSheet(pane_header_qss())
        layout.addWidget(title)

        self.chosen_list = _ReorderList()
        self.chosen_list.setObjectName("preset_list")
        self.chosen_list.reordered.connect(self._on_reordered)
        layout.addWidget(self.chosen_list)
        return pane

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        if self._defaults is not None:
            restore = QPushButton(tr("Restore Defaults"))
            restore.clicked.connect(self._restore_defaults)
            row.addWidget(restore)
        row.addStretch()
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        self.apply_btn = QPushButton(tr("Apply"))
        self.apply_btn.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.apply_btn)
        pin_dialog_default(self.apply_btn, cancel, *([restore] if self._defaults is not None else []))
        return row

    def _restore_defaults(self) -> None:
        self._chosen = list(self._defaults or [])
        chosen = set(self._chosen)
        for slider_id, toggle in self.available_toggles.items():
            toggle.blockSignals(True)
            toggle.setChecked(slider_id in chosen)
            toggle.blockSignals(False)
        self._rebuild_chosen()

    def _on_item_toggled(self, slider_id: str, checked: bool) -> None:
        if checked and slider_id not in self._chosen:
            self._chosen.append(slider_id)
        elif not checked and slider_id in self._chosen:
            self._chosen.remove(slider_id)
        self._rebuild_chosen()

    def _rebuild_chosen(self) -> None:
        self.chosen_list.clear()
        for slider_id in self._chosen:
            item = QListWidgetItem(self._label_by_id[slider_id])
            item.setData(_ID_ROLE, slider_id)
            self.chosen_list.addItem(item)

    def _on_reordered(self) -> None:
        """The dropped list is the truth; read the order back off it."""
        self._chosen = [self.chosen_list.item(i).data(_ID_ROLE) for i in range(self.chosen_list.count())]

    def selected_ids(self) -> list[str]:
        return list(self._chosen)
