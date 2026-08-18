import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import dialog_pane_qss, hint_label, pane_header_qss
from negpy.desktop.view.styles.theme import THEME

_ID_ROLE = Qt.ItemDataRole.UserRole


class FavouritesDialog(QDialog):
    """Column-chooser for the Favourites panel: tick sliders on the left, order them on the
    right. Takes plain (id, category, label) triples rather than widgets so it stays testable
    without a live controls panel."""

    def __init__(
        self,
        parent,
        choices: list[tuple[str, str, str]],
        selected: list[str],
        *,
        title: str = "Edit Favourites",
        chosen_header: str = "FAVOURITES",
        hint: str = "Favourites mirror the real controls — editing one here is the same as editing it in its own panel.",
        defaults: list[str] | None = None,
    ):
        super().__init__(parent)
        self._choices = choices
        known = {slider_id for slider_id, _, _ in choices}
        self._chosen = [slider_id for slider_id in selected if slider_id in known]
        self._label_by_id = {slider_id: label for slider_id, _, label in choices}
        self._chosen_header = chosen_header
        self._defaults = [slider_id for slider_id in defaults if slider_id in known] if defaults is not None else None

        self.setWindowTitle(title)
        self.setStyleSheet(f"QDialog {{ background: {THEME.bg_dark}; }}")
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

    def _build_available(self) -> QWidget:
        pane = QWidget()
        pane.setStyleSheet(dialog_pane_qss())
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, THEME.space_xl, 0)
        layout.setSpacing(THEME.space_sm)

        title = QLabel("AVAILABLE")
        title.setStyleSheet(pane_header_qss())
        layout.addWidget(title)

        self.available_list = QListWidget()
        self.available_list.setObjectName("preset_list")
        self.available_list.itemChanged.connect(self._on_item_toggled)
        layout.addWidget(self.available_list)

        self.available_list.blockSignals(True)
        current_category = None
        for slider_id, category, label in self._choices:
            if category != current_category:
                current_category = category
                header = QListWidgetItem(category.upper())
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                font = QFont(header.font())
                font.setBold(True)
                header.setFont(font)
                header.setForeground(QColor(THEME.text_muted))
                self.available_list.addItem(header)
            item = QListWidgetItem(label)
            item.setData(_ID_ROLE, slider_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if slider_id in self._chosen else Qt.CheckState.Unchecked)
            self.available_list.addItem(item)
        self.available_list.blockSignals(False)
        return pane

    def _build_chosen(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_sm)

        title = QLabel(self._chosen_header)
        title.setStyleSheet(pane_header_qss())
        layout.addWidget(title)

        self.chosen_list = QListWidget()
        self.chosen_list.setObjectName("preset_list")
        layout.addWidget(self.chosen_list)

        row = QHBoxLayout()
        self.up_btn = QPushButton()
        self.up_btn.setIcon(qta.icon("fa5s.arrow-up", color=THEME.text_primary))
        self.up_btn.setToolTip("Move up")
        self.up_btn.setFixedWidth(36)
        self.up_btn.clicked.connect(self._move_up)

        self.down_btn = QPushButton()
        self.down_btn.setIcon(qta.icon("fa5s.arrow-down", color=THEME.text_primary))
        self.down_btn.setToolTip("Move down")
        self.down_btn.setFixedWidth(36)
        self.down_btn.clicked.connect(self._move_down)

        self.remove_btn = QPushButton()
        self.remove_btn.setIcon(qta.icon("fa5s.times", color=THEME.text_primary))
        self.remove_btn.setToolTip("Remove from favourites")
        self.remove_btn.setFixedWidth(36)
        self.remove_btn.clicked.connect(self._remove_selected)

        for btn in (self.up_btn, self.down_btn, self.remove_btn):
            row.addWidget(btn)
        row.addStretch()
        layout.addLayout(row)
        return pane

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        if self._defaults is not None:
            restore = QPushButton("Restore Defaults")
            restore.clicked.connect(self._restore_defaults)
            row.addWidget(restore)
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setProperty("primary", True)
        self.apply_btn.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.apply_btn)
        return row

    def _restore_defaults(self) -> None:
        self._chosen = list(self._defaults or [])
        chosen = set(self._chosen)
        self.available_list.blockSignals(True)
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            slider_id = item.data(_ID_ROLE)
            if slider_id is not None:
                item.setCheckState(Qt.CheckState.Checked if slider_id in chosen else Qt.CheckState.Unchecked)
        self.available_list.blockSignals(False)
        self._rebuild_chosen()

    def _on_item_toggled(self, item: QListWidgetItem) -> None:
        slider_id = item.data(_ID_ROLE)
        if slider_id is None:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        if checked and slider_id not in self._chosen:
            self._chosen.append(slider_id)
        elif not checked and slider_id in self._chosen:
            self._chosen.remove(slider_id)
        self._rebuild_chosen()

    def _rebuild_chosen(self, select_row: int = -1) -> None:
        self.chosen_list.clear()
        for slider_id in self._chosen:
            item = QListWidgetItem(self._label_by_id[slider_id])
            item.setData(_ID_ROLE, slider_id)
            self.chosen_list.addItem(item)
        if 0 <= select_row < len(self._chosen):
            self.chosen_list.setCurrentRow(select_row)

    def _set_checked(self, slider_id: str, checked: bool) -> None:
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            if item.data(_ID_ROLE) == slider_id:
                self.available_list.blockSignals(True)
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                self.available_list.blockSignals(False)
                return

    def _move_up(self) -> None:
        row = self.chosen_list.currentRow()
        if row > 0:
            self._chosen[row - 1], self._chosen[row] = self._chosen[row], self._chosen[row - 1]
            self._rebuild_chosen(row - 1)

    def _move_down(self) -> None:
        row = self.chosen_list.currentRow()
        if 0 <= row < len(self._chosen) - 1:
            self._chosen[row + 1], self._chosen[row] = self._chosen[row], self._chosen[row + 1]
            self._rebuild_chosen(row + 1)

    def _remove_selected(self) -> None:
        row = self.chosen_list.currentRow()
        if 0 <= row < len(self._chosen):
            self._set_checked(self._chosen.pop(row), False)
            self._rebuild_chosen(min(row, len(self._chosen) - 1))

    def selected_ids(self) -> list[str]:
        return list(self._chosen)
