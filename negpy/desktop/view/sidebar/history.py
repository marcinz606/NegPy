from negpy.kernel.system.i18n import tr
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QHBoxLayout, QInputDialog, QListWidget, QListWidgetItem, QMenu, QMessageBox, QVBoxLayout

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import header_row, section_subheader
from negpy.desktop.view.styles.theme import THEME

_INDEX_ROLE = Qt.ItemDataRole.UserRole

_WORK_PRINT_TOOLTIP = tr(
    "Named versions of this frame — the printer's work prints. Unlike the edit history they "
    "are never pruned and never dropped by a later edit. Click one to make it live (undoable); "
    "right-click to export, rename or delete it."
)


class HistoryPanel(BaseSidebar):
    """Work prints (named versions, kept) above the edit-history steps (pruned)."""

    SIDE_MARGIN = THEME.space_xl

    def _init_ui(self) -> None:
        self.work_print_header = section_subheader(tr("WORK PRINTS"))
        self.save_btn = self._icon_action("fa5s.save", "")
        self.save_btn.setToolTip(tooltip_with_shortcut(tr("Save Work Print: keep the current edit as a named version"), "save_work_print"))
        self.layout.addLayout(header_row(self.work_print_header, self.save_btn))

        self.work_prints = QListWidget()
        self.work_prints.setToolTip(_WORK_PRINT_TOOLTIP)
        self.work_prints.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.work_prints.setMaximumHeight(120)

        # Global setting, default off; the toggle stays in view even when the print list is empty.
        self.apply_metadata_btn = self._labeled_toggle(
            "",
            tr("Load Metadata"),
            self.controller.session.work_print_apply_metadata(),
            tr("When a work print loads, restore its saved metadata. Off (the default) keeps the metadata on screen now."),
        )
        # The list and its toggle share one box: the toggle sits at the list's bottom-right.
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(self.work_prints)
        apply_row = QHBoxLayout()
        apply_row.addStretch(1)
        apply_row.addWidget(self.apply_metadata_btn)
        box.addLayout(apply_row)
        self.layout.addLayout(box, 0)

        self.layout.addWidget(section_subheader(tr("EDIT HISTORY")))

        self.list = QListWidget()
        self.list.setToolTip(tr("Click a step to jump to it (last 100 edits kept)."))
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layout.addWidget(self.list, 1)

        self.refresh()

    def _connect_signals(self) -> None:
        self.list.itemClicked.connect(self._on_item_clicked)
        self.list.customContextMenuRequested.connect(self._on_context_menu)
        self.work_prints.itemClicked.connect(lambda item: self.controller.session.load_work_print(item.text()))
        self.work_prints.customContextMenuRequested.connect(self._on_work_print_menu)
        self.save_btn.clicked.connect(self.save_work_print)
        self.apply_metadata_btn.toggled.connect(self._on_apply_metadata_toggled)
        self.controller.session.history_changed.connect(self.refresh)
        self.controller.session.work_prints_changed.connect(self.refresh)
        self.controller.session.file_selected.connect(lambda _: self.refresh())

    def refresh(self) -> None:
        session = self.controller.session
        names = session.work_prints()
        self.work_prints.clear()
        self.work_prints.addItems(names)
        # An empty list is a strip of dead space in a panel most people use for undo.
        self.work_prints.setVisible(bool(names))
        self.work_print_header.setVisible(bool(names))
        self.save_btn.setEnabled(bool(self.controller.state.current_file_hash))

        self.list.clear()
        for row in reversed(self.controller.history_steps()):  # newest on top
            item = QListWidgetItem(row["label"])
            item.setData(_INDEX_ROLE, row["index"])
            if row["is_current"]:
                font = item.font()
                font.setWeight(QFont.Weight.Bold)
                item.setFont(font)
                self.list.setCurrentItem(item)
            self.list.addItem(item)

    def save_work_print(self) -> None:
        session = self.controller.session
        if not self.controller.state.current_file_hash:
            return
        name, ok = QInputDialog.getText(self, tr("Save Work Print"), tr("Name:"), text=session.next_work_print_name())
        name = name.strip()
        if not (ok and name):
            return
        if name in session.work_prints():
            replace = QMessageBox.question(self, tr("Replace Work Print"), tr("“{name}” already exists. Replace it?", name=name))
            if replace != QMessageBox.StandardButton.Yes:
                return
        session.save_work_print(name)

    def _on_apply_metadata_toggled(self, checked: bool) -> None:
        self.controller.session.set_work_print_apply_metadata(checked)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self.controller.jump_to_history_step(item.data(_INDEX_ROLE))

    def _on_context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        export_action = menu.addAction(tr("Export This Version…"))
        if menu.exec(self.list.mapToGlobal(pos)) is export_action:
            self.controller.export_history_step(item.data(_INDEX_ROLE))

    def _on_work_print_menu(self, pos) -> None:
        item = self.work_prints.itemAt(pos)
        if item is None:
            return
        name = item.text()
        menu = QMenu(self)
        export_action = menu.addAction(tr("Export This Version…"))
        rename_action = menu.addAction(tr("Rename…"))
        delete_action = menu.addAction(tr("Delete…"))
        chosen = menu.exec(self.work_prints.mapToGlobal(pos))
        if chosen is export_action:
            self.controller.export_work_print(name)
        elif chosen is rename_action:
            new_name, ok = QInputDialog.getText(self, tr("Rename Work Print"), tr("Name:"), text=name)
            new_name = new_name.strip()
            if not (ok and new_name) or new_name == name:
                return
            exists = new_name in self.controller.session.work_prints()
            if (
                exists
                and QMessageBox.question(self, tr("Replace Work Print"), tr("“{new_name}” already exists. Replace it?", new_name=new_name))
                != QMessageBox.StandardButton.Yes
            ):
                return
            self.controller.session.rename_work_print(name, new_name, replace_existing=exists)
        elif chosen is delete_action:
            if confirm_delete_named(
                self,
                tr("Work Print"),
                name,
                informative=tr("Work prints are never pruned by later edits — this is the only way one goes away."),
            ):
                self.controller.session.delete_work_print(name)
