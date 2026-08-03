from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QInputDialog, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton

from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME

_INDEX_ROLE = Qt.ItemDataRole.UserRole

_WORK_PRINT_TOOLTIP = (
    "Named versions of this frame — the printer's work prints. Unlike the edit history they "
    "are never pruned and never dropped by a later edit. Click one to make it live (undoable); "
    "right-click to export, rename or delete it."
)


class HistoryPanel(BaseSidebar):
    """Work prints (named versions, kept) above the edit-history steps (pruned)."""

    SIDE_MARGIN = THEME.space_xl

    def _init_ui(self) -> None:
        self.work_print_header = section_subheader("WORK PRINTS")
        self.layout.addWidget(self.work_print_header)

        self.work_prints = QListWidget()
        self.work_prints.setToolTip(_WORK_PRINT_TOOLTIP)
        self.work_prints.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.work_prints.setMaximumHeight(120)
        self.layout.addWidget(self.work_prints, 0)

        self.save_btn = QPushButton("Save work print")
        self.save_btn.setToolTip(tooltip_with_shortcut("Keep the current edit as a named version", "save_work_print"))
        self.layout.addWidget(self.save_btn)

        self.layout.addWidget(section_subheader("EDIT HISTORY"))

        self.list = QListWidget()
        self.list.setToolTip("Click a step to jump to it (last 100 edits kept).")
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layout.addWidget(self.list, 1)

        self.refresh()

    def _connect_signals(self) -> None:
        self.list.itemClicked.connect(self._on_item_clicked)
        self.list.customContextMenuRequested.connect(self._on_context_menu)
        self.work_prints.itemClicked.connect(lambda item: self.controller.session.load_work_print(item.text()))
        self.work_prints.customContextMenuRequested.connect(self._on_work_print_menu)
        self.save_btn.clicked.connect(self.save_work_print)
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
        name, ok = QInputDialog.getText(self, "Save work print", "Name:", text=session.next_work_print_name())
        name = name.strip()
        if not (ok and name):
            return
        if name in session.work_prints():
            replace = QMessageBox.question(self, "Replace work print", f"“{name}” already exists. Replace it?")
            if replace != QMessageBox.StandardButton.Yes:
                return
        session.save_work_print(name)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self.controller.jump_to_history_step(item.data(_INDEX_ROLE))

    def _on_context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        export_action = menu.addAction("Export this version…")
        if menu.exec(self.list.mapToGlobal(pos)) is export_action:
            self.controller.export_history_step(item.data(_INDEX_ROLE))

    def _on_work_print_menu(self, pos) -> None:
        item = self.work_prints.itemAt(pos)
        if item is None:
            return
        name = item.text()
        menu = QMenu(self)
        export_action = menu.addAction("Export this version…")
        rename_action = menu.addAction("Rename…")
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(self.work_prints.mapToGlobal(pos))
        if chosen is export_action:
            self.controller.export_work_print(name)
        elif chosen is rename_action:
            new_name, ok = QInputDialog.getText(self, "Rename work print", "Name:", text=name)
            if ok:
                self.controller.session.rename_work_print(name, new_name.strip())
        elif chosen is delete_action:
            self.controller.session.delete_work_print(name)
