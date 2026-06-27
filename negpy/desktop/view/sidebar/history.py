from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMenu

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader

_INDEX_ROLE = Qt.ItemDataRole.UserRole


class HistoryPanel(BaseSidebar):
    """Scrollable list of edit-history steps; click to jump, right-click to export."""

    def _init_ui(self) -> None:
        self.layout.addWidget(section_subheader("EDIT HISTORY"))

        self.list = QListWidget()
        self.list.setToolTip("Click a step to jump to it (last 100 edits kept).")
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layout.addWidget(self.list, 1)

        self.refresh()

    def _connect_signals(self) -> None:
        self.list.itemClicked.connect(self._on_item_clicked)
        self.list.customContextMenuRequested.connect(self._on_context_menu)
        self.controller.session.history_changed.connect(self.refresh)
        self.controller.session.file_selected.connect(lambda _: self.refresh())

    def refresh(self) -> None:
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
