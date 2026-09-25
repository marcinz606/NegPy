import os

import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.roll import BATCH_ANALYSIS_DISABLED_TOOLTIP, BATCH_ANALYSIS_TOOLTIP
from negpy.desktop.view.confirm import (
    confirm_delete_named,
    confirm_delete_several,
    confirm_load_roll,
    warn_invalid_roll_name,
)
from negpy.desktop.view.widgets.rename_roll_dialog import RenameRollDialog
from negpy.desktop.view.styles.templates import TOOLBAR_BUTTON_HEIGHT, TOOLBAR_ICON_SIZE, hint_label, wrap_tooltip
from negpy.desktop.view.widgets.overflow_bar import OverflowBar
from negpy.desktop.view.widgets.sort_button import SortButton
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.text import count_of
from negpy.services.assets import rolls
from negpy.services.assets.library import folder_counts, folder_label, summarize_counts
from negpy.services.assets.presets import is_valid_preset_name

_ROLL_ID_ROLE = Qt.ItemDataRole.UserRole
_MISSING_ROLE = Qt.ItemDataRole.UserRole + 1
_NAME_ROLE = Qt.ItemDataRole.UserRole + 2
_FOLDER_ROLE = Qt.ItemDataRole.UserRole + 3


class LibraryTree(QWidget):
    """The library: every Roll imported or built, in one flat-colored tree.

    A folder is only ever an import source, never something browsed live: importing
    recognizes it (or, for a parent full of scan folders, recognizes each folder
    with images under it as its own roll in one pass), and from then on the roll -- not the path
    -- is what is opened, renamed or deleted. NegPy owns nothing on disk here.
    Refresh finds new roll folders under those parents and marks rolls whose folder is gone.

    Click selects, double-click or Enter opens a roll -- opening is the expensive
    step, so it waits for the second click.
    """

    rolls_changed = pyqtSignal()  # a roll was imported, renamed or deleted
    folder_roll_created = pyqtSignal(str)  # a folder was recognized as a roll for the first time

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.repo = controller.session.repo
        self._sort_order, self._sort_descending = self._saved_sort()
        self._init_ui()
        self.reload()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # No title of its own: the section header above already names it. The same
        # OverflowBar of toolbar buttons the Film Strip uses, so both sections' rows read
        # as one control set.
        self.toolbar = OverflowBar(height=TOOLBAR_BUTTON_HEIGHT, spacing=4)

        self.import_btn = QToolButton()
        self.import_btn.setIcon(qta.icon("fa5s.plus", color=THEME.text_primary))
        self.import_btn.setToolTip(wrap_tooltip("Import — a folder as a roll, or its subfolders as one roll each"))
        self.import_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        import_menu = QMenu(self.import_btn)
        import_menu.addAction("Import Folder as a Roll…").triggered.connect(self.prompt_import_folder)
        import_menu.addAction("Import Subfolders as Rolls…").triggered.connect(self.prompt_import_subfolders)
        self.import_btn.setMenu(import_menu)

        self.refresh_btn = QToolButton()
        self.refresh_btn.setIcon(qta.icon("fa5s.sync-alt", color=THEME.text_primary))
        self.refresh_btn.setToolTip(
            wrap_tooltip("Find new rolls in folders imported as subfolders, and re-read every roll's frame count from disk")
        )
        self.refresh_btn.clicked.connect(self.refresh)

        # Opt-in (Preferences); hidden until then. Decodes and embeds every photo
        # under library_roots once, so search by meaning can rank the whole library,
        # not just the open roll -- an explicit action, never automatic.
        self.index_btn = QToolButton()
        self.index_btn.setIcon(qta.icon("fa5s.database", color=THEME.text_primary))
        self.index_btn.setToolTip(wrap_tooltip("Index the library so search by meaning can rank every roll, not just the loaded one"))
        self.index_btn.clicked.connect(self.controller.index_library)

        self.sort_btn = SortButton((("name", "Name"), ("date", "Date")), "Sort the roll list")
        self.sort_btn.show_order(self._sort_order, self._sort_descending)
        self.sort_btn.order_selected.connect(lambda order: self.set_sort(order, self._sort_descending, save=True))
        self.sort_btn.direction_selected.connect(lambda descending: self.set_sort(self._sort_order, descending, save=True))

        self.filters_btn = QToolButton()
        self.filters_btn.setIcon(qta.icon("fa5s.filter", color=THEME.text_primary))
        self.filters_btn.setToolTip(wrap_tooltip("Discovery Filters — folder names that importing subfolders and Refresh skip"))
        self.filters_btn.clicked.connect(self.edit_discovery_filters)

        for btn in (self.import_btn, self.refresh_btn, self.index_btn, self.filters_btn, self.sort_btn):
            btn.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
            btn.setFixedHeight(TOOLBAR_BUTTON_HEIGHT)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        for widget, label in (
            (self.import_btn, "Import"),
            (self.refresh_btn, "Refresh"),
            (self.index_btn, "Index Library"),
            (self.filters_btn, "Discovery Filters…"),
            (self.sort_btn, "Sort"),
        ):
            self.toolbar.add_button(widget, label)
        # Opt-in, so it starts off; sync_ui turns it on with the feature.
        self.toolbar.set_button_visible(self.index_btn, False)
        layout.addWidget(self.toolbar)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setStyleSheet(
            f"QTreeWidget::item:selected {{ background: {THEME.accent_primary}; color: {THEME.text_on_accent}; }}"
            f"QTreeWidget::item:hover:!selected {{ background: {THEME.surface_hover_faint}; }}"
        )
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.itemSelectionChanged.connect(self._recolor_counts)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        # A scoped shortcut rather than an event filter: a filter object that outlives the tree, or
        # is collected before it, aborts Qt during teardown.
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self.tree)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(self.open_selection)
        layout.addWidget(self.tree, 1)

        self.empty_label = hint_label("Import a folder to build your library")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty_label)

    # --- importing -----------------------------------------------------------

    def prompt_import_folder(self) -> bool:
        """Import Folder as a Roll: recognize one folder and open it.

        Returns whether anything was imported, so the empty-state prompt (Library
        button with nothing in the library yet) knows whether it can proceed.
        """
        start = self.repo.get_global_setting("last_open_folder", "") or ""
        path = QFileDialog.getExistingDirectory(self, "Import Folder as a Roll", start)
        if not path:
            return False
        self.repo.save_global_setting("last_open_folder", path)
        images, _ = folder_counts(path)
        if not images:
            return self.import_subfolders(path)
        if not confirm_load_roll(self, self.repo, images, folder_label(path)):
            return False
        is_new = rolls.folder_roll_id_for_path(self.repo, path) is None
        self.controller.open_library_folder(path)
        self.reload()
        self.rolls_changed.emit()
        if is_new:
            self.folder_roll_created.emit(path)
        return True

    def prompt_import_subfolders(self) -> bool:
        """Import Subfolders as Rolls: every folder with images under a chosen parent,
        at any depth, becomes its own roll without opening any of them."""
        start = self.repo.get_global_setting("last_open_folder", "") or ""
        parent = QFileDialog.getExistingDirectory(self, "Import Subfolders as Rolls", start)
        if not parent:
            return False
        self.repo.save_global_setting("last_open_folder", parent)
        return self.import_subfolders(parent)

    def import_subfolders(self, parent: str) -> bool:
        """Recognize every roll folder under *parent*, without opening any of them."""
        roll_ids = self.controller.import_subfolders_as_rolls(parent)
        if not roll_ids:
            self.controller.set_status(f"No folders with images found in “{folder_label(parent)}”", 4000)
            return False
        self.controller.set_status(f"Imported {count_of(len(roll_ids), 'roll')} from “{folder_label(parent)}”", 3000)
        self.reload()
        self.rolls_changed.emit()
        return True

    def edit_discovery_filters(self) -> bool:
        """Edit the folder-name filters discovery skips, one per line."""
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Discovery Filters",
            "Skip folders whose name contains a line.\nA line with * must match the whole name (raw_*).",
            "\n".join(rolls.discovery_filters(self.repo)),
        )
        if ok:
            rolls.set_discovery_filters(self.repo, text.splitlines())
            self.refresh()
        return ok

    # --- rolls -----------------------------------------------------------------

    def _saved_sort(self) -> tuple[str, bool]:
        """The roll list's own sort. Before it had one it followed the Film Strip, so an
        unset one starts from the Film Strip's (Scene reads as Name: rolls have no scenes)."""
        order = self.repo.get_global_setting("library_sort_order")
        descending = self.repo.get_global_setting("library_sort_descending")
        if order is None:
            order = self.repo.get_global_setting("file_sort_order") or "name"
            descending = self.repo.get_global_setting("file_sort_descending")
        return ("date" if order == "date" else "name"), bool(descending)

    def set_sort(self, order: str, descending: bool, save: bool = False) -> None:
        """Order the roll list, independently of the Film Strip's own sort."""
        self.sort_btn.show_order(order, descending)
        if save:
            self.repo.save_global_setting("library_sort_order", order)
            self.repo.save_global_setting("library_sort_descending", descending)
        if (order, descending) == (self._sort_order, self._sort_descending):
            return
        self._sort_order = order
        self._sort_descending = descending
        self.reload()

    def _sorted(self, entries: list) -> list:
        if self._sort_order == "date":
            return sorted(entries, key=lambda pair: pair[1].get("created_at", 0.0), reverse=self._sort_descending)
        return sorted(entries, key=lambda pair: pair[1].get("name", "").casefold(), reverse=self._sort_descending)

    def sync_ui(self) -> None:
        """Shows Index Library only once the feature is on, and only once the model
        is actually downloaded -- clicking it before that would have nothing to run."""
        from negpy.services.assets import semantic_model

        enabled = self.controller.state.semantic_search_enabled
        self.toolbar.set_button_visible(self.index_btn, enabled)
        self.index_btn.setEnabled(enabled and semantic_model.clip_model_ready())

    def refresh(self) -> None:
        """Refresh: rediscover rolls under the imported parents, then re-read every count."""
        found, dropped = self.controller.rediscover_rolls()
        self.reload()
        if found or dropped:
            parts = [f"Found {count_of(found, 'new roll')}"] if found else []
            parts += [f"removed {count_of(dropped, 'filtered roll')}"] if dropped else []
            self.controller.set_status(" · ".join(parts).capitalize(), 3000)
            self.rolls_changed.emit()

    def reload(self) -> None:
        selected = self._selected_roll_id()
        collapsed = {item.data(0, _FOLDER_ROLE) for item in self._folder_items() if not item.isExpanded()}
        self.tree.clear()
        entries = self._sorted(rolls.all_rolls_sorted(self.repo))
        self.empty_label.setVisible(not entries)
        # A folder roll named "a/b/c" sits under folder rows "a" and "a/b", placed where
        # their first roll falls in the sort.
        folders: dict[str, QTreeWidgetItem] = {}
        for roll_id, entry in entries:
            parts = entry.get("name", "").split(rolls.ROLL_PATH_SEP) if entry.get("kind") == "folder" else [entry.get("name", "")]
            parent = self.tree.invisibleRootItem()
            for depth in range(1, len(parts)):
                key = rolls.ROLL_PATH_SEP.join(parts[:depth])
                if key not in folders:
                    folder_path = entry.get("folder_path", "")
                    for _ in range(len(parts) - depth):
                        folder_path = os.path.dirname(folder_path)
                    folders[key] = self._make_folder_item(parts[depth - 1], folder_path)
                    parent.addChild(folders[key])
                parent = folders[key]
            parent.addChild(self._make_item(roll_id, entry, parts[-1]))
        for item in folders.values():
            item.setText(1, count_of(self._leaf_count(item), "roll"))
            item.setExpanded(item.data(0, _FOLDER_ROLE) not in collapsed)
        if selected:
            self._select_roll(selected)

    @staticmethod
    def _make_folder_item(name: str, folder_path: str) -> QTreeWidgetItem:
        """A folder row: groups rolls, is not a roll itself, so it opens and selects nothing."""
        item = QTreeWidgetItem([name, ""])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setData(0, _FOLDER_ROLE, folder_path)
        item.setToolTip(0, folder_path)
        item.setIcon(0, qta.icon("fa5s.folder-open", color=THEME.text_secondary))
        item.setForeground(1, QColor(THEME.text_muted))
        return item

    def _leaf_count(self, item: QTreeWidgetItem) -> int:
        return sum(
            self._leaf_count(child) if child.data(0, _ROLL_ID_ROLE) is None else 1
            for child in (item.child(i) for i in range(item.childCount()))
        )

    def _all_items(self) -> list[QTreeWidgetItem]:
        items = []
        it = QTreeWidgetItemIterator(self.tree)
        while it.value():
            items.append(it.value())
            it += 1
        return items

    def _roll_items(self) -> list[QTreeWidgetItem]:
        return [item for item in self._all_items() if item.data(0, _ROLL_ID_ROLE) is not None]

    def _folder_items(self) -> list[QTreeWidgetItem]:
        return [item for item in self._all_items() if item.data(0, _FOLDER_ROLE)]

    def _make_item(self, roll_id: str, entry: dict, label: str) -> QTreeWidgetItem:
        is_folder = entry.get("kind") == "folder"
        folder_path = entry.get("folder_path", "")
        missing = is_folder and not os.path.isdir(folder_path)
        count = "folder missing" if missing else summarize_counts(self._frame_count(entry), 0)
        item = QTreeWidgetItem([label, count])
        item.setData(0, _ROLL_ID_ROLE, roll_id)
        item.setData(0, _NAME_ROLE, entry.get("name", ""))
        item.setData(0, _MISSING_ROLE, missing)
        # Roll kind reads off the icon's shape: a folder roll is a folder, a virtual one
        # the search it was built from. Colour is spoken for elsewhere (film mode, channels).
        item.setIcon(0, qta.icon("fa5s.folder" if is_folder else "fa5s.search", color=THEME.text_secondary))
        item.setForeground(1, QColor(self._count_color(item)))
        if missing:
            item.setToolTip(0, f"Folder not found on disk: {folder_path}")
            item.setToolTip(1, "Folder not found on disk — move it back, or delete the roll")
        else:
            item.setToolTip(0, folder_path if is_folder else "Built from a search or a hand-picked set of frames")
        return item

    @staticmethod
    def _count_color(item: QTreeWidgetItem) -> str:
        if item.isSelected():
            return THEME.text_on_accent
        return THEME.warn_amber if item.data(0, _MISSING_ROLE) else THEME.text_muted

    def _frame_count(self, entry: dict) -> int:
        if entry.get("kind") == "folder":
            images, _ = folder_counts(entry.get("folder_path", ""))
            return images + len(entry.get("extra_paths") or ())
        return len(entry.get("member_paths") or ())

    def _selected_roll_id(self):
        item = self.tree.currentItem()
        return item.data(0, _ROLL_ID_ROLE) if item is not None else None

    def _select_roll(self, roll_id) -> None:
        for item in self._roll_items():
            if item.data(0, _ROLL_ID_ROLE) == roll_id:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
                return

    def _recolor_counts(self) -> None:
        # A per-item brush is out of a stylesheet's reach, so the count column has to be repainted
        # by hand or it stays grey on the accent red.
        for item in self._roll_items():
            item.setForeground(1, QColor(self._count_color(item)))

    # --- opening -----------------------------------------------------------

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        roll_id = item.data(0, _ROLL_ID_ROLE)
        if roll_id is not None:
            self.controller.open_roll(roll_id)

    def open_selection(self) -> None:
        """Enter: open the current roll."""
        item = self.tree.currentItem()
        if item is not None and item.data(0, _ROLL_ID_ROLE) is not None:
            self.controller.open_roll(item.data(0, _ROLL_ID_ROLE))

    def _selected_roll_items(self) -> list[tuple]:
        return [
            (item.data(0, _ROLL_ID_ROLE), item.data(0, _NAME_ROLE))
            for item in self.tree.selectedItems()
            if item.data(0, _ROLL_ID_ROLE) is not None
        ]

    def _show_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        if item is not None and item.data(0, _FOLDER_ROLE):
            menu.addAction("Delete…").triggered.connect(lambda: self._delete_folder(item.data(0, _FOLDER_ROLE), item.text(0)))
            menu.addSeparator()
        elif item is not None and item.data(0, _ROLL_ID_ROLE) is not None:
            roll_id = item.data(0, _ROLL_ID_ROLE)
            selection = self._selected_roll_items()
            if len(selection) > 1 and roll_id in dict(selection):
                menu.addAction(f"Delete {len(selection)} Rolls…").triggered.connect(lambda: self._delete_rolls(selection))
            else:
                name = item.data(0, _NAME_ROLE)
                menu.addAction("Open").triggered.connect(lambda: self.controller.open_roll(roll_id))
                is_active = roll_id == self.controller.state.active_roll_id
                analyze_action = menu.addAction("Roll Analysis")
                analyze_action.setEnabled(is_active)
                analyze_action.setToolTip(BATCH_ANALYSIS_TOOLTIP if is_active else BATCH_ANALYSIS_DISABLED_TOOLTIP)
                analyze_action.triggered.connect(self.controller.request_batch_normalization)
                menu.addAction("Rename…").triggered.connect(lambda: self._rename_roll(roll_id, name))
                menu.addAction("Delete…").triggered.connect(lambda: self._delete_roll(roll_id, name))
            menu.addSeparator()
        menu.addAction("Import Folder as a Roll…").triggered.connect(self.prompt_import_folder)
        menu.addAction("Import Subfolders as Rolls…").triggered.connect(self.prompt_import_subfolders)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _rename_roll(self, roll_id: str, current_name: str) -> None:
        entry = rolls.roll_for_id(self.repo, roll_id)
        is_folder = bool(entry) and entry.get("kind") == "folder"
        # Rename changes the roll's own folder name; the folder rows above it stay.
        prefix = ""
        if is_folder and rolls.ROLL_PATH_SEP in current_name:
            prefix, current_name = current_name.rsplit(rolls.ROLL_PATH_SEP, 1)

        dlg = RenameRollDialog(current_name, self, folder_backed=is_folder)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, rename_folder = dlg.name(), dlg.rename_folder()

        if not name or (name == current_name and not rename_folder):
            return
        if not is_valid_preset_name(name):
            warn_invalid_roll_name(self, "Rename Roll")
            return

        if rename_folder:
            if not self.controller.request_rename_roll(roll_id, name, True):
                QMessageBox.warning(
                    self,
                    "Rename Roll",
                    "Could not rename the folder on disk — check that no other folder already has that name, "
                    "and that you have permission to rename it here.",
                )
                return
        rolls.rename_roll(self.repo, roll_id, f"{prefix}{rolls.ROLL_PATH_SEP}{name}" if prefix else name)

        self.reload()
        self.rolls_changed.emit()

    def _delete_roll(self, roll_id: str, name: str) -> None:
        if confirm_delete_named(self, "Roll", name, informative="This only forgets the roll — nothing on disk is touched."):
            rolls.delete_roll(self.repo, roll_id)
            self.reload()
            self.rolls_changed.emit()

    def _delete_folder(self, folder_path: str, name: str) -> None:
        if confirm_delete_named(self, "Folder", name, informative="This forgets every roll in it — nothing on disk is touched."):
            rolls.delete_folder_rolls(self.repo, folder_path)
            self.reload()
            self.rolls_changed.emit()

    def _delete_rolls(self, selection: list[tuple]) -> None:
        names = [name for _roll_id, name in selection]
        if confirm_delete_several(self, "Roll", names, informative="This only forgets the roll records — nothing on disk is touched."):
            for roll_id, _name in selection:
                rolls.delete_roll(self.repo, roll_id)
            self.reload()
            self.rolls_changed.emit()
