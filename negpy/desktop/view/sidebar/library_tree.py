import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QMenu,
    QMessageBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
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
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.text import count_of
from negpy.services.assets import rolls
from negpy.services.assets.library import folder_counts, folder_label, summarize_counts
from negpy.services.assets.presets import is_valid_preset_name

_ROLL_ID_ROLE = Qt.ItemDataRole.UserRole


class LibraryTree(QWidget):
    """The library: every Roll imported or built, in one flat, flat-colored list.

    A folder is only ever an import source, never something browsed live: importing
    recognizes it (or, for a parent full of scan folders, recognizes each immediate
    subfolder as its own roll in one pass), and from then on the roll -- not the path
    -- is what is opened, renamed or deleted. NegPy owns nothing on disk here;
    re-importing after Finder reorganizes a folder just re-reads its count.

    Click selects, double-click or Enter opens a roll -- opening is the expensive
    step, so it waits for the second click.
    """

    rolls_changed = pyqtSignal()  # a roll was imported, renamed or deleted
    folder_roll_created = pyqtSignal(str)  # a folder was recognized as a roll for the first time

    def __init__(self, controller, trailing_widgets: tuple[QWidget, ...] = ()):
        super().__init__()
        self.controller = controller
        self.repo = controller.session.repo
        self._sort_order = "name"
        self._sort_descending = False
        self._trailing_widgets = trailing_widgets
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
        self.refresh_btn.setToolTip(wrap_tooltip("Re-read every roll's frame count from disk"))
        self.refresh_btn.clicked.connect(self.reload)

        # Opt-in (Preferences); hidden until then. Decodes and embeds every photo
        # under library_roots once, so search by meaning can rank the whole library,
        # not just the open roll -- an explicit action, never automatic.
        self.index_btn = QToolButton()
        self.index_btn.setIcon(qta.icon("fa5s.database", color=THEME.text_primary))
        self.index_btn.setToolTip(wrap_tooltip("Index the library so search by meaning can rank every roll, not just the loaded one"))
        self.index_btn.clicked.connect(self.controller.index_library)

        for btn in (self.import_btn, self.refresh_btn, self.index_btn):
            btn.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
            btn.setFixedHeight(TOOLBAR_BUTTON_HEIGHT)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        for widget, label in (
            (self.import_btn, "Import"),
            (self.refresh_btn, "Refresh"),
            (self.index_btn, "Index Library"),
            *((w, "Sort") for w in self._trailing_widgets),
        ):
            self.toolbar.add_button(widget, label)
        # Opt-in, so it starts off; sync_ui turns it on with the feature.
        self.toolbar.set_button_visible(self.index_btn, False)
        layout.addWidget(self.toolbar)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        # Every roll is a top-level item, never a child -- the branch/twisty gutter Qt
        # reserves by default has nothing to show and only pushes the icon right.
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(0)
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
            self.controller.set_status(f"No images directly in “{folder_label(path)}”", 4000)
            return False
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
        """Import Subfolders as Rolls: every immediate subfolder of a chosen parent
        becomes its own roll, one level only, without opening any of them."""
        start = self.repo.get_global_setting("last_open_folder", "") or ""
        parent = QFileDialog.getExistingDirectory(self, "Import Subfolders as Rolls", start)
        if not parent:
            return False
        self.repo.save_global_setting("last_open_folder", parent)
        roll_ids = self.controller.import_subfolders_as_rolls(parent)
        if not roll_ids:
            self.controller.set_status(f"No subfolders found in “{folder_label(parent)}”", 4000)
            return False
        self.controller.set_status(f"Imported {count_of(len(roll_ids), 'roll')}", 3000)
        self.reload()
        self.rolls_changed.emit()
        return True

    # --- rolls -----------------------------------------------------------------

    def set_sort(self, order: str, descending: bool) -> None:
        """Order rolls the way the film strip orders frames."""
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

    def reload(self) -> None:
        selected = self._selected_roll_id()
        self.tree.clear()
        entries = self._sorted(rolls.all_rolls_sorted(self.repo))
        self.empty_label.setVisible(not entries)
        for roll_id, entry in entries:
            self.tree.addTopLevelItem(self._make_item(roll_id, entry))
        if selected:
            self._select_roll(selected)

    def _make_item(self, roll_id: str, entry: dict) -> QTreeWidgetItem:
        is_folder = entry.get("kind") == "folder"
        item = QTreeWidgetItem([entry.get("name", ""), summarize_counts(self._frame_count(entry), 0)])
        item.setData(0, _ROLL_ID_ROLE, roll_id)
        # Roll kind reads off the icon's shape: a folder roll is a folder, a virtual one
        # the search it was built from. Colour is spoken for elsewhere (film mode, channels).
        item.setIcon(0, qta.icon("fa5s.folder" if is_folder else "fa5s.search", color=THEME.text_secondary))
        item.setForeground(1, QColor(THEME.text_muted))
        item.setToolTip(0, entry.get("folder_path", "") if is_folder else "Built from a search or a hand-picked set of frames")
        return item

    def _frame_count(self, entry: dict) -> int:
        if entry.get("kind") == "folder":
            images, _ = folder_counts(entry.get("folder_path", ""))
            return images + len(entry.get("extra_paths") or ())
        return len(entry.get("member_paths") or ())

    def _selected_roll_id(self):
        item = self.tree.currentItem()
        return item.data(0, _ROLL_ID_ROLE) if item is not None else None

    def _select_roll(self, roll_id) -> None:
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, _ROLL_ID_ROLE) == roll_id:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
                return

    def _recolor_counts(self) -> None:
        # A per-item brush is out of a stylesheet's reach, so the count column has to be repainted
        # by hand or it stays grey on the accent red.
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setForeground(1, QColor(THEME.text_on_accent) if item.isSelected() else QColor(THEME.text_muted))

    # --- opening -----------------------------------------------------------

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        self.controller.open_roll(item.data(0, _ROLL_ID_ROLE))

    def open_selection(self) -> None:
        """Enter: open the current roll."""
        item = self.tree.currentItem()
        if item is not None:
            self.controller.open_roll(item.data(0, _ROLL_ID_ROLE))

    def _selected_roll_items(self) -> list[tuple]:
        return [(item.data(0, _ROLL_ID_ROLE), item.text(0)) for item in self.tree.selectedItems()]

    def _show_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        if item is not None:
            roll_id = item.data(0, _ROLL_ID_ROLE)
            selection = self._selected_roll_items()
            if len(selection) > 1 and roll_id in dict(selection):
                menu.addAction(f"Delete {len(selection)} Rolls…").triggered.connect(lambda: self._delete_rolls(selection))
            else:
                name = item.text(0)
                menu.addAction("Open").triggered.connect(lambda: self.controller.open_roll(roll_id))
                is_active = roll_id == self.controller.state.active_roll_id
                analyze_action = menu.addAction("Batch Analysis")
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
        else:
            rolls.rename_roll(self.repo, roll_id, name)

        self.reload()
        self.rolls_changed.emit()

    def _delete_roll(self, roll_id: str, name: str) -> None:
        if confirm_delete_named(self, "Roll", name, informative="This only forgets the roll — nothing on disk is touched."):
            rolls.delete_roll(self.repo, roll_id)
            self.reload()
            self.rolls_changed.emit()

    def _delete_rolls(self, selection: list[tuple]) -> None:
        names = [name for _roll_id, name in selection]
        if confirm_delete_several(self, "Roll", names, informative="This only forgets the roll records — nothing on disk is touched."):
            for roll_id, _name in selection:
                rolls.delete_roll(self.repo, roll_id)
            self.reload()
            self.rolls_changed.emit()
