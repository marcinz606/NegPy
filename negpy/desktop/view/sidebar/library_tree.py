from negpy.kernel.system.i18n import language, tr
import os

import qtawesome as qta
from PyQt6.QtCore import QDateTime, QLocale, QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QKeySequence, QPainter, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QStyle,
    QStyledItemDelegate,
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
from negpy.desktop.view.keyboard_shortcuts import _close_roll
from negpy.desktop.view.shortcut_registry import label_with_shortcut
from negpy.desktop.view.styles.templates import TOOLBAR_BUTTON_HEIGHT, TOOLBAR_ICON_SIZE, hint_label, labeled_action, wrap_tooltip
from negpy.desktop.view.widgets.flat_rows import ICON, more_press, more_rect, paint_fill, paint_more, paint_pill, two_line_rects
from negpy.desktop.view.widgets.sort_button import SortButton
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.text import count_of
from negpy.services.assets import rolls
from negpy.services.assets.composites import frame_count, saved_composites
from negpy.services.assets.library import folder_counts, folder_image_paths, folder_label
from negpy.services.assets.triplets import saved_triplets
from negpy.services.assets.presets import is_valid_preset_name
from negpy.services.assets.gear import GearProfiles
from negpy.features.process.models import ProcessMode

_ROLL_ID_ROLE = Qt.ItemDataRole.UserRole
_MISSING_ROLE = Qt.ItemDataRole.UserRole + 1
_NAME_ROLE = Qt.ItemDataRole.UserRole + 2
_FOLDER_ROLE = Qt.ItemDataRole.UserRole + 3
# The line under a roll's name (count, film mode, stock, scenes, date), or a folder row's roll count.
_DETAIL_ROLE = Qt.ItemDataRole.UserRole + 4
# The film-mode dot ahead of that line; "" for none.
_DOT_ROLE = Qt.ItemDataRole.UserRole + 5
_FRAMES_ROLE = Qt.ItemDataRole.UserRole + 6
# A folder row's roll-name prefix ("a/b"): the row's identity, since rolls from two disk trees can share it.
_PREFIX_ROLE = Qt.ItemDataRole.UserRole + 7

# The roll-name prefixes of the folder rows the user collapsed, kept across sessions.
_COLLAPSED_KEY = "library_collapsed_folders"

_ROLL_ROW = 46
_FOLDER_ROW = 30

# The Setup card's names and the Film Strip's mode colors.
_MODES = {
    ProcessMode.C41: (tr("Color"), THEME.mode_c41),
    ProcessMode.BW: (tr("B&W"), THEME.mode_bw),
    ProcessMode.E6: (tr("Slide"), THEME.mode_e6),
}


def _short_date(timestamp: float) -> str:
    date = QDateTime.fromSecsSinceEpoch(int(timestamp)).date()
    same_year = date.year() == QDateTime.currentDateTime().date().year()
    # The UI's language, not the system's: the month reads in the language around it.
    return QLocale(language()).toString(date, "d MMM" if same_year else "d MMM yyyy")


class _RollDelegate(QStyledItemDelegate):
    """A roll row: its name over a line that says what it is, the open roll marked with an Open
    pill, ⋮ on hover. A folder row is one line: the folder and how many rolls it holds."""

    def __init__(self, tree: QTreeWidget, active_roll):
        super().__init__(tree)
        self._active_roll = active_roll

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), _ROLL_ROW if index.data(_ROLL_ID_ROLE) is not None else _FOLDER_ROW)

    def paint(self, p: QPainter, option, index) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(0, 1, 0, -1)
        roll_id = index.data(_ROLL_ID_ROLE)
        is_open = roll_id is not None and roll_id == self._active_roll()
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        paint_fill(p, rect, current=is_open, hover=hover and roll_id is not None, selected=selected)

        font = QFont(option.font)
        font.setPixelSize(THEME.font_size_base)
        small = QFont(font)
        small.setPixelSize(THEME.font_size_small)
        x = rect.left() + THEME.space_lg
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None:
            icon.paint(p, QRect(x, rect.center().y() - ICON // 2, ICON, ICON))
            x += ICON + THEME.space_lg
        right = more_rect(option.rect).left() - THEME.space_md
        name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        detail = index.data(_DETAIL_ROLE) or ""

        if roll_id is None:
            p.setFont(small)
            p.setPen(QColor(THEME.text_hint))
            width = QFontMetrics(small).horizontalAdvance(detail)
            p.drawText(QRect(right - width, rect.top(), width, rect.height()), Qt.AlignmentFlag.AlignVCenter, detail)
            p.setFont(font)
            p.setPen(QColor(THEME.text_secondary))
            text = QFontMetrics(font).elidedText(name, Qt.TextElideMode.ElideRight, right - width - THEME.space_md - x)
            p.drawText(QRect(x, rect.top(), right - x, rect.height()), Qt.AlignmentFlag.AlignVCenter, text)
            p.restore()
            return

        if is_open:
            right = paint_pill(p, right, rect.center().y(), tr("Open"), small, accent=True) - THEME.space_md
        name_font = QFont(font)
        name_font.setWeight(QFont.Weight(THEME.weight_semibold if is_open else THEME.weight_regular))
        name_metrics, small_metrics = QFontMetrics(name_font), QFontMetrics(small)
        name_line, line = two_line_rects(rect, x, right, name_metrics, small_metrics)
        p.setFont(name_font)
        p.setPen(QColor(THEME.text_primary))
        p.drawText(name_line, Qt.AlignmentFlag.AlignVCenter, name_metrics.elidedText(name, Qt.TextElideMode.ElideRight, right - x))
        # The dot's slot stays when a roll records no film mode, so the lines align.
        dot = index.data(_DOT_ROLE)
        if dot:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(dot))
            p.drawEllipse(line.left(), line.center().y() - 3, 7, 7)
        line.setLeft(line.left() + 7 + THEME.space_md)
        p.setFont(small)
        p.setPen(QColor(THEME.warn_amber if index.data(_MISSING_ROLE) else THEME.text_hint))
        p.drawText(line, Qt.AlignmentFlag.AlignVCenter, small_metrics.elidedText(detail, Qt.TextElideMode.ElideRight, line.width()))
        if hover:
            paint_more(p, option.rect)
        p.restore()


class _RollTree(QTreeWidget):
    """A press on a roll's ⋮, or a right click, asks for that row's menu instead of picking it."""

    menu_requested = pyqtSignal(QPoint)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menu_requested)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = more_press(self, event)
        if pos is not None and self.itemAt(pos).data(0, _ROLL_ID_ROLE) is not None:
            self.menu_requested.emit(pos)
            return
        super().mousePressEvent(event)


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
        self._gear = None
        # True while the tree opens folders on its own, so the saved state ignores those expands.
        self._auto_expanding = False
        self._shown_open_roll = None
        saved = self.repo.get_global_setting(_COLLAPSED_KEY, [])
        self._collapsed = {key for key in saved if isinstance(key, str)} if isinstance(saved, list) else set()
        self._init_ui()
        # Deferred: files_changed fires before the session saves its triplets.
        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(0)
        self._count_timer.timeout.connect(self.refresh_active_count)
        controller.session.files_changed.connect(self._count_timer.start)
        self.reload()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_sm)

        # Like the canvas bar's Roll mode: the filter takes the free width, then Sort, Import and ⋮.
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(tr("Filter rolls…"))
        self.filter_input.setToolTip(tr("Show the rolls whose name or line under it holds this text"))
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.addAction(qta.icon("fa5s.search", color=THEME.text_secondary), QLineEdit.ActionPosition.LeadingPosition)
        self.filter_input.setFixedHeight(TOOLBAR_BUTTON_HEIGHT)
        self.filter_input.setMinimumWidth(40)
        self.filter_input.textChanged.connect(lambda _text: self._apply_filter())

        self.sort_btn = SortButton((("name", tr("Name")), ("date", tr("Date"))), tr("Sort the roll list"))
        self.sort_btn.order_selected.connect(lambda order: self.set_sort(order, self._sort_descending, save=True))
        self.sort_btn.direction_selected.connect(lambda descending: self.set_sort(self._sort_order, descending, save=True))

        self.import_btn = QToolButton()
        self.import_btn.setIcon(qta.icon("mdi6.folder-plus-outline", color=THEME.text_primary))
        self.import_btn.setText(tr("Import"))
        self.import_btn.setToolTip(wrap_tooltip(tr("Import a folder as a roll, or its subfolders as one roll each")))
        self.import_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        import_menu = QMenu(self.import_btn)
        import_menu.addAction(tr("Import Folder as a Roll…")).triggered.connect(self.prompt_import_folder)
        import_menu.addAction(tr("Import Subfolders as Rolls…")).triggered.connect(self.prompt_import_subfolders)
        self.import_btn.setMenu(import_menu)

        self.more_btn = QToolButton()
        self.more_btn.setObjectName("toolbar_overflow")
        self.more_btn.setIcon(qta.icon("fa5s.ellipsis-v", color=THEME.text_primary))
        self.more_btn.setToolTip(tr("Refresh, Discovery Filters and Index Library"))
        self.more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more_menu = QMenu(self.more_btn)
        more_menu.setToolTipsVisible(True)
        self.refresh_action = more_menu.addAction(tr("Refresh"))
        self.refresh_action.setToolTip(tr("Find new rolls in imported folders, and recount the frames of each roll."))
        self.refresh_action.triggered.connect(self.refresh)
        self.filters_action = more_menu.addAction(tr("Discovery Filters…"))
        self.filters_action.setToolTip(tr("Discovery Filters: folder names that importing subfolders and Refresh skip"))
        self.filters_action.triggered.connect(self.edit_discovery_filters)
        # Opt-in (Preferences); hidden until then. Decodes and embeds every photo
        # under library_roots once, so search by meaning can rank the whole library,
        # not just the open roll -- an explicit action, never automatic.
        self.index_action = more_menu.addAction(tr("Index Library"))
        self.index_action.setToolTip(tr("Index the library, so search by meaning covers every roll."))
        self.index_action.triggered.connect(self.controller.index_library)
        self.index_action.setVisible(False)
        self.more_btn.setMenu(more_menu)

        for btn in (self.sort_btn, self.import_btn, self.more_btn):
            btn.setProperty("boxed", True)
            btn.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
            btn.setFixedHeight(TOOLBAR_BUTTON_HEIGHT)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        for btn in (self.sort_btn, self.import_btn):
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setProperty("dropdown", True)
        self.set_sort(self._sort_order, self._sort_descending)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(THEME.space_sm)
        toolbar.addWidget(self.filter_input, 1)
        for btn in (self.sort_btn, self.import_btn, self.more_btn):
            toolbar.addWidget(btn)
        layout.addLayout(toolbar)

        self.tree = _RollTree()
        self.tree.setObjectName("roll_tree")
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setItemDelegate(_RollDelegate(self.tree, lambda: self.controller.state.active_roll_id))
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.menu_requested.connect(self._show_context_menu)
        self.tree.itemExpanded.connect(lambda item: self._remember_folder(item, collapsed=False))
        self.tree.itemCollapsed.connect(lambda item: self._remember_folder(item, collapsed=True))
        # A scoped shortcut rather than an event filter: a filter object that outlives the tree, or
        # is collected before it, aborts Qt during teardown.
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self.tree)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(self.open_selection)
        layout.addWidget(self.tree, 1)

        self.tally_label = hint_label("")
        layout.addWidget(self.tally_label)

        # An empty library is told what a roll is, with the import it needs in reach.
        self.empty_label = QWidget()
        empty_col = QVBoxLayout(self.empty_label)
        empty_col.setContentsMargins(THEME.space_2xl, THEME.space_2xl * 2, THEME.space_2xl, THEME.space_2xl * 2)
        empty_col.setSpacing(THEME.space_lg)
        icon = QLabel()
        icon.setPixmap(qta.icon("mdi6.bookshelf", color=THEME.text_muted).pixmap(34, 34))
        title = QLabel(tr("No rolls yet"))
        title.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_header}px; font-weight: {THEME.weight_semibold};")
        hint = hint_label(tr("A roll is a folder of scans, or a set of frames you saved."))
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        first = labeled_action(
            "mdi6.folder-plus-outline", tr("Import Folder as a Roll…"), tr("Make one folder of scans a roll and open it"), primary=True
        )
        first.clicked.connect(self.prompt_import_folder)
        many = labeled_action("", tr("Import Subfolders as Rolls…"), tr("Make each folder of scans under a parent a roll of its own"))
        many.clicked.connect(self.prompt_import_subfolders)
        for widget in (icon, title, hint, first, many):
            empty_col.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.empty_label)
        layout.addStretch(0)

    def prompt_import_folder(self) -> bool:
        """Import Folder as a Roll: recognize one folder and open it.

        Returns whether anything was imported, so the empty-state prompt (Library
        button with nothing in the library yet) knows whether it can proceed.
        """
        start = self.repo.get_global_setting("last_open_folder", "") or ""
        path = QFileDialog.getExistingDirectory(self, tr("Import Folder as a Roll"), start)
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
        parent = QFileDialog.getExistingDirectory(self, tr("Import Subfolders as Rolls"), start)
        if not parent:
            return False
        self.repo.save_global_setting("last_open_folder", parent)
        return self.import_subfolders(parent)

    def import_subfolders(self, parent: str) -> bool:
        """Recognize every roll folder under *parent*, without opening any of them."""
        roll_ids = self.controller.import_subfolders_as_rolls(parent)
        if not roll_ids:
            self.controller.set_status(tr("No folders with images found in “{value}”", value=folder_label(parent)), 4000)
            return False
        self.controller.set_status(
            tr("Imported {value} from “{value1}”", value=count_of(len(roll_ids), "roll"), value1=folder_label(parent)), 3000
        )
        self.reload()
        self.rolls_changed.emit()
        return True

    def edit_discovery_filters(self) -> bool:
        """Edit the folder-name filters discovery skips, one per line."""
        text, ok = QInputDialog.getMultiLineText(
            self,
            tr("Discovery Filters"),
            tr(
                "One filter per line. A folder that matches is skipped, with everything in it.\n"
                "Text matches any part of the name, ignoring case: export skips old_exports.\n"
                "* is a wildcard, and the filter must match the whole name: raw_* skips raw_01."
            ),
            "\n".join(rolls.discovery_filters(self.repo)),
        )
        if ok:
            rolls.set_discovery_filters(self.repo, text.splitlines())
            self.refresh()
        return ok

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
        self.sort_btn.setText(self.sort_btn.order_action(order).text())
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
        self.index_action.setVisible(enabled)
        self.index_action.setEnabled(enabled and semantic_model.clip_model_ready())
        # The open roll's mark follows the session, which never rebuilds the tree.
        self.tree.viewport().update()
        if self.controller.state.active_roll_id != self._shown_open_roll:
            self._show_open_roll()

    def refresh(self) -> None:
        """Refresh: rediscover rolls under the imported parents, then re-read every count."""
        found, dropped = self.controller.rediscover_rolls()
        self.reload()
        if found or dropped:
            new, gone = count_of(found, "new roll"), count_of(dropped, "filtered roll")
            if found and dropped:
                message = tr("Found {value}, removed {value1}", value=new, value1=gone)
            else:
                message = tr("Found {value}", value=new) if found else tr("Removed {value}", value=gone)
            self.controller.set_status(message, 3000)
            self.rolls_changed.emit()

    def reload(self) -> None:
        # Selecting a roll expands its folder to show it.
        self._auto_expanding = True
        try:
            self._rebuild()
        finally:
            self._auto_expanding = False
        self._show_open_roll()

    def _show_open_roll(self) -> None:
        """Open the folders above the open roll and scroll to its row."""
        roll_id = self._shown_open_roll = self.controller.state.active_roll_id
        item = next((item for item in self._roll_items() if item.data(0, _ROLL_ID_ROLE) == roll_id), None)
        if item is None:
            return
        self._auto_expanding = True
        try:
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
        finally:
            self._auto_expanding = False
        self.tree.scrollToItem(item)

    def _rebuild(self) -> None:
        selected = self._selected_roll_id()
        self.tree.clear()
        entries = self._sorted(rolls.all_rolls_sorted(self.repo))
        self.empty_label.setVisible(not entries)
        self.tree.setVisible(bool(entries))
        stock_ids = {(entry.get("defaults") or {}).get("film_stock_id") for _roll_id, entry in entries} - {None, ""}
        self._gear = GearProfiles.load_library() if stock_ids else None
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
                    folders[key].setData(0, _PREFIX_ROLE, key)
                    parent.addChild(folders[key])
                parent = folders[key]
            parent.addChild(self._make_item(roll_id, entry, parts[-1]))
        for item in folders.values():
            item.setData(0, _DETAIL_ROLE, count_of(self._leaf_count(item), "roll"))
            item.setExpanded(item.data(0, _PREFIX_ROLE) not in self._collapsed)
        if not self._collapsed <= folders.keys():
            self._collapsed &= folders.keys()
            self._save_collapsed()
        # Without folder rows there is nothing to expand, so the rows start at the edge.
        self.tree.setRootIsDecorated(bool(folders))
        self._apply_filter()
        if selected:
            self._select_roll(selected)

    def _remember_folder(self, item: QTreeWidgetItem, collapsed: bool) -> None:
        key = item.data(0, _PREFIX_ROLE)
        if self._auto_expanding or not key or (key in self._collapsed) == collapsed:
            return
        if collapsed:
            self._collapsed.add(key)
        else:
            self._collapsed.discard(key)
        self._save_collapsed()

    def _save_collapsed(self) -> None:
        self.repo.save_global_setting(_COLLAPSED_KEY, sorted(self._collapsed))

    @staticmethod
    def _make_folder_item(name: str, folder_path: str) -> QTreeWidgetItem:
        """A folder row: groups rolls, is not a roll itself, so it opens and selects nothing."""
        item = QTreeWidgetItem([name])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setData(0, _FOLDER_ROLE, folder_path)
        item.setToolTip(0, folder_path)
        item.setIcon(0, qta.icon("mdi6.folder-open-outline", color=THEME.text_secondary))
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
        item = QTreeWidgetItem([label])
        item.setData(0, _ROLL_ID_ROLE, roll_id)
        item.setData(0, _NAME_ROLE, entry.get("name", ""))
        item.setData(0, _MISSING_ROLE, missing)
        self._set_detail(item, roll_id, entry)
        # Roll kind reads off the icon's shape: a folder roll is a folder, a virtual one
        # the search it was built from.
        item.setIcon(0, qta.icon("mdi6.folder-outline" if is_folder else "mdi6.text-search", color=THEME.text_secondary))
        if missing:
            item.setToolTip(0, tr("Folder not found on disk: {path}", path=folder_path))
            item.setToolTip(1, tr("The folder is not on disk, so move it back or delete the roll."))
        else:
            item.setToolTip(0, folder_path if is_folder else tr("Built from a search or a hand-picked set of frames"))
        return item

    def _set_detail(self, item: QTreeWidgetItem, roll_id: str, entry: dict) -> None:
        """The line under a roll's name: its photos, film mode, stock, scenes and import date."""
        if item.data(0, _MISSING_ROLE):
            item.setData(0, _DETAIL_ROLE, tr("folder missing"))
            item.setData(0, _DOT_ROLE, "")
            item.setData(0, _FRAMES_ROLE, 0)
            return
        defaults = entry.get("defaults") or {}
        frames = self._frame_count(roll_id, entry)
        item.setData(0, _FRAMES_ROLE, frames)
        parts = [count_of(frames, "photo")]
        mode = defaults.get("process_mode")
        label, color = _MODES.get(ProcessMode(mode), ("", "")) if mode else ("", "")
        if label:
            parts.append(label)
        stock = self._gear.get_film_stock(defaults.get("film_stock_id", "")) if self._gear is not None else None
        if stock is not None:
            parts.append(stock.resolved_display_name)
        if entry.get("scenes"):
            parts.append(count_of(len(entry["scenes"]), "scene"))
        if entry.get("created_at"):
            parts.append(_short_date(entry["created_at"]))
        item.setData(0, _DETAIL_ROLE, " · ".join(parts))
        item.setData(0, _DOT_ROLE, color)

    def _apply_filter(self) -> None:
        """Hide the rolls the filter box rules out, a folder row with them, and say how many show."""
        needle = self.filter_input.text().strip().casefold()
        shown = 0
        rolls_ = self._roll_items()
        for item in rolls_:
            visible = not needle or needle in f"{item.text(0)} {item.data(0, _DETAIL_ROLE) or ''}".casefold()
            item.setHidden(not visible)
            shown += visible
        for item in reversed(self._folder_items()):
            item.setHidden(not any(not item.child(i).isHidden() for i in range(item.childCount())))
        total = len(rolls_)
        if needle:
            text = tr("{shown} of {total}", shown=shown, total=count_of(total, "roll"))
        else:
            frames = sum(item.data(0, _FRAMES_ROLE) or 0 for item in rolls_)
            text = f"{count_of(total, 'roll')} · {count_of(frames, 'photo')}"
        self.tally_label.setText(text)
        self.tally_label.setVisible(bool(total))

    def _frame_count(self, roll_id: str, entry: dict) -> int:
        if entry.get("kind") == "folder":
            skip = {os.path.normcase(os.path.normpath(p)) for p in entry.get("excluded_paths") or ()}
            own = [p for p in folder_image_paths(entry.get("folder_path", "")) if os.path.normcase(os.path.normpath(p)) not in skip]
            paths = [*own, *(entry.get("extra_paths") or ())]
        else:
            paths = entry.get("member_paths") or ()
        # Discovery re-attaches a remembered triplet in Trichrome Mode only.
        triplets = saved_triplets(self.repo) if self.controller.rgb_scan_mode_for_roll(roll_id) else {}
        return frame_count(paths, saved_composites(self.repo), triplets)

    def refresh_active_count(self) -> None:
        """Re-read the open roll's count: a capture or a grouping changes it under a built tree."""
        roll_id = self.controller.state.active_roll_id
        entry = rolls.roll_for_id(self.repo, roll_id) if roll_id else None
        for item in self._roll_items() if entry else ():
            if item.data(0, _ROLL_ID_ROLE) == roll_id and not item.data(0, _MISSING_ROLE):
                self._set_detail(item, roll_id, entry)
                self._apply_filter()

    def _selected_roll_id(self):
        item = self.tree.currentItem()
        return item.data(0, _ROLL_ID_ROLE) if item is not None else None

    def _select_roll(self, roll_id) -> None:
        for item in self._roll_items():
            if item.data(0, _ROLL_ID_ROLE) == roll_id:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
                return

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
            menu.addAction(tr("Delete…")).triggered.connect(lambda: self._delete_folder(item.data(0, _FOLDER_ROLE), item.text(0)))
            menu.addSeparator()
        elif item is not None and item.data(0, _ROLL_ID_ROLE) is not None:
            roll_id = item.data(0, _ROLL_ID_ROLE)
            selection = self._selected_roll_items()
            if len(selection) > 1 and roll_id in dict(selection):
                menu.addAction(tr("Delete {value} Rolls…", value=len(selection))).triggered.connect(lambda: self._delete_rolls(selection))
            else:
                name = item.data(0, _NAME_ROLE)
                menu.addAction(tr("Open")).triggered.connect(lambda: self.controller.open_roll(roll_id))
                is_active = roll_id == self.controller.state.active_roll_id
                if is_active:
                    menu.addAction(label_with_shortcut(tr("Close Roll…"), "close_roll")).triggered.connect(
                        lambda: _close_roll(self, self.controller)
                    )
                analyze_action = menu.addAction(tr("Roll Analysis"))
                analyze_action.setEnabled(is_active)
                analyze_action.setToolTip(BATCH_ANALYSIS_TOOLTIP if is_active else BATCH_ANALYSIS_DISABLED_TOOLTIP)
                analyze_action.triggered.connect(self.controller.request_batch_normalization)
                menu.addAction(tr("Rename…")).triggered.connect(lambda: self._rename_roll(roll_id, name))
                menu.addAction(tr("Delete…")).triggered.connect(lambda: self._delete_roll(roll_id, name))
            menu.addSeparator()
        menu.addAction(tr("Import Folder as a Roll…")).triggered.connect(self.prompt_import_folder)
        menu.addAction(tr("Import Subfolders as Rolls…")).triggered.connect(self.prompt_import_subfolders)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _rename_roll(self, roll_id: str, current_name: str) -> None:
        entry = rolls.roll_for_id(self.repo, roll_id)
        is_folder = bool(entry) and entry.get("kind") == "folder"
        # Rename changes the roll's own folder name; the folder rows above it stay.
        prefix = ""
        if is_folder and rolls.ROLL_PATH_SEP in current_name:
            prefix, current_name = current_name.rsplit(rolls.ROLL_PATH_SEP, 1)

        dlg = RenameRollDialog(current_name, self, folder_backed=is_folder, repo=self.repo)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, rename_folder = dlg.name(), dlg.rename_folder()

        if not name or (name == current_name and not rename_folder):
            return
        if not is_valid_preset_name(name):
            warn_invalid_roll_name(self, tr("Rename Roll"))
            return

        if rename_folder:
            if not self.controller.request_rename_roll(roll_id, name, True):
                QMessageBox.warning(
                    self,
                    tr("Rename Roll"),
                    tr("Could not rename the folder on disk. Check that no other folder has that name and that you may rename it."),
                )
                return
        rolls.rename_roll(self.repo, roll_id, f"{prefix}{rolls.ROLL_PATH_SEP}{name}" if prefix else name)

        self.reload()
        self.rolls_changed.emit()

    def _delete_roll(self, roll_id: str, name: str) -> None:
        if confirm_delete_named(self, tr("Roll"), name, informative=tr("This only forgets the roll. Nothing on disk changes.")):
            rolls.delete_roll(self.repo, roll_id)
            self.reload()
            self.rolls_changed.emit()

    def _delete_folder(self, folder_path: str, name: str) -> None:
        if confirm_delete_named(self, tr("Folder"), name, informative=tr("This forgets every roll in it. Nothing on disk changes.")):
            rolls.delete_folder_rolls(self.repo, folder_path)
            self.reload()
            self.rolls_changed.emit()

    def _delete_rolls(self, selection: list[tuple]) -> None:
        names = [name for _roll_id, name in selection]
        if confirm_delete_several(self, tr("Rolls"), names, informative=tr("This only forgets the roll records. Nothing on disk changes.")):
            for roll_id, _name in selection:
                rolls.delete_roll(self.repo, roll_id)
            self.reload()
            self.rolls_changed.emit()
