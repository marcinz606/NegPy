import os

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME
from negpy.services.assets.library import folder_counts, summarize_counts

_PATH_ROLE = Qt.ItemDataRole.UserRole
_IS_ROOT_ROLE = Qt.ItemDataRole.UserRole + 1
_MTIME_ROLE = Qt.ItemDataRole.UserRole + 2
_PLACEHOLDER = "__unpopulated__"


def _subdirectories(path: str) -> list:
    try:
        entries = [e for e in os.scandir(path) if e.is_dir() and not e.name.startswith(".")]
    except OSError:
        return []
    return entries


def _has_subdirectory(path: str) -> bool:
    try:
        return any(e.is_dir() and not e.name.startswith(".") for e in os.scandir(path))
    except OSError:
        return False


class LibraryTree(QWidget):
    """The folder tree of the user's library roots, read straight from disk.

    NegPy owns nothing here: a folder is a folder on the filesystem, so reorganizing
    in Finder needs no reconciling, and nothing in this panel ever moves a file.
    Children are read on expand — a root pointing at a large archive costs one
    directory listing, not a crawl.

    Click selects (ctrl/shift builds a set), double-click or Enter opens. Loading a
    roll is the expensive step, so it waits for the second click rather than firing
    while you are still picking which folders you meant.
    """

    folders_activated = pyqtSignal(list)  # paths to open — one folder, or a whole selection
    folders_appended = pyqtSignal(list)  # "Add to session": load without replacing
    roots_changed = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.repo = controller.session.repo
        self._sort_order = "name"
        self._sort_descending = False
        self._init_ui()
        self.reload()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # No title of its own — the section header above already names it.
        header = QHBoxLayout()
        header.setSpacing(4)
        header.addStretch(1)

        self.add_root_btn = QToolButton()
        self.add_root_btn.setIcon(qta.icon("fa5s.plus", color=THEME.text_primary))
        self.add_root_btn.setToolTip("Add a library folder")
        self.add_root_btn.setFixedSize(20, 20)
        self.add_root_btn.clicked.connect(self.add_root)

        self.refresh_btn = QToolButton()
        self.refresh_btn.setIcon(qta.icon("fa5s.sync-alt", color=THEME.text_primary))
        self.refresh_btn.setToolTip("Re-read the folders from disk")
        self.refresh_btn.setFixedSize(20, 20)
        self.refresh_btn.clicked.connect(self._on_refresh)

        header.addWidget(self.add_root_btn)
        header.addWidget(self.refresh_btn)
        layout.addLayout(header)

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
            f"QTreeWidget::item:selected {{ background: {THEME.accent_primary}; color: #FFFFFF; }}"
            f"QTreeWidget::item:hover:!selected {{ background: rgba(255, 255, 255, 18); }}"
        )
        self.tree.itemExpanded.connect(self._on_expanded)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.itemSelectionChanged.connect(self._recolour_counts)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        # Scoped shortcuts rather than an event filter: the filter object outliving (or
        # being collected before) the tree aborts Qt during teardown.
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self.tree)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(self.open_selection)
        layout.addWidget(self.tree, 1)

        self.empty_label = QLabel("Add a folder to browse your library")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px;")
        layout.addWidget(self.empty_label)

    # --- roots -------------------------------------------------------------

    def roots(self) -> list[str]:
        saved = self.repo.get_global_setting("library_roots", []) or []
        return [p for p in saved if isinstance(p, str)]

    def primary_root(self) -> str:
        return next((r for r in self.roots() if os.path.isdir(r)), "")

    def _save_roots(self, roots: list[str]) -> None:
        self.repo.save_global_setting("library_roots", roots)
        self.reload()
        self.roots_changed.emit()

    def add_root(self, path: str = "") -> str:
        """Add a library folder, asking for one when not given. Returns the path added."""
        if not path:
            start = self.repo.get_global_setting("last_open_folder", "") or ""
            path = QFileDialog.getExistingDirectory(self, "Choose your library folder", start)
        if not path:
            return ""
        roots = self.roots()
        if path not in roots:
            self._save_roots([path, *roots])
        return path

    def remove_root(self, path: str) -> None:
        self._save_roots([p for p in self.roots() if p != path])

    def _on_refresh(self) -> None:
        self.reload()
        self.roots_changed.emit()

    # --- tree --------------------------------------------------------------

    def set_sort(self, order: str, descending: bool) -> None:
        """Order folders the way the film strip orders frames."""
        if (order, descending) == (self._sort_order, self._sort_descending):
            return
        self._sort_order = order
        self._sort_descending = descending
        self.reload()

    def _sorted(self, entries: list) -> list:
        if self._sort_order == "date":
            return sorted(entries, key=lambda e: _entry_mtime(e), reverse=self._sort_descending)
        return sorted(entries, key=lambda e: e.name.lower(), reverse=self._sort_descending)

    def reload(self) -> None:
        expanded = self._expanded_paths()
        selected = self._selected_paths()
        self.tree.clear()
        roots = self.roots()
        self.empty_label.setVisible(not roots)
        for root in roots:
            self.tree.addTopLevelItem(self._make_item(root, os.path.basename(root.rstrip(os.sep)) or root, is_root=True))
        self._restore(expanded, selected)

    def _make_item(self, path: str, label: str, is_root: bool = False, mtime: float = 0.0) -> QTreeWidgetItem:
        images, subfolders = folder_counts(path)
        item = QTreeWidgetItem([label, summarize_counts(images, subfolders)])
        item.setData(0, _PATH_ROLE, path)
        item.setData(0, _IS_ROOT_ROLE, is_root)
        item.setData(0, _MTIME_ROLE, mtime)
        item.setIcon(0, qta.icon("fa5s.folder", color=THEME.text_secondary))
        item.setForeground(1, QColor(THEME.text_muted))
        item.setToolTip(0, path)
        if subfolders:
            item.addChild(QTreeWidgetItem([_PLACEHOLDER, ""]))
        return item

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1 or item.child(0).text(0) != _PLACEHOLDER:
            return
        item.takeChildren()
        for entry in self._sorted(_subdirectories(item.data(0, _PATH_ROLE))):
            item.addChild(self._make_item(entry.path, entry.name, mtime=_entry_mtime(entry)))

    def _walk(self, item: QTreeWidgetItem, visit) -> None:
        visit(item)
        for i in range(item.childCount()):
            self._walk(item.child(i), visit)

    def _each_top_level(self, visit) -> None:
        for i in range(self.tree.topLevelItemCount()):
            self._walk(self.tree.topLevelItem(i), visit)

    def _expanded_paths(self) -> set:
        found: set = set()
        self._each_top_level(lambda item: item.isExpanded() and found.add(item.data(0, _PATH_ROLE)))
        return found

    def _selected_paths(self) -> set:
        return {item.data(0, _PATH_ROLE) for item in self.tree.selectedItems()}

    def _restore(self, expanded: set, selected: set) -> None:
        if not expanded and not selected:
            return

        def walk(item: QTreeWidgetItem) -> None:
            if item.data(0, _PATH_ROLE) in expanded:
                item.setExpanded(True)  # populates through _on_expanded
            if item.data(0, _PATH_ROLE) in selected:
                item.setSelected(True)
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))

    def reveal(self, path: str) -> None:
        """Select and expand a folder, scrolling it into view.

        Walks down from its root expanding as it goes — children only exist once their
        parent has been expanded, so a deep folder cannot simply be searched for.
        """
        target = os.path.normpath(path)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            root = os.path.normpath(item.data(0, _PATH_ROLE))
            if target != root and not target.startswith(root + os.sep):
                continue
            relative = os.path.relpath(target, root)
            for part in [] if relative == "." else relative.split(os.sep):
                item.setExpanded(True)  # populates through _on_expanded
                child = next((item.child(j) for j in range(item.childCount()) if item.child(j).text(0) == part), None)
                if child is None:
                    break
                item = child
            self.tree.clearSelection()
            item.setSelected(True)
            self.tree.setCurrentItem(item)
            item.setExpanded(True)
            self.tree.scrollToItem(item)
            return

    def select_parent(self) -> None:
        """Move the selection one folder up — the tree's answer to "go up"."""
        current = self.tree.currentItem()
        parent = current.parent() if current is not None else None
        if parent is None:
            return
        self.tree.clearSelection()
        parent.setSelected(True)
        self.tree.setCurrentItem(parent)

    def _recolour_counts(self) -> None:
        # A per-item brush is out of a stylesheet's reach, so the count column has to be
        # repainted by hand or it stays grey on the accent red.
        self._each_top_level(lambda item: item.setForeground(1, QColor("#FFFFFF") if item.isSelected() else QColor(THEME.text_muted)))

    # --- opening -----------------------------------------------------------

    def selected_paths(self) -> list[str]:
        return [item.data(0, _PATH_ROLE) for item in self.tree.selectedItems() if item.data(0, _PATH_ROLE)]

    def _activate(self, item: QTreeWidgetItem) -> None:
        """Open a row: the whole selection when the row is part of one, else just it."""
        path = item.data(0, _PATH_ROLE)
        selected = self.selected_paths()
        paths = selected if path in selected and len(selected) > 1 else [path]
        self.folders_activated.emit(paths)

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        self._activate(item)

    def open_selection(self) -> None:
        """Enter: open every selected folder."""
        selected = self.selected_paths()
        if selected:
            self.folders_activated.emit(selected)

    def _show_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            path = item.data(0, _PATH_ROLE)
            selected = self.selected_paths()
            paths = selected if path in selected and len(selected) > 1 else [path]
            label = f"Open {len(paths)} folders" if len(paths) > 1 else "Open folder"
            menu.addAction(label).triggered.connect(lambda: self.folders_activated.emit(paths))
            menu.addAction("Add to session").triggered.connect(lambda: self.folders_appended.emit(paths))
            menu.addSeparator()
            if item.data(0, _IS_ROOT_ROLE):
                menu.addAction("Remove from library").triggered.connect(lambda: self.remove_root(path))
        menu.addAction("Add library folder…").triggered.connect(lambda: self.add_root())
        menu.addAction("Refresh").triggered.connect(self._on_refresh)
        menu.exec(self.tree.viewport().mapToGlobal(pos))


def _entry_mtime(entry) -> float:
    try:
        return entry.stat().st_mtime
    except OSError:
        return 0.0
