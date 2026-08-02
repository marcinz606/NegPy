import os

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME

_PATH_ROLE = Qt.ItemDataRole.UserRole
_IS_ROOT_ROLE = Qt.ItemDataRole.UserRole + 1
_PLACEHOLDER = "__unpopulated__"


def _subdirectories(path: str) -> list[os.DirEntry]:
    try:
        entries = [e for e in os.scandir(path) if e.is_dir() and not e.name.startswith(".")]
    except OSError:
        return []
    return sorted(entries, key=lambda e: e.name.lower())


def _has_subdirectory(path: str) -> bool:
    try:
        return any(e.is_dir() and not e.name.startswith(".") for e in os.scandir(path))
    except OSError:
        return False


class LibraryTree(QWidget):
    """The folder tree of the user's library roots, read straight from disk.

    NegPy owns nothing here: a folder is a folder on the filesystem, so reorganizing
    in Finder needs no reconciling, and nothing in this panel ever moves a file.
    Children are scanned on expand — a root pointing at a large archive costs one
    directory listing, not a crawl.
    """

    folder_opened = pyqtSignal(str, bool)  # (folder path, add to session rather than replace)
    roots_changed = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.repo = controller.session.repo
        self._init_ui()
        self.reload()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(4)
        title = QLabel("Library")
        title.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 10px; font-weight: bold;")
        header.addWidget(title)
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
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.itemExpanded.connect(self._on_expanded)
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
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

    def _save_roots(self, roots: list[str]) -> None:
        self.repo.save_global_setting("library_roots", roots)
        self.reload()
        self.roots_changed.emit()

    def add_root(self) -> None:
        start = self.repo.get_global_setting("last_open_folder", "") or ""
        folder = QFileDialog.getExistingDirectory(self, "Add Library Folder", start)
        if not folder:
            return
        roots = self.roots()
        if folder not in roots:
            self._save_roots([*roots, folder])

    def remove_root(self, path: str) -> None:
        self._save_roots([p for p in self.roots() if p != path])

    def _on_refresh(self) -> None:
        self.reload()
        self.roots_changed.emit()

    # --- tree --------------------------------------------------------------

    def reload(self) -> None:
        expanded = self._expanded_paths()
        self.tree.clear()
        roots = self.roots()
        self.empty_label.setVisible(not roots)
        for root in roots:
            item = self._make_item(root, os.path.basename(root.rstrip(os.sep)) or root, is_root=True)
            self.tree.addTopLevelItem(item)
        self._restore_expanded(expanded)

    def _make_item(self, path: str, label: str, is_root: bool = False) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label])
        item.setData(0, _PATH_ROLE, path)
        item.setData(0, _IS_ROOT_ROLE, is_root)
        item.setIcon(0, qta.icon("fa5s.folder", color=THEME.text_secondary))
        item.setToolTip(0, path)
        if _has_subdirectory(path):
            item.addChild(QTreeWidgetItem([_PLACEHOLDER]))
        return item

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1 or item.child(0).text(0) != _PLACEHOLDER:
            return
        item.takeChildren()
        for entry in _subdirectories(item.data(0, _PATH_ROLE)):
            item.addChild(self._make_item(entry.path, entry.name))

    def _expanded_paths(self) -> set[str]:
        found: set[str] = set()

        def walk(item: QTreeWidgetItem) -> None:
            if item.isExpanded():
                found.add(item.data(0, _PATH_ROLE))
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        return found

    def _restore_expanded(self, paths: set[str]) -> None:
        if not paths:
            return

        def walk(item: QTreeWidgetItem) -> None:
            if item.data(0, _PATH_ROLE) in paths:
                item.setExpanded(True)  # populates through _on_expanded
                for i in range(item.childCount()):
                    walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))

    def _on_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        self.folder_opened.emit(item.data(0, _PATH_ROLE), False)

    def _show_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            path = item.data(0, _PATH_ROLE)
            menu.addAction("Open folder").triggered.connect(lambda: self.folder_opened.emit(path, False))
            menu.addAction("Add to session").triggered.connect(lambda: self.folder_opened.emit(path, True))
            menu.addSeparator()
            if item.data(0, _IS_ROOT_ROLE):
                menu.addAction("Remove from library").triggered.connect(lambda: self.remove_root(path))
        menu.addAction("Add library folder…").triggered.connect(self.add_root)
        menu.addAction("Refresh").triggered.connect(self._on_refresh)
        menu.exec(self.tree.viewport().mapToGlobal(pos))
