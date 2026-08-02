from PyQt6.QtWidgets import (
    QWidget,
    QSplitter,
    QVBoxLayout,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal, Qt, QThread
from negpy.desktop.controller import AppController
from negpy.desktop.view.sidebar.header import SidebarHeader
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.desktop.view.sidebar.library_tree import LibraryTree
from negpy.kernel.system.version import check_for_updates

_TREE_DEFAULT_HEIGHT = 180


class UpdateCheckWorker(QThread):
    """Background worker to check for new releases."""

    finished = pyqtSignal(str)

    def run(self):
        new_ver = check_for_updates()
        if new_ver:
            self.finished.emit(new_ver)


class SessionPanel(QWidget):
    """
    Left sidebar panel containing the library folder tree, the filmstrip file
    browser and the update check.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = SidebarHeader(self.controller)
        layout.addWidget(self.header)

        self.update_label = QLabel("")
        self.update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_label.setObjectName("update_label")
        self.update_label.setOpenExternalLinks(True)
        self.update_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.update_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_label.setVisible(False)
        layout.addWidget(self.update_label)

        self.update_worker = UpdateCheckWorker()
        self.update_worker.finished.connect(self._on_update_found)
        self.update_worker.start()

        self.library_tree = LibraryTree(self.controller)
        self.file_browser = FileBrowser(self.controller)

        # A splitter rather than a fixed stack: the tree is a navigator, and how much
        # of the panel it deserves depends entirely on how deep the user's folders go.
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.library_tree)
        self.splitter.addWidget(self.file_browser)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(True)
        layout.addWidget(self.splitter, 1)

        self._splitter_restored = False
        # Hidden until the user has a library: an empty tree is just a taller panel.
        self.library_tree.setVisible(bool(self.controller.library_roots()))

    def showEvent(self, event) -> None:
        # Sizes are restored here, not in the constructor: QSplitter scales whatever it
        # is given to the space it actually has, and before the first show that is zero.
        super().showEvent(event)
        if not self._splitter_restored:
            self._splitter_restored = True
            self._restore_splitter()

    def _restore_splitter(self) -> None:
        saved = self.controller.session.repo.get_global_setting("library_tree_height")
        height = int(saved) if saved is not None else _TREE_DEFAULT_HEIGHT
        self.splitter.setSizes([max(0, height), max(1, self.splitter.height() - height)])

    def _connect_signals(self) -> None:
        # The tree browses; the film strip owns the prompt and the folder tiles, so a
        # folder entered from the tree and one entered from the sheet behave the same.
        self.library_tree.folder_opened.connect(self.file_browser.browse_folder)
        self.library_tree.roots_changed.connect(self._on_roots_changed)
        self.splitter.splitterMoved.connect(self._save_splitter)

    def _on_roots_changed(self) -> None:
        # Folders moved or a root was added — the cached walk describes a tree that
        # no longer exists.
        self.controller.invalidate_library_walk()
        if self.controller.library_roots():
            self.library_tree.setVisible(True)

    def _save_splitter(self, *_args) -> None:
        self.controller.session.repo.save_global_setting("library_tree_height", self.splitter.sizes()[0])

    def toggle_library_tree(self) -> None:
        """Show/hide the folder tree; adding a root reveals it again.

        Keyed off isHidden (explicitly hidden) rather than isVisible (also false when
        the whole panel is docked away), so the toggle still flips while the session
        panel is closed instead of always resolving to 'show'."""
        self.library_tree.setVisible(self.library_tree.isHidden())

    def _on_update_found(self, version: str) -> None:
        self.update_label.setText(
            '<a href="https://github.com/marcinz606/NegPy/releases" '
            'style="color:#558B2F; text-decoration:none;">'
            f"⬇ Update Available: v{version}</a>"
        )
        self.update_label.setVisible(True)
