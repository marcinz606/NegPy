from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal, Qt, QThread
from negpy.desktop.controller import AppController
from negpy.desktop.view.sidebar.header import SidebarHeader
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.kernel.system.version import check_for_updates


class UpdateCheckWorker(QThread):
    """Background worker to check for new releases."""

    finished = pyqtSignal(str)

    def run(self):
        new_ver = check_for_updates()
        if new_ver:
            self.finished.emit(new_ver)


class SessionPanel(QWidget):
    """
    Left sidebar panel containing the film strip (which holds the library folder
    tree) and the update check.
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

        self.file_browser = FileBrowser(self.controller)
        self.library_tree = self.file_browser.library_tree
        layout.addWidget(self.file_browser, 1)

        # Hidden until the user has a library: an empty section is just clutter.
        self.file_browser.library_section.setVisible(bool(self.controller.library_roots()))

    def _connect_signals(self) -> None:
        # The tree navigates, the film strip loads: browsing folders is free and stays
        # with the tree, while the prompt and the frames belong to the strip.
        self.library_tree.folders_activated.connect(self.file_browser.load_folders)
        self.library_tree.folders_appended.connect(lambda paths: self.file_browser.load_folders(paths, add_to_session=True))
        self.library_tree.roots_changed.connect(self._on_roots_changed)
        self.file_browser.library_requested.connect(self.show_library)
        self.file_browser.browse_requested.connect(self.library_tree.reveal)
        self.file_browser.sort_changed.connect(lambda: self.library_tree.set_sort(*self.file_browser.sort_choice()))
        self.controller.library_cleared.connect(self._on_library_cleared)

    def show_library(self, ask_if_unset: bool = True) -> None:
        """Reveal the library's primary folder in the tree, asking for one if unset.

        The panel's resting state: with nothing loaded there is nothing else to show,
        and a list of rolls beats a blank sheet.
        """
        primary = self.library_tree.primary_root()
        if not primary:
            if not ask_if_unset or not self.library_tree.add_root():
                return
            primary = self.library_tree.primary_root()
        self.file_browser.library_section.setVisible(True)
        self.file_browser.library_section.expand()
        self.library_tree.reveal(primary)

    def browse_parent(self) -> None:
        """Alt+Up: move the tree's selection one folder up."""
        self.library_tree.select_parent()

    def _on_roots_changed(self) -> None:
        # Folders moved or a root was added — the cached walk describes a tree that
        # no longer exists.
        self.library_tree.reload()
        self.controller.invalidate_library_walk()
        if self.controller.library_roots():
            self.file_browser.library_section.setVisible(True)

    def _on_library_cleared(self) -> None:
        self._on_roots_changed()
        self.file_browser.library_section.setVisible(False)

    def toggle_library_tree(self) -> None:
        """Fold the folder section away, or bring it back."""
        section = self.file_browser.library_section
        section.toggle_button.setChecked(not section.toggle_button.isChecked())

    def _on_update_found(self, version: str) -> None:
        self.update_label.setText(
            '<a href="https://github.com/marcinz606/NegPy/releases" '
            'style="color:#558B2F; text-decoration:none;">'
            f"⬇ Update Available: v{version}</a>"
        )
        self.update_label.setVisible(True)
