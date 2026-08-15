from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from negpy.desktop.controller import AppController
from negpy.desktop.view.sidebar.header import SidebarHeader
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.update_dialog import UpdateDialog, start_update_check
from negpy.kernel.system.updater import UpdateInfo
from negpy.kernel.system.version import get_app_version


class SessionPanel(QWidget):
    """
    Left sidebar panel containing the film strip (which holds the library folder
    tree) and the update check.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.update_info: Optional[UpdateInfo] = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        repo = self.controller.session.repo
        persisted = repo.get_global_setting("section_expanded_app_header")
        self.header = SidebarHeader(self.controller, expanded=bool(persisted) if persisted is not None else True)
        self.header.expanded_changed.connect(lambda v: repo.save_global_setting("section_expanded_app_header", v))
        layout.addWidget(self.header)

        self.update_label = QLabel("")
        self.update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_label.setObjectName("update_label")
        self.update_label.setOpenExternalLinks(False)
        self.update_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.update_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_label.setVisible(False)
        self.update_label.linkActivated.connect(lambda _href: self.show_update_dialog())
        layout.addWidget(self.update_label)

        start_update_check(self._on_update_checked)

        self.file_browser = FileBrowser(self.controller)
        self.library_tree = self.file_browser.library_tree
        layout.addWidget(self.file_browser, 1)

        # Hidden until the user has a library: an empty section is just clutter.
        self.file_browser.library_section.setVisible(bool(self.controller.library_roots()))

    def _connect_signals(self) -> None:
        # The tree navigates and the film strip loads: browsing folders is free and stays with the
        # tree, while the prompt and the frames belong to the strip.
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
        # Folders moved or a root was added, so the cached walk describes a tree that no longer
        # exists.
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

    def _on_update_checked(self, info: Optional[UpdateInfo]) -> None:
        if info is None:
            return
        self.update_info = info
        self.update_label.setText(
            f'<a href="#update" style="color:{THEME.status_success}; text-decoration:none;">⬇ Update Available: v{info.version}</a>'
        )
        self.update_label.setToolTip(
            "Install this update — NegPy downloads it, closes, and reopens on the new version"
            if info.can_self_install
            else "See what is new and where to download it"
        )
        self.update_label.setVisible(True)

    def show_update_dialog(self) -> None:
        """Open the update window. Silent while the startup check has found nothing."""
        if self.update_info is None:
            return
        UpdateDialog(self.update_info, self).exec()

    def check_for_updates(self) -> None:
        """Re-run the release check on demand and report either way."""
        if self.update_info is not None:
            self.show_update_dialog()
            return
        start_update_check(self._on_manual_check)

    def _on_manual_check(self, info: Optional[UpdateInfo]) -> None:
        if info is None:
            QMessageBox.information(self, "NegPy", f"NegPy {get_app_version()} is up to date.")
            return
        self._on_update_checked(info)
        self.show_update_dialog()
