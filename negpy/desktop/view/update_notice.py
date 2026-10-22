from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

from negpy.desktop.view.widgets.update_dialog import UpdateDialog, start_update_check
from negpy.kernel.system.i18n import tr
from negpy.kernel.system.updater import UpdateInfo
from negpy.kernel.system.version import get_app_version


class UpdateNotice(QObject):
    """The release check at startup and on demand; the toolbar's ⋯ menu shows what it finds."""

    update_found = pyqtSignal(str)

    def __init__(self, window: QWidget, repo) -> None:
        super().__init__(window)
        self._window = window
        self._repo = repo
        self.update_info: Optional[UpdateInfo] = None
        self._checking = False
        start_update_check(self._on_update_checked)

    def _on_update_checked(self, info: Optional[UpdateInfo]) -> None:
        if info is None:
            return
        self.update_info = info
        self.update_found.emit(info.version)

    def show_update_dialog(self) -> None:
        """Open the update window. Silent while the startup check has found nothing."""
        if self.update_info is None:
            return
        UpdateDialog(self.update_info, self._window, repo=self._repo).exec()

    def check_for_updates(self) -> None:
        """Re-run the release check on demand and report either way; one check at a time."""
        if self.update_info is not None:
            self.show_update_dialog()
            return
        if self._checking:
            return
        self._checking = True
        start_update_check(self._on_manual_check)

    def _on_manual_check(self, info: Optional[UpdateInfo]) -> None:
        self._checking = False
        if info is None:
            QMessageBox.information(self._window, tr("Check for Updates"), tr("NegPy {value} is up to date.", value=get_app_version()))
            return
        self._on_update_checked(info)
        self.show_update_dialog()
