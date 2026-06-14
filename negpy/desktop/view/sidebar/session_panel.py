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
    Left sidebar panel containing the filmstrip file browser and update check.
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

        self.update_label = QLabel("")
        self.update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_label.setObjectName("update_label")
        self.update_label.setVisible(False)
        layout.addWidget(self.update_label)

        self.header = SidebarHeader(self.controller)
        layout.addWidget(self.header)

        self.update_worker = UpdateCheckWorker()
        self.update_worker.finished.connect(self._on_update_found)
        self.update_worker.start()

        self.file_browser = FileBrowser(self.controller)
        layout.addWidget(self.file_browser, 1)

    def _connect_signals(self) -> None:
        pass

    def _on_update_found(self, version: str) -> None:
        self.update_label.setText(f"Update Available: v{version}")
        self.update_label.setVisible(True)
