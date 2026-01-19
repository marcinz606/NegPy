import os
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListView,
    QFileDialog,
    QHBoxLayout,
    QGroupBox,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal, QSize, Qt, QThread
from PyQt6.QtGui import QPixmap

import qtawesome as qta
from src.desktop.controller import AppController
from src.desktop.view.styles.theme import THEME
from src.kernel.system.paths import get_resource_path
from src.kernel.system.version import get_app_version, check_for_updates
from src.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from src.infrastructure.loaders.helpers import get_supported_raw_wildcards


class UpdateCheckWorker(QThread):
    """Background worker to check for new releases."""

    finished = pyqtSignal(str)

    def run(self):
        new_ver = check_for_updates()
        if new_ver:
            self.finished.emit(new_ver)


class FilesSidebar(QWidget):
    """
    Asset management panel for loading and selecting images.
    """

    file_selected = pyqtSignal(str)

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        header_container = QVBoxLayout()
        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel()
        icon_pix = QPixmap(get_resource_path("media/icons/icon.png"))
        if not icon_pix.isNull():
            icon_label.setPixmap(
                icon_pix.scaled(
                    32,
                    32,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        name_label = QLabel("NegPy")
        name_label.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #eee; margin-left: 5px;"
        )

        header.addWidget(icon_label)
        header.addWidget(name_label)
        header_container.addLayout(header)

        self.ver_label = QLabel(f"v{get_app_version()}")
        self.ver_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ver_label.setStyleSheet("font-size: 18px; color: #888; margin-top: -5px;")
        header_container.addWidget(self.ver_label)

        self.update_label = QLabel("")
        self.update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_label.setStyleSheet(
            "font-size: 11px; color: #2e7d32; font-weight: bold;"
        )
        self.update_label.setVisible(False)
        header_container.addWidget(self.update_label)

        layout.addLayout(header_container)

        self.update_worker = UpdateCheckWorker()
        self.update_worker.finished.connect(self._on_update_found)
        self.update_worker.start()

        action_group = QGroupBox("")
        action_layout = QHBoxLayout(action_group)

        self.add_files_btn = QPushButton(" File")
        self.add_files_btn.setIcon(
            qta.icon("fa5s.file-import", color=THEME.text_primary)
        )
        self.add_folder_btn = QPushButton(" Folder")
        self.add_folder_btn.setIcon(
            qta.icon("fa5s.folder-plus", color=THEME.text_primary)
        )
        self.unload_btn = QPushButton(" Clear")
        self.unload_btn.setIcon(qta.icon("fa5s.times-circle", color=THEME.text_primary))

        action_layout.addWidget(self.add_files_btn)
        action_layout.addWidget(self.add_folder_btn)
        action_layout.addWidget(self.unload_btn)


        layout.addWidget(action_group)


        self.list_view = QListView()
        self.list_view.setModel(self.session.asset_model)
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setIconSize(QSize(100, 100))
        self.list_view.setGridSize(QSize(120, 130))
        self.list_view.setSpacing(10)
        self.list_view.setWordWrap(True)
        self.list_view.setAlternatingRowColors(False)
        self.list_view.setStyleSheet(
            "QListView::item { border: 1px solid #333; border-radius: 4px; padding: 5px; } "
            "QListView::item:selected { background-color: #094771; border: 1px solid #007acc; }"
        )

        layout.addWidget(self.list_view)

    def _connect_signals(self) -> None:
        self.add_files_btn.clicked.connect(self._on_add_files)
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        self.unload_btn.clicked.connect(self.session.clear_files)
        self.list_view.clicked.connect(self._on_item_clicked)

    def _on_add_files(self) -> None:
        wildcards = get_supported_raw_wildcards()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            f"Supported Images ({wildcards})",
        )
        if files:
            self.session.add_files(files)
            self.controller.generate_missing_thumbnails()

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            valid_exts = tuple(SUPPORTED_RAW_EXTENSIONS)
            paths = []
            for f in os.listdir(folder):
                if f.lower().endswith(valid_exts):
                    paths.append(os.path.join(folder, f))

            if paths:
                self.session.add_files(paths)
                self.controller.generate_missing_thumbnails()

    def _on_item_clicked(self, index) -> None:
        self.session.select_file(index.row())

    def _on_update_found(self, version: str) -> None:
        self.update_label.setText(f"Update Available: v{version}")
        self.update_label.setVisible(True)
