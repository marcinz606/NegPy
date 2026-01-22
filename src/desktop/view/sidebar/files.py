import os
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListView,
    QFileDialog,
    QHBoxLayout,
    QGroupBox,
)
from PyQt6.QtCore import pyqtSignal, QSize, QTimer

import qtawesome as qta
from src.desktop.controller import AppController
from src.desktop.view.styles.theme import THEME
from src.infrastructure.filesystem.watcher import FolderWatchService
from src.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from src.infrastructure.loaders.helpers import get_supported_raw_wildcards


class FileBrowser(QWidget):
    """
    Asset management panel for loading and selecting images.
    """

    file_selected = pyqtSignal(str)

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session

        self.scan_timer = QTimer(self)
        self.scan_timer.setInterval(2000)  # Check every 2 seconds
        self.scan_timer.timeout.connect(self._scan_folder)

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        action_group = QGroupBox("")
        action_layout = QVBoxLayout(action_group)

        btns_row = QHBoxLayout()
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

        btns_row.addWidget(self.add_files_btn)
        btns_row.addWidget(self.add_folder_btn)
        btns_row.addWidget(self.unload_btn)
        action_layout.addLayout(btns_row)

        self.hot_folder_btn = QPushButton(" Hot Folder Mode")
        self.hot_folder_btn.setCheckable(True)
        self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=THEME.text_primary))
        self.hot_folder_btn.setToolTip(
            "Automatically load new images from the current folder"
        )
        self._update_hot_folder_style(False)
        action_layout.addWidget(self.hot_folder_btn)

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
        self.hot_folder_btn.toggled.connect(self._on_hot_folder_toggled)

    def _on_hot_folder_toggled(self, checked: bool) -> None:
        self._update_hot_folder_style(checked)
        if checked:
            self.scan_timer.start()
        else:
            self.scan_timer.stop()

    def _update_hot_folder_style(self, checked: bool) -> None:
        if checked:
            self.hot_folder_btn.setStyleSheet(
                f"background-color: {THEME.accent_primary}; color: white; font-weight: bold;"
            )
            self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color="white"))
        else:
            self.hot_folder_btn.setStyleSheet("")
            self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=THEME.text_primary))

    def _scan_folder(self) -> None:
        if not self.session.state.uploaded_files:
            return

        # Watch directory of the most recently added file
        last_file = self.session.state.uploaded_files[-1]
        folder_path = os.path.dirname(last_file["path"])
        existing = {f["path"] for f in self.session.state.uploaded_files}

        new_files = FolderWatchService.scan_for_new_files(folder_path, existing)
        if new_files:
            self.session.add_files(new_files)
            self.controller.generate_missing_thumbnails()

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
