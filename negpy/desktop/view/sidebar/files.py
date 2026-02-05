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
from PyQt6.QtCore import pyqtSignal, QSize, QTimer, QItemSelectionModel, Qt

import qtawesome as qta
from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards


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
        self.scan_timer.setInterval(2000)
        self.scan_timer.timeout.connect(self._scan_folder)

        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(200)
        self.selection_timer.timeout.connect(self._commit_selection)

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
        self.add_files_btn.setIcon(qta.icon("fa5s.file-import", color=THEME.text_primary))
        self.add_folder_btn = QPushButton(" Folder")
        self.add_folder_btn.setIcon(qta.icon("fa5s.folder-plus", color=THEME.text_primary))
        self.unload_btn = QPushButton(" Clear")
        self.unload_btn.setIcon(qta.icon("fa5s.times-circle", color=THEME.text_primary))

        btns_row.addWidget(self.add_files_btn)
        btns_row.addWidget(self.add_folder_btn)
        btns_row.addWidget(self.unload_btn)
        action_layout.addLayout(btns_row)

        hot_sync_row = QHBoxLayout()
        self.hot_folder_btn = QPushButton(" Hot Folder Mode")
        self.hot_folder_btn.setCheckable(True)
        self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=THEME.text_primary))
        self.hot_folder_btn.setToolTip("Automatically load new images from the current folder")
        self._update_hot_folder_style(False)

        self.sync_btn = QPushButton(" Sync Edits")
        self.sync_btn.setIcon(qta.icon("fa5s.sync", color=THEME.text_primary))
        self.sync_btn.setToolTip("Apply current settings to all selected images (excluding crop/rotation)")

        hot_sync_row.addWidget(self.hot_folder_btn)
        hot_sync_row.addWidget(self.sync_btn)
        action_layout.addLayout(hot_sync_row)

        layout.addWidget(action_group)

        self.list_view = QListView()
        self.list_view.setModel(self.session.asset_model)
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
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
        self.list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.hot_folder_btn.toggled.connect(self._on_hot_folder_toggled)
        self.sync_btn.clicked.connect(self.session.sync_selected_settings)
        self.session.state_changed.connect(self.sync_ui)

    def sync_ui(self) -> None:
        """Updates list selection to match session state."""
        selection_model = self.list_view.selectionModel()
        current_indices = {idx.row() for idx in selection_model.selectedIndexes()}
        target_indices = set(self.session.state.selected_indices)

        if current_indices == target_indices:
            active_idx = self.session.state.selected_file_idx
            if active_idx >= 0:
                qt_idx = self.session.asset_model.index(active_idx, 0)
                if self.list_view.currentIndex() != qt_idx:
                    self.list_view.setCurrentIndex(qt_idx)
            return

        selection_model.blockSignals(True)
        try:
            selection_model.clearSelection()
            for idx in self.session.state.selected_indices:
                qt_idx = self.session.asset_model.index(idx, 0)
                selection_model.select(qt_idx, QItemSelectionModel.SelectionFlag.Select)

            active_idx = self.session.state.selected_file_idx
            if active_idx >= 0:
                qt_idx = self.session.asset_model.index(active_idx, 0)
                self.list_view.setCurrentIndex(qt_idx)
                self.list_view.scrollTo(qt_idx)
        finally:
            selection_model.blockSignals(False)

    def _on_selection_changed(self, selected, deselected) -> None:
        self.selection_timer.start()

    def _commit_selection(self) -> None:
        """Sends current UI selection to the session after debounce."""
        indices = [idx.row() for idx in self.list_view.selectionModel().selectedIndexes()]
        if set(indices) != set(self.session.state.selected_indices):
            self.session.update_selection(indices)

    def _on_hot_folder_toggled(self, checked: bool) -> None:
        self._update_hot_folder_style(checked)
        if checked:
            self.scan_timer.start()
        else:
            self.scan_timer.stop()

    def _update_hot_folder_style(self, checked: bool) -> None:
        if checked:
            self.hot_folder_btn.setStyleSheet(f"background-color: {THEME.accent_primary}; color: white; font-weight: bold;")
            self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color="white"))
        else:
            self.hot_folder_btn.setStyleSheet("")
            self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=THEME.text_primary))

    def _scan_folder(self) -> None:
        if not self.session.state.uploaded_files:
            return

        last_file = self.session.state.uploaded_files[-1]
        folder_path = os.path.dirname(last_file["path"])
        existing = {f["path"] for f in self.session.state.uploaded_files}

        new_files = FolderWatchService.scan_for_new_files(folder_path, existing)
        if new_files:
            self.controller.request_asset_discovery(new_files)

    def _on_add_files(self) -> None:
        wildcards = get_supported_raw_wildcards()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            f"Supported Images ({wildcards})",
        )
        if files:
            self.controller.request_asset_discovery(files)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.controller.request_asset_discovery([folder])

    def _on_item_clicked(self, index) -> None:
        from PyQt6.QtWidgets import QApplication

        modifiers = QApplication.keyboardModifiers()
        indices = [idx.row() for idx in self.list_view.selectionModel().selectedIndexes()]

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.session.update_selection(indices)
        else:
            self.session.select_file(index.row(), selection_override=indices)
