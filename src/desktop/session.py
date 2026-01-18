from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QAbstractListModel, QModelIndex, Qt
from src.domain.models import WorkspaceConfig
from src.infrastructure.storage.repository import StorageRepository


class ToolMode(Enum):
    NONE = auto()
    WB_PICK = auto()
    CROP_MANUAL = auto()
    DUST_PICK = auto()


@dataclass
class AppState:
    """
    Reactive state object for the desktop session.
    """

    current_file_path: Optional[str] = None
    current_file_hash: Optional[str] = None
    config: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    workspace_color_space: str = "Adobe RGB"
    is_processing: bool = False
    active_tool: ToolMode = ToolMode.NONE
    uploaded_files: List[Dict[str, Any]] = field(default_factory=list)
    thumbnails: Dict[str, Any] = field(
        default_factory=dict
    )  # filename -> QIcon/QPixmap
    selected_file_idx: int = -1
    active_adjustment_idx: int = 0
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    preview_raw: Optional[Any] = None
    original_res: tuple[int, int] = (0, 0)
    clipboard: Optional[WorkspaceConfig] = None

    # ICC Management
    icc_profile_path: Optional[str] = None
    icc_invert: bool = False
    apply_icc_to_export: bool = False


class AssetListModel(QAbstractListModel):
    """
    Model for the uploaded files list with thumbnail support.
    """

    def __init__(self, state: AppState):
        super().__init__()
        self._state = state

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._state.uploaded_files)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._state.uploaded_files):
            return None

        file_info = self._state.uploaded_files[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return file_info["name"]

        if role == Qt.ItemDataRole.DecorationRole:
            return self._state.thumbnails.get(file_info["name"])

        if role == Qt.ItemDataRole.ToolTipRole:
            return file_info["path"]

        return None

    def refresh(self) -> None:
        self.layoutChanged.emit()


class DesktopSessionManager(QObject):
    """
    Manages application state, file list, and configuration persistence.
    """

    state_changed = pyqtSignal()
    file_selected = pyqtSignal(str)  # Emits file path when active file changes

    def __init__(self, repo: StorageRepository):
        super().__init__()
        self.repo = repo
        self.state = AppState()
        self.asset_model = AssetListModel(self.state)

    def select_file(self, index: int) -> None:
        """
        Changes active file and hydrates state from repository.
        """
        if 0 <= index < len(self.state.uploaded_files):
            # Save current before switching
            if self.state.current_file_hash:
                self.repo.save_file_settings(
                    self.state.current_file_hash, self.state.config
                )

            file_info = self.state.uploaded_files[index]
            self.state.selected_file_idx = index
            self.state.current_file_path = file_info["path"]
            self.state.current_file_hash = file_info["hash"]

            # Load settings for new file
            saved_config = self.repo.load_file_settings(file_info["hash"])
            self.state.config = saved_config or WorkspaceConfig()

            self.file_selected.emit(file_info["path"])
            self.state_changed.emit()

    def next_file(self) -> None:
        if self.state.selected_file_idx < len(self.state.uploaded_files) - 1:
            self.select_file(self.state.selected_file_idx + 1)

    def prev_file(self) -> None:
        if self.state.selected_file_idx > 0:
            self.select_file(self.state.selected_file_idx - 1)

    def update_config(self, config: WorkspaceConfig, persist: bool = True) -> None:
        """
        Updates global config and optionally saves to disk.
        """
        self.state.config = config
        if persist and self.state.current_file_hash:
            self.repo.save_file_settings(self.state.current_file_hash, config)
        self.state_changed.emit()

    def reset_settings(self) -> None:
        """
        Reverts current file to default configuration.
        """
        self.update_config(WorkspaceConfig())

    def copy_settings(self) -> None:
        import copy

        self.state.clipboard = copy.deepcopy(self.state.config)
        self.state_changed.emit()

    def paste_settings(self) -> None:
        if self.state.clipboard:
            import copy

            self.update_config(copy.deepcopy(self.state.clipboard))

    def add_files(self, file_paths: List[str]) -> None:
        """
        Adds new files to the session, calculating hashes and updating the model.
        """
        import os
        from src.kernel.image.logic import calculate_file_hash

        for path in file_paths:
            f_hash = calculate_file_hash(path)
            # Avoid duplicates
            if any(f["hash"] == f_hash for f in self.state.uploaded_files):
                continue

            self.state.uploaded_files.append(
                {"name": os.path.basename(path), "path": path, "hash": f_hash}
            )

        self.asset_model.refresh()
        self.state_changed.emit()

    def clear_files(self) -> None:
        """
        Purges all loaded files from the session.
        """
        self.state.uploaded_files.clear()
        self.state.thumbnails.clear()
        self.state.selected_file_idx = -1
        self.state.current_file_path = None
        self.state.current_file_hash = None
        self.state.config = WorkspaceConfig()

        self.asset_model.refresh()
        self.state_changed.emit()
