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

    # Hardware Acceleration
    gpu_enabled: bool = True


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
    settings_saved = pyqtSignal()
    file_selected = pyqtSignal(str)  # Emits file path when active file changes

    def __init__(self, repo: StorageRepository):
        super().__init__()
        self.repo = repo
        self.state = AppState()
        self.asset_model = AssetListModel(self.state)

        # Load global hardware settings
        saved_gpu = self.repo.get_global_setting("gpu_enabled")
        if saved_gpu is not None:
            self.state.gpu_enabled = bool(saved_gpu)

    def set_gpu_enabled(self, enabled: bool) -> None:
        """Updates and persists the hardware acceleration preference."""
        if self.state.gpu_enabled != enabled:
            self.state.gpu_enabled = enabled
            self.repo.save_global_setting("gpu_enabled", enabled)
            self.state_changed.emit()

    def _apply_sticky_settings(
        self, config: WorkspaceConfig, only_global: bool = False
    ) -> WorkspaceConfig:
        """
        Overlays globally persisted settings onto the config.
        If only_global is True, only non-look settings (Export) are applied.
        """
        from dataclasses import replace
        from src.domain.models import (
            ExportConfig,
            LabConfig,
            ToningConfig,
            RetouchConfig,
        )

        # --- Global Infrastructure Settings (Always Applied) ---
        sticky_export = self.repo.get_global_setting("last_export_config")
        if sticky_export:
            valid_keys = ExportConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in sticky_export.items() if k in valid_keys}
            new_export = ExportConfig(**filtered)
            config = replace(config, export=new_export)

        if only_global:
            return config

        # --- Look & Style Settings (Applied to new files) ---

        # 1. Process Mode
        sticky_mode = self.repo.get_global_setting("last_process_mode")
        if sticky_mode:
            config = replace(config, process_mode=sticky_mode)

        # 2. Analysis Buffer, Density, Grade, CMY, Toe, Shoulder
        sticky_buffer = self.repo.get_global_setting("last_analysis_buffer")
        sticky_density = self.repo.get_global_setting("last_density")
        sticky_grade = self.repo.get_global_setting("last_grade")
        sticky_cyan = self.repo.get_global_setting("last_wb_cyan")
        sticky_magenta = self.repo.get_global_setting("last_wb_magenta")
        sticky_yellow = self.repo.get_global_setting("last_wb_yellow")
        sticky_camera_wb = self.repo.get_global_setting("last_use_camera_wb")

        sticky_toe = self.repo.get_global_setting("last_toe")
        sticky_toe_w = self.repo.get_global_setting("last_toe_width")
        sticky_toe_h = self.repo.get_global_setting("last_toe_hardness")
        sticky_shoulder = self.repo.get_global_setting("last_shoulder")
        sticky_shoulder_w = self.repo.get_global_setting("last_shoulder_width")
        sticky_shoulder_h = self.repo.get_global_setting("last_shoulder_hardness")

        new_exp = config.exposure
        if sticky_buffer is not None:
            new_exp = replace(new_exp, analysis_buffer=float(sticky_buffer))
        if sticky_density is not None:
            new_exp = replace(new_exp, density=float(sticky_density))
        if sticky_grade is not None:
            new_exp = replace(new_exp, grade=float(sticky_grade))
        if sticky_cyan is not None:
            new_exp = replace(new_exp, wb_cyan=float(sticky_cyan))
        if sticky_magenta is not None:
            new_exp = replace(new_exp, wb_magenta=float(sticky_magenta))
        if sticky_yellow is not None:
            new_exp = replace(new_exp, wb_yellow=float(sticky_yellow))
        if sticky_camera_wb is not None:
            new_exp = replace(new_exp, use_camera_wb=bool(sticky_camera_wb))

        if sticky_toe is not None:
            new_exp = replace(new_exp, toe=float(sticky_toe))
        if sticky_toe_w is not None:
            new_exp = replace(new_exp, toe_width=float(sticky_toe_w))
        if sticky_toe_h is not None:
            new_exp = replace(new_exp, toe_hardness=float(sticky_toe_h))
        if sticky_shoulder is not None:
            new_exp = replace(new_exp, shoulder=float(sticky_shoulder))
        if sticky_shoulder_w is not None:
            new_exp = replace(new_exp, shoulder_width=float(sticky_shoulder_w))
        if sticky_shoulder_h is not None:
            new_exp = replace(new_exp, shoulder_hardness=float(sticky_shoulder_h))

        config = replace(config, exposure=new_exp)
        # 3. Aspect Ratio & Offset
        sticky_ratio = self.repo.get_global_setting("last_aspect_ratio")
        sticky_offset = self.repo.get_global_setting("last_autocrop_offset")

        new_geo = config.geometry
        if sticky_ratio:
            new_geo = replace(new_geo, autocrop_ratio=sticky_ratio)
        if sticky_offset is not None:
            new_geo = replace(new_geo, autocrop_offset=int(sticky_offset))

        config = replace(config, geometry=new_geo)

        # 4. Lab Settings
        sticky_lab = self.repo.get_global_setting("last_lab_config")
        if sticky_lab:
            valid_keys = LabConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in sticky_lab.items() if k in valid_keys}
            config = replace(config, lab=LabConfig(**filtered))

        # 5. Toning Settings
        sticky_toning = self.repo.get_global_setting("last_toning_config")
        if sticky_toning:
            valid_keys = ToningConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in sticky_toning.items() if k in valid_keys}
            config = replace(config, toning=ToningConfig(**filtered))

        # 6. Retouch Settings
        sticky_retouch = self.repo.get_global_setting("last_retouch_config")
        if sticky_retouch:
            valid_keys = RetouchConfig.__dataclass_fields__.keys()
            # Never carry over manual spots to other files
            filtered = {
                k: v
                for k, v in sticky_retouch.items()
                if k in valid_keys and k != "manual_dust_spots"
            }
            config = replace(config, retouch=replace(config.retouch, **filtered))

        return config

    def _persist_sticky_settings(self, config: WorkspaceConfig) -> None:
        """
        Saves current Mode, Ratio, and Export settings to global storage.
        """
        from dataclasses import asdict

        self.repo.save_global_setting("last_process_mode", config.process_mode)
        self.repo.save_global_setting(
            "last_analysis_buffer", config.exposure.analysis_buffer
        )
        self.repo.save_global_setting("last_density", config.exposure.density)
        self.repo.save_global_setting("last_grade", config.exposure.grade)
        self.repo.save_global_setting("last_wb_cyan", config.exposure.wb_cyan)
        self.repo.save_global_setting("last_wb_magenta", config.exposure.wb_magenta)
        self.repo.save_global_setting("last_wb_yellow", config.exposure.wb_yellow)
        self.repo.save_global_setting(
            "last_use_camera_wb", config.exposure.use_camera_wb
        )

        self.repo.save_global_setting("last_toe", config.exposure.toe)
        self.repo.save_global_setting("last_toe_width", config.exposure.toe_width)
        self.repo.save_global_setting("last_toe_hardness", config.exposure.toe_hardness)
        self.repo.save_global_setting("last_shoulder", config.exposure.shoulder)
        self.repo.save_global_setting(
            "last_shoulder_width", config.exposure.shoulder_width
        )
        self.repo.save_global_setting(
            "last_shoulder_hardness", config.exposure.shoulder_hardness
        )

        self.repo.save_global_setting(
            "last_aspect_ratio", config.geometry.autocrop_ratio
        )
        self.repo.save_global_setting(
            "last_autocrop_offset", config.geometry.autocrop_offset
        )
        self.repo.save_global_setting("last_export_config", asdict(config.export))
        self.repo.save_global_setting("last_lab_config", asdict(config.lab))
        self.repo.save_global_setting("last_toning_config", asdict(config.toning))
        self.repo.save_global_setting("last_retouch_config", asdict(config.retouch))

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
                # No settings_saved emit here to avoid global refresh,
                # but we want the controller to update the icon
                self.settings_saved.emit()

            file_info = self.state.uploaded_files[index]
            self.state.selected_file_idx = index
            self.state.current_file_path = file_info["path"]
            self.state.current_file_hash = file_info["hash"]

            # Load settings for new file
            saved_config = self.repo.load_file_settings(file_info["hash"])

            if saved_config:
                # If we have saved config, use it but sync global export settings
                self.state.config = self._apply_sticky_settings(
                    saved_config, only_global=True
                )
            else:
                # If no saved config, use defaults and apply ALL sticky look settings
                self.state.config = self._apply_sticky_settings(
                    WorkspaceConfig(), only_global=False
                )

            self.file_selected.emit(file_info["path"])
            self.state_changed.emit()

    def next_file(self) -> None:
        if self.state.selected_file_idx < len(self.state.uploaded_files) - 1:
            self.select_file(self.state.selected_file_idx + 1)

    def prev_file(self) -> None:
        if self.state.selected_file_idx > 0:
            self.select_file(self.state.selected_file_idx - 1)

    def update_config(self, config: WorkspaceConfig, persist: bool = False) -> None:
        """
        Updates global config and optionally saves to disk.
        """
        self.state.config = config

        if persist:
            # Only perform disk I/O if explicitly requested (e.g. manual save, file change)
            self._persist_sticky_settings(config)
            if self.state.current_file_hash:
                self.repo.save_file_settings(self.state.current_file_hash, config)
                self.settings_saved.emit()

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

    def remove_current_file(self) -> None:
        """
        Removes the currently selected file from the session.
        """
        idx = self.state.selected_file_idx
        if 0 <= idx < len(self.state.uploaded_files):
            # Remove file and thumbnail
            file_info = self.state.uploaded_files.pop(idx)
            self.state.thumbnails.pop(file_info["name"], None)

            # Reset selection or pick next/prev
            if not self.state.uploaded_files:
                self.state.selected_file_idx = -1
                self.state.current_file_path = None
                self.state.current_file_hash = None
                self.state.preview_raw = None
                self.state.config = WorkspaceConfig()
            else:
                # Clamp index to new bounds
                new_idx = min(idx, len(self.state.uploaded_files) - 1)
                self.select_file(new_idx)

            self.asset_model.refresh()
            self.state_changed.emit()
