from enum import Enum, auto
from dataclasses import dataclass, field, replace
from typing import Dict, Any, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QAbstractListModel, QModelIndex, Qt
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository


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
    thumbnails: Dict[str, Any] = field(default_factory=dict)  # filename -> QIcon/QPixmap
    selected_file_idx: int = -1
    selected_indices: List[int] = field(default_factory=list)
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

    def _apply_sticky_settings(self, config: WorkspaceConfig, only_global: bool = False) -> WorkspaceConfig:
        """
        Overlays globally persisted settings onto the config.
        If only_global is True, only non-look settings (Export) are applied.
        """
        from negpy.domain.models import (
            ExportConfig,
            LabConfig,
            ToningConfig,
            RetouchConfig,
        )

        sticky_export = self.repo.get_global_setting("last_export_config")
        if sticky_export:
            valid_keys = ExportConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in sticky_export.items() if k in valid_keys}
            new_export = ExportConfig(**filtered)
            config = replace(config, export=new_export)

        if only_global:
            return config

        sticky_mode = self.repo.get_global_setting("last_process_mode")
        sticky_buffer = self.repo.get_global_setting("last_analysis_buffer")
        sticky_roll_average = self.repo.get_global_setting("last_use_roll_average")
        sticky_floors = self.repo.get_global_setting("last_locked_floors")
        sticky_ceils = self.repo.get_global_setting("last_locked_ceils")
        sticky_roll_name = self.repo.get_global_setting("last_roll_name")

        new_process = config.process
        if sticky_mode:
            new_process = replace(new_process, process_mode=sticky_mode)
        if sticky_buffer is not None:
            new_process = replace(new_process, analysis_buffer=float(sticky_buffer))
        if sticky_roll_average is not None:
            new_process = replace(new_process, use_roll_average=bool(sticky_roll_average))
        if sticky_floors:
            new_process = replace(new_process, locked_floors=tuple(sticky_floors))
        if sticky_ceils:
            new_process = replace(new_process, locked_ceils=tuple(sticky_ceils))
        if sticky_roll_name:
            new_process = replace(new_process, roll_name=str(sticky_roll_name))

        config = replace(config, process=new_process)

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

        sticky_ratio = self.repo.get_global_setting("last_aspect_ratio")
        sticky_offset = self.repo.get_global_setting("last_autocrop_offset")

        new_geo = config.geometry
        if sticky_ratio:
            new_geo = replace(new_geo, autocrop_ratio=sticky_ratio)
        if sticky_offset is not None:
            new_geo = replace(new_geo, autocrop_offset=int(sticky_offset))

        config = replace(config, geometry=new_geo)

        sticky_lab = self.repo.get_global_setting("last_lab_config")
        if sticky_lab:
            valid_keys = LabConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in sticky_lab.items() if k in valid_keys}
            config = replace(config, lab=LabConfig(**filtered))

        sticky_toning = self.repo.get_global_setting("last_toning_config")
        if sticky_toning:
            valid_keys = ToningConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in sticky_toning.items() if k in valid_keys}
            config = replace(config, toning=ToningConfig(**filtered))

        sticky_retouch = self.repo.get_global_setting("last_retouch_config")
        if sticky_retouch:
            valid_keys = RetouchConfig.__dataclass_fields__.keys()
            # Never carry over manual spots to other files
            filtered = {k: v for k, v in sticky_retouch.items() if k in valid_keys and k != "manual_dust_spots"}
            config = replace(config, retouch=replace(config.retouch, **filtered))

        return config

    def _persist_sticky_settings(self, config: WorkspaceConfig) -> None:
        """
        Saves current settings to global storage.
        """
        from dataclasses import asdict

        self.repo.save_global_setting("last_process_mode", config.process.process_mode)
        self.repo.save_global_setting("last_analysis_buffer", config.process.analysis_buffer)
        self.repo.save_global_setting("last_use_roll_average", config.process.use_roll_average)
        self.repo.save_global_setting("last_locked_floors", config.process.locked_floors)
        self.repo.save_global_setting("last_locked_ceils", config.process.locked_ceils)
        self.repo.save_global_setting("last_roll_name", config.process.roll_name)

        self.repo.save_global_setting("last_density", config.exposure.density)
        self.repo.save_global_setting("last_grade", config.exposure.grade)
        self.repo.save_global_setting("last_wb_cyan", config.exposure.wb_cyan)
        self.repo.save_global_setting("last_wb_magenta", config.exposure.wb_magenta)
        self.repo.save_global_setting("last_wb_yellow", config.exposure.wb_yellow)
        self.repo.save_global_setting("last_use_camera_wb", config.exposure.use_camera_wb)

        self.repo.save_global_setting("last_toe", config.exposure.toe)
        self.repo.save_global_setting("last_toe_width", config.exposure.toe_width)
        self.repo.save_global_setting("last_toe_hardness", config.exposure.toe_hardness)
        self.repo.save_global_setting("last_shoulder", config.exposure.shoulder)
        self.repo.save_global_setting("last_shoulder_width", config.exposure.shoulder_width)
        self.repo.save_global_setting("last_shoulder_hardness", config.exposure.shoulder_hardness)

        self.repo.save_global_setting("last_aspect_ratio", config.geometry.autocrop_ratio)
        self.repo.save_global_setting("last_autocrop_offset", config.geometry.autocrop_offset)
        self.repo.save_global_setting("last_export_config", asdict(config.export))
        self.repo.save_global_setting("last_lab_config", asdict(config.lab))
        self.repo.save_global_setting("last_toning_config", asdict(config.toning))
        self.repo.save_global_setting("last_retouch_config", asdict(config.retouch))

    def select_file(self, index: int, selection_override: Optional[List[int]] = None) -> None:
        """
        Changes active file and hydrates state from repository.
        """
        if 0 <= index < len(self.state.uploaded_files):
            # Save current before switching
            if self.state.current_file_hash:
                self.repo.save_file_settings(self.state.current_file_hash, self.state.config)
                self.settings_saved.emit()

            file_info = self.state.uploaded_files[index]
            self.state.selected_file_idx = index
            self.state.selected_indices = selection_override if selection_override is not None else [index]
            self.state.current_file_path = file_info["path"]
            self.state.current_file_hash = file_info["hash"]

            saved_config = self.repo.load_file_settings(file_info["hash"])

            if saved_config:
                self.state.config = self._apply_sticky_settings(saved_config, only_global=True)
            else:
                self.state.config = self._apply_sticky_settings(WorkspaceConfig(), only_global=False)

            self.file_selected.emit(file_info["path"])
            self.state_changed.emit()

    def update_selection(self, indices: List[int]) -> None:
        """Updates the list of currently selected indices."""
        self.state.selected_indices = indices
        self.state_changed.emit()

    def sync_selected_settings(self) -> None:
        """
        Synchronizes current settings to all other selected files,
        excluding file-specific parameters (crop, rotation, retouch).
        """
        if not self.state.selected_indices or self.state.selected_file_idx == -1:
            return

        source_config = self.state.config

        for idx in self.state.selected_indices:
            if idx == self.state.selected_file_idx:
                continue

            if 0 <= idx < len(self.state.uploaded_files):
                target_info = self.state.uploaded_files[idx]
                target_hash = target_info["hash"]

                target_config = self.repo.load_file_settings(target_hash)
                if not target_config:
                    target_config = WorkspaceConfig()

                merged_geo = replace(
                    source_config.geometry,
                    manual_crop_rect=target_config.geometry.manual_crop_rect,
                    fine_rotation=target_config.geometry.fine_rotation,
                )

                merged_retouch = replace(source_config.retouch, manual_dust_spots=target_config.retouch.manual_dust_spots)

                new_config = replace(source_config, geometry=merged_geo, retouch=merged_retouch)

                self.repo.save_file_settings(target_hash, new_config)

        self.settings_saved.emit()

    def next_file(self) -> None:
        if self.state.selected_file_idx < len(self.state.uploaded_files) - 1:
            self.select_file(self.state.selected_file_idx + 1)

    def prev_file(self) -> None:
        if self.state.selected_file_idx > 0:
            self.select_file(self.state.selected_file_idx - 1)

    def update_config(self, config: WorkspaceConfig, persist: bool = False, render: bool = True) -> None:
        """
        Updates global config and optionally saves to disk.
        """
        self.state.config = config

        if persist:
            self._persist_sticky_settings(config)
            if self.state.current_file_hash:
                self.repo.save_file_settings(self.state.current_file_hash, config)
                self.settings_saved.emit()

        if render:
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

    def add_files(self, file_paths: List[str], validated_info: Optional[List[Dict]] = None) -> None:
        """
        Adds new files to the session.
        """
        import os
        from negpy.kernel.image.logic import calculate_file_hash

        if validated_info:
            for info in validated_info:
                if any(f["hash"] == info["hash"] for f in self.state.uploaded_files):
                    continue
                self.state.uploaded_files.append(info)
        else:
            for path in file_paths:
                try:
                    f_hash = calculate_file_hash(path)
                    if f_hash.startswith("err_"):
                        continue

                    if any(f["hash"] == f_hash for f in self.state.uploaded_files):
                        continue

                    self.state.uploaded_files.append({"name": os.path.basename(path), "path": path, "hash": f_hash})
                except Exception as e:
                    from negpy.kernel.system.logging import get_logger

                    get_logger(__name__).error(f"Failed to add {path}: {e}")

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
            file_info = self.state.uploaded_files.pop(idx)
            self.state.thumbnails.pop(file_info["name"], None)

            if not self.state.uploaded_files:
                self.state.selected_file_idx = -1
                self.state.current_file_path = None
                self.state.current_file_hash = None
                self.state.preview_raw = None
                self.state.config = WorkspaceConfig()
            else:
                new_idx = min(idx, len(self.state.uploaded_files) - 1)
                self.select_file(new_idx)

            self.asset_model.refresh()
            self.state_changed.emit()
