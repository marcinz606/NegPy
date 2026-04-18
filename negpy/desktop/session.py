import os
import threading
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, pyqtSignal

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import APP_CONFIG


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
    metrics_lock: threading.Lock = field(default_factory=threading.Lock, init=False, compare=False, repr=False)
    preview_raw: Optional[Any] = None
    original_res: tuple[int, int] = (0, 0)
    clipboard: Optional[WorkspaceConfig] = None

    # ICC Management
    icc_profile_path: Optional[str] = None
    icc_invert: bool = False
    apply_icc_to_export: bool = False

    # Hardware Acceleration
    gpu_enabled: bool = True

    # High Quality / Full Resoluiton Preview Toggle
    hq_preview: bool = False

    # Canvas background color swatch index (0=Black, 1=Dark Grey, 2=Mid Grey)
    canvas_bg_index: int = 0

    # History tracking
    undo_index: int = 0
    max_history_index: int = 0


class AssetListModel(QAbstractListModel):
    """
    Model for the uploaded files list with thumbnail support.
    """

    def __init__(self, state: AppState):
        super().__init__()
        self._state = state
        self._sort_order = "name"  # "name" | "date"
        self._sort_descending = False
        self._sorted_indices: list[int] = []
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        files = self._state.uploaded_files
        indices = list(range(len(files)))
        if self._sort_order == "name":
            indices.sort(key=lambda i: files[i]["name"].lower(), reverse=self._sort_descending)
        else:

            def _mtime(i: int) -> float:
                try:
                    return os.path.getmtime(files[i]["path"])
                except OSError:
                    return 0.0

            indices.sort(key=_mtime, reverse=self._sort_descending)
        self._sorted_indices = indices

    def set_sort_order(self, order: str) -> None:
        self._sort_order = order
        self._rebuild_indices()
        self.layoutChanged.emit()

    def set_sort_descending(self, descending: bool) -> None:
        self._sort_descending = descending
        self._rebuild_indices()
        self.layoutChanged.emit()

    def display_to_actual(self, display_row: int) -> int:
        if display_row < 0 or display_row >= len(self._sorted_indices):
            return -1
        return self._sorted_indices[display_row]

    def actual_to_display(self, actual_idx: int) -> int:
        try:
            return self._sorted_indices.index(actual_idx)
        except ValueError:
            return -1

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._sorted_indices)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._sorted_indices):
            return None

        file_info = self._state.uploaded_files[self._sorted_indices[index.row()]]

        if role == Qt.ItemDataRole.DisplayRole:
            return file_info["name"]

        if role == Qt.ItemDataRole.DecorationRole:
            return self._state.thumbnails.get(file_info["name"])

        if role == Qt.ItemDataRole.ToolTipRole:
            return file_info["path"]

        return None

    def refresh(self) -> None:
        self._rebuild_indices()
        self.layoutChanged.emit()


class DesktopSessionManager(QObject):
    """
    Manages application state, file list, and configuration persistence.
    """

    state_changed = pyqtSignal()
    history_changed = pyqtSignal()  # Emitted when undo/redo/persist happens
    settings_saved = pyqtSignal()
    file_selected = pyqtSignal(str)  # Emits file path when active file changes

    def __init__(self, repo: StorageRepository):
        super().__init__()
        self.repo = repo
        self.state = AppState()
        self.asset_model = AssetListModel(self.state)
        self._config_dirty = False  # True only after an explicit persist=True edit

        # Load global hardware settings
        saved_gpu = self.repo.get_global_setting("gpu_enabled")
        if saved_gpu is not None:
            self.state.gpu_enabled = bool(saved_gpu)

        saved_hq = self.repo.get_global_setting("hq_preview")
        if saved_hq is not None:
            self.state.hq_preview = bool(saved_hq)

        saved_bg = self.repo.get_global_setting("canvas_bg_index")
        if saved_bg is not None:
            self.state.canvas_bg_index = int(saved_bg)

        saved_icc_path = self.repo.get_global_setting("icc_profile_path")
        if saved_icc_path and os.path.exists(saved_icc_path):
            self.state.icc_profile_path = saved_icc_path
        saved_icc_invert = self.repo.get_global_setting("icc_invert")
        if saved_icc_invert is not None:
            self.state.icc_invert = bool(saved_icc_invert)
        saved_icc_apply = self.repo.get_global_setting("icc_apply_to_export")
        if saved_icc_apply is not None:
            self.state.apply_icc_to_export = bool(saved_icc_apply)

    def set_gpu_enabled(self, enabled: bool) -> None:
        """Updates and persists the hardware acceleration preference."""
        if self.state.gpu_enabled != enabled:
            self.state.gpu_enabled = enabled
            self.repo.save_global_setting("gpu_enabled", enabled)
            self.state_changed.emit()

    def set_hq_preview(self, enabled: bool) -> None:
        """Updates and persists the HQ preview preference."""
        if self.state.hq_preview != enabled:
            self.state.hq_preview = enabled
            self.repo.save_global_setting("hq_preview", enabled)
            self.state_changed.emit()

    def set_canvas_bg(self, index: int) -> None:
        """Updates and persists the canvas background color index."""
        if self.state.canvas_bg_index != index:
            self.state.canvas_bg_index = index
            self.repo.save_global_setting("canvas_bg_index", index)

    def save_icc_prefs(self) -> None:
        """Persists current ICC profile settings."""
        self.repo.save_global_setting("icc_profile_path", self.state.icc_profile_path)
        self.repo.save_global_setting("icc_invert", self.state.icc_invert)
        self.repo.save_global_setting("icc_apply_to_export", self.state.apply_icc_to_export)

    def _apply_sticky_settings(self, config: WorkspaceConfig, only_global: bool = False) -> WorkspaceConfig:
        """
        Overlays globally persisted settings onto the config.

        Two tiers:
        - only_global=True  (file has a sidecar): only export preferences are overlaid.
        - only_global=False (new file, no sidecar): workflow settings are overlaid
          (process mode, roll name, analysis buffer, geometry defaults, export).
          Per-image look settings (exposure, lab, toning, retouch) are intentionally
          NOT carried over so that fresh files always start from clean defaults.
        """
        from negpy.domain.models import ExportConfig

        sticky_export = self.repo.get_global_setting("last_export_config")
        if sticky_export:
            valid_keys = ExportConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in sticky_export.items() if k in valid_keys}
            config = replace(config, export=ExportConfig(**filtered))

        if only_global:
            return config

        # Workflow settings — safe to carry across all files on a roll
        sticky_mode = self.repo.get_global_setting("last_process_mode")
        sticky_buffer = self.repo.get_global_setting("last_analysis_buffer")
        sticky_drange_clip = self.repo.get_global_setting("last_drange_clip")
        sticky_roll_average = self.repo.get_global_setting("last_use_roll_average")
        sticky_floors = self.repo.get_global_setting("last_locked_floors")
        sticky_ceils = self.repo.get_global_setting("last_locked_ceils")
        sticky_roll_name = self.repo.get_global_setting("last_roll_name")

        new_process = config.process
        if sticky_mode:
            new_process = replace(new_process, process_mode=sticky_mode)
        if sticky_buffer is not None:
            new_process = replace(new_process, analysis_buffer=float(sticky_buffer))
        if sticky_drange_clip is not None:
            new_process = replace(new_process, drange_clip=float(sticky_drange_clip))
        if sticky_roll_average is not None:
            new_process = replace(new_process, use_roll_average=bool(sticky_roll_average))
        if sticky_floors:
            new_process = replace(new_process, locked_floors=tuple(sticky_floors))
        if sticky_ceils:
            new_process = replace(new_process, locked_ceils=tuple(sticky_ceils))
        if sticky_roll_name:
            new_process = replace(new_process, roll_name=str(sticky_roll_name))
        config = replace(config, process=new_process)

        sticky_ratio = self.repo.get_global_setting("last_aspect_ratio")
        sticky_offset = self.repo.get_global_setting("last_autocrop_offset")
        new_geo = config.geometry
        if sticky_ratio:
            new_geo = replace(new_geo, autocrop_ratio=sticky_ratio)
        if sticky_offset is not None:
            new_geo = replace(new_geo, autocrop_offset=int(sticky_offset))
        config = replace(config, geometry=new_geo)

        # Exposure, lab, toning, retouch are per-image look decisions and are
        # deliberately excluded here — fresh files start from WorkspaceConfig defaults.
        # Exception: use_camera_wb and dust_remove are workflow preferences, not image-specific looks.
        sticky_camera_wb = self.repo.get_global_setting("last_use_camera_wb")
        if sticky_camera_wb is not None:
            config = replace(config, exposure=replace(config.exposure, use_camera_wb=bool(sticky_camera_wb)))

        # Exception: dust_remove is a workflow preference, not an image-specific look.
        sticky_dust = self.repo.get_global_setting("last_dust_remove")
        if sticky_dust is not None:
            config = replace(config, retouch=replace(config.retouch, dust_remove=bool(sticky_dust)))

        return config

    def _persist_sticky_settings(self, config: WorkspaceConfig) -> None:
        """
        Saves current settings to global storage.
        """
        from dataclasses import asdict

        self.repo.save_global_setting("last_process_mode", config.process.process_mode)
        self.repo.save_global_setting("last_analysis_buffer", config.process.analysis_buffer)
        self.repo.save_global_setting("last_drange_clip", config.process.drange_clip)
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
        self.repo.save_global_setting("last_shoulder", config.exposure.shoulder)
        self.repo.save_global_setting("last_shoulder_width", config.exposure.shoulder_width)

        self.repo.save_global_setting("last_aspect_ratio", config.geometry.autocrop_ratio)
        self.repo.save_global_setting("last_autocrop_offset", config.geometry.autocrop_offset)
        self.repo.save_global_setting("last_export_config", asdict(config.export))
        self.repo.save_global_setting("last_lab_config", asdict(config.lab))
        self.repo.save_global_setting("last_toning_config", asdict(config.toning))
        self.repo.save_global_setting("last_retouch_config", asdict(config.retouch))
        self.repo.save_global_setting("last_dust_remove", config.retouch.dust_remove)

    def select_file(self, index: int, selection_override: Optional[List[int]] = None) -> None:
        """
        Changes active file and hydrates state from repository.
        """
        if 0 <= index < len(self.state.uploaded_files):
            # Save current before switching, but only if user actually made explicit edits
            if self.state.current_file_hash and self._config_dirty:
                self.repo.save_file_settings(self.state.current_file_hash, self.state.config)
                self.settings_saved.emit()
            self._config_dirty = False

            file_info = self.state.uploaded_files[index]
            self.state.selected_file_idx = index
            self.state.selected_indices = selection_override if selection_override is not None else [index]
            self.state.current_file_path = file_info["path"]
            self.state.current_file_hash = file_info["hash"]

            # Restore history state for file
            self.state.undo_index = self.repo.get_max_history_index(file_info["hash"])
            self.state.max_history_index = self.state.undo_index

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

                merged_process = replace(
                    source_config.process,
                    local_floors=target_config.process.local_floors,
                    local_ceils=target_config.process.local_ceils,
                )

                new_config = replace(
                    source_config,
                    geometry=merged_geo,
                    retouch=merged_retouch,
                    process=merged_process,
                )

                self.repo.save_file_settings(target_hash, new_config)

        self.settings_saved.emit()

    def next_file(self) -> None:
        if self.state.selected_file_idx < len(self.state.uploaded_files) - 1:
            self.select_file(self.state.selected_file_idx + 1)

    def prev_file(self) -> None:
        if self.state.selected_file_idx > 0:
            self.select_file(self.state.selected_file_idx - 1)

    def update_config(self, config: WorkspaceConfig, persist: bool = False, render: bool = True, record_history: bool = True) -> None:
        """
        Updates global config and optionally saves to disk.
        """
        if persist and record_history and self.state.current_file_hash:
            # If editing after an undo, drop the now-orphaned future branch
            if self.state.undo_index < self.state.max_history_index:
                self.repo.truncate_history_above(self.state.current_file_hash, self.state.undo_index)
            self.repo.save_history_step(self.state.current_file_hash, self.state.undo_index, self.state.config)
            self.state.undo_index += 1
            self.state.max_history_index = self.state.undo_index

            if self.state.undo_index > APP_CONFIG.max_history_steps:
                self.repo.prune_history(self.state.current_file_hash, max_steps=APP_CONFIG.max_history_steps)

            self.history_changed.emit()

        self.state.config = config

        if persist:
            self._config_dirty = True
            self._persist_sticky_settings(config)
            if self.state.current_file_hash:
                self.repo.save_file_settings(self.state.current_file_hash, config)
                self.settings_saved.emit()

        if render:
            self.state_changed.emit()

    def undo(self) -> None:
        if self.state.undo_index > 0 and self.state.current_file_hash:
            if self.state.undo_index == self.state.max_history_index:
                self.repo.save_history_step(self.state.current_file_hash, self.state.undo_index, self.state.config)

            self.state.undo_index -= 1
            prev_config = self.repo.load_history_step(self.state.current_file_hash, self.state.undo_index)
            if prev_config:
                self.state.config = prev_config
                self._config_dirty = True
                self.state_changed.emit()
                self.history_changed.emit()

    def redo(self) -> None:
        if self.state.undo_index < self.state.max_history_index and self.state.current_file_hash:
            self.state.undo_index += 1
            next_config = self.repo.load_history_step(self.state.current_file_hash, self.state.undo_index)
            if next_config:
                self.state.config = next_config
                self._config_dirty = True
                self.state_changed.emit()
                self.history_changed.emit()

    def reset_settings(self) -> None:
        """
        Reverts current file to default configuration and clears history.
        """
        if self.state.current_file_hash:
            self.repo.clear_history(self.state.current_file_hash)
            self.state.undo_index = 0
            self.state.max_history_index = 0
            self.history_changed.emit()

        self._config_dirty = False
        self.update_config(WorkspaceConfig())
        self.state_changed.emit()

    def reset_section(self, section: str) -> None:
        """Reset a single feature section to its default config."""
        from negpy.features.exposure.models import ExposureConfig
        from negpy.features.lab.models import LabConfig
        from negpy.features.toning.models import ToningConfig
        from negpy.features.geometry.models import GeometryConfig
        from negpy.features.process.models import ProcessConfig
        from negpy.features.retouch.models import RetouchConfig

        defaults = {
            "exposure": ExposureConfig(),
            "lab": LabConfig(),
            "toning": ToningConfig(),
            "geometry": GeometryConfig(),
            "process": ProcessConfig(),
            "retouch": RetouchConfig(),
        }
        if section not in defaults:
            return
        new_config = replace(self.state.config, **{section: defaults[section]})
        self.update_config(new_config, persist=True)

    def copy_settings(self) -> None:
        import copy

        self.state.clipboard = copy.deepcopy(self.state.config)
        self.state_changed.emit()

    def paste_settings(self) -> None:
        if self.state.clipboard and self.state.current_file_hash:
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
        self._config_dirty = False

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
                self.state.selected_indices = []
                self.state.current_file_path = None
                self.state.current_file_hash = None
                self.state.preview_raw = None
                self.state.config = WorkspaceConfig()
            else:
                new_idx = min(idx, len(self.state.uploaded_files) - 1)
                self.select_file(new_idx)

            self.asset_model.refresh()
            self.state_changed.emit()
