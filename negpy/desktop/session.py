import os
import re
import threading
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, pyqtSignal

from negpy.desktop.view.canvas.crop_guides import CropGuide
from negpy.domain.models import ExportPreset, WorkspaceConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.sidecar import load_or_promote


class ToolMode(Enum):
    NONE = auto()
    WB_PICK = auto()
    CROP_MANUAL = auto()
    DUST_PICK = auto()
    SCRATCH_PICK = auto()
    LOCAL_DRAW = auto()
    ANALYSIS_DRAW = auto()


@dataclass
class AppState:
    """
    Reactive state object for the desktop session.
    """

    current_file_path: Optional[str] = None
    current_file_hash: Optional[str] = None
    source_cs: str = ""
    config: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    workspace_color_space: str = "Adobe RGB"
    is_processing: bool = False
    active_tool: ToolMode = ToolMode.NONE
    # Colour page region (0 Global, 1 Shadows, 2 Highlights): scopes the WB
    # picker so a pick writes the selected region's CMY fields.
    wb_pick_region: int = 0
    uploaded_files: List[Dict[str, Any]] = field(default_factory=list)
    thumbnails: Dict[str, Any] = field(default_factory=dict)  # filename -> QIcon/QPixmap
    source_exif: Dict[str, Any] = field(default_factory=dict)  # file_hash -> piexif dict
    selected_file_idx: int = -1
    selected_indices: List[int] = field(default_factory=list)
    active_adjustment_idx: int = 0
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_lock: threading.Lock = field(default_factory=threading.Lock, init=False, compare=False, repr=False)
    preview_raw: Optional[Any] = None
    preview_ir: Optional[Any] = None  # downsampled IR float32 [0,1] (H,W); None if source has no IR
    has_ir: bool = False
    original_res: tuple[int, int] = (0, 0)
    clipboard: Optional[WorkspaceConfig] = None

    # ICC Management
    icc_input_path: Optional[str] = None
    icc_output_path: Optional[str] = None
    # Effective monitor ICC profile bytes used by every preview → display transform;
    # None = treat the display as sRGB. Resolved from the override (if any) else the
    # auto-detected profile below.
    monitor_icc_bytes: Optional[bytes] = None
    # Raw profile auto-detected from the active screen (drives the "As detected" option).
    monitor_icc_detected_bytes: Optional[bytes] = None
    # User override: a ColorSpace value (e.g. "Display P3") or None = use detected.
    monitor_profile_override: Optional[str] = None
    # Soft-proof toggle: when off, Output/Input ICC affect export only, not the preview.
    # Defaults on so the preview is true to export by default.
    soft_proof_enabled: bool = True

    # Hardware Acceleration
    gpu_enabled: bool = True

    # High Quality / Full Resoluiton Preview Toggle
    hq_preview: bool = False

    # Process-mode autodetect on file load (opt-in)
    autodetect_enabled: bool = False

    # Canvas background color swatch index (0=Black, 1=Dark Grey, 2=Mid Grey)
    canvas_bg_index: int = 0

    # Crop tool composition guide (CropGuide value); display-only, so not in GeometryConfig
    crop_guide: str = "thirds"
    crop_guide_orientation: int = 0

    # Reverse scroll-wheel zoom direction on the image viewer (scroll up = zoom out).
    invert_zoom_scroll: bool = False

    # Local adjustments UI state (not persisted in workspace config)
    local_selected_mask: int = -1
    show_local_overlay: bool = True

    # History tracking
    undo_index: int = 0
    max_history_index: int = 0

    # Dirty flag: True when explicit persist=True edits have been made since last file open/switch
    is_dirty: bool = False

    # True when the active file has no saved config yet (gates process-mode autodetect)
    current_file_is_new: bool = False

    # True while the before/after view shows the un-graded auto baseline instead of edits
    compare_mode: bool = False

    # Export presets (globally managed, not per-file)
    export_presets: List[ExportPreset] = field(default_factory=list)

    # Flat "for editing elsewhere" master output (digital intermediate).
    # When on, export and the optional preview-peek use the flat render intent.
    flat_output: bool = False
    flat_format: str = "TIFF"  # "TIFF" (16-bit) or "DNG" (linear)
    # Transient: preview is currently peeking the flat render (not persisted).
    flat_peek: bool = False


class AssetListModel(QAbstractListModel):
    """
    Model for the uploaded files list with thumbnail support.
    """

    def __init__(self, state: AppState):
        super().__init__()
        self._state = state
        self._sort_order = "name"  # "name" | "date"
        self._sort_descending = False
        self._filter_text: str = ""
        self._filter_regex: bool = False
        self._filter_pattern: Optional[re.Pattern] = None
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

        if self._filter_text:
            if self._filter_pattern is not None:
                pattern = self._filter_pattern
                indices = [i for i in indices if pattern.search(files[i]["name"])]
            else:
                needle = self._filter_text
                indices = [i for i in indices if needle in files[i]["name"].lower()]

        self._sorted_indices = indices

    def set_sort_order(self, order: str) -> None:
        self._sort_order = order
        self._rebuild_indices()
        self.layoutChanged.emit()

    def set_sort_descending(self, descending: bool) -> None:
        self._sort_descending = descending
        self._rebuild_indices()
        self.layoutChanged.emit()

    def set_filter(self, text: str, regex: bool) -> bool:
        """Updates filter. Returns True on success, False if regex failed to compile."""
        text = text.strip()
        if not text:
            self._filter_text = ""
            self._filter_regex = regex
            self._filter_pattern = None
            self._rebuild_indices()
            self.layoutChanged.emit()
            return True

        if regex:
            try:
                pattern = re.compile(text, re.IGNORECASE)
            except re.error:
                return False
            self._filter_text = text
            self._filter_regex = True
            self._filter_pattern = pattern
        else:
            self._filter_text = text.lower()
            self._filter_regex = False
            self._filter_pattern = None

        self._rebuild_indices()
        self.layoutChanged.emit()
        return True

    def visible_actual_indices(self) -> set[int]:
        return set(self._sorted_indices)

    def visible_actual_indices_ordered(self) -> list[int]:
        return list(self._sorted_indices)

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


_ASPECT_LABELS = {
    "process": "Process",
    "crop": "Crop",
    "rotation": "Rotation",
    "exposure": "Exposure",
    "color": "Color",
    "finish": "Finish",
    "bounds_luma": "Tonal span",
    "bounds_colour": "Colour balance",
}

_VALID_ASPECTS = frozenset(_ASPECT_LABELS)


def _source_effective_bounds(process) -> Optional[tuple]:
    """The floors/ceils a source frame is currently rendering with.

    Roll baseline when the source is on one, else its per-frame meter. Returns
    None when the source was never analysed (all-zero) — nothing to broadcast.
    """
    if process.is_locked_initialized and (process.use_luma_average or process.use_colour_average):
        return process.locked_floors, process.locked_ceils
    if process.is_local_initialized:
        return process.local_floors, process.local_ceils
    return None


def build_synced_config(
    source: WorkspaceConfig,
    target: WorkspaceConfig,
    aspects: frozenset,
    src_bounds: Optional[tuple],
) -> WorkspaceConfig:
    """Pure per-target merge for a bulk "Apply to selected" action.

    `aspects` is a subset of _VALID_ASPECTS, checked independently in the Sync
    Settings dialog. `src_bounds` is (floors, ceils) from _source_effective_bounds,
    needed when bounds_luma/bounds_colour is checked. Builds the result by starting
    from `target` and overlaying only the checked aspects, so anything not covered
    by an aspect (flatfield, rgbscan, metadata, export, dust spots, per-frame local
    bounds) always stays the target's own.
    """
    out = target

    if "process" in aspects:
        out = replace(
            out,
            process=replace(
                source.process,
                local_floors=out.process.local_floors,
                local_ceils=out.process.local_ceils,
                locked_floors=out.process.locked_floors,
                locked_ceils=out.process.locked_ceils,
                use_luma_average=out.process.use_luma_average,
                use_colour_average=out.process.use_colour_average,
            ),
        )

    if "crop" in aspects:
        sg = source.geometry
        out = replace(
            out,
            geometry=replace(
                out.geometry,
                auto_crop_enabled=sg.auto_crop_enabled,
                autocrop_offset=sg.autocrop_offset,
                autocrop_ratio=sg.autocrop_ratio,
                autocrop_mode=sg.autocrop_mode,
                manual_crop_rect=sg.manual_crop_rect,
            ),
        )

    if "rotation" in aspects:
        sg = source.geometry
        out = replace(
            out,
            geometry=replace(
                out.geometry,
                rotation=sg.rotation,
                fine_rotation=sg.fine_rotation,
                flip_horizontal=sg.flip_horizontal,
                flip_vertical=sg.flip_vertical,
            ),
        )

    if "exposure" in aspects:
        out = replace(out, exposure=source.exposure)

    if "color" in aspects:
        out = replace(out, lab=source.lab, toning=source.toning)

    if "finish" in aspects:
        out = replace(
            out,
            retouch=replace(source.retouch, manual_dust_spots=out.retouch.manual_dust_spots),
            finish=source.finish,
        )

    if aspects & {"bounds_luma", "bounds_colour"}:
        floors, ceils = src_bounds
        # locked_* is a shared pair, so always write it; toggle only the selected axis.
        changes: dict = {"locked_floors": floors, "locked_ceils": ceils}
        if "bounds_luma" in aspects:
            changes["use_luma_average"] = True
        if "bounds_colour" in aspects:
            changes["use_colour_average"] = True
        out = replace(out, process=replace(out.process, **changes))

    return out


def resolve_asset_rgbscan(params: WorkspaceConfig, asset: dict) -> WorkspaceConfig:
    """Overlay a frame's own RGB-scan triplet paths (from the asset dict) onto its export
    params — the authoritative source select_file uses. A non-triplet frame gets rgbscan
    reset so a batch frame never inherits the currently-open frame's leaked/stale triplet."""
    green, blue = asset.get("green_path"), asset.get("blue_path")
    if green and blue:
        align = bool(asset.get("align", params.rgbscan.align))
        return replace(params, rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=align))
    return replace(params, rgbscan=RgbScanConfig())


class DesktopSessionManager(QObject):
    """
    Manages application state, file list, and configuration persistence.
    """

    state_changed = pyqtSignal()
    files_changed = pyqtSignal()  # File list additions only — does not trigger sidebar sync
    history_changed = pyqtSignal()  # Emitted when undo/redo/persist happens
    settings_saved = pyqtSignal()
    active_file_changing = pyqtSignal()  # Outgoing file about to be replaced — last chance to snapshot it
    settings_copied = pyqtSignal()
    settings_pasted = pyqtSignal()
    settings_synced = pyqtSignal(str)  # Bulk "Apply to selected" done — carries a status message
    file_selected = pyqtSignal(str)  # Emits file path when active file changes
    session_emptied = pyqtSignal()  # Last file removed — the viewer must blank the stale frame

    @property
    def _config_dirty(self) -> bool:
        return self.state.is_dirty

    @_config_dirty.setter
    def _config_dirty(self, value: bool) -> None:
        self.state.is_dirty = value

    def __init__(self, repo: StorageRepository):
        super().__init__()
        self.repo = repo
        self.state = AppState()
        self.asset_model = AssetListModel(self.state)
        # is_dirty initialised to False via AppState default

        # Load global hardware settings
        saved_gpu = self.repo.get_global_setting("gpu_enabled")
        if saved_gpu is not None:
            self.state.gpu_enabled = bool(saved_gpu)

        saved_hq = self.repo.get_global_setting("hq_preview")
        if saved_hq is not None:
            self.state.hq_preview = bool(saved_hq)
        if APP_CONFIG.force_hq_preview is not None:
            self.state.hq_preview = APP_CONFIG.force_hq_preview

        saved_autodetect = self.repo.get_global_setting("autodetect_enabled")
        if saved_autodetect is not None:
            self.state.autodetect_enabled = bool(saved_autodetect)

        saved_bg = self.repo.get_global_setting("canvas_bg_index")
        if saved_bg is not None:
            self.state.canvas_bg_index = int(saved_bg)

        saved_guide = self.repo.get_global_setting("crop_guide")
        if saved_guide in set(CropGuide):
            self.state.crop_guide = str(saved_guide)
        saved_guide_orient = self.repo.get_global_setting("crop_guide_orientation")
        if saved_guide_orient is not None:
            self.state.crop_guide_orientation = int(saved_guide_orient) % 8

        saved_invert_zoom = self.repo.get_global_setting("invert_zoom_scroll")
        if saved_invert_zoom is not None:
            self.state.invert_zoom_scroll = bool(saved_invert_zoom)

        saved_icc_in = self.repo.get_global_setting("icc_input_path")
        if saved_icc_in and os.path.exists(saved_icc_in):
            self.state.icc_input_path = saved_icc_in
        saved_icc_out = self.repo.get_global_setting("icc_output_path")
        if saved_icc_out and os.path.exists(saved_icc_out):
            self.state.icc_output_path = saved_icc_out
        saved_monitor_override = self.repo.get_global_setting("monitor_profile_override")
        if saved_monitor_override:
            self.state.monitor_profile_override = saved_monitor_override
        saved_soft_proof = self.repo.get_global_setting("soft_proof_enabled")
        if saved_soft_proof is not None:
            self.state.soft_proof_enabled = bool(saved_soft_proof)

        saved_flat_output = self.repo.get_global_setting("flat_output")
        if saved_flat_output is not None:
            self.state.flat_output = bool(saved_flat_output)
        saved_flat_format = self.repo.get_global_setting("flat_format")
        if saved_flat_format in ("TIFF", "DNG"):
            self.state.flat_format = saved_flat_format

        self.state.export_presets = self.repo.load_export_presets()

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

    def set_autodetect_enabled(self, enabled: bool) -> None:
        """Updates and persists the process-mode autodetect preference."""
        if self.state.autodetect_enabled != enabled:
            self.state.autodetect_enabled = enabled
            self.repo.save_global_setting("autodetect_enabled", enabled)
            self.state_changed.emit()

    def set_canvas_bg(self, index: int) -> None:
        """Updates and persists the canvas background color index."""
        if self.state.canvas_bg_index != index:
            self.state.canvas_bg_index = index
            self.repo.save_global_setting("canvas_bg_index", index)

    def set_crop_guide(self, guide: str) -> None:
        """Updates and persists the crop composition guide."""
        if self.state.crop_guide != guide:
            self.state.crop_guide = guide
            self.repo.save_global_setting("crop_guide", guide)

    def set_crop_guide_orientation(self, orientation: int) -> None:
        """Updates and persists the crop guide orientation step."""
        if self.state.crop_guide_orientation != orientation:
            self.state.crop_guide_orientation = orientation
            self.repo.save_global_setting("crop_guide_orientation", orientation)

    def save_icc_prefs(self) -> None:
        """Persists current ICC profile settings."""
        self.repo.save_global_setting("icc_input_path", self.state.icc_input_path)
        self.repo.save_global_setting("icc_output_path", self.state.icc_output_path)
        self.repo.save_global_setting("monitor_profile_override", self.state.monitor_profile_override)
        self.repo.save_global_setting("soft_proof_enabled", self.state.soft_proof_enabled)

    def save_export_presets(self) -> None:
        """Persists current export presets."""
        self.repo.save_export_presets(self.state.export_presets)

    def save_flat_output_prefs(self) -> None:
        """Persists the flat ('for editing elsewhere') output preferences."""
        self.repo.save_global_setting("flat_output", self.state.flat_output)
        self.repo.save_global_setting("flat_format", self.state.flat_format)

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

        sticky_protect = self.repo.get_global_setting("last_protect_original_metadata")
        if sticky_protect is not None:
            config = replace(
                config,
                metadata=replace(config.metadata, protect_original_metadata=bool(sticky_protect)),
            )

        # Flat-field reference and distortion k1 are rig-global: the active profile's
        # values always override the per-file ones. New files default to enabled when a
        # profile is active; saved files keep their toggle.
        active_ff = self.repo.get_global_setting("flatfield_active_profile")
        ff_rec = self.repo.get_flatfield_profile(active_ff) if active_ff else None
        ff_path, ff_k1 = ff_rec if ff_rec else ("", 0.0)
        config = replace(config, flatfield=replace(config.flatfield, reference_path=ff_path, k1=ff_k1))

        # Temperature roll-locks (per region): re-aim each locked region's M/Y
        # pair at its Kelvin target, keeping the frame's own off-locus tint.
        for lock_key, m_field, y_field in (
            ("wb_temp_lock", "wb_magenta", "wb_yellow"),
            ("wb_temp_lock_shadow", "shadow_magenta", "shadow_yellow"),
            ("wb_temp_lock_highlight", "highlight_magenta", "highlight_yellow"),
        ):
            locked_k = self.repo.get_global_setting(lock_key)
            if locked_k is not None:
                from negpy.features.exposure.logic import kelvin_to_wb

                m2, y2 = kelvin_to_wb(float(locked_k), getattr(config.exposure, m_field), getattr(config.exposure, y_field))
                config = replace(config, exposure=replace(config.exposure, **{m_field: m2, y_field: y2}))

        if only_global:
            return config

        config = replace(config, flatfield=replace(config.flatfield, apply=bool(ff_path)))

        # Workflow settings — safe to carry across all files on a roll
        sticky_mode = self.repo.get_global_setting("last_process_mode")
        sticky_buffer = self.repo.get_global_setting("last_analysis_buffer")
        sticky_luma_range_clip = self.repo.get_global_setting("last_luma_range_clip")
        sticky_color_range_clip = self.repo.get_global_setting("last_color_range_clip")
        # Roll-average baseline is roll-scoped (written per-file by Batch Analysis /
        # a saved roll), never seeded onto fresh files.
        sticky_crosstalk_strength = self.repo.get_global_setting("last_crosstalk_strength")
        sticky_crosstalk_matrix = self.repo.get_global_setting("last_crosstalk_matrix")
        sticky_crosstalk_profile = self.repo.get_global_setting("last_crosstalk_profile")

        new_process = config.process
        if sticky_mode:
            new_process = replace(new_process, process_mode=sticky_mode)
        if sticky_buffer is not None:
            new_process = replace(new_process, analysis_buffer=float(sticky_buffer))
        if sticky_luma_range_clip is not None:
            new_process = replace(new_process, luma_range_clip=float(sticky_luma_range_clip))
        if sticky_color_range_clip is not None:
            new_process = replace(new_process, color_range_clip=float(sticky_color_range_clip))
        if sticky_crosstalk_strength is not None:
            new_process = replace(new_process, crosstalk_strength=float(sticky_crosstalk_strength))
        if sticky_crosstalk_matrix:
            new_process = replace(new_process, crosstalk_matrix=tuple(sticky_crosstalk_matrix))
        if sticky_crosstalk_profile:
            new_process = replace(new_process, crosstalk_profile=str(sticky_crosstalk_profile))
        config = replace(config, process=new_process)

        sticky_ratio = self.repo.get_global_setting("last_aspect_ratio")
        sticky_autocrop_mode = self.repo.get_global_setting("last_autocrop_mode")
        sticky_offset = self.repo.get_global_setting("last_autocrop_offset")
        sticky_flip_h = self.repo.get_global_setting("last_flip_horizontal")
        sticky_flip_v = self.repo.get_global_setting("last_flip_vertical")
        new_geo = config.geometry
        if sticky_ratio:
            new_geo = replace(new_geo, autocrop_ratio=sticky_ratio)
        if sticky_autocrop_mode:
            new_geo = replace(new_geo, autocrop_mode=str(sticky_autocrop_mode))
        if sticky_offset is not None:
            new_geo = replace(new_geo, autocrop_offset=int(sticky_offset))
        if sticky_flip_h is not None:
            new_geo = replace(new_geo, flip_horizontal=bool(sticky_flip_h))
        if sticky_flip_v is not None:
            new_geo = replace(new_geo, flip_vertical=bool(sticky_flip_v))
        config = replace(config, geometry=new_geo)

        sticky_lab = self.repo.get_global_setting("last_lab_config")
        if sticky_lab:
            from negpy.features.lab.models import LabConfig

            valid_keys = LabConfig.__dataclass_fields__.keys()
            config = replace(config, lab=LabConfig(**{k: v for k, v in sticky_lab.items() if k in valid_keys}))

        # Exposure, toning, retouch are per-image look decisions and are
        # deliberately excluded here — fresh files start from WorkspaceConfig defaults.
        # Exception: linear_raw and dust_remove are workflow preferences, not image-specific looks.
        sticky_linear_raw = self.repo.get_global_setting("last_linear_raw")
        if sticky_linear_raw is not None:
            config = replace(config, process=replace(config.process, linear_raw=bool(sticky_linear_raw)))

        # Processing toggles (Auto Density / Auto Grade / Shadow Neutral / Paper
        # White) are workflow preferences, not per-image looks: carry them to
        # fresh files unless explicitly changed per file.
        new_exp = config.exposure
        for key, attr in (
            ("last_auto_exposure", "auto_exposure"),
            ("last_auto_normalize_contrast", "auto_normalize_contrast"),
            ("last_paper_dmin", "paper_dmin"),
        ):
            val = self.repo.get_global_setting(key)
            if val is not None:
                new_exp = replace(new_exp, **{attr: bool(val)})
        config = replace(config, exposure=new_exp)

        # Paper stock is roll-wide; render guards cross-mode leak.
        sticky_paper = self.repo.get_global_setting("last_paper_profile")
        if sticky_paper:
            config = replace(config, exposure=replace(config.exposure, paper_profile=str(sticky_paper)))

        # Exception: dust_remove is a workflow preference, not an image-specific look.
        sticky_dust = self.repo.get_global_setting("last_dust_remove")
        if sticky_dust is not None:
            config = replace(config, retouch=replace(config.retouch, dust_remove=bool(sticky_dust)))

        return config

    def _persist_sticky_settings(self, config: WorkspaceConfig) -> None:
        """
        Saves current settings to global storage in a single transaction.
        """
        from dataclasses import asdict

        self.repo.save_global_settings(
            {
                "last_process_mode": config.process.process_mode,
                "last_analysis_buffer": config.process.analysis_buffer,
                "last_luma_range_clip": config.process.luma_range_clip,
                "last_color_range_clip": config.process.color_range_clip,
                "last_crosstalk_strength": config.process.crosstalk_strength,
                "last_crosstalk_matrix": config.process.crosstalk_matrix,
                "last_crosstalk_profile": config.process.crosstalk_profile,
                "last_density": config.exposure.density,
                "last_grade": config.exposure.grade,
                "last_wb_cyan": config.exposure.wb_cyan,
                "last_wb_magenta": config.exposure.wb_magenta,
                "last_wb_yellow": config.exposure.wb_yellow,
                "last_linear_raw": config.process.linear_raw,
                "last_auto_exposure": config.exposure.auto_exposure,
                "last_auto_normalize_contrast": config.exposure.auto_normalize_contrast,
                "last_paper_dmin": config.exposure.paper_dmin,
                "last_paper_profile": config.exposure.paper_profile,
                "last_toe": config.exposure.toe,
                "last_toe_width": config.exposure.toe_width,
                "last_shoulder": config.exposure.shoulder,
                "last_shoulder_width": config.exposure.shoulder_width,
                "last_aspect_ratio": config.geometry.autocrop_ratio,
                "last_autocrop_mode": config.geometry.autocrop_mode,
                "last_autocrop_offset": config.geometry.autocrop_offset,
                "last_flip_horizontal": config.geometry.flip_horizontal,
                "last_flip_vertical": config.geometry.flip_vertical,
                "last_export_config": asdict(config.export),
                "last_lab_config": asdict(config.lab),
                "last_toning_config": asdict(config.toning),
                "last_retouch_config": asdict(config.retouch),
                "last_dust_remove": config.retouch.dust_remove,
                "last_protect_original_metadata": config.metadata.protect_original_metadata,
            }
        )

    def select_file(self, index: int, selection_override: Optional[List[int]] = None) -> None:
        """
        Changes active file and hydrates state from repository.
        """
        if 0 <= index < len(self.state.uploaded_files):
            # Save current before switching, but only if user actually made explicit edits
            if self.state.current_file_hash and self._config_dirty:
                self.repo.save_file_settings(self.state.current_file_hash, self.state.config)
                self.settings_saved.emit()
                self.active_file_changing.emit()
            self._config_dirty = False

            file_info = self.state.uploaded_files[index]
            self.state.selected_file_idx = index
            self.state.selected_indices = selection_override if selection_override is not None else [index]
            self.state.current_file_path = file_info["path"]
            self.state.current_file_hash = file_info["hash"]

            # Read source EXIF for metadata display
            from negpy.infrastructure.loaders.helpers import read_exif_from_file

            exif = read_exif_from_file(file_info["path"])
            if exif:
                self.state.source_exif[file_info["hash"]] = exif
            elif file_info["hash"] in self.state.source_exif:
                del self.state.source_exif[file_info["hash"]]

            # Restore history state for file
            self.state.undo_index = self.repo.get_max_history_index(file_info["hash"])
            self.state.max_history_index = self.state.undo_index

            saved_config = load_or_promote(self.repo, file_info["hash"], file_info["path"])
            self.state.current_file_is_new = saved_config is None

            if saved_config:
                self.state.config = self._apply_sticky_settings(saved_config, only_global=True)
            else:
                self.state.config = self._apply_sticky_settings(WorkspaceConfig(), only_global=False)

            # RGB-scan triplet: the green/blue exposures travel with the asset entry.
            green, blue = file_info.get("green_path"), file_info.get("blue_path")
            if green and blue:
                from negpy.features.rgbscan.models import RgbScanConfig

                align = bool(file_info.get("align", self.state.config.rgbscan.align))
                self.state.config = replace(
                    self.state.config, rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=align)
                )

            self.file_selected.emit(file_info["path"])
            self.state_changed.emit()
            self._persist_session()

    def update_selection(self, indices: List[int]) -> None:
        """Updates the list of currently selected indices."""
        self.state.selected_indices = indices
        self.state_changed.emit()

    def sync_selected_settings(self, aspects: frozenset, scope: str = "selection") -> int:
        """
        Apply the active frame's settings to other frames. Returns the count changed.

        aspects: subset of _VALID_ASPECTS (process/crop/rotation/exposure/color/
                 finish/bounds_luma/bounds_colour), checked independently.
        scope:   "selection" (the multi-selected frames) or "roll" (all loaded frames).
        """
        aspects = frozenset(aspects) & _VALID_ASPECTS
        if self.state.selected_file_idx == -1 or not aspects:
            return 0

        source_config = self.state.config

        src_bounds = None
        needs_bounds = bool(aspects & {"bounds_luma", "bounds_colour"})
        if needs_bounds:
            src_bounds = _source_effective_bounds(source_config.process)
            if src_bounds is None:
                self.settings_synced.emit("Render the source frame before syncing bounds")
                return 0

        target_indices = self.asset_model.visible_actual_indices_ordered() if scope == "roll" else self.state.selected_indices

        count = 0
        for idx in target_indices:
            if idx == self.state.selected_file_idx or not (0 <= idx < len(self.state.uploaded_files)):
                continue
            target_hash = self.state.uploaded_files[idx]["hash"]
            target_config = self.repo.load_file_settings(target_hash) or WorkspaceConfig()
            self.repo.save_file_settings(target_hash, build_synced_config(source_config, target_config, aspects, src_bounds))
            count += 1

        if count:
            label = ", ".join(_ASPECT_LABELS[a] for a in _ASPECT_LABELS if a in aspects)
            if scope == "roll":
                msg = f"{label} synced to whole roll ({count} frames)"
            else:
                msg = f"{label} synced to {count} frame{'s' if count != 1 else ''}"
            self.settings_synced.emit(msg)
            self.settings_saved.emit()
        return count

    def next_file(self) -> None:
        display_idx = self.asset_model.actual_to_display(self.state.selected_file_idx)
        if display_idx == -1:
            return
        if display_idx < self.asset_model.rowCount() - 1:
            self.select_file(self.asset_model.display_to_actual(display_idx + 1))

    def prev_file(self) -> None:
        display_idx = self.asset_model.actual_to_display(self.state.selected_file_idx)
        if display_idx == -1:
            return
        if display_idx > 0:
            self.select_file(self.asset_model.display_to_actual(display_idx - 1))

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

    def jump_to_step(self, index: int) -> None:
        """Load an arbitrary history step (random-access undo/redo)."""
        if not self.state.current_file_hash:
            return
        if index == self.state.undo_index or not (0 <= index <= self.state.max_history_index):
            return

        # Preserve the live top before stepping away (same guard as undo()).
        if self.state.undo_index == self.state.max_history_index:
            self.repo.save_history_step(self.state.current_file_hash, self.state.undo_index, self.state.config)

        config = self.repo.load_history_step(self.state.current_file_hash, index)
        if config is None:
            return
        self.state.undo_index = index
        self.state.config = config
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
        from negpy.features.finish.models import FinishConfig
        from negpy.features.geometry.models import GeometryConfig
        from negpy.features.lab.models import LabConfig
        from negpy.features.local.models import LocalAdjustmentsConfig
        from negpy.features.process.models import ProcessConfig
        from negpy.features.retouch.models import RetouchConfig
        from negpy.features.toning.models import ToningConfig

        defaults = {
            "exposure": ExposureConfig(),
            "lab": LabConfig(),
            "local": LocalAdjustmentsConfig(),
            "toning": ToningConfig(),
            "geometry": GeometryConfig(),
            "process": ProcessConfig(),
            "retouch": RetouchConfig(),
            "finish": FinishConfig(),
        }
        if section not in defaults:
            return
        new_config = replace(self.state.config, **{section: defaults[section]})
        if section == "local":
            self.state.local_selected_mask = -1
        self.update_config(new_config, persist=True)

    def copy_settings(self, include_bounds: bool = False) -> None:
        import copy

        cfg = copy.deepcopy(self.state.config)
        if not include_bounds:
            cfg = replace(
                cfg,
                process=replace(
                    cfg.process,
                    local_floors=(0.0, 0.0, 0.0),
                    local_ceils=(0.0, 0.0, 0.0),
                    lock_bounds=False,
                ),
            )
        self.state.clipboard = cfg
        self.state_changed.emit()
        self.settings_copied.emit()

    def copy_settings_with_bounds(self) -> None:
        self.copy_settings(include_bounds=True)

    def paste_settings(self) -> None:
        if self.state.clipboard and self.state.current_file_hash:
            import copy

            self.update_config(copy.deepcopy(self.state.clipboard), persist=True)
            self.settings_pasted.emit()

    def _persist_session(self) -> None:
        """Saves the open-file manifest (paths + active) for restore on next launch."""
        paths = [f["path"] for f in self.state.uploaded_files]
        self.repo.save_global_setting("session_files", paths)
        self.repo.save_global_setting("session_active_path", self.state.current_file_path)
        # RGB-scan triplets keep their green/blue exposures here so restore can rebuild
        # the merged asset (re-discovery from the red path alone cannot regroup it).
        triplets = {
            f["path"]: [f["green_path"], f["blue_path"], bool(f.get("align", True))]
            for f in self.state.uploaded_files
            if f.get("green_path") and f.get("blue_path")
        }
        self.repo.save_global_setting("session_triplets", triplets)

    def add_files(self, file_paths: List[str], validated_info: Optional[List[Dict]] = None) -> None:
        """
        Adds new files to the session.
        """
        import os

        from negpy.kernel.image.logic import calculate_file_hash

        if validated_info:
            for info in validated_info:
                same_path_idx = next(
                    (i for i, existing in enumerate(self.state.uploaded_files) if existing["path"] == info["path"]),
                    None,
                )
                if same_path_idx is not None:
                    old = self.state.uploaded_files[same_path_idx]
                    self.state.thumbnails.pop(old["name"], None)
                    self.state.uploaded_files[same_path_idx] = info
                    continue
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
        self.files_changed.emit()
        self._persist_session()

    def set_triplet(self, index: int, red_path: str, green_path: str, blue_path: str, align: bool = True) -> None:
        """Reassign the R/G/B exposures of an RGB-scan asset, then reload it."""
        import os

        from negpy.kernel.image.logic import calculate_file_hash

        if not (0 <= index < len(self.state.uploaded_files)):
            return
        name = os.path.splitext(os.path.basename(red_path))[0] + " (RGB)"
        self.state.uploaded_files[index] = {
            "name": name,
            "path": red_path,
            "hash": calculate_file_hash(red_path),
            "green_path": green_path,
            "blue_path": blue_path,
            "align": align,
        }
        self.asset_model.refresh()
        self.files_changed.emit()
        self.select_file(index)

    def _reset_active_image_state(self) -> None:
        """Clears everything tied to the previously displayed image after the session
        emptied, then announces it via `session_emptied` so the viewer blanks the
        stale frame instead of keeping an image that can no longer be removed."""
        self.state.selected_file_idx = -1
        self.state.selected_indices = []
        self.state.current_file_path = None
        self.state.current_file_hash = None
        self.state.preview_raw = None
        self.state.preview_ir = None
        self.state.has_ir = False
        self.state.config = WorkspaceConfig()
        self._config_dirty = False
        with self.state.metrics_lock:
            self.state.last_metrics.clear()
        self.session_emptied.emit()

    def clear_files(self) -> None:
        """
        Purges all loaded files from the session.
        """
        self.state.uploaded_files.clear()
        self.state.thumbnails.clear()
        self._reset_active_image_state()

        self.asset_model.refresh()
        self.state_changed.emit()
        self._persist_session()

    def remove_current_file(self) -> None:
        """
        Removes the currently selected file from the session.
        """
        idx = self.state.selected_file_idx
        if 0 <= idx < len(self.state.uploaded_files):
            file_info = self.state.uploaded_files.pop(idx)
            self.state.thumbnails.pop(file_info["name"], None)

            if not self.state.uploaded_files:
                self._reset_active_image_state()
            else:
                new_idx = min(idx, len(self.state.uploaded_files) - 1)
                self.select_file(new_idx)

            self.asset_model.refresh()
            self.state_changed.emit()
            self._persist_session()

    def remove_selected_files(self) -> None:
        """
        Removes all currently selected files from the session.
        """
        indices = sorted(set(self.state.selected_indices), reverse=True)
        if not indices:
            return

        for idx in indices:
            if 0 <= idx < len(self.state.uploaded_files):
                file_info = self.state.uploaded_files.pop(idx)
                self.state.thumbnails.pop(file_info["name"], None)

        if not self.state.uploaded_files:
            self._reset_active_image_state()
        else:
            new_idx = min(min(indices), len(self.state.uploaded_files) - 1)
            self.select_file(new_idx)

        self.asset_model.refresh()
        self.state_changed.emit()
        self._persist_session()
