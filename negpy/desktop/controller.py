import os
import time
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QCheckBox, QMessageBox

from negpy.desktop.converters import ImageConverter
from negpy.desktop.render_memo import RenderMemo
from negpy.desktop.session import AppState, DesktopSessionManager, ToolMode, resolve_asset_rgbscan
from negpy.desktop.workers.export import ExportTask, ExportWorker, find_export_conflicts
from negpy.desktop.workers.render import (
    AssetDiscoveryTask,
    AssetDiscoveryWorker,
    BatchAutoCropInput,
    BatchAutoCropResult,
    BatchAutoCropTask,
    BatchAutoCropWorker,
    NormalizationTask,
    NormalizationWorker,
    PreviewLoadTask,
    PreviewLoadWorker,
    RenderTask,
    RenderWorker,
    ThumbnailUpdateTask,
    ThumbnailWorker,
)
from negpy.desktop.workers.scan_worker import ScanRequest, ScanWorker
from negpy.desktop.workers.capture_worker import (
    CalibrationRequest,
    CaptureRequest,
    CaptureWorker,
    LiveViewRequest,
)
from negpy.domain.models import (
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
    WorkspaceConfig,
    canonical_crop_ratio,
    export_blocked,
    flat_export_config,
    flat_master_config,
    preset_from_export_config,
    resolve_preset_export,
)
from negpy.services.assets.half_frame import base_hash, slice_half
from negpy.services.assets.sidecar import load_or_promote, write_sidecar
from negpy.features.exposure.logic import (
    calculate_wb_shifts,
    calculate_wb_shifts_from_log,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.geometry.logic import apply_fine_rotation, detect_closest_aspect_ratio, enforce_roi_aspect_ratio
from negpy.features.geometry.models import FINE_ROTATION_LIMIT, AutocropMode
from negpy.features.lab.models import LabConfig
from negpy.features.local.models import LocalAdjustmentsConfig
from negpy.features.process.models import ProcessConfig, ProcessMode, invalidate_local_bounds
from negpy.kernel.system.paths import get_resource_path
from negpy.features.retouch.logic import fallback_source_offset, select_source_offset
from negpy.features.retouch.models import HEAL_SIZE_REF, RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)

_THUMB_FAILED_MSG = "thumbnail failed — file may be unreadable"


@dataclass(frozen=True)
class _PendingCaptureImport:
    """Capture intent carried across asynchronous discovery and session hydration."""

    process_mode: Optional[ProcessMode] = None
    detect_mode: bool = False


def _capture_import_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _autocrop_fingerprint(config: WorkspaceConfig, workspace_color_space: str) -> tuple:
    """Identity of every setting that changes detection pixels or crop coordinates."""
    geometry = config.geometry
    flatfield = config.flatfield
    rgbscan = config.rgbscan
    return (
        int(geometry.rotation),
        round(float(geometry.fine_rotation), 7),
        bool(geometry.flip_horizontal),
        bool(geometry.flip_vertical),
        str(geometry.autocrop_mode),
        str(geometry.autocrop_ratio),
        int(geometry.autocrop_offset),
        bool(flatfield.apply),
        str(flatfield.reference_path),
        round(float(flatfield.k1), 9),
        bool(config.process.linear_raw),
        bool(rgbscan.enabled),
        str(rgbscan.green_path),
        str(rgbscan.blue_path),
        bool(rgbscan.align),
        str(workspace_color_space),
    )


@dataclass(frozen=True)
class _DiscoveryRequest:
    paths: tuple[str, ...]
    auto_open: bool
    restore_triplets: Optional[dict]
    replace_existing: bool
    reselect_path: Optional[str]
    rgb_scan: bool
    half_frame: bool


def baseline_compare_config(config: WorkspaceConfig) -> WorkspaceConfig:
    """
    The 'before' config for the before/after view: reset the creative sections to defaults
    while keeping process (mode + normalization bounds), geometry/crop, export and metadata,
    so it shows the un-graded auto conversion of the same framed image.
    """
    return replace(
        config,
        exposure=ExposureConfig(),
        lab=LabConfig(),
        local=LocalAdjustmentsConfig(),
        toning=ToningConfig(),
        finish=FinishConfig(),
        retouch=RetouchConfig(),
    )


def history_step_label(prev: Optional[WorkspaceConfig], config: WorkspaceConfig, index: int) -> str:
    """List label for a history step: index + which config sections changed vs. the previous step."""
    if prev is None:
        return f"{index} · base"
    changed = [f.name for f in fields(config) if getattr(prev, f.name) != getattr(config, f.name)]
    return f"{index} · {', '.join(changed)}" if changed else f"{index} · —"


class AppController(QObject):
    """
    Main application orchestrator.
    Manages UI state synchronization, background workers, and render flow.
    """

    image_updated = pyqtSignal()
    preview_loaded = pyqtSignal()
    metrics_available = pyqtSignal(dict)
    loading_started = pyqtSignal()
    load_failed = pyqtSignal()
    export_progress = pyqtSignal(int, int, str)
    export_finished = pyqtSignal(float, int)
    render_requested = pyqtSignal(RenderTask)
    preview_load_requested = pyqtSignal(PreviewLoadTask)
    normalization_requested = pyqtSignal(NormalizationTask)
    batch_autocrop_requested = pyqtSignal(BatchAutoCropTask)
    analysis_buffer_preview_requested = pyqtSignal(float)
    rotation_guide_requested = pyqtSignal()
    crop_guide_changed = pyqtSignal()
    dust_overlay_changed = pyqtSignal()
    asset_discovery_requested = pyqtSignal(AssetDiscoveryTask)
    thumbnail_requested = pyqtSignal(list)
    thumbnail_update_requested = pyqtSignal(ThumbnailUpdateTask)
    tool_sync_requested = pyqtSignal()
    config_updated = pyqtSignal()
    monitor_profile_changed = pyqtSignal()
    compare_changed = pyqtSignal(bool)
    flat_output_changed = pyqtSignal(bool)
    flat_peek_changed = pyqtSignal(bool)
    zoom_requested = pyqtSignal(float)
    zoom_changed = pyqtSignal(float)
    _render_cleanup_requested = pyqtSignal()
    status_message_requested = pyqtSignal(str, int)
    status_progress_requested = pyqtSignal(int, int)
    batch_started = pyqtSignal(str, bool)  # title, abortable
    batch_progress = pyqtSignal(int, int, str)  # current, total, label
    batch_finished = pyqtSignal()
    pixel_readout_rgb = pyqtSignal(object)  # (r255, g255, b255) tuple or None
    densitometer_readout = pyqtSignal(object)  # DensitometerReading or None
    tone_drag_changed = pyqtSignal(str)  # exposure field being slider-dragged; "" = drag ended
    scan_devices_requested = pyqtSignal()
    scan_requested = pyqtSignal(ScanRequest)
    scan_devices_ready = pyqtSignal(list)
    scan_progress = pyqtSignal(float)
    scan_finished = pyqtSignal(str)
    scan_error = pyqtSignal(str)
    scan_started = pyqtSignal()
    capture_light_requested = pyqtSignal(int, int, int, int, str)
    capture_requested = pyqtSignal(CaptureRequest)
    capture_light_set = pyqtSignal(int, int, int, int)
    capture_progress = pyqtSignal(float)
    capture_finished = pyqtSignal(list)
    capture_cancelled = pyqtSignal()
    capture_error = pyqtSignal(str)
    capture_status = pyqtSignal(str)
    live_view_requested = pyqtSignal(LiveViewRequest)
    live_view_stop_requested = pyqtSignal()
    camera_session_close_requested = pyqtSignal()
    live_view_focus_magnifier_requested = pyqtSignal(bool)
    live_view_focus_magnifier_pos_requested = pyqtSignal(int, int)
    live_view_camera_setting_requested = pyqtSignal(str, int)
    capture_live_view_started = pyqtSignal(str)
    calibration_requested = pyqtSignal(CalibrationRequest)
    capture_calibration_progress = pyqtSignal(float, str)
    capture_calibration_finished = pyqtSignal(object)
    capture_calibration_exposure = pyqtSignal(str)  # "over"/"under": target unreachable, aborted, no preset
    poll_connection_requested = pyqtSignal(str)  # light port (auto-poll)
    connection_polled = pyqtSignal(dict)  # {usb_ok, usb_model, light_ok, light_detail}
    poll_light_temp_requested = pyqtSignal(str)  # light port (temp-only poll, runs even mid-live-view)
    light_temp_polled = pyqtSignal(object)  # Scanlight LED temperature °C, or None

    def __init__(self, session_manager: DesktopSessionManager):
        super().__init__()
        self.session = session_manager
        self.state: AppState = session_manager.state
        self._thumb_config: Optional[WorkspaceConfig] = None
        self._first_render_t0: Optional[float] = None
        self._export_start_time = 0.0
        self._export_failures = 0
        self._discovery_running = False
        self._auto_open_after_discovery = False
        self._replace_after_discovery = False
        self._reselect_after_discovery: Optional[str] = None
        self._pending_capture_imports: Dict[str, _PendingCaptureImport] = {}
        self._pending_asset_discoveries: List[_DiscoveryRequest] = []
        self._active_discovery_keys: frozenset[str] = frozenset()
        self._pending_scanned_file: Optional[str] = None
        self._gpu_fallback_notified = False
        self._cleaned_up = False
        self._active_batch: Optional[str] = None
        self._active_batch_title = ""
        self._active_batch_abortable = False
        self._batch_serial = 0
        self._active_batch_token: Optional[int] = None
        self._autocrop_batch_token: Optional[int] = None
        self._autocrop_dispatched = 0
        self._autocrop_preflight_skipped = 0
        self._autocrop_cancel_requested = False

        self.preview_service = PreviewManager()
        self.batch_autocrop_preview_service = PreviewManager()
        self.watcher = FolderWatchService()
        self.asset_store = LocalAssetStore(APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir)
        self.asset_store.initialize()

        # Thread management
        self.render_thread = QThread()
        self.render_worker = RenderWorker()
        self.render_worker.moveToThread(self.render_thread)
        self.render_thread.start()

        self.export_thread = QThread()
        self.export_worker = ExportWorker()
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.start()

        self.thumb_thread = QThread()
        self.thumb_worker = ThumbnailWorker(self.asset_store)
        self.thumb_worker.moveToThread(self.thumb_thread)
        self.thumb_thread.start()

        self.norm_thread = QThread()
        self.norm_worker = NormalizationWorker(self.preview_service, self.session.repo)
        self.norm_worker.moveToThread(self.norm_thread)
        self.batch_autocrop_worker = BatchAutoCropWorker(self.batch_autocrop_preview_service)
        self.batch_autocrop_worker.moveToThread(self.norm_thread)
        self.norm_thread.start()

        self.discovery_thread = QThread()
        self.discovery_worker = AssetDiscoveryWorker()
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_thread.start()

        self.preview_load_thread = QThread()
        self.preview_load_worker = PreviewLoadWorker(self.preview_service)
        self.preview_load_worker.moveToThread(self.preview_load_thread)
        self.preview_load_thread.start()

        self.scan_thread = QThread()
        self.scan_worker = ScanWorker()
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.start()

        self.capture_thread = QThread()
        self.capture_worker = CaptureWorker()
        self.capture_worker.moveToThread(self.capture_thread)
        # Started lazily on first capture use (_ensure_capture_thread): a *running* QThread aborts
        # if destroyed without quit(), and controller unit tests build AppController without ever
        # scanning — leaving the thread unstarted keeps it invisible to their teardown loops (so
        # upstream tests needn't know about it), and the app starts it the moment the Camera
        # Scanning tab polls or the user acts.
        self._capture_thread_started = False

        self.canvas: Any = None
        self._is_rendering = False
        self._pending_render_task: Any = None

        # Last displayed render per frame — navigate-back paints it instantly
        # while the authoritative render refreshes underneath.
        self._render_memo = RenderMemo()

        self._render_debounce = QTimer()
        self._render_debounce.setSingleShot(True)
        self._render_debounce.setInterval(80)
        self._render_debounce.timeout.connect(self.request_render)

        self._crop_bounds_dirty = False

        self._cursor_readout_timer = QTimer()
        self._cursor_readout_timer.setSingleShot(True)
        self._cursor_readout_timer.setInterval(33)
        self._cursor_readout_timer.timeout.connect(self._emit_pixel_readout)
        self._pending_cursor_nx: Optional[float] = None
        self._pending_cursor_ny: Optional[float] = None
        self._prefetch_gen = 0
        self._preview_load_t0 = 0.0
        self._requested_file_path: str = ""

        self._connect_signals()

    def register_canvas(self, canvas: Any) -> None:
        """
        Registers the canvas and connects its signals.
        """
        self.canvas = canvas
        self.zoom_requested.connect(self.canvas.set_zoom)
        self.canvas.zoom_changed.connect(self.zoom_changed.emit)
        self.canvas.cursor_position_changed.connect(self.on_cursor_moved)
        self.canvas.cursor_left_canvas.connect(self.on_cursor_left)

        from negpy.desktop.view.canvas.toolbar import CANVAS_COLORS

        idx = self.state.canvas_bg_index
        _, (r, g, b), _ = CANVAS_COLORS[idx]
        self.canvas.set_background_color(r, g, b)

    def on_cursor_moved(self, nx: float, ny: float) -> None:
        self._pending_cursor_nx = nx
        self._pending_cursor_ny = ny
        if not self._cursor_readout_timer.isActive():
            self._cursor_readout_timer.start()

    def on_cursor_left(self) -> None:
        self._pending_cursor_nx = None
        self._pending_cursor_ny = None
        self.pixel_readout_rgb.emit(None)
        self.densitometer_readout.emit(None)

    def _emit_pixel_readout(self) -> None:
        nx, ny = self._pending_cursor_nx, self._pending_cursor_ny
        if nx is None or ny is None or self.canvas is None:
            return
        rgb = self.canvas.get_pixel_rgb(nx, ny)
        if rgb is None:
            return
        r, g, b = rgb
        r255 = int(round(max(0.0, min(1.0, r)) * 255))
        g255 = int(round(max(0.0, min(1.0, g)) * 255))
        b255 = int(round(max(0.0, min(1.0, b)) * 255))
        self.pixel_readout_rgb.emit((r255, g255, b255))
        self.densitometer_readout.emit(self._compute_densitometer_reading(nx, ny, rgb))

    def _compute_densitometer_reading(self, nx: float, ny: float, display_rgb: tuple) -> Optional[Any]:
        """Probe the normalized-log frame under the cursor; None when unavailable."""
        from negpy.features.exposure.densitometer import compute_reading, map_display_to_norm

        metrics = self.state.last_metrics
        nl = metrics.get("normalized_log")
        bounds = metrics.get("final_bounds") or metrics.get("log_bounds")
        if nl is None or bounds is None or self.canvas is None:
            return None
        disp = self.canvas.display_size()
        if disp is None:
            return None
        if isinstance(nl, np.ndarray):
            norm_h, norm_w = nl.shape[:2]
        else:
            norm_w, norm_h = nl.width, nl.height
        pos = map_display_to_norm(
            nx,
            ny,
            disp[0],
            disp[1],
            self.canvas.content_rect(),
            metrics.get("active_roi"),
            self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW),
            norm_w,
            norm_h,
        )
        if pos is None:
            return None
        x, y = pos
        try:
            if isinstance(nl, np.ndarray):
                val = nl[y, x]
            else:
                val = nl.readback_region(x, y, 1, 1)[0, 0]
        except Exception:
            return None
        return compute_reading((float(val[0]), float(val[1]), float(val[2])), bounds, display_rgb)

    def set_status(self, message: str, timeout: int = 0) -> None:
        self.status_message_requested.emit(message, timeout)

    def _connect_signals(self) -> None:
        self.render_requested.connect(self.render_worker.process)
        self._render_cleanup_requested.connect(self.render_worker.cleanup)
        self.render_worker.finished.connect(self._on_render_finished)
        self.render_worker.metrics_updated.connect(self._on_metrics_updated)
        self.render_worker.error.connect(self._on_render_error)

        self.export_worker.progress.connect(self.export_progress.emit)
        self.export_worker.progress.connect(self._on_batch_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.cancelled.connect(self._on_export_batch_cancelled)
        self.export_worker.error.connect(self._on_render_error)
        self.export_worker.error.connect(self._on_export_task_error)

        self.thumbnail_requested.connect(self.thumb_worker.generate)
        self.thumb_worker.progress.connect(self._on_thumbnail_progress)
        self.thumbnail_update_requested.connect(self.thumb_worker.update_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_worker.error.connect(self._on_render_error)
        self.thumb_worker.error.connect(self._on_thumbnail_batch_error)

        self.normalization_requested.connect(self.norm_worker.process)
        self.norm_worker.progress.connect(self._on_normalization_progress)
        self.norm_worker.finished.connect(self._on_normalization_finished)
        self.norm_worker.cancelled.connect(self._on_normalization_cancelled)
        self.norm_worker.error.connect(self._on_render_error)
        self.norm_worker.error.connect(self._on_normalization_error)

        self.batch_autocrop_requested.connect(self.batch_autocrop_worker.process)
        self.batch_autocrop_worker.progress.connect(self._on_batch_autocrop_progress)
        self.batch_autocrop_worker.finished.connect(self._on_batch_autocrop_finished)
        self.batch_autocrop_worker.cancelled.connect(self._on_batch_autocrop_cancelled)
        self.batch_autocrop_worker.error.connect(self._on_batch_autocrop_error)

        self.asset_discovery_requested.connect(self.discovery_worker.process)
        self.discovery_worker.progress.connect(self._on_discovery_progress)
        self.discovery_worker.finished.connect(self._on_discovery_finished)
        self.discovery_worker.error.connect(self._on_render_error)
        self.discovery_worker.error.connect(self._on_discovery_batch_error)

        self.preview_load_requested.connect(self.preview_load_worker.process)
        self.preview_load_worker.splash.connect(self._on_splash_preview)
        self.preview_load_worker.finished.connect(self._on_preview_loaded)
        self.preview_load_worker.error.connect(self._on_render_error)
        self.preview_load_worker.load_failed.connect(self._on_preview_load_failed)

        self.scan_devices_requested.connect(self.scan_worker.list_devices)
        self.scan_worker.devices_ready.connect(self.scan_devices_ready.emit)
        self.scan_worker.progress.connect(self.scan_progress.emit)
        self.scan_worker.finished.connect(self._on_scan_finished)
        self.scan_worker.error.connect(self.scan_error.emit)
        self.scan_requested.connect(self.scan_worker.run_scan)
        self.capture_light_requested.connect(self.capture_worker.set_light)
        self.capture_requested.connect(self.capture_worker.run_capture)
        self.capture_worker.light_set.connect(self.capture_light_set.emit)
        self.capture_worker.progress.connect(self.capture_progress.emit)
        self.capture_worker.finished.connect(self._on_capture_finished)
        self.capture_worker.cancelled.connect(self.capture_cancelled.emit)
        self.capture_worker.error.connect(self.capture_error.emit)
        self.capture_worker.status.connect(self.capture_status.emit)
        self.live_view_requested.connect(self.capture_worker.start_live_view)
        self.live_view_stop_requested.connect(self.capture_worker.stop_live_view)
        self.camera_session_close_requested.connect(self.capture_worker.close_camera_session)
        self.live_view_focus_magnifier_requested.connect(self.capture_worker.set_focus_magnifier)
        self.live_view_focus_magnifier_pos_requested.connect(self.capture_worker.set_focus_magnifier_pos)
        self.live_view_camera_setting_requested.connect(self.capture_worker.set_camera_setting)
        self.capture_worker.live_view_started.connect(self.capture_live_view_started.emit)
        self.calibration_requested.connect(self.capture_worker.run_calibration)
        self.capture_worker.calibration_progress.connect(self.capture_calibration_progress.emit)
        self.capture_worker.calibration_finished.connect(self.capture_calibration_finished.emit)
        self.capture_worker.calibration_exposure.connect(self.capture_calibration_exposure.emit)
        self.poll_connection_requested.connect(self.capture_worker.poll_connection)
        self.capture_worker.poll_status.connect(self.connection_polled.emit)
        self.poll_light_temp_requested.connect(self.capture_worker.poll_light_temp)
        self.capture_worker.light_temp_polled.connect(self.light_temp_polled.emit)

        self.session.active_file_changing.connect(lambda: self._update_thumbnail_from_state(force_readback=True))
        self.session.session_emptied.connect(self._render_memo.clear)
        self.session.file_selected.connect(self.load_file)
        self.session.state_changed.connect(self.config_updated.emit)
        self.session.state_changed.connect(self._render_debounce.start)
        self.session.files_changed.connect(self._render_debounce.start)

    def generate_missing_thumbnails(self) -> None:
        missing = [f for f in self.state.uploaded_files if f["name"] not in self.state.thumbnails]
        if missing:
            if self._begin_batch("thumbnails", "Generating thumbnails", abortable=False) is None:
                return
            self._thumb_requested = [f["name"] for f in missing]
            self.set_status("GENERATING THUMBNAILS...")
            self.thumbnail_requested.emit(missing)

    def _on_thumbnail_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"THUMBNAIL {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _on_thumbnails_finished(self, new_thumbs: Dict[str, Any]) -> None:
        self.status_progress_requested.emit(0, 0)
        self._end_batch("thumbnails")
        for name, pil_img in new_thumbs.items():
            if pil_img:
                u8_arr = np.array(pil_img.convert("RGB"))
                self.state.thumbnails[name] = QIcon(QPixmap.fromImage(ImageConverter.to_qimage(u8_arr)))

        # Consume the request list: update_rendered() re-emits this same signal with
        # single-file dicts after every settled render, and evaluating those against
        # a stale batch list would falsely badge every other frame.
        requested = getattr(self, "_thumb_requested", [])
        self._thumb_requested = []
        failed = {n for n in requested if not new_thumbs.get(n)}
        for f in self.state.uploaded_files:
            if f["name"] in failed:
                f.setdefault("decode_failed", _THUMB_FAILED_MSG)
            elif f["name"] in new_thumbs and f.get("decode_failed") == _THUMB_FAILED_MSG:
                del f["decode_failed"]
        self.session.asset_model.refresh()

    # --- Batch progress popup -------------------------------------------------

    def _begin_batch(self, owner: str, title: str, abortable: bool) -> Optional[int]:
        """Claim the shared batch lane and return its generation token."""
        if self._active_batch is not None:
            self.set_status(f"{self._active_batch_title} is already running", 3000)
            return None
        self._batch_serial += 1
        self._active_batch = owner
        self._active_batch_title = title
        self._active_batch_abortable = abortable
        self._active_batch_token = self._batch_serial
        self.batch_started.emit(title, abortable)
        return self._active_batch_token

    def _batch_busy(self, requested: str) -> bool:
        if self._active_batch is None:
            return False
        self.set_status(f"Cannot start {requested} while {self._active_batch_title} is running", 3000)
        return True

    def _end_batch(self, owner: str, token: Optional[int] = None) -> bool:
        """Release only the batch generation that owns the progress lane."""
        if self._active_batch != owner:
            return False
        if token is not None and token != self._active_batch_token:
            return False
        self._active_batch = None
        self._active_batch_title = ""
        self._active_batch_abortable = False
        self._active_batch_token = None
        self.batch_finished.emit()
        if self._pending_asset_discoveries and not self._discovery_running:
            QTimer.singleShot(0, self._start_next_asset_discovery)
        return True

    def _on_batch_progress(self, current: int, total: int, name: str) -> None:
        self.batch_progress.emit(current, total, name)

    def _on_batch_cancelled(self, owner: str) -> None:
        self.set_status("Aborted", 3000)
        self._end_batch(owner)

    def _on_export_batch_cancelled(self) -> None:
        owner = self._active_batch if self._active_batch in ("export", "contact_sheet") else "export"
        self._on_batch_cancelled(owner)

    def _on_discovery_batch_error(self, _message: str) -> None:
        self._discovery_running = False
        self._end_batch("discovery")

    def _on_thumbnail_batch_error(self, _message: str) -> None:
        self._on_batch_error("thumbnails")

    def _on_normalization_cancelled(self) -> None:
        self._on_batch_cancelled("normalization")

    def _on_normalization_error(self, _message: str) -> None:
        self._on_batch_error("normalization")

    def _on_batch_error(self, owner: str) -> None:
        self._end_batch(owner)

    def abort_active_batch(self) -> None:
        """Requests cancellation of the running abortable batch (export or analysis)."""
        if self._active_batch in ("export", "contact_sheet"):
            self.export_worker.cancel()
        elif self._active_batch == "normalization":
            self.norm_worker.cancel()
        elif self._active_batch == "autocrop":
            self._autocrop_cancel_requested = True
            self.batch_autocrop_worker.cancel(self._autocrop_batch_token)

    def saved_session_paths(self) -> List[str]:
        """Returns last session's file paths that still exist on disk."""
        paths = self.session.repo.get_global_setting("session_files", []) or []
        return [p for p in paths if os.path.exists(p)]

    def restore_session(self) -> None:
        """Re-loads the previous session's files and reselects the active one."""
        paths = self.saved_session_paths()
        if not paths:
            return
        active = self.session.repo.get_global_setting("session_active_path")
        self._pending_scanned_file = active if active in paths else paths[0]
        triplets = self.session.repo.get_global_setting("session_triplets", {}) or {}
        self.request_asset_discovery(paths, auto_open=True, restore_triplets=triplets)

    def request_asset_discovery(
        self,
        paths: List[str],
        auto_open: bool = False,
        restore_triplets: Optional[dict] = None,
        replace_existing: bool = False,
        reselect_path: Optional[str] = None,
    ) -> None:
        """
        Starts asynchronous discovery of supported assets.
        Requests arriving while hashing is in progress are queued in order.

        `replace_existing` rebuilds the asset list from the results (instead of
        appending) and reselects `reselect_path` — used when re-running discovery
        over already-loaded files (e.g. an RGB-scan mode toggle).
        """
        request = _DiscoveryRequest(
            paths=tuple(paths),
            auto_open=auto_open,
            restore_triplets=restore_triplets,
            replace_existing=replace_existing,
            reselect_path=reselect_path,
            rgb_scan=bool(self.session.repo.get_global_setting("rgbscan_mode", False)),
            half_frame=bool(self.session.repo.get_global_setting("half_frame_mode", False)),
        )
        if self._discovery_running:
            self._pending_asset_discoveries.append(request)
            return

        if self._active_batch is not None:
            self._pending_asset_discoveries.append(request)
            self.set_status(f"Queued asset discovery until {self._active_batch_title} finishes", 3000)
            return

        self._start_asset_discovery(request)

    def _start_asset_discovery(self, request: _DiscoveryRequest) -> None:
        """Start one request; callers ensure only one discovery is active."""

        from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS

        if self._begin_batch("discovery", "Hashing files", abortable=False) is None:
            self._pending_asset_discoveries.insert(0, request)
            return
        self._discovery_running = True
        self._auto_open_after_discovery = request.auto_open
        self._replace_after_discovery = request.replace_existing
        self._reselect_after_discovery = request.reselect_path
        self._active_discovery_keys = frozenset(_capture_import_key(path) for path in request.paths)
        self.set_status("SCANNING FOR ASSETS...")
        task = AssetDiscoveryTask(
            paths=list(request.paths),
            supported_extensions=tuple(SUPPORTED_RAW_EXTENSIONS),
            rgb_scan=request.rgb_scan,
            restore_triplets=request.restore_triplets,
            half_frame=request.half_frame,
        )
        self.asset_discovery_requested.emit(task)

    def _start_next_asset_discovery(self) -> None:
        if self._pending_asset_discoveries and not self._discovery_running and self._active_batch is None:
            self._start_asset_discovery(self._pending_asset_discoveries.pop(0))

    def set_rgb_scan_mode(self, enabled: bool) -> None:
        """Persist the RGB-scan toggle and re-discover already-loaded assets so the
        mode regroups/ungroups triplets in place (not only on the next folder load)."""
        self.session.repo.save_global_setting("rgbscan_mode", bool(enabled))
        if enabled:
            # Narrowband LEDs are what RGB-scan triplets are captured with — correcting
            # for them is the point of the toggle, so switch it on together.
            self.session.repo.save_global_setting("last_narrowband_scan", True)
        files = self.session.state.uploaded_files
        if not files:
            return
        if enabled and not self.state.config.process.narrowband_scan:
            self.session.update_config(
                replace(self.state.config, process=replace(self.state.config.process, narrowband_scan=True)), persist=True
            )
            self.request_render()
        paths: List[str] = []
        for f in files:
            paths.append(f["path"])
            for k in ("green_path", "blue_path"):
                if f.get(k):
                    paths.append(f[k])
        self.request_asset_discovery(paths, replace_existing=True, reselect_path=self.state.current_file_path)

    def set_half_frame_mode(self, enabled: bool) -> None:
        """Persist the half-frame toggle and re-discover already-loaded assets so the
        mode splits/collapses frames in place (not only on the next folder load)."""
        self.session.repo.save_global_setting("half_frame_mode", bool(enabled))
        files = self.session.state.uploaded_files
        if not files:
            return
        paths: List[str] = []
        for f in files:
            paths.append(f["path"])
            for k in ("green_path", "blue_path"):
                if f.get(k):
                    paths.append(f[k])
        self.request_asset_discovery(paths, replace_existing=True, reselect_path=self.state.current_file_path)

    def _on_discovery_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"HASHING {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _on_discovery_finished(self, valid_assets: List[Dict]) -> None:
        """
        Adds discovered assets to the session and starts thumbnail generation.
        """
        ended_batch = self._end_batch("discovery")
        if not ended_batch and self._active_batch is None:
            # Preserve the completion signal for direct invocations and late
            # delivery without releasing a newer batch owner.
            self.batch_finished.emit()
        self.status_progress_requested.emit(0, 0)
        self._discovery_running = False
        auto_open = self._auto_open_after_discovery
        self._auto_open_after_discovery = False
        replace_existing = self._replace_after_discovery
        reselect_path = self._reselect_after_discovery
        self._replace_after_discovery = False
        self._reselect_after_discovery = None
        active_discovery_keys = self._active_discovery_keys
        self._active_discovery_keys = frozenset()
        pending_scan = getattr(self, "_pending_scanned_file", None)

        if replace_existing and valid_assets:
            # Re-run over already-loaded files (e.g. RGB-scan toggle): rebuild the list
            # so dedup-by-hash doesn't drop a regrouped red, then reselect the active frame.
            self.session.state.uploaded_files.clear()
            self.session.add_files([], validated_info=valid_assets)
            self.generate_missing_thumbnails()
            idx = next(
                (
                    i
                    for i, f in enumerate(self.session.state.uploaded_files)
                    if reselect_path in (f.get("path"), f.get("green_path"), f.get("blue_path"))
                ),
                0,
            )
            self.session.select_file(idx)
            self._start_next_asset_discovery()
            return

        selected_pending_scan = False
        if valid_assets:
            first_new_idx = len(self.session.state.uploaded_files)
            self.session.add_files([], validated_info=valid_assets)
            self.generate_missing_thumbnails()
            if pending_scan and self._select_file_by_path(pending_scan):
                selected_pending_scan = True
            elif auto_open and not self.state.current_file_path and len(self.session.state.uploaded_files) > first_new_idx:
                self.session.select_file(first_new_idx)
        else:
            self.set_status("NO SUPPORTED ASSETS FOUND", 3000)
            self.status_progress_requested.emit(0, 0)

        if pending_scan:
            pending_key = _capture_import_key(pending_scan)
            if selected_pending_scan:
                # select_file emits load_file synchronously in the real session. Pop again
                # as a fallback for alternate session implementations and tests.
                self._pending_capture_imports.pop(pending_key, None)
                self._pending_scanned_file = None
            elif pending_key in active_discovery_keys:
                # This request finished without the intended primary asset. Drop only its
                # metadata; a later capture may already be waiting in the FIFO queue.
                self._pending_capture_imports.pop(pending_key, None)
                self._pending_scanned_file = None
        self._start_next_asset_discovery()

    def _file_hash_for_path(self, file_path: str) -> Optional[str]:
        if self.state.current_file_path == file_path and self.state.current_file_hash:
            return self.state.current_file_hash
        for f in self.state.uploaded_files:
            if f.get("path") == file_path:
                return f.get("hash")
        return None

    def _active_half(self) -> Optional[tuple[int, float]]:
        """(half, split_x) of the active asset, or None for whole-frame assets."""
        h = self.state.current_file_hash
        if not h:
            return None
        for f in self.state.uploaded_files:
            if f.get("hash") == h:
                half = int(f.get("half") or 0)
                return (half, float(f.get("split_x") or 0.5)) if half else None
        return None

    def _render_memo_key(self) -> str:
        """Identity of everything that shapes the displayed render of the current
        config: the edit itself plus every display-path input. Any mismatch is a
        memo miss, so navigate-back only skips straight to pixels that would be
        reproduced exactly."""
        import hashlib
        import json

        proofing = self.state.soft_proof_enabled
        narrowband = self.state.config.process.narrowband_scan
        parts = (
            json.dumps(self.state.config.to_dict(), sort_keys=True, default=str),
            self.state.hq_preview,
            self.state.workspace_color_space,
            self.state.gpu_enabled,
            proofing,
            self.effective_input_icc() if (proofing or narrowband) else None,
            self.effective_output_icc() if proofing else None,
            hashlib.md5(self.state.monitor_icc_bytes).hexdigest() if self.state.monitor_icc_bytes else "",
        )
        return hashlib.md5(repr(parts).encode()).hexdigest()

    def load_file(self, file_path: str, preserve_zoom: bool = False, force_detect: bool = False) -> None:
        """
        Dispatches RAW decode to a background worker to keep the UI thread free.
        """
        self._prefetch_gen += 1
        self._preview_load_t0 = time.perf_counter()
        self._requested_file_path = file_path

        # Navigate-back fast path: the frame's last render is memoized and nothing
        # that shaped it has changed (select_file already hydrated its config), so
        # paint it now — no spinner, no toasts — and let the real render refresh
        # metrics quietly underneath.
        target_hash = self._file_hash_for_path(file_path)
        memo = self._render_memo.get(target_hash, self._render_memo_key()) if target_hash else None

        if not preserve_zoom:
            self.zoom_requested.emit(1.0)
        if memo is None:
            self.loading_started.emit()
        self._thumb_config = None

        self._render_cleanup_requested.emit()
        # The cleanup destroys the GPU textures last_metrics still points at; drop the
        # densitometer's probe source so hover readouts go quiet until the next render.
        self.state.last_metrics.pop("normalized_log", None)

        if memo is not None:
            with self.state.metrics_lock:
                self.state.last_metrics["base_positive"] = memo["base_positive"]
                self.state.last_metrics["content_rect"] = memo.get("content_rect")
                self.state.last_metrics["splash"] = False
            self.image_updated.emit()

        self.state.preview_raw = None
        self.state.preview_ir = None
        self.state.has_ir = False
        self.state.original_res = (0, 0)

        pending_import = self._pending_capture_imports.pop(_capture_import_key(file_path), None)
        if pending_import is not None and pending_import.process_mode is not None:
            process = self.state.config.process
            process = replace(
                process,
                process_mode=pending_import.process_mode,
                **invalidate_local_bounds(process),
            )
            self.state.config = replace(self.state.config, process=process)
            self.state.is_dirty = True

        rgbscan = self.state.config.rgbscan
        self.preview_load_requested.emit(
            PreviewLoadTask(
                file_path=file_path,
                workspace_color_space=self.state.workspace_color_space,
                use_camera_wb=not self.state.config.process.linear_raw,
                full_resolution=self.state.hq_preview,
                file_hash=base_hash(self._file_hash_for_path(file_path)),  # halves share the per-file decode cache
                # A memoized frame is already painted — the embedded-JPEG splash
                # would repaint stale pixels over it.
                use_splash=memo is None,
                detect_mode=(
                    pending_import.detect_mode
                    if pending_import is not None
                    else force_detect or (self.state.autodetect_enabled and self.state.current_file_is_new)
                ),
                green_path=rgbscan.green_path if rgbscan.enabled else "",
                blue_path=rgbscan.blue_path if rgbscan.enabled else "",
                align=rgbscan.align,
            )
        )

    def _split_active_half(self, raw: Any, dims: Any) -> tuple[Any, Any]:
        """Slice a full-frame decode/splash down to the active half asset (no-op otherwise).

        Both halves decode identically, so slicing by the current selection is safe
        even for a stale same-path load; cached buffers are read-only, hence the copy.
        """
        half_info = self._active_half()
        if half_info is None:
            return raw, dims
        half, split_x = half_info
        raw = np.ascontiguousarray(slice_half(raw, half, split_x))
        if dims is not None:
            h0, w0 = dims
            xs = min(max(int(round(w0 * split_x)), 1), w0 - 1)
            dims = (h0, xs) if half == 1 else (h0, w0 - xs)
        return raw, dims

    def _on_splash_preview(self, file_path: str, raw: Any, dims: Any) -> None:
        if self._requested_file_path != file_path:
            return
        raw, dims = self._split_active_half(raw, dims)
        self.state.original_res = dims
        # Paint the embedded sRGB thumbnail directly — no pipeline; the real render replaces it.
        with self.state.metrics_lock:
            self.state.last_metrics["base_positive"] = raw
            self.state.last_metrics["splash"] = True
        self.image_updated.emit()

    def _on_preview_load_failed(self, file_path: str, message: str) -> None:
        for f in self.state.uploaded_files:
            if f["path"] == file_path:
                f["decode_failed"] = message
                self.session.asset_model.refresh()
                return

    def _on_preview_loaded(self, file_path: str, raw: Any, dims: Any, source_cs: str, ir_preview: Any, detected_mode: str) -> None:
        for f in self.state.uploaded_files:
            if f["path"] == file_path and f.pop("decode_failed", None) is not None:
                self.session.asset_model.refresh()
        if self._requested_file_path != file_path:
            return
        logger.info(
            "load-timing preview_e2e %.0fms (load request -> decoded buffer) %s",
            (time.perf_counter() - self._preview_load_t0) * 1000,
            file_path,
        )
        raw, dims = self._split_active_half(raw, dims)
        if ir_preview is not None:
            ir_preview, _ = self._split_active_half(ir_preview, None)
        self.state.preview_raw = raw
        self.state.preview_ir = ir_preview
        self.state.has_ir = ir_preview is not None
        if not self.state.has_ir and self.state.dust_overlay_mode == "ir":
            self.state.dust_overlay_mode = "off"
        self.state.original_res = dims
        self.state.current_file_path = file_path
        self.state.source_cs = source_cs
        self._apply_detected_mode(detected_mode)
        self.preview_loaded.emit()
        self.config_updated.emit()
        self._first_render_t0 = time.perf_counter()
        self.request_render()
        self._schedule_prefetch_neighbors()

    def _schedule_prefetch_neighbors(self) -> None:
        from negpy.desktop.prefetch_logic import neighbor_paths_and_hashes

        g = self._prefetch_gen

        def _run() -> None:
            if g != self._prefetch_gen:
                return
            idx = self.state.selected_file_idx
            files = self.state.uploaded_files
            for path, h in neighbor_paths_and_hashes(files, idx):
                # Match the cache key load_file will use for this neighbour: its own saved
                # linear_raw, not the current file's. Otherwise the warm buffer lands under
                # the wrong key and navigation re-decodes anyway.
                saved = self.session.repo.load_file_settings(h) if h else None
                linear_raw = saved.process.linear_raw if saved else False
                self.preview_load_requested.emit(
                    PreviewLoadTask(
                        file_path=path,
                        workspace_color_space=self.state.workspace_color_space,
                        use_camera_wb=not linear_raw,
                        # Half-size only: a full-res HQ neighbour (~720MB) evicts the
                        # active buffer; the cache key separates resolutions.
                        full_resolution=False,
                        file_hash=base_hash(h),
                        use_splash=False,
                        for_cache_warm=True,
                    )
                )

        QTimer.singleShot(50, _run)

    def _apply_detected_mode(self, detected_mode: str) -> None:
        """
        Silently apply the autodetected process mode for a new file. Never overrides
        a saved or user-edited mode (the worker only runs detection on new files).
        """
        if not detected_mode or detected_mode == self.state.config.process.process_mode:
            return
        new_proc = replace(
            self.state.config.process,
            process_mode=ProcessMode(detected_mode),
            **invalidate_local_bounds(self.state.config.process),
        )
        self.state.config = replace(self.state.config, process=new_proc)
        self.state.is_dirty = True

    def toggle_autodetect(self, enabled: bool) -> None:
        self.session.set_autodetect_enabled(enabled)
        if enabled and self.state.current_file_path:
            self.load_file(self.state.current_file_path, preserve_zoom=True, force_detect=True)

    def toggle_hq_preview(self) -> None:
        self.session.set_hq_preview(not self.state.hq_preview)
        if self.state.current_file_path:
            self.load_file(self.state.current_file_path, preserve_zoom=True)

    def handle_canvas_clicked(self, nx: float, ny: float) -> None:
        if self.state.active_tool == ToolMode.WB_PICK:
            self._handle_wb_pick(nx, ny)
        elif self.state.active_tool == ToolMode.DUST_PICK:
            self._handle_dust_pick(nx, ny)

    def set_active_tool(self, mode: ToolMode) -> None:
        # Both the crop and analysis-region tools show the full uncropped frame, so
        # crossing into/out of that set must re-render to swap the preview.
        uncropped = {ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW}
        preview_mode_changed = (self.state.active_tool in uncropped) != (mode in uncropped)
        leaving_crop = self.state.active_tool == ToolMode.CROP_MANUAL and mode != ToolMode.CROP_MANUAL
        self.state.active_tool = mode
        self.tool_sync_requested.emit()
        if leaving_crop and self._crop_bounds_dirty:
            # Recompute bounds once now the final crop is committed.
            new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
            self.session.update_config(replace(self.state.config, process=new_proc), render=False)
            self._crop_bounds_dirty = False
        if preview_mode_changed:
            if leaving_crop:
                # Same spinner/overlay treatment as an initial file load: the bounds
                # recompute above plus this render can take a noticeable moment on a
                # large HQ frame, and image_updated (fired when the render lands)
                # dismisses it — guaranteed since request_render() runs right below.
                self.loading_started.emit()
            self.request_render()

    def cancel_active_tool(self) -> None:
        if self.state.active_tool != ToolMode.NONE:
            self.set_active_tool(ToolMode.NONE)

    def show_rotation_guide(self) -> None:
        """Request the canvas show the fine-rotation alignment grid."""
        self.rotation_guide_requested.emit()

    def set_crop_guide(self, guide: str) -> None:
        self.session.set_crop_guide(guide)
        self.crop_guide_changed.emit()

    def cycle_crop_guide_orientation(self) -> None:
        self.session.set_crop_guide_orientation((self.state.crop_guide_orientation + 1) % 8)
        self.crop_guide_changed.emit()

    def cycle_dust_overlay(self) -> None:
        """Advance the dust-detection overlay: Off → Marked → IR → Off
        (IR skipped when the scan has no IR channel). Repaint only — the data is
        already in state.last_metrics / state.preview_ir, no re-render needed."""
        seq = ["off", "marked", "ir"]
        if not self.state.has_ir:
            seq.remove("ir")
        cur = self.state.dust_overlay_mode if self.state.dust_overlay_mode in seq else "off"
        self.state.dust_overlay_mode = seq[(seq.index(cur) + 1) % len(seq)]
        self.dust_overlay_changed.emit()

    def handle_crop_rect_changed(self, nx1: float, ny1: float, nx2: float, ny2: float, persist: bool) -> None:
        """Live-updates (persist=False) or commits (persist=True) the manual crop rect
        while the crop tool is open. The tool stays active afterwards — darktable-style
        continuous adjustment, not a one-shot drag-then-close."""
        if self.state.active_tool != ToolMode.CROP_MANUAL:
            return
        new_geo = replace(
            self.state.config.geometry,
            manual_crop_rect=(
                min(nx1, nx2),
                min(ny1, ny2),
                max(nx1, nx2),
                max(ny1, ny2),
            ),
            auto_crop_enabled=False,
        )
        # Defer the bounds recompute to crop-tool close; clearing here re-normalizes every drag step.
        self._crop_bounds_dirty = True
        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=persist)
        if persist:
            self.request_render()
        else:
            self._render_debounce.start()

    def handle_crop_rotation_changed(self, angle: float, persist: bool) -> None:
        """Live-updates (persist=False) or commits (persist=True) fine rotation from the
        crop tool's edge rotation handles. Writes the same geometry.fine_rotation the
        sidebar slider drives, so handle drag and slider fine-tuning compose; the crop
        rect is display-space and stays put while the image rotates under it."""
        if self.state.active_tool != ToolMode.CROP_MANUAL:
            return
        new_geo = replace(self.state.config.geometry, fine_rotation=angle)
        # Defer the bounds recompute to crop-tool close, like the rect drag.
        self._crop_bounds_dirty = True
        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=persist)
        self.rotation_guide_requested.emit()
        if persist:
            self.request_render()
        else:
            self._render_debounce.start()

    def handle_straighten_completed(self, delta_deg: float) -> None:
        """Applies the straighten tool's measured correction on top of the current
        fine rotation and closes the tool (one-shot, like a Lightroom straighten
        line). ``delta_deg`` is stored-convention (positive = CCW on screen) and
        display-space, so it composes additively under flips/90° turns."""
        if self.state.active_tool != ToolMode.STRAIGHTEN:
            return
        current = self.state.config.geometry.fine_rotation
        new_angle = float(np.clip(current + delta_deg, -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT))
        new_geo = replace(self.state.config.geometry, fine_rotation=new_angle)
        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=True)
        self.rotation_guide_requested.emit()
        self.set_active_tool(ToolMode.NONE)
        self.request_render()

    def confirm_manual_crop(self) -> None:
        """Close the crop tool (committing the current rect) — invoked by a double-click
        inside the crop box so the user needn't return to the Crop button."""
        if self.state.active_tool == ToolMode.CROP_MANUAL:
            self.set_active_tool(ToolMode.NONE)

    def set_crop_ratio(self, ratio: str) -> None:
        """Sets the sidebar Ratio picker's target ratio. If a manual crop box is
        already drawn, reshapes it to the new ratio in place — same center, shrunk
        to fit within its current footprint (enforce_roi_aspect_ratio, the same
        centered-reshape auto-crop uses) — instead of leaving the box visually
        stale until the user redrags it.

        Deliberately does NOT invalidate the metering bounds, unlike the other crop
        entry points. Those clear them because the crop decides whether the film
        rebate is inside the metered region (resolve_analysis_region meters within
        context.active_roi), and letting clear base into the meter wrecks the
        bounds. A ratio change can't do that: both this reshape and autocrop's
        _enforce_ratio_by_occupancy only ever shrink the box inside a footprint
        that already excludes the rebate, so the new ROI is a subset of the old
        one. Re-metering there can only drift the per-channel floors/ceils — i.e.
        a visible colour shift from what is supposed to be a pure reframe."""
        geom = self.state.config.geometry
        if ratio == geom.autocrop_ratio:
            return
        new_geo = replace(geom, autocrop_ratio=ratio)

        rect = geom.manual_crop_rect
        img = self.state.preview_raw
        if rect is not None and img is not None:
            h, w = img.shape[:2]
            if geom.rotation in (1, 3):
                h, w = w, h
            nx1, ny1, nx2, ny2 = rect
            roi_px = (round(ny1 * h), round(ny2 * h), round(nx1 * w), round(nx2 * w))
            y1, y2, x1, x2 = enforce_roi_aspect_ratio(roi_px, h, w, ratio)
            new_geo = replace(new_geo, manual_crop_rect=(x1 / w, y1 / h, x2 / w, y2 / h))

        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=True)
        # Same spinner/overlay treatment as reset_crop/apply_auto_crop: the base
        # stage still re-runs (geometry is part of its cache key), which can take a
        # noticeable moment on a large HQ frame.
        self.loading_started.emit()
        self.request_render()

    def handle_analysis_rect_changed(self, nx1: float, ny1: float, nx2: float, ny2: float, persist: bool) -> None:
        """Live-update (persist=False) or commit (persist=True) the freehand analysis
        region while the tool is open. Setting a region re-meters the frame, so a commit
        clears the per-file bounds (unless bounds are locked) and re-renders."""
        if self.state.active_tool != ToolMode.ANALYSIS_DRAW:
            return
        rect = (min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2))
        proc = replace(self.state.config.process, analysis_rect=rect)
        if persist:
            proc = replace(proc, **invalidate_local_bounds(proc))
        self.session.update_config(replace(self.state.config, process=proc), persist=persist)
        if persist:
            self.request_render()
        else:
            self._render_debounce.start()

    def clear_analysis_region(self) -> None:
        """Drop the freehand analysis region; metering falls back to the Analysis Buffer slider."""
        if self.state.config.process.analysis_rect is None:
            return
        proc = replace(self.state.config.process, analysis_rect=None)
        proc = replace(proc, **invalidate_local_bounds(proc))
        self.session.update_config(replace(self.state.config, process=proc), persist=True)
        self.request_render()

    def confirm_analysis_region(self) -> None:
        """Close the analysis-region tool (double-click inside the region)."""
        if self.state.active_tool == ToolMode.ANALYSIS_DRAW:
            self.set_active_tool(ToolMode.NONE)

    def reset_crop(self) -> None:
        self._crop_bounds_dirty = False
        new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(self.state.config.geometry, manual_crop_rect=None, auto_crop_enabled=False),
                process=new_proc,
            )
        )
        # Same spinner/overlay treatment as an initial file load: the bounds
        # recompute above can take a noticeable moment on a large HQ frame.
        self.loading_started.emit()
        self.request_render()

    def apply_auto_crop(self) -> None:
        # Autocrop supersedes a manual crop in progress: leave the tool.
        if self.state.active_tool == ToolMode.CROP_MANUAL:
            self.state.active_tool = ToolMode.NONE
            self.tool_sync_requested.emit()
        self._crop_bounds_dirty = False
        new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(
                    self.state.config.geometry,
                    manual_crop_rect=None,
                    auto_crop_enabled=True,
                ),
                process=new_proc,
            )
        )
        self.loading_started.emit()
        self.request_render()

    def _config_for_autocrop_asset(self, asset: dict) -> WorkspaceConfig:
        """Resolve per-asset settings, including unsaved edits on the active frame."""
        if asset.get("hash") == self.state.current_file_hash:
            return resolve_asset_rgbscan(self.state.config, asset)
        return self.session.config_for_asset(asset)

    def request_batch_auto_crop(self) -> None:
        """Analyze visible landscape frames together and persist explicit safe crops."""
        if self._batch_busy("Auto Crop All"):
            return
        if self.state.config.geometry.autocrop_mode != AutocropMode.IMAGE:
            self.set_status("Auto Crop All currently supports Image only mode", 4000)
            return
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return

        frames: list[BatchAutoCropInput] = []
        preflight_skipped = 0
        for asset in visible_files:
            config = self._config_for_autocrop_asset(asset)
            if config.geometry.manual_crop_rect is not None or config.geometry.autocrop_mode != AutocropMode.IMAGE:
                preflight_skipped += 1
                continue
            frames.append(
                BatchAutoCropInput(
                    file_info=asset,
                    config=config,
                    fingerprint=_autocrop_fingerprint(config, self.state.workspace_color_space),
                )
            )

        if not frames:
            self.set_status(f"Auto Crop All preserved {preflight_skipped} frame(s); nothing to analyze", 4000)
            return

        token = self._begin_batch("autocrop", "Auto cropping roll", abortable=True)
        if token is None:
            return
        self._autocrop_batch_token = token
        self._autocrop_dispatched = len(frames)
        self._autocrop_preflight_skipped = preflight_skipped
        self._autocrop_cancel_requested = False
        self.set_status(f"Auto cropping {len(frames)} frame(s)...")
        self.batch_autocrop_requested.emit(
            BatchAutoCropTask(
                frames=frames,
                workspace_color_space=self.state.workspace_color_space,
                generation=token,
            )
        )

    def _on_batch_autocrop_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"Auto crop {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _on_batch_autocrop_finished(self, results: list[BatchAutoCropResult]) -> None:
        token = self._autocrop_batch_token
        if self._active_batch != "autocrop" or token is None or token != self._active_batch_token:
            return  # stale completion from an older generation
        if self._autocrop_cancel_requested:
            self._on_batch_autocrop_cancelled()
            return

        saved = 0
        conflicted = 0
        failed = 0
        active_changed = False
        try:
            for result in results:
                asset = result.file_info
                try:
                    latest = self._config_for_autocrop_asset(asset)
                    if latest.geometry.manual_crop_rect is not None:
                        conflicted += 1
                        continue
                    if _autocrop_fingerprint(latest, self.state.workspace_color_space) != result.fingerprint:
                        conflicted += 1
                        continue

                    rect = result.manual_crop_rect
                    if len(rect) != 4 or not (0.0 <= rect[0] < rect[2] <= 1.0 and 0.0 <= rect[1] < rect[3] <= 1.0):
                        conflicted += 1
                        continue
                    fine_rotation = latest.geometry.fine_rotation + result.correction_angle
                    if not np.isfinite(fine_rotation) or abs(fine_rotation) > FINE_ROTATION_LIMIT:
                        conflicted += 1
                        continue

                    new_geometry = replace(
                        latest.geometry,
                        manual_crop_rect=tuple(float(value) for value in rect),
                        auto_crop_enabled=False,
                        fine_rotation=float(fine_rotation),
                    )
                    new_process = replace(latest.process, **invalidate_local_bounds(latest.process))
                    updated = replace(latest, geometry=new_geometry, process=new_process)
                    if asset.get("hash") == self.state.current_file_hash:
                        self.session.persist_active_batch_config(updated)
                        active_changed = True
                    else:
                        self.session.repo.save_file_settings(asset["hash"], updated, file_path=asset["path"])
                    saved += 1
                except Exception:
                    failed += 1
                    logger.exception("Auto Crop All could not persist %s", asset.get("path", asset.get("hash", "frame")))
        finally:
            self._end_batch("autocrop", token)
            self._autocrop_batch_token = None
            self._autocrop_cancel_requested = False
            self.status_progress_requested.emit(0, 0)

        unresolved = max(0, self._autocrop_dispatched - len(results))
        preserved = self._autocrop_preflight_skipped + conflicted
        failure_suffix = f", failed {failed}" if failed else ""
        self.set_status(
            f"Auto Crop All: saved {saved}, preserved {preserved}, unchanged {unresolved}{failure_suffix}",
            5000,
        )
        if active_changed:
            self.config_updated.emit()
            self.request_render()

    def _on_batch_autocrop_cancelled(self) -> None:
        token = self._autocrop_batch_token
        if token is None:
            return
        self._end_batch("autocrop", token)
        self._autocrop_batch_token = None
        self._autocrop_cancel_requested = False
        self.status_progress_requested.emit(0, 0)
        self.set_status("Auto Crop All aborted; no crops were saved", 4000)

    def _on_batch_autocrop_error(self, message: str) -> None:
        token = self._autocrop_batch_token
        if token is None:
            return
        self._end_batch("autocrop", token)
        self._autocrop_batch_token = None
        self._autocrop_cancel_requested = False
        self.status_progress_requested.emit(0, 0)
        logger.error("Auto Crop All failed: %s", message)
        self.set_status(f"Auto Crop All failed: {message}", 5000)

    def detect_aspect_ratio(self) -> None:
        img = self.state.preview_raw
        if img is None:
            return

        geom = self.state.config.geometry
        transformed = img
        if geom.rotation != 0:
            transformed = np.rot90(transformed, k=geom.rotation)
        if geom.flip_horizontal:
            transformed = np.ascontiguousarray(np.fliplr(transformed))
        if geom.flip_vertical:
            transformed = np.ascontiguousarray(np.flipud(transformed))
        if geom.fine_rotation != 0.0:
            transformed = apply_fine_rotation(transformed, geom.fine_rotation)

        # Detection can match a portrait-oriented frame to a portrait-only AspectRatio
        # (e.g. "2:3") that the ratio picker doesn't display — canonicalize so the
        # stored ratio always matches an entry the picker can show (see
        # domain.models.CROP_RATIO_CHOICES; the crop tool auto-orients regardless).
        new_ratio = canonical_crop_ratio(detect_closest_aspect_ratio(transformed, fallback=geom.autocrop_ratio))
        if new_ratio == geom.autocrop_ratio:
            return

        new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(geom, autocrop_ratio=new_ratio),
                process=new_proc,
            ),
            persist=True,
            render=False,
        )
        # Emit manually so UI syncs (combo dropdown updates), but without triggering
        # a render via the state_changed debounce.
        self.config_updated.emit()
        if geom.auto_crop_enabled:
            self.request_render()

    def save_current_edits(self) -> None:
        if self.state.current_file_hash:
            self.session.update_config(self.state.config, persist=True)
            self._update_thumbnail_from_state(force_readback=True)

    def clear_retouch(self) -> None:
        from negpy.desktop.view.confirm import confirm_clear_heals

        conf = self.state.config.retouch
        count = len(conf.manual_dust_spots) + len(conf.manual_heal_strokes)
        if count == 0:
            return
        # Wiping every heal is not step-recoverable like single-heal undo — confirm.
        if not confirm_clear_heals(None, count):
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=[], manual_heal_strokes=[]),
            ),
            persist=True,
        )
        self.request_render()

    def delete_heal(self, kind: str, index: int) -> None:
        """Removes one placed heal by identity ("stroke"/"spot", index) — lets the
        user pick off a bad patch directly instead of unwinding newer heals first."""
        strokes = list(self.state.config.retouch.manual_heal_strokes)
        spots = list(self.state.config.retouch.manual_dust_spots)
        if kind == "stroke" and 0 <= index < len(strokes):
            strokes.pop(index)
        elif kind == "spot" and 0 <= index < len(spots):
            spots.pop(index)
        else:
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=spots, manual_heal_strokes=strokes),
            ),
            persist=True,
        )
        self.request_render()

    def undo_last_retouch(self) -> None:
        """
        Removes the most recently added heal (strokes first, then legacy spots).
        """
        strokes = list(self.state.config.retouch.manual_heal_strokes)
        spots = list(self.state.config.retouch.manual_dust_spots)
        if strokes:
            strokes.pop()
        elif spots:
            spots.pop()
        else:
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=spots, manual_heal_strokes=strokes),
            ),
            persist=True,
        )
        self.request_render()

    def _handle_dust_pick(self, nx: float, ny: float) -> None:
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)
        self._commit_heal_stroke([(rx, ry)])

    def handle_heal_stroke_completed(self, viewport_pts: list) -> None:
        """Commits a scratch-tool polyline (viewport-normalized points)."""
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None or not viewport_pts:
            return
        raw_pts = [CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_pts]
        self._commit_heal_stroke(raw_pts)

    def _commit_heal_stroke(self, raw_pts: list) -> None:
        conf = self.state.config.retouch
        size = float(conf.manual_dust_size)
        index = len(conf.manual_heal_strokes)

        # Score the clone source on the source-frame preview. Brush size is a
        # diameter at HEAL_SIZE_REF scale (same convention as the pipeline
        # radius and the overlay cursor).
        offset = (0.0, 0.0)
        preview = self.state.preview_raw
        if preview is not None:
            scale = max(preview.shape[:2]) / float(HEAL_SIZE_REF)
            offset = select_source_offset(preview, raw_pts, 0.5 * size * scale, index)
        else:
            offset = fallback_source_offset(index, size, (self.state.original_res[1], self.state.original_res[0]))

        stroke = ([[rx, ry] for rx, ry in raw_pts], size, float(offset[0]), float(offset[1]))
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_heal_strokes=conf.manual_heal_strokes + [stroke]),
            ),
            persist=True,
        )
        self.request_render()

    def handle_lasso_completed(self, viewport_vertices: list) -> None:
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None or len(viewport_vertices) < 3:
            return

        raw_vertices = tuple(CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_vertices)

        from negpy.features.local.models import PolygonMask

        mask = PolygonMask(vertices=raw_vertices, strength=0.3)
        local = self.state.config.local
        new_masks = local.masks + (mask,)
        new_local = replace(local, masks=new_masks)
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.state.local_selected_mask = len(new_masks) - 1
        self.set_active_tool(ToolMode.NONE)  # auto-exit draw mode once the polygon closes
        self.config_updated.emit()
        self.request_render()

    def handle_local_mask_edited(self, index: int, viewport_vertices: list) -> None:
        """Replace a mask's vertices after an on-canvas drag/add edit (persist on release)."""
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        local = self.state.config.local
        if uv_grid is None or not (0 <= index < len(local.masks)) or len(viewport_vertices) < 3:
            return
        raw_vertices = tuple(CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_vertices)
        masks = list(local.masks)
        masks[index] = replace(masks[index], vertices=raw_vertices)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.config_updated.emit()
        self.request_render()

    def delete_local_vertex(self, index: int, vertex_index: int) -> None:
        """Remove one vertex from a mask (keeps a minimum of 3)."""
        local = self.state.config.local
        if not (0 <= index < len(local.masks)):
            return
        mask = local.masks[index]
        if len(mask.vertices) <= 3 or not (0 <= vertex_index < len(mask.vertices)):
            return
        verts = mask.vertices[:vertex_index] + mask.vertices[vertex_index + 1 :]
        masks = list(local.masks)
        masks[index] = replace(mask, vertices=verts)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.config_updated.emit()
        self.request_render()

    def select_local_mask(self, index: int) -> None:
        self.state.local_selected_mask = index
        self.config_updated.emit()

    def set_local_mask_visible(self, index: int, visible: bool) -> None:
        """Show/hide one mask's outline on the canvas (view-only; no re-render)."""
        if not (0 <= index < len(self.state.config.local.masks)):
            return
        hidden = set(self.state.local_hidden_masks)
        if visible:
            hidden.discard(index)
        else:
            hidden.add(index)
        self.state.local_hidden_masks = hidden
        self.session.persist_hidden_masks()
        if self.canvas:
            self.canvas.overlay.update()

    def delete_local_mask(self, index: int) -> None:
        local = self.state.config.local
        if not (0 <= index < len(local.masks)):
            return
        from negpy.desktop.view.confirm import confirm_delete_mask

        if not confirm_delete_mask(None):
            return
        new_masks = local.masks[:index] + local.masks[index + 1 :]
        new_local = replace(local, masks=new_masks)
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)

        sel = self.state.local_selected_mask
        self.state.local_selected_mask = -1 if sel == index else (sel - 1 if sel > index else sel)
        self.state.local_hidden_masks = {j - 1 if j > index else j for j in self.state.local_hidden_masks if j != index}
        self.session.persist_hidden_masks()

        self.config_updated.emit()
        self.request_render()

    def update_selected_local_mask(self, **changes) -> None:
        local = self.state.config.local
        idx = self.state.local_selected_mask
        if not (0 <= idx < len(local.masks)):
            return
        masks = list(local.masks)
        masks[idx] = replace(masks[idx], **changes)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.request_render()

    def _handle_wb_pick(self, nx: float, ny: float) -> None:
        """
        Samples color from viewport coordinates and updates WB shifts to neutralize.
        """
        with self.state.metrics_lock:
            metrics = dict(self.state.last_metrics)

        img = metrics.get("normalized_log")
        is_log = True
        if img is None:
            img = metrics.get("base_positive")
            is_log = False

        if img is None:
            return

        roi = metrics.get("active_roi")
        radius = 4

        if isinstance(img, GPUTexture):
            h, w = img.height, img.width
            if roi and is_log:
                ry1, ry2, rx1, rx2 = roi
                center_y = int(np.clip(ry1 + ny * (ry2 - ry1), 0, h - 1))
                center_x = int(np.clip(rx1 + nx * (rx2 - rx1), 0, w - 1))
            else:
                center_y = int(np.clip(ny * h, 0, h - 1))
                center_x = int(np.clip(nx * w, 0, w - 1))
            x0 = max(center_x - radius, 0)
            y0 = max(center_y - radius, 0)
            rw = min(center_x + radius, w) - x0
            rh = min(center_y + radius, h) - y0
            sampled = img.readback_region(x0, y0, rw, rh).mean(axis=(0, 1))
        elif isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            if roi and is_log:
                ry1, ry2, rx1, rx2 = roi
                center_y = int(np.clip(ry1 + ny * (ry2 - ry1), 0, h - 1))
                center_x = int(np.clip(rx1 + nx * (rx2 - rx1), 0, w - 1))
            else:
                center_y = int(np.clip(ny * h, 0, h - 1))
                center_x = int(np.clip(nx * w, 0, w - 1))
            y0 = max(center_y - radius, 0)
            y1_ = min(center_y + radius, h)
            x0 = max(center_x - radius, 0)
            x1_ = min(center_x + radius, w)
            sampled = img[y0:y1_, x0:x1_].mean(axis=(0, 1))
        else:
            return

        exp = self.state.config.exposure
        bounds = metrics.get("final_bounds") or metrics.get("log_bounds")  # CPU/GPU key names
        if is_log:
            new_m, new_y = calculate_wb_shifts_from_log(sampled[:3], bounds)
        else:
            delta_m, delta_y = calculate_wb_shifts(sampled[:3])
            damping = 0.4
            new_m = exp.wb_magenta + delta_m * damping
            new_y = exp.wb_yellow + delta_y * damping

        region = self.state.wb_pick_region
        if region == 0:
            new_exp = replace(
                exp,
                wb_cyan=0.0,
                wb_magenta=float(np.clip(new_m, -1.0, 1.0)),
                wb_yellow=float(np.clip(new_y, -1.0, 1.0)),
            )
        else:
            # Store the residual over the global pair in the region's fields.
            # Filtration offsets are range-normalized, regional ones absolute
            # density — convert by the stretch range. Assumes the picked patch
            # sits in its region (weight ~1).
            c_field, m_field, y_field = (
                ("shadow_cyan", "shadow_magenta", "shadow_yellow"),
                ("highlight_cyan", "highlight_magenta", "highlight_yellow"),
            )[region - 1]
            rng_m = rng_y = 1.0
            if is_log and bounds is not None:
                rng_m = max(abs(bounds.ceils[1] - bounds.floors[1]), 1e-6)
                rng_y = max(abs(bounds.ceils[2] - bounds.floors[2]), 1e-6)
            dm = (new_m - exp.wb_magenta) / rng_m
            dy = (new_y - exp.wb_yellow) / rng_y
            new_exp = replace(
                exp,
                **{
                    c_field: 0.0,
                    m_field: float(np.clip(dm, -1.0, 1.0)),
                    y_field: float(np.clip(dy, -1.0, 1.0)),
                },
            )
        self.session.update_config(replace(self.state.config, exposure=new_exp), persist=True, record_history=True)
        self.request_render()

    def request_batch_normalization(self) -> None:
        """
        Initiates background analysis for batch normalization.
        """
        if self._batch_busy("Batch Analysis"):
            return
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return

        total = len(visible_files)
        cropped = 0
        for f in visible_files:
            p = self.session.repo.load_file_settings(f["hash"])
            if p and (p.geometry.manual_crop_rect or p.geometry.auto_crop_enabled):
                cropped += 1

        if cropped == 0:
            crop_status = f"Crop status: 0 of {total} files are cropped."
            crop_warning = (
                "Strongly recommended: crop all images in this session before running "
                "Batch Analysis. Without a crop, the Analysis Buffer's small centered "
                "margin isn't enough to exclude sprocket holes and empty space outside "
                "the actual frame — that unwanted region gets included in the luma and "
                "color average, producing a less accurate result for every file."
            )
        elif cropped < total:
            crop_status = f"Crop status: {cropped} of {total} files are cropped."
            crop_warning = (
                f"Strongly recommended: crop the remaining {total - cropped} file(s) "
                "before running Batch Analysis. Uncropped files rely on the Analysis "
                "Buffer's small centered margin, which isn't enough to exclude sprocket "
                "holes and empty space outside the actual frame — that unwanted region "
                "gets included in the luma and color average, producing a less accurate "
                "result for every file."
            )
        else:
            crop_status = f"Crop status: all {total} files are cropped."
            crop_warning = "Analysis will run on each file's cropped negative area."

        sheet_note = ""
        if self.session.asset_model.sheet_filter != "all":
            sheet_note = f"Note: the Sheet filter is on — only the {total} visible frame(s) are analyzed.\n\n"

        reply = QMessageBox.question(
            None,
            "Batch Analysis",
            f"{sheet_note}"
            f"{crop_status}\n"
            f"{crop_warning}\n\n"
            "Batch Analysis measures the exposure bounds of every file and applies "
            "their average to the whole roll, so all your frames share a consistent "
            "baseline.\n\n"
            "Two settings from the image you have open right now are applied to every "
            "file before averaging:\n"
            "  • Analysis Buffer — shrinks the analyzed region inward, excluding a "
            "margin around the edges (film borders, light leaks, the scanner mask).\n"
            "  • Luma Range Clip — how aggressively the highlight/shadow tails are "
            "clipped when setting each file's bounds.\n"
            "Set both on the current frame before running.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        token = self._begin_batch("normalization", "Analyzing roll", abortable=True)
        if token is None:
            return
        self.set_status("Starting Batch Normalization...")
        task = NormalizationTask(
            files=visible_files,
            workspace_color_space=self.state.workspace_color_space,
            override_analysis_buffer=self.state.config.process.analysis_buffer,
            override_luma_range_clip=self.state.config.process.luma_range_clip,
            override_color_range_clip=self.state.config.process.color_range_clip,
            override_crosstalk_strength=self.state.config.process.crosstalk_strength,
            override_crosstalk_matrix=self.state.config.process.crosstalk_matrix,
        )
        self.normalization_requested.emit(task)

    def _on_normalization_progress(self, current: int, total: int, name: str, has_crop: bool) -> None:
        """
        Updates UI status during batch analysis.
        """
        marker = "cropped" if has_crop else "full frame"
        self.set_status(f"Analyzing {current}/{total}: {name} [{marker}]...")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, f"{name} [{marker}]")

    def _on_normalization_finished(self, locked_floors: tuple, locked_ceils: tuple) -> None:
        """
        Applies averaged normalization baseline to all files.
        """
        self._end_batch("normalization")
        for f_info in self.state.uploaded_files:
            p = self.session.repo.load_file_settings(f_info["hash"]) or replace(self.state.config)
            new_process = replace(
                p.process,
                use_luma_average=True,
                use_colour_average=True,
                locked_floors=locked_floors,
                locked_ceils=locked_ceils,
                roll_name=None,
            )
            new_p = replace(p, process=new_process)
            # The active file records its step via update_config(persist=True) below.
            if f_info["hash"] != self.state.current_file_hash:
                self.session.push_external_history(f_info["hash"], p, new_p)
            self.session.repo.save_file_settings(f_info["hash"], new_p, file_path=f_info["path"])

        # Update current state
        new_process = replace(
            self.state.config.process,
            use_luma_average=True,
            use_colour_average=True,
            locked_floors=locked_floors,
            locked_ceils=locked_ceils,
            roll_name=None,
        )
        self.session.update_config(replace(self.state.config, process=new_process), persist=True)

        self.set_status("batch analysis complete", timeout=3000)
        self.status_progress_requested.emit(0, 0)
        self.request_render()

    def save_current_normalization_as_roll(self, name: str) -> None:
        """
        Persists current batch normalization values as a named roll.
        """
        proc = self.state.config.process
        self.session.repo.save_normalization_roll(name, proc.locked_floors, proc.locked_ceils)
        self.session.update_config(
            replace(self.state.config, process=replace(proc, roll_name=name)),
            persist=True,
            render=False,
        )
        self.set_status(f"Roll '{name}' saved", 2000)

    def apply_normalization_roll(self, name: str) -> None:
        """
        Loads and applies a named normalization roll to the entire session.
        """
        data = self.session.repo.load_normalization_roll(name)
        if data:
            locked_floors, locked_ceils = data
            for f_info in self.state.uploaded_files:
                p = self.session.repo.load_file_settings(f_info["hash"]) or replace(self.state.config)
                new_process = replace(
                    p.process,
                    use_luma_average=True,
                    use_colour_average=True,
                    locked_floors=locked_floors,
                    locked_ceils=locked_ceils,
                    roll_name=name,
                )
                new_p = replace(p, process=new_process)
                if f_info["hash"] != self.state.current_file_hash:
                    self.session.push_external_history(f_info["hash"], p, new_p)
                self.session.repo.save_file_settings(f_info["hash"], new_p, file_path=f_info["path"])

            new_process = replace(
                self.state.config.process,
                use_luma_average=True,
                use_colour_average=True,
                locked_floors=locked_floors,
                locked_ceils=locked_ceils,
                roll_name=name,
            )
            self.session.update_config(replace(self.state.config, process=new_process), persist=True)
            self.set_status(f"Applied Roll '{name}'", 2000)
            self.request_render()

    def clear_roll_baseline(self) -> None:
        """Roll Analysis section reset: take the current frame off the roll baseline
        (both averaging axes + named roll) and re-meter it per-frame."""
        new_process = replace(
            self.state.config.process,
            use_luma_average=False,
            use_colour_average=False,
            roll_name=None,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.session.update_config(replace(self.state.config, process=new_process), persist=True)
        self.request_render()

    def reanalyze_current_file(self) -> None:
        """
        Clears cached local floors and forces a fresh analysis render.
        """
        new_process = replace(
            self.state.config.process,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.session.update_config(replace(self.state.config, process=new_process))
        self.request_render()

    def set_active_flatfield_profile(self, name: str) -> None:
        """
        Selects the globally active flat-field reference profile (or clears it when
        ``name`` is empty). Applies its path to the current image and re-renders.
        """
        self.session.repo.save_global_setting("flatfield_active_profile", name or "")
        rec = self.session.repo.get_flatfield_profile(name) if name else None
        path, k1 = rec if rec else ("", 0.0)
        new_ff = replace(self.state.config.flatfield, reference_path=path or "", apply=bool(path), k1=k1)
        self.session.update_config(replace(self.state.config, flatfield=new_ff), persist=True)
        self.request_render()

    def save_flatfield_profile(self, name: str, path: str) -> None:
        """
        Saves a reference image as a named flat-field profile and makes it active.
        """
        self.session.repo.save_flatfield_profile(name, path)
        self.set_active_flatfield_profile(name)
        self.set_status(f"Flat-field profile '{name}' saved", 2000)

    def delete_flatfield_profile(self, name: str) -> None:
        """
        Removes a flat-field profile; clears the active correction if it was selected.
        """
        if not name:
            return
        self.session.repo.delete_flatfield_profile(name)
        if self.session.repo.get_global_setting("flatfield_active_profile") == name:
            self.set_active_flatfield_profile("")

    def load_gear_library(self):
        from negpy.services.assets.gear import GearProfiles

        return GearProfiles.load_library()

    def save_gear_library(self, library) -> None:
        from negpy.services.assets.gear import GearProfiles

        GearProfiles.save_library(library)

    def set_flatfield_enabled(self, enabled: bool) -> None:
        """
        Per-image toggle to enable/disable flat-field correction for the current frame.
        """
        new_ff = replace(self.state.config.flatfield, apply=enabled)
        self.session.update_config(replace(self.state.config, flatfield=new_ff), persist=True)
        self.request_render()

    def set_flatfield_k1(self, k1: float) -> None:
        """
        Sets the rig's radial distortion. Saved into the active flat-field profile (so it
        applies to every frame on that rig), not the per-image recipe.
        """
        new_ff = replace(self.state.config.flatfield, k1=k1)
        self.session.update_config(replace(self.state.config, flatfield=new_ff), persist=True)
        active = self.session.repo.get_global_setting("flatfield_active_profile") or ""
        if active:
            rec = self.session.repo.get_flatfield_profile(active)
            path = rec[0] if rec else ""
            self.session.repo.save_flatfield_profile(active, path, k1)
        self.request_render()

    # ── Scanner integration ───────────────────────────────────────────

    def request_scan_devices(self) -> None:
        """Request device enumeration on the scan worker thread."""
        self.scan_devices_requested.emit()

    def start_scan(self, req: ScanRequest) -> None:
        """Start a scan. The UI connects to scan signals for state updates."""
        self.scan_started.emit()
        self.scan_requested.emit(req)

    def cancel_scan(self) -> None:
        self.scan_worker.cancel()

    def _on_scan_finished(self, path: str) -> None:
        """Auto-add scanned file to NegPy file list and select it."""
        self.scan_finished.emit(path)
        self._pending_scanned_file = path
        self.request_asset_discovery([path])

    def _select_file_by_path(self, path: str) -> bool:
        """Find a file by path in uploaded_files and select it."""
        for i, f_info in enumerate(self.session.state.uploaded_files):
            if f_info.get("path") == path:
                self.session.select_file(i)
                return True
        return False

    # ── Scanlight capture integration ─────────────────────────────────

    def _ensure_capture_thread(self) -> None:
        """Start the capture worker's thread on first use (lazy). Every capture entry point that
        emits to the worker calls this first, so the thread is running when the queued cross-thread
        signal is delivered. The live-view sub-controls and cancel skip it: they only run once a
        session is already up (started here) or touch the worker's thread-safe cancel Event."""
        if not self._capture_thread_started:
            self.capture_thread.start()
            self._capture_thread_started = True

    def set_scanlight_color(self, r: int, g: int, b: int, w: int = 0, port: str = "") -> None:
        """Live light control (no capture): RGB for preview, or white (w) for focus."""
        self._ensure_capture_thread()
        self.capture_light_requested.emit(r, g, b, w, port)

    def start_capture(self, req: CaptureRequest) -> None:
        """Start a capture; the Scanlight sidebar tracks state via signals."""
        self._ensure_capture_thread()
        self._last_capture_req = req
        self.capture_requested.emit(req)

    def cancel_capture(self) -> None:
        self.capture_worker.cancel()

    def start_live_view(self, req: LiveViewRequest) -> None:
        self._ensure_capture_thread()
        self.live_view_requested.emit(req)

    def stop_live_view(self) -> None:
        self.live_view_stop_requested.emit()

    def close_camera_session(self) -> None:
        """Release the held PTP session. Call once neither the scan window nor the
        preset-calibration pop-up is open — some bodies (Fuji) get stuck in a
        tethered-capture state until the session is cleanly exited, and leaving it
        open past the last consuming window makes the next connection attempt hang."""
        if self._capture_thread_started:
            self.camera_session_close_requested.emit()

    def set_focus_magnifier(self, on: bool) -> None:
        self.live_view_focus_magnifier_requested.emit(on)

    def set_focus_magnifier_pos(self, x: int, y: int) -> None:
        self.live_view_focus_magnifier_pos_requested.emit(x, y)

    def set_camera_setting(self, which: str, raw: int) -> None:
        self.live_view_camera_setting_requested.emit(which, raw)

    def start_calibration(self, req: CalibrationRequest) -> None:
        self._ensure_capture_thread()
        self.calibration_requested.emit(req)

    def poll_connection(self, port: str) -> None:
        self._ensure_capture_thread()
        self.poll_connection_requested.emit(port)

    def poll_light_temp(self, port: str) -> None:
        self._ensure_capture_thread()
        self.poll_light_temp_requested.emit(port)

    def _on_capture_finished(self, paths: list) -> None:
        """Feed the captured frame(s) into NegPy. A 3-file RGB triplet → RGB-Scan negative
        (C-41) pipeline; a single white-light slide → E-6/positive; a normal white-light
        camera scan → an ordinary single RAW (RGB-Scan off, process left to NegPy)."""
        self.capture_finished.emit(paths)
        if not paths:
            return
        req = getattr(self, "_last_capture_req", None)
        white = bool(req is not None and req.white_mode)
        rgb = bool(req is not None and getattr(req, "rgb_mode", True))
        # RGB-Scan (triplet merge) is on only for an actual RGB triplet — off for a single
        # white-light slide OR a normal (non-Scanlight) camera scan.
        self.session.repo.save_global_setting("rgbscan_mode", rgb and not white)
        if white:  # slides/B&W force a positive process
            mode = (req.white_process_mode or "auto").lower()
            target = {"e-6": ProcessMode.E6, "b&w": ProcessMode.BW}.get(mode)
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(
                process_mode=target,
                detect_mode=target is None,
            )
        elif rgb:
            # Independently exposed RGB channels have no broadband orange-mask signal for
            # the normal classifier. They are negative scans unless capture metadata says
            # otherwise, so carry C-41 through discovery instead of guessing from the merge.
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(process_mode=ProcessMode.C41)
        self._pending_scanned_file = paths[0]
        self.request_asset_discovery(list(paths))

    def effective_output_icc(self) -> Optional[str]:
        """Output profile the preview proofs through: a custom override, else the
        profile for the selected export color space. None means no proof (Same as Source)."""
        return self.state.icc_output_path or ColorSpaceRegistry.get_icc_path(self.state.config.export.export_color_space)

    def effective_input_icc(self, process: Optional[ProcessConfig] = None) -> Optional[str]:
        """Source profile for color management: an explicit Input ICC wins; else the
        bundled RGBScan profile when Narrowband Scan is on; else None."""
        p = process if process is not None else self.state.config.process
        if self.state.icc_input_path:
            return self.state.icc_input_path
        if p.narrowband_scan:
            return get_resource_path("icc/RGBScan.icc")
        return None

    def proof_active(self) -> bool:
        """True when the preview should soft-proof: the toggle is on and an input or
        output profile is available, or Narrowband Scan supplies an implicit input
        profile. Off → preview is the edit on the monitor."""
        if self.state.config.process.narrowband_scan:
            return True
        return self.state.soft_proof_enabled and bool(self.state.icc_input_path or self.effective_output_icc())

    def set_soft_proof(self, enabled: bool) -> None:
        """Toggle preview soft-proofing through the Output/Input ICC (preview only)."""
        if self.state.soft_proof_enabled == enabled:
            return
        self.state.soft_proof_enabled = enabled
        self.session.save_icc_prefs()
        self.request_render()

    def _apply_monitor_profile(self) -> None:
        """Resolve the effective display profile (override else detected), push it to
        every preview path, and re-render. Display-only; export is unaffected."""
        from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space

        override = self.state.monitor_profile_override
        effective = icc_bytes_for_space(override) if override else self.state.monitor_icc_detected_bytes
        self.state.monitor_icc_bytes = effective
        if self.canvas is not None:
            self.canvas.set_monitor_profile(effective)
        self.request_render()
        self.monitor_profile_changed.emit()

    def set_monitor_detected(self, detected_bytes: Optional[bytes]) -> None:
        """Record the auto-detected screen profile and re-resolve the effective one."""
        self.state.monitor_icc_detected_bytes = detected_bytes
        self._apply_monitor_profile()

    def set_monitor_override(self, cs_name: Optional[str]) -> None:
        """Set the manual display-profile override (None = use detected) and persist it."""
        self.state.monitor_profile_override = cs_name
        self.session.save_icc_prefs()
        self._apply_monitor_profile()

    def request_render(
        self, readback_metrics: bool = True, config_override: Optional[WorkspaceConfig] = None, ephemeral: bool = False
    ) -> None:
        """
        Dispatches a render task to the worker thread.
        Direct callers bypass the debounce; the timer is cancelled to avoid a duplicate.

        config_override renders an alternate config (e.g. the before/after baseline) without
        mutating session state; pass readback_metrics=False so it doesn't disturb
        histogram/bounds persistence.
        """
        self._render_debounce.stop()

        # Any non-compare render (a user edit, navigation, etc.) exits before/after compare.
        if config_override is None and self.state.compare_mode:
            self.state.compare_mode = False
            self.compare_changed.emit(False)

        # Likewise, any direct render exits the flat preview-peek.
        if config_override is None and self.state.flat_peek:
            self.state.flat_peek = False
            self.flat_peek_changed.emit(False)

        if self.state.preview_raw is None:
            return

        preview_raw = self.state.preview_raw
        if preview_raw is None:
            return

        target_size = float(APP_CONFIG.preview_render_size)
        if self.state.hq_preview:
            target_size = float(max(preview_raw.shape[:2]))

        # Soft-proof gating: Output/Input ICC only touch the preview when the toggle is
        # on; otherwise the preview is the edit shown on the monitor (export unaffected).
        # Narrowband Scan supplies an implicit input profile regardless of the toggle.
        proofing = self.state.soft_proof_enabled
        narrowband = self.state.config.process.narrowband_scan
        icc_input = self.effective_input_icc() if (proofing or narrowband) else None
        effective_output = self.effective_output_icc() if proofing else None

        crop_preview_full = self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW)
        # Only a plain render of the saved edit is reproducible on navigate-back;
        # overrides (compare/flat peek), splash and tool previews are not memoized.
        memo_key = ""
        if config_override is None and not ephemeral and not crop_preview_full:
            memo_key = self._render_memo_key()

        task = RenderTask(
            buffer=preview_raw,
            config=config_override if config_override is not None else self.state.config,
            source_hash=self.state.current_file_hash or "preview",
            preview_size=target_size,
            icc_input_path=icc_input,
            icc_output_path=effective_output,
            color_space=self.state.workspace_color_space,
            gpu_enabled=self.state.gpu_enabled,
            readback_metrics=readback_metrics,
            ir_buffer=self.state.preview_ir,
            monitor_icc_bytes=self.state.monitor_icc_bytes,
            crop_preview_full=crop_preview_full,
            ephemeral=ephemeral,
            memo_key=memo_key,
        )

        if self._is_rendering:
            self._pending_render_task = task
            return

        self._is_rendering = True
        self.render_requested.emit(task)

    def _baseline_compare_config(self) -> WorkspaceConfig:
        return baseline_compare_config(self.state.config)

    def toggle_compare(self) -> None:
        """Toggle the before/after view between current edits and the auto baseline."""
        if self.state.preview_raw is None:
            return
        if self.state.compare_mode:
            self.state.compare_mode = False
            self.compare_changed.emit(False)
            self.request_render()
        else:
            self.state.compare_mode = True
            self.compare_changed.emit(True)
            self.request_render(readback_metrics=False, config_override=self._baseline_compare_config())

    # --- Flat ("for editing elsewhere") master output -----------------------

    def set_flat_output(self, enabled: bool) -> None:
        """Toggle the flat digital-intermediate output intent (export + peek)."""
        if self.state.flat_output == enabled:
            return
        self.state.flat_output = enabled
        self.session.save_flat_output_prefs()
        # Flat masters default to full resolution; only honour Print/Pixels when the
        # user explicitly selects those modes in the export panel.
        if enabled and self.state.config.export.export_resolution_mode == ExportResolutionMode.PRINT.value:
            self.session.update_config(
                replace(
                    self.state.config,
                    export=replace(
                        self.state.config.export,
                        export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
                    ),
                ),
                persist=True,
            )
        self.flat_output_changed.emit(enabled)
        # If a peek is active and flat output was turned off, drop back to the edit.
        if not enabled and self.state.flat_peek:
            self.toggle_flat_peek(force=False)

    def _flat_export_format(self) -> str:
        return ExportFormat.DNG if self.state.flat_format == "DNG" else ExportFormat.TIFF

    def set_flat_format(self, fmt: str) -> None:
        """Set the flat master file format ('TIFF' 16-bit or 'DNG' linear)."""
        fmt = fmt if fmt in ("TIFF", "DNG") else "TIFF"
        if self.state.flat_format == fmt:
            return
        self.state.flat_format = fmt
        self.session.save_flat_output_prefs()

    def toggle_flat_peek(self, force: Optional[bool] = None) -> None:
        """Preview the flat master render in the canvas without changing the saved edit.

        ``force`` sets an explicit state; otherwise toggles. Mutually exclusive with
        the before/after compare view.
        """
        if self.state.preview_raw is None:
            return
        target = (not self.state.flat_peek) if force is None else force
        if target == self.state.flat_peek:
            return

        if target and self.state.compare_mode:
            self.state.compare_mode = False
            self.compare_changed.emit(False)

        self.state.flat_peek = target
        self.flat_peek_changed.emit(target)

        if target:
            self.request_render(readback_metrics=False, config_override=flat_master_config(self.state.config))
        else:
            self.request_render()

    def _enabled_presets(self) -> List[ExportPreset]:
        return [p for p in self.state.export_presets if p.enabled]

    def _validate_preset_paths(self, presets: List[ExportPreset]) -> bool:
        """Returns True if all absolute-path presets have a valid directory configured."""
        from PyQt6.QtWidgets import QFileDialog

        for p in presets:
            if p.output_mode == ExportPresetOutputMode.ABSOLUTE and not p.output_path.strip():
                new_path = QFileDialog.getExistingDirectory(None, f"Select output folder for preset '{p.name}'", os.path.expanduser("~"))
                if not new_path:
                    return False
                p.output_path = new_path
                self.session.save_export_presets()
        return True

    def _batch_params_for(self, f: dict) -> WorkspaceConfig:
        """Resolve a visible frame's export params: its saved DB config (else the current
        config), with its own RGB-scan green/blue re-injected from the asset dict — the
        same authoritative source individual export gets via select_file."""
        params = self.session.repo.load_file_settings(f["hash"]) or self.state.config
        return resolve_asset_rgbscan(params, f)

    def _tasks_for_file(
        self,
        file_info: dict,
        params: WorkspaceConfig,
        presets: List[ExportPreset],
        bounds_override=None,
        source_exif=None,
        metadata_config=None,
    ) -> List[ExportTask]:
        tasks = []
        for preset in presets:
            task_params, export_settings = resolve_preset_export(preset, params)
            export_settings.icc_input_path = self.effective_input_icc(task_params.process)
            tasks.append(
                ExportTask(
                    file_info=file_info,
                    params=task_params,
                    export_settings=export_settings,
                    gpu_enabled=self.state.gpu_enabled,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=metadata_config,
                    working_color_space=self.state.workspace_color_space,
                )
            )
        return tasks

    def _ensure_valid_export_path(self) -> Optional[str]:
        """
        Checks if the current export path is valid. If not, prompts the user.
        Returns the valid path or None if the user cancelled.
        """
        export_path = self.state.config.export.export_path
        if self.state.config.export.output_mode == ExportPresetOutputMode.SAME_AS_SOURCE:
            return export_path  # path irrelevant when exporting to source folder
        if export_path.strip().lower() in ["export", "/export", ""]:
            from PyQt6.QtWidgets import QFileDialog

            new_path = QFileDialog.getExistingDirectory(None, "Select Export Directory", os.path.expanduser("~"))
            if new_path:
                new_export = replace(self.state.config.export, export_path=new_path)
                self.session.update_config(replace(self.state.config, export=new_export), persist=True)
                return new_path
            return None
        return export_path

    def history_steps(self) -> List[Dict[str, Any]]:
        """Rows for the History panel: one dict {index, label, is_current} per edit step."""
        file_hash = self.state.current_file_hash
        if not file_hash:
            return []
        configs = dict(self.session.repo.load_all_history(file_hash))
        # The live top step may not be persisted yet — it lives in state.config.
        configs[self.state.undo_index] = self.state.config

        rows: List[Dict[str, Any]] = []
        for i in range(self.state.max_history_index + 1):
            config = configs.get(i)
            if config is None:
                continue
            rows.append(
                {
                    "index": i,
                    "label": history_step_label(configs.get(i - 1), config, i),
                    "is_current": i == self.state.undo_index,
                }
            )
        return rows

    def jump_to_history_step(self, index: int) -> None:
        self.session.jump_to_step(index)

    def export_history_step(self, index: int) -> None:
        """Load a history step, then export it through the normal export path."""
        self.session.jump_to_step(index)
        self.request_export()

    def request_export(self) -> None:
        """Exports the current file using the settings currently shown in the Export panel."""
        if self._batch_busy("export"):
            return
        if not self.state.current_file_path:
            return

        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        export_conf = replace(
            self.state.config.export,
            export_path=export_path,
            icc_input_path=self.effective_input_icc(),
            icc_output_path=self.state.icc_output_path,
        )
        params = self.state.config
        if self.state.flat_output:
            params = flat_master_config(params)
            export_conf = flat_export_config(export_conf, fmt=self._flat_export_format())
        source_exif = self.state.source_exif.get(self.state.current_file_hash or "")

        self._run_export_tasks(
            [
                ExportTask(
                    file_info={
                        "name": os.path.basename(self.state.current_file_path),
                        "path": self.state.current_file_path,
                        "hash": self.state.current_file_hash,
                    },
                    params=params,
                    export_settings=preset_from_export_config(export_conf),
                    gpu_enabled=self.state.gpu_enabled,
                    source_exif=source_exif,
                    metadata_config=self.state.config.metadata,
                    working_color_space=self.state.workspace_color_space,
                )
            ]
        )

    def request_export_selected(self) -> None:
        """Batch-exports the currently selected files using each file's own saved settings."""
        selected = [self.state.uploaded_files[i] for i in self.state.selected_indices if 0 <= i < len(self.state.uploaded_files)]
        self.request_batch_export(files=[f for f in selected if not f.get("excluded")])

    def request_batch_export(self, override_settings: bool = False, files: list[dict] | None = None) -> None:
        """Batch-exports the given files (all visible by default) using current settings, optionally applied to all."""
        if self._batch_busy("export"):
            return
        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        current_export = replace(self.state.config.export, export_path=export_path)
        icc_output = self.state.icc_output_path
        sync_metadata = self.state.config.metadata.sync_to_batch

        if files is None:
            files = [
                self.state.uploaded_files[i]
                for i in self.session.asset_model.visible_actual_indices_ordered()
                if not self.state.uploaded_files[i].get("excluded")
            ]

        if len(files) > 1 and not self._confirm_bulk_export(f"Export {len(files)} frames?"):
            return

        if self.state.config.export.export_sidecars_enabled:
            self._write_edit_sidecars(files)

        flat = self.state.flat_output
        flat_fmt = self._flat_export_format()

        tasks = []
        for f in files:
            params = self._batch_params_for(f)

            if override_settings:
                params = replace(params, export=current_export)
            else:
                # Always use current session export path/mode/format even for
                # per-file exports. Per-file configs from the DB bypass
                # _apply_sticky_settings and may have stale ABSOLUTE/export_path
                # or DNG export_fmt values that don't match what the UI shows.
                params = replace(
                    params,
                    export=replace(
                        params.export,
                        output_mode=current_export.output_mode,
                        export_path=current_export.export_path,
                        output_subfolder=current_export.output_subfolder,
                        export_fmt=current_export.export_fmt,
                        export_color_space=current_export.export_color_space,
                    ),
                )

            final_export = replace(
                params.export,
                icc_input_path=self.effective_input_icc(params.process),
                icc_output_path=icc_output,
            )

            if flat:
                params = flat_master_config(params)
                final_export = flat_export_config(final_export, fmt=flat_fmt)

            bounds_override = None
            if f["hash"] == self.state.current_file_hash:
                with self.state.metrics_lock:
                    bounds_override = self.state.last_metrics.get("log_bounds")

            source_exif = self.state.source_exif.get(f["hash"])
            metadata_config = self.state.config.metadata if sync_metadata else params.metadata

            tasks.append(
                ExportTask(
                    file_info=f,
                    params=params,
                    export_settings=preset_from_export_config(final_export),
                    gpu_enabled=self.state.gpu_enabled,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=metadata_config,
                    working_color_space=self.state.workspace_color_space,
                )
            )

        if tasks:
            self._run_export_tasks(tasks)

    def _preset_export_files_for_selection(self) -> list[dict]:
        """Selected filmstrip frames in display order; single selection exports the preview frame."""
        n = len(self.state.uploaded_files)
        selected = [i for i in self.state.selected_indices if 0 <= i < n]

        if len(selected) <= 1:
            if not self.state.current_file_path or not (0 <= self.state.selected_file_idx < n):
                return []
            file_info = self.state.uploaded_files[self.state.selected_file_idx]
            if file_info.get("excluded"):
                return []
            return [file_info]

        selected_set = set(selected)
        visible_order = self.session.asset_model.visible_actual_indices_ordered()
        ordered = [i for i in visible_order if i in selected_set]
        for i in sorted(selected_set):
            if i not in ordered:
                ordered.append(i)
        files = [self.state.uploaded_files[i] for i in ordered]
        return [f for f in files if not f.get("excluded")]

    def _build_preset_export_tasks(self, files: list[dict], presets: List[ExportPreset]) -> List[ExportTask]:
        sync_metadata = self.state.config.metadata.sync_to_batch
        tasks: List[ExportTask] = []
        for f in files:
            params = self._batch_params_for(f)

            bounds_override = None
            if f["hash"] == self.state.current_file_hash:
                with self.state.metrics_lock:
                    bounds_override = self.state.last_metrics.get("log_bounds")

            source_exif = self.state.source_exif.get(f["hash"])
            metadata_config = self.state.config.metadata if sync_metadata else params.metadata

            tasks.extend(
                self._tasks_for_file(
                    f,
                    params,
                    presets,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=metadata_config,
                )
            )
        return tasks

    def _confirm_bulk_export(self, text: str) -> bool:
        reply = QMessageBox.question(
            None,
            "Export",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _dispatch_preset_export(self, files: list[dict]) -> None:
        if self._batch_busy("export"):
            return
        if not files:
            return

        presets = self._enabled_presets()
        if not presets:
            QMessageBox.information(None, "No presets enabled", "Enable at least one export preset in the Export panel.")
            return

        if not self._validate_preset_paths(presets):
            return

        if len(files) > 1:
            n_frames = len(files)
            n_presets = len(presets)
            n_files = n_frames * n_presets
            preset_word = "preset" if n_presets == 1 else "presets"
            file_word = "file" if n_files == 1 else "files"
            if not self._confirm_bulk_export(f"Export {n_frames} frames through {n_presets} {preset_word} ({n_files} {file_word})?"):
                return

        if self.state.config.export.export_sidecars_enabled:
            self._write_edit_sidecars(files)

        tasks = self._build_preset_export_tasks(files, presets)
        if tasks:
            self._run_export_tasks(tasks)

    def request_preset_export(self) -> None:
        """Initiates high-resolution export for the current file using enabled presets."""
        if not self.state.current_file_path:
            return

        file_info = {
            "name": os.path.basename(self.state.current_file_path),
            "path": self.state.current_file_path,
            "hash": self.state.current_file_hash,
        }
        self._dispatch_preset_export([file_info])

    def request_preset_export_selected(self) -> None:
        """Initiates preset export for every selected filmstrip frame."""
        files = self._preset_export_files_for_selection()
        self._dispatch_preset_export(files)

    def request_preset_batch_export(self) -> None:
        """Initiates batch export for all visible files using enabled presets."""
        visible_files = [
            self.state.uploaded_files[i]
            for i in self.session.asset_model.visible_actual_indices_ordered()
            if not self.state.uploaded_files[i].get("excluded")
        ]
        self._dispatch_preset_export(visible_files)

    def _contact_sheet_output_dir(self, visible_files: list) -> Optional[str]:
        """Resolve the contact sheet output folder (custom path or export destination rules)."""
        custom = self.state.config.export.contact_sheet_output_path.strip()
        if custom:
            return custom
        if self.state.config.export.output_mode == ExportPresetOutputMode.SAME_AS_SOURCE:
            return os.path.dirname(visible_files[0]["path"])
        return self._ensure_valid_export_path()

    def request_contact_sheet(self) -> None:
        """Renders all visible files small and writes darkroom contact sheet(s)."""
        if self._batch_busy("contact sheet"):
            return
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return

        out_dir = self._contact_sheet_output_dir(visible_files)
        if not out_dir:
            return

        if len(visible_files) > 1 and not self._confirm_bulk_export(f"Render a contact sheet from {len(visible_files)} frames?"):
            return

        tasks = []
        for f in visible_files:
            params = self._batch_params_for(f)
            tasks.append(
                ExportTask(
                    file_info=f,
                    params=params,
                    export_settings=params.export,
                    gpu_enabled=self.state.gpu_enabled,
                    working_color_space=self.state.workspace_color_space,
                )
            )

        cs = self.state.config.export
        self._export_start_time = time.time()
        self._export_failures = 0
        if self._begin_batch("contact_sheet", "Contact sheet", abortable=True) is None:
            return
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_contact_sheet",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
            Q_ARG(str, out_dir),
            Q_ARG(int, cs.contact_sheet_cell_px),
            Q_ARG(int, cs.contact_sheet_gap),
            Q_ARG(int, cs.contact_sheet_margin),
            Q_ARG(int, cs.contact_sheet_max_tiles),
        )

    def _write_edit_sidecars(self, files: list[dict]) -> int:
        """Write a .negpy edit sidecar next to each source (each frame's own saved edits). Returns count written."""
        repo = self.session.repo
        written = 0
        for f in files:
            half = int(f.get("half") or 0)
            params = load_or_promote(repo, f["hash"], f["path"], half=half) or self.state.config
            try:
                write_sidecar(f["path"], params, half=half)
                written += 1
            except Exception as exc:
                logger.warning("Sidecar write failed for %s: %s", f.get("path"), exc)
        return written

    def export_edit_sidecars(self) -> None:
        """Explicit batch sidecar export for all visible files (ignores the on-export toggle)."""
        visible_files = [
            self.state.uploaded_files[i]
            for i in self.session.asset_model.visible_actual_indices_ordered()
            if not self.state.uploaded_files[i].get("excluded")
        ]
        if not visible_files:
            return
        written = self._write_edit_sidecars(visible_files)
        self.set_status(f"Wrote {written} edit sidecar(s)", 4000)

    def _run_export_tasks(self, tasks: List[ExportTask]) -> None:
        # Reject unencodable format/colour-space pairings before anything else.
        blocked = [t for t in tasks if export_blocked(t.export_settings.export_fmt, t.export_settings.export_color_space)]
        if blocked:
            names = ", ".join(sorted({t.file_info.get("name", "?") for t in blocked})[:5])
            QMessageBox.warning(
                None,
                "Export",
                f"JPEG XL can't tag the selected colour space ({names}).\n"
                "Choose sRGB, P3 D65, Rec 2020 or Greyscale, or a different format.",
            )
            return

        # Then confirm any overwrites before dispatching to the worker.
        tasks = self._resolve_export_conflicts(tasks)
        if not tasks:
            return

        self._export_start_time = time.time()
        self._export_failures = 0
        if self._begin_batch("export", "Exporting", abortable=True) is None:
            return
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_batch",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
        )

    def _resolve_export_conflicts(self, tasks: List[ExportTask]) -> Optional[List[ExportTask]]:
        """Decide how to handle existing destination files before dispatching an export.

        If the "Overwrite existing files" preference is on, overwrite silently (no prompt)
        — for single Export and Export All alike. Otherwise, if the batch would clobber
        existing files, prompt (Overwrite / Rename / Cancel); the dialog's "always
        overwrite without asking" toggle persists the preference. Returns the tasks to run
        (overwrite flag set to the chosen action) or None to cancel the whole export."""
        if not tasks:
            return tasks

        if self.state.config.export.overwrite:
            return [replace(t, export_settings=replace(t.export_settings, overwrite=True)) for t in tasks]

        conflicts = find_export_conflicts(tasks)
        if not conflicts:
            return tasks

        choice, remember = self._prompt_overwrite_conflicts(conflicts)
        if choice is None:
            return None
        if remember and choice:
            self._set_overwrite_preference(True)
        return [replace(t, export_settings=replace(t.export_settings, overwrite=choice)) for t in tasks]

    def _set_overwrite_preference(self, value: bool) -> None:
        """Persist the global 'Overwrite existing files' preference (syncs the Export tab
        checkbox and the sticky default) without touching edit history or re-rendering."""
        cfg = self.state.config
        if bool(cfg.export.overwrite) == value:
            return
        new_config = replace(cfg, export=replace(cfg.export, overwrite=value))
        self.session.update_config(new_config, persist=True, render=True, record_history=False)

    @staticmethod
    def _prompt_overwrite_conflicts(conflicts: List[str]) -> tuple[Optional[bool], bool]:
        """Ask how to handle existing destination files. Returns (choice, remember):
        choice is True (overwrite), False (rename with a numbered suffix) or None (cancel);
        remember is whether the user asked to always overwrite without being asked again."""
        n = len(conflicts)
        names = "\n".join("  • " + os.path.basename(p) for p in conflicts[:8])
        if n > 8:
            names += f"\n  … and {n - 8} more"

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Warning)
        if n == 1:
            box.setWindowTitle("File already exists")
            box.setText(f"“{os.path.basename(conflicts[0])}” already exists in the export folder.")
        else:
            box.setWindowTitle("Files already exist")
            box.setText(f"{n} files already exist in the export destination.")
        box.setInformativeText(f"{names}\n\nOverwrite, save with a new name, or cancel?")

        remember_check = QCheckBox("Always overwrite without asking")
        remember_check.setToolTip("Turns on the Export panel's “Overwrite existing files” option; stays on until you turn it off.")
        box.setCheckBox(remember_check)

        overwrite_label = "Overwrite" if n == 1 else "Overwrite All"
        rename_label = "Rename" if n == 1 else "Rename All"
        overwrite_btn = box.addButton(overwrite_label, QMessageBox.ButtonRole.DestructiveRole)
        rename_btn = box.addButton(rename_label, QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(rename_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()

        clicked = box.clickedButton()
        remember = remember_check.isChecked()
        if clicked is overwrite_btn:
            return True, remember
        if clicked is rename_btn:
            return False, remember
        return None, False

    def _on_render_finished(self, _result: Any, metrics: Dict[str, Any]) -> None:
        self._is_rendering = False

        if self._first_render_t0 is not None and not metrics.get("ephemeral"):
            logger.info(
                "load-timing first_render %.0fms (buffer -> painted) %s",
                (time.perf_counter() - self._first_render_t0) * 1000,
                self.state.current_file_path,
            )
            self._first_render_t0 = None

        # Config is replaced wholesale on every edit, so identity detects any change.
        should_update_thumb = (
            self._pending_render_task is None and not metrics.get("ephemeral") and self.state.config is not self._thumb_config
        )

        with self.state.metrics_lock:
            self.state.last_metrics.update(metrics)
            self.state.last_metrics["splash"] = False

        # Memoize the displayed pixels for instant navigate-back. ndarray only:
        # GPU textures are destroyed on navigation (in the default soft-proof
        # path the displayed buffer is already a CPU array). Stored by reference
        # — display buffers are read-only downstream.
        result = metrics.get("base_positive")
        if metrics.get("memo_key") and isinstance(result, np.ndarray) and metrics.get("source_hash") == self.state.current_file_hash:
            self._render_memo.store(
                metrics["source_hash"],
                metrics["memo_key"],
                {"base_positive": result, "content_rect": metrics.get("content_rect")},
            )

        if metrics.get("gpu_fallback") and not self._gpu_fallback_notified:
            self._gpu_fallback_notified = True
            self.set_status("GPU acceleration failed — using CPU", 5000)

        self.image_updated.emit()

        if should_update_thumb:
            self._thumb_config = self.state.config
            # persist=False: refresh in-memory only; disk JPEG written on switch/save/export.
            self._update_thumbnail_from_state(force_readback=True, persist=False)

        if self._pending_render_task:
            task = self._pending_render_task
            self._pending_render_task = None
            self._is_rendering = True
            self.render_requested.emit(task)

    def _on_metrics_updated(self, metrics: Dict[str, Any]) -> None:
        """
        Handles late-arriving metrics and persists analysis results.
        """
        with self.state.metrics_lock:
            self.state.last_metrics.update(metrics)
        if "ir_degenerate" in metrics:
            self.state.ir_degenerate = bool(metrics["ir_degenerate"])
        self.metrics_available.emit(metrics)

        # Don't persist bounds from an ephemeral (splash) render or a render of a different
        # file (late metric after a fast switch) — they aren't this frame's bounds.
        if metrics.get("ephemeral"):
            return
        src = metrics.get("source_hash")
        if src is not None and src != self.state.current_file_hash:
            return

        # Persist the per-frame *base* (not the final mix) — re-feeding a mix as the next
        # base stacks edits. Skip only when both axes ride the roll baseline.
        proc = self.state.config.process
        bounds = metrics.get("log_bounds_base") or metrics.get("log_bounds")
        if bounds and not (proc.use_luma_average and proc.use_colour_average):
            changes = {}
            if not proc.lock_bounds and (bounds.floors != proc.local_floors or bounds.ceils != proc.local_ceils):
                changes["local_floors"] = bounds.floors
                changes["local_ceils"] = bounds.ceils

            if changes:
                new_process = replace(self.state.config.process, **changes)
                self.session.update_config(
                    replace(self.state.config, process=new_process),
                    persist=True,
                    render=False,
                    record_history=False,
                )
                # render=False: the displayed pixels already reflect these measured
                # bounds — move the frame's memo entry to the updated config's key
                # so the first navigate-back after an initial render still hits.
                self._render_memo.rekey(src or self.state.current_file_hash or "", self._render_memo_key())

    def _on_render_error(self, message: str) -> None:
        self.state.is_processing = self._is_rendering = False
        logger.error(f"Worker failure: {message}")
        self.set_status(f"Failed to load file: {message}", 5000)
        self.load_failed.emit()

        if self._pending_render_task:
            task = self._pending_render_task
            self._pending_render_task = None
            self._is_rendering = True
            self.render_requested.emit(task)

    def _on_export_task_error(self, _message: str) -> None:
        self._export_failures += 1

    def _on_export_finished(self) -> None:
        elapsed = time.time() - self._export_start_time
        owner = self._active_batch if self._active_batch in ("export", "contact_sheet") else "export"
        self._end_batch(owner)
        self.export_finished.emit(elapsed, self._export_failures)
        self._update_thumbnail_from_state(force_readback=True)

    def _update_thumbnail_from_state(self, force_readback: bool = False, persist: bool = True) -> None:
        if not self.state.current_file_path or not self.state.current_file_hash:
            return
        with self.state.metrics_lock:
            metrics = dict(self.state.last_metrics)
        buffer = metrics.get("base_positive")

        if isinstance(buffer, GPUTexture):
            t0 = time.perf_counter()
            buffer = buffer.readback()
            logger.info(
                "thumb-refresh readback %.1fms %s",
                (time.perf_counter() - t0) * 1000,
                os.path.basename(self.state.current_file_path),
            )

        if buffer is not None and not isinstance(buffer, np.ndarray):
            buffer = metrics.get("analysis_buffer")
        if buffer is None or not isinstance(buffer, np.ndarray):
            return

        self.thumbnail_update_requested.emit(
            ThumbnailUpdateTask(
                filename=os.path.basename(self.state.current_file_path),
                file_hash=self.state.current_file_hash,
                buffer=buffer,
                color_space=self.state.workspace_color_space,
                persist=persist,
            )
        )

    def cleanup(self) -> None:
        """
        Total system evacuation on exit.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._render_debounce.stop()
        self._cursor_readout_timer.stop()
        if self.render_thread.isRunning():
            self.render_thread.quit()
            self.render_thread.wait()
        if self.export_thread.isRunning():
            self.export_thread.quit()
            self.export_thread.wait()
        if self.thumb_thread.isRunning():
            self.thumb_thread.quit()
            self.thumb_thread.wait()
        self._autocrop_cancel_requested = True
        self.batch_autocrop_worker.cancel(self._autocrop_batch_token)
        if self.norm_thread.isRunning():
            self.norm_thread.quit()
            self.norm_thread.wait()
        if self.discovery_thread.isRunning():
            self.discovery_thread.quit()
            self.discovery_thread.wait()
        if self.preview_load_thread.isRunning():
            self.preview_load_thread.quit()
            self.preview_load_thread.wait()
        self.scan_worker.cancel()
        if self.scan_thread.isRunning():
            self.scan_thread.quit()
            self.scan_thread.wait()
        self.capture_worker.shutdown()
        if self.capture_thread.isRunning():
            self.capture_thread.quit()
            self.capture_thread.wait()
        self.render_worker.destroy_all()

        # All GPU-touching threads are now joined; release the wgpu device.
        GPUDevice.destroy_singleton()
