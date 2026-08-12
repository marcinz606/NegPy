import math
import os
import time
from collections import Counter
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QCheckBox, QMessageBox

from negpy.kernel.system.text import count_of, plural
from negpy.desktop.converters import ImageConverter
from negpy.desktop.render_memo import RenderMemo
from negpy.desktop.session import (
    AppState,
    DesktopSessionManager,
    ToolMode,
    resolve_asset_hdr,
    resolve_asset_rgbscan,
    resolve_asset_stitch,
)
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
    TestStripTask,
    ThumbnailUpdateTask,
    ThumbnailWorker,
)
from negpy.desktop.workers.scan_worker import BatchRequest, RollPreviewRequest, ScanRequest, ScanWorker
from negpy.desktop.workers.library import LibrarySearchTask, LibrarySearchWorker
from negpy.desktop.workers.hdr import HdrTask, HdrWorker
from negpy.desktop.workers.stitch import StitchTask, StitchWorker
from negpy.features.hdr.models import hdr_frame_paths, hdr_hash, hdr_name, hdr_stem
from negpy.features.process.logic import effective_linear_raw
from negpy.features.stitch.models import stitch_hash, stitch_name
from negpy.desktop.workers.capture_worker import (
    CalibrationRequest,
    CaptureRequest,
    CaptureWorker,
    LiveViewRequest,
)
from negpy.domain.models import (
    ColorSpace,
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
from negpy.services.assets.half_frame import half_hash
from negpy.services.assets.sidecar import load_or_promote, write_sidecar
from negpy.features.exposure.analysis import (
    RING_GRID,
    STRIP_GRID,
    proof_grid,
    ring_cells,
    ring_overrides,
    rotate_grid,
    strip_cells,
    strip_overrides,
)
from negpy.features.exposure.logic import (
    calculate_wb_shifts,
    calculate_wb_shifts_from_log,
)
from negpy.features.altprocess.models import AltProcess
from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.geometry.logic import apply_fine_rotation, detect_closest_aspect_ratio, enforce_roi_aspect_ratio
from negpy.features.geometry.models import FINE_ROTATION_LIMIT, AutocropMode
from negpy.features.lab.models import LabConfig
from negpy.features.local.models import LocalAdjustmentsConfig
from negpy.features.exposure.transfer import is_transparency_transfer
from negpy.features.process.models import ProcessConfig, ProcessMode, invalidate_local_bounds, scan_setup_values
from negpy.services.assets.thumbnails import asset_thumbnail_key
from negpy.kernel.system.paths import get_resource_path
from negpy.features.retouch.logic import downsample_ir, trace_scratch
from negpy.features.retouch.models import RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.infrastructure.capture.settings import WhiteCaptureMode
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.services.rendering.source_identity import source_token
from negpy.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)

_THUMB_FAILED_MSG = "thumbnail failed — file may be unreadable"
# Busy toasts are cleared when the frame lands; the timeout is only a backstop for a
# render that dies without reaching _on_render_finished.
_BUSY_TOAST_MS = 30000


@dataclass(frozen=True)
class _PendingCaptureImport:
    """Capture intent carried across asynchronous discovery and session hydration."""

    process_mode: Optional[ProcessMode] = None
    detect_mode: bool = False
    capture_roll: str = ""
    capture_frame: Optional[int] = None


def _interactive_proxy(raw: Optional[Any]) -> Optional[Any]:
    """Preview-resolution stand-in for an HQ buffer; None when one is not needed.

    Every downstream cost scales with this buffer, so interactive frames render
    against it rather than the full-resolution original.
    """
    if not isinstance(raw, np.ndarray) or raw.ndim < 2:
        return None
    long_edge = max(raw.shape[:2])
    if long_edge <= APP_CONFIG.preview_render_size:
        return None
    scale = APP_CONFIG.preview_render_size / float(long_edge)
    w, h = max(1, round(raw.shape[1] * scale)), max(1, round(raw.shape[0] * scale))
    return cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)


def _interactive_ir_proxy(ir: Optional[Any], proxy: Optional[Any]) -> Optional[Any]:
    """IR plane matched to ``proxy``'s shape, or None when no proxy is in use.

    Not a plain resize: a defect is a *minimum* in IR transmittance, which area
    averaging removes.
    """
    if proxy is None or not isinstance(ir, np.ndarray):
        return None
    h, w = proxy.shape[:2]
    if ir.shape[:2] == (h, w):
        return ir
    return downsample_ir(ir, max(h, w), dims=(w, h))


def _capture_import_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _component_paths(files: List[Dict]) -> List[str]:
    """Every source file behind the loaded assets, composites decomposed into their parts.

    Re-discovery over an asset list that only saw primaries would drop the rest."""
    paths: List[str] = []
    for f in files:
        paths.append(f["path"])
        paths.extend(f[k] for k in ("green_path", "blue_path") if f.get(k))
        paths.extend(f.get("stitch_paths") or ())
        paths.extend(p for t in f.get("stitch_triplets") or () for p in t if p)
    return list(dict.fromkeys(paths))


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
        round(float(geometry.autocrop_rebate_trim), 4),
        bool(flatfield.apply),
        str(flatfield.profile_id),
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
    restore_stitches: Optional[dict] = None
    restore_hdr: Optional[dict] = None
    half_frame_profile: Optional[dict] = None  # {crop_rect, split_x, gutter_thickness}


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


# Solved knee fields under their slider names (sidebar/tone.py).
_KNEE_LABELS = {
    "shadow_grade": "Shadows Grade",
    "highlight_grade": "Highlights Grade",
    "midtone_gamma": "Snap",
}


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
    # Emitted before the GPU engine frees its texture pool; the canvas samples a
    # pooled texture directly and must drop it first.
    gpu_textures_released = pyqtSignal()
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
    zones_overlay_changed = pyqtSignal(bool)
    grain_focuser_changed = pyqtSignal(bool)
    printing_notes_changed = pyqtSignal(bool)
    printing_notes_requested = pyqtSignal()  # the canvas holds the annotated pixels
    strip_requested = pyqtSignal(TestStripTask)
    test_strip_changed = pyqtSignal(bool)  # True = mosaic is up, False = cleared or building
    zone_pins_changed = pyqtSignal()
    zone_arm_changed = pyqtSignal(object)  # armed zone, or None
    asset_discovery_requested = pyqtSignal(AssetDiscoveryTask)
    library_search_requested = pyqtSignal(LibrarySearchTask)
    library_search_finished = pyqtSignal(int)  # frames found (0 = nothing matched)
    library_cleared = pyqtSignal()  # roots forgotten elsewhere — the panel must re-read them
    stitch_requested = pyqtSignal(object)
    hdr_requested = pyqtSignal(object)
    thumbnail_requested = pyqtSignal(list)
    thumbnail_update_requested = pyqtSignal(ThumbnailUpdateTask)
    tool_sync_requested = pyqtSignal()
    config_updated = pyqtSignal()
    monitor_profile_changed = pyqtSignal()
    compare_changed = pyqtSignal(bool)
    flat_output_changed = pyqtSignal(bool)
    linear_output_changed = pyqtSignal(bool)
    flat_peek_changed = pyqtSignal(bool)
    zoom_requested = pyqtSignal(float)
    zoom_changed = pyqtSignal(float)
    _render_cleanup_requested = pyqtSignal(object)  # texture to spare, or None
    status_message_requested = pyqtSignal(str, int)
    status_progress_requested = pyqtSignal(int, int)
    batch_started = pyqtSignal(str, bool)  # title, abortable
    batch_progress = pyqtSignal(int, int, str)  # current, total, label
    batch_finished = pyqtSignal()
    pixel_readout_rgb = pyqtSignal(object)  # (r255, g255, b255) tuple or None
    densitometer_readout = pyqtSignal(object)  # DensitometerReading or None
    tone_drag_changed = pyqtSignal(str)  # exposure field being slider-dragged; "" = drag ended
    scan_devices_requested = pyqtSignal()
    scan_backend_requested = pyqtSignal(str)
    scan_requested = pyqtSignal(ScanRequest)
    scan_devices_ready = pyqtSignal(list)
    scan_progress = pyqtSignal(float, str)  # progress, phase name
    scan_finished = pyqtSignal(str)
    scan_error = pyqtSignal(str)
    scan_started = pyqtSignal()
    scan_cancelled = pyqtSignal()
    scan_ejected = pyqtSignal(bool)
    scan_eject_error = pyqtSignal(str)
    scan_frame_done = pyqtSignal(int, str)  # batch: frame number, rgb path
    scan_batch_finished = pyqtSignal(list)  # batch: all completed rgb paths
    scan_batch_requested = pyqtSignal(BatchRequest)
    scan_eject_requested = pyqtSignal(str)
    scan_roll_preview_requested = pyqtSignal(RollPreviewRequest)
    scan_roll_preview_ready = pyqtSignal(object)  # one RollPreview per strip slot
    scan_roll_preview_finished = pyqtSignal()
    capture_light_requested = pyqtSignal(int, int, int, int, str)
    capture_requested = pyqtSignal(CaptureRequest)
    capture_light_set = pyqtSignal(int, int, int, int)
    capture_progress = pyqtSignal(float)
    capture_channel = pyqtSignal(str)  # "R"/"G"/"B" as each triplet channel starts
    capture_camera_setting_applied = pyqtSignal(str)  # a set_camera_setting call ran to completion
    capture_live_view_failed = pyqtSignal(str)  # preview thread died after retries; session dropped
    capture_live_view_unsupported = pyqtSignal(str)  # body advertises no preview; none was attempted
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
        self.flush_export_settings: Optional[Callable[[], None]] = None

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
        # Shares the export thread: the batch lane serializes them anyway.
        self.stitch_worker = StitchWorker()
        self.stitch_worker.moveToThread(self.export_thread)
        self.hdr_worker = HdrWorker()
        self.hdr_worker.moveToThread(self.export_thread)
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
        # Shares the discovery thread: a library search ends in a discovery anyway,
        # and neither should ever run while the other is walking the disk.
        self.library_worker = LibrarySearchWorker()
        self.library_worker.moveToThread(self.discovery_thread)
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
        self._busy_toast = False
        self._pending_render_task: Any = None

        # Last displayed render per frame — navigate-back paints it instantly
        # while the authoritative render refreshes underneath.
        self._render_memo = RenderMemo()
        # (source_hash, memo_key, content_rect) of the on-screen GPU render; load_file
        # files its texture under this on the way out.
        self._last_render_identity: Optional[tuple] = None
        self._render_memo.large_entries = self.state.hq_preview
        # Test strips, keyed density/grade-blind (see _strip_memo_key). Four mosaics an
        # entry, so the conservative budget.
        self._strip_memo = RenderMemo()
        self._strip_memo.large_entries = True

        self._render_debounce = QTimer()
        self._render_debounce.setSingleShot(True)
        self._render_debounce.setInterval(80)
        self._render_debounce.timeout.connect(self.request_render)

        self._crop_bounds_dirty = False
        self._zone_preview_shown = False
        self._pin_dragging = False
        self._pin_solution: Optional[Any] = None

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
        from negpy.features.exposure.densitometer import compute_reading

        bounds = self.state.last_metrics.get("final_bounds") or self.state.last_metrics.get("log_bounds")
        if bounds is None:
            return None
        val = self._sample_normalized_log(nx, ny)
        if val is None:
            return None
        return compute_reading(val, bounds, display_rgb)

    def _sample_normalized_log(self, nx: float, ny: float, radius: int = 0) -> Optional[Tuple[float, float, float]]:
        """Mean of the (2·radius+1)² normalized-log patch at content-normalized nx,ny;
        None when no frame is probed. Shared by the hover probe (1×1) and zone pins."""
        from negpy.features.exposure.densitometer import map_display_to_norm

        metrics = self.state.last_metrics
        nl = metrics.get("normalized_log")
        if nl is None or self.canvas is None:
            return None
        disp = self.canvas.display_size()
        if disp is None:
            return None
        if isinstance(nl, np.ndarray):
            norm_h, norm_w = nl.shape[:2]
        else:
            norm_w, norm_h = nl.width, nl.height
        # nx,ny arrive content-normalized (the overlay subtracts the border) —
        # passing content_rect here would compensate twice.
        pos = map_display_to_norm(
            nx,
            ny,
            disp[0],
            disp[1],
            None,
            metrics.get("active_roi"),
            self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW),
            norm_w,
            norm_h,
        )
        if pos is None:
            return None
        x, y = pos
        x0, x1 = max(0, x - radius), min(norm_w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(norm_h, y + radius + 1)
        try:
            if isinstance(nl, np.ndarray):
                val = nl[y0:y1, x0:x1].reshape(-1, nl.shape[2]).mean(axis=0)
            else:
                region = np.asarray(nl.readback_region(x0, y0, x1 - x0, y1 - y0), dtype=np.float32)
                val = region[..., :3].reshape(-1, 3).mean(axis=0)
        except Exception:
            return None
        return (float(val[0]), float(val[1]), float(val[2]))

    def set_status(self, message: str, timeout: int = 0) -> None:
        self.status_message_requested.emit(message, timeout)

    def _connect_signals(self) -> None:
        self.render_requested.connect(self.render_worker.process)
        self.strip_requested.connect(self.render_worker.build_strip)
        self._render_cleanup_requested.connect(self.render_worker.cleanup)
        self.render_worker.strip_finished.connect(self.on_strip_finished)
        self.render_worker.strip_progress.connect(self.on_strip_progress)
        self.render_worker.busy.connect(self._on_render_busy)
        self.render_worker.finished.connect(self._on_render_finished)
        self.render_worker.metrics_updated.connect(self._on_metrics_updated)
        self.render_worker.error.connect(self._on_render_error)
        self.render_worker.error.connect(self._on_strip_error)

        self.export_worker.progress.connect(self.export_progress.emit)
        self.export_worker.progress.connect(self._on_batch_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.cancelled.connect(self._on_export_batch_cancelled)
        self.export_worker.error.connect(self._on_render_error)
        self.export_worker.error.connect(self._on_export_task_error)

        self.stitch_requested.connect(self.stitch_worker.run)
        self.stitch_worker.progress.connect(self._on_batch_progress)
        self.stitch_worker.registered.connect(self._on_stitch_registered)
        self.stitch_worker.cancelled.connect(self._on_stitch_cancelled)
        self.stitch_worker.error.connect(self._on_stitch_error)

        self.hdr_requested.connect(self.hdr_worker.run)
        self.hdr_worker.progress.connect(self._on_batch_progress)
        self.hdr_worker.solved.connect(self._on_hdr_solved)
        self.hdr_worker.cancelled.connect(self._on_hdr_cancelled)
        self.hdr_worker.error.connect(self._on_hdr_error)

        self.thumbnail_requested.connect(self.thumb_worker.generate)
        self.thumb_worker.progress.connect(self._on_thumbnail_progress)
        self.thumbnail_update_requested.connect(self.thumb_worker.update_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_worker.rendered_finished.connect(self._on_rendered_thumbnail)
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
        self.library_search_requested.connect(self.library_worker.search)
        self.library_worker.progress.connect(self._on_library_walk_progress)
        self.library_worker.finished.connect(self._on_library_search_finished)
        self.library_worker.error.connect(self._on_render_error)

        self.preview_load_requested.connect(self.preview_load_worker.process)
        self.preview_load_worker.splash.connect(self._on_splash_preview)
        self.preview_load_worker.finished.connect(self._on_preview_loaded)
        self.preview_load_worker.error.connect(self._on_render_error)
        self.preview_load_worker.load_failed.connect(self._on_preview_load_failed)

        self.scan_devices_requested.connect(self.scan_worker.list_devices)
        self.scan_backend_requested.connect(self.scan_worker.set_backend)
        self.scan_worker.devices_ready.connect(self.scan_devices_ready.emit)
        self.scan_worker.progress.connect(self.scan_progress.emit)
        self.scan_worker.finished.connect(self._on_scan_finished)
        self.scan_worker.error.connect(self.scan_error.emit)
        self.scan_requested.connect(self.scan_worker.run_scan)
        self.scan_batch_requested.connect(self.scan_worker.run_batch)
        self.scan_eject_requested.connect(self.scan_worker.eject)
        self.scan_worker.cancelled.connect(self.scan_cancelled.emit)
        self.scan_worker.frame_done.connect(self.scan_frame_done.emit)
        self.scan_worker.batch_finished.connect(self._on_scan_batch_finished)
        self.scan_worker.ejected.connect(self.scan_ejected.emit)
        self.scan_worker.eject_error.connect(self.scan_eject_error.emit)
        self.scan_roll_preview_requested.connect(self.scan_worker.run_roll_preview)
        self.scan_worker.roll_preview_ready.connect(self.scan_roll_preview_ready.emit)
        self.scan_worker.roll_preview_finished.connect(self.scan_roll_preview_finished.emit)
        self.capture_light_requested.connect(self.capture_worker.set_light)
        self.capture_requested.connect(self.capture_worker.run_capture)
        self.capture_worker.light_set.connect(self.capture_light_set.emit)
        self.capture_worker.progress.connect(self.capture_progress.emit)
        self.capture_worker.channel.connect(self.capture_channel.emit)
        self.capture_worker.camera_setting_applied.connect(self.capture_camera_setting_applied.emit)
        self.capture_worker.live_view_failed.connect(self.capture_live_view_failed.emit)
        self.capture_worker.live_view_unsupported.connect(self.capture_live_view_unsupported.emit)
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

        self.session.active_file_changing.connect(self._update_thumbnail_from_state)
        self.session.session_emptied.connect(self._render_memo.clear)
        self.session.session_emptied.connect(self._strip_memo.clear)
        self.session.file_selected.connect(self._on_file_selected_load)
        self.session.state_changed.connect(self.config_updated.emit)
        self.session.state_changed.connect(self._render_debounce.start)
        self.session.files_changed.connect(self._render_debounce.start)

    def generate_missing_thumbnails(self) -> None:
        missing = [f for f in self.state.uploaded_files if asset_thumbnail_key(f) not in self.state.thumbnails]
        if missing:
            if self._begin_batch("thumbnails", "Generating thumbnails", abortable=False) is None:
                return
            self._thumb_requested = [asset_thumbnail_key(f) for f in missing]
            self.set_status("GENERATING THUMBNAILS...")
            self.thumbnail_requested.emit(missing)

    def clear_thumbnail_cache(self) -> None:
        """Drops cached thumbnails on disk and in memory, then regenerates loaded ones."""
        self.asset_store.clear_thumbnails()
        # Must precede generate_missing_thumbnails: it only enqueues names absent here.
        self.state.thumbnails.clear()
        self.state.rendered_thumbnails.clear()
        self.session.asset_model.refresh()
        self.generate_missing_thumbnails()

    def _on_thumbnail_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"THUMBNAIL {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _set_thumbnail(self, key: str, pil_img: Any) -> None:
        u8_arr = np.array(pil_img.convert("RGB"))
        self.state.thumbnails[key] = QIcon(QPixmap.fromImage(ImageConverter.to_qimage(u8_arr)))

    def _on_thumbnails_finished(self, new_thumbs: Dict[str, Any]) -> None:
        self.status_progress_requested.emit(0, 0)
        self._end_batch("thumbnails")
        for key, pil_img in new_thumbs.items():
            # A frame that already rendered on the canvas has the correct (inverted)
            # thumbnail; don't let this batch overwrite it with the source-decode placeholder.
            if pil_img and key not in self.state.rendered_thumbnails:
                self._set_thumbnail(key, pil_img)

        requested = getattr(self, "_thumb_requested", [])
        self._thumb_requested = []
        failed = {k for k in requested if not new_thumbs.get(k)}
        for f in self.state.uploaded_files:
            key = asset_thumbnail_key(f)
            if key in failed:
                f.setdefault("decode_failed", _THUMB_FAILED_MSG)
            elif key in new_thumbs and f.get("decode_failed") == _THUMB_FAILED_MSG:
                del f["decode_failed"]
        self.session.asset_model.refresh()

    def _on_rendered_thumbnail(self, new_thumbs: Dict[str, Any]) -> None:
        """A canvas render produced a thumbnail — it supersedes any batch placeholder."""
        for key, pil_img in new_thumbs.items():
            if pil_img:
                self._set_thumbnail(key, pil_img)
                self.state.rendered_thumbnails.add(key)
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
        elif self._active_batch == "stitch":
            self.stitch_worker.cancel()
        elif self._active_batch == "hdr":
            self.hdr_worker.cancel()

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
        stitches = self.session.repo.get_global_setting("session_stitches", {}) or {}
        merges = self.session.repo.get_global_setting("session_hdr_merges", {}) or {}
        self.request_asset_discovery(paths, auto_open=True, restore_triplets=triplets, restore_stitches=stitches, restore_hdr=merges)

    def request_asset_discovery(
        self,
        paths: List[str],
        auto_open: bool = False,
        restore_triplets: Optional[dict] = None,
        replace_existing: bool = False,
        reselect_path: Optional[str] = None,
        restore_stitches: Optional[dict] = None,
        restore_hdr: Optional[dict] = None,
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
            restore_stitches=restore_stitches,
            restore_hdr=restore_hdr,
            half_frame_profile=self.half_frame_profile(),
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
            restore_stitches=request.restore_stitches,
            restore_hdr=request.restore_hdr,
            half_frame_profile=request.half_frame_profile,
        )
        self.asset_discovery_requested.emit(task)

    def _start_next_asset_discovery(self) -> None:
        if self._pending_asset_discoveries and not self._discovery_running and self._active_batch is None:
            self._start_asset_discovery(self._pending_asset_discoveries.pop(0))

    # --- Library (folders on disk) --------------------------------------------

    def library_roots(self) -> List[str]:
        saved = self.session.repo.get_global_setting("library_roots", []) or []
        return [p for p in saved if isinstance(p, str)]

    def open_library_folder(self, folder: str, add_to_session: bool = False) -> None:
        self.open_library_folders([folder], add_to_session=add_to_session)

    def open_library_folders(self, folders: List[str], add_to_session: bool = False) -> None:
        """Load one or several folders' frames. Replacing the session costs nothing —
        every edit lives in the database under its own content hash, not in the file list."""
        present = [f for f in folders if os.path.isdir(f)]
        if not present:
            self.set_status("Folder is no longer on disk", 3000)
            return
        self.request_asset_discovery(
            present,
            auto_open=True,
            replace_existing=not add_to_session,
            reselect_path=self.state.current_file_path if add_to_session else None,
        )

    def invalidate_library_walk(self) -> None:
        """Drop the cached traversal so the next search re-reads the folders."""
        QMetaObject.invokeMethod(self.library_worker, "invalidate", Qt.ConnectionType.QueuedConnection)

    def request_library_search(self, query: str, rewalk: bool = False) -> None:
        """Search every library root, not just the loaded frames, and open the matches.

        Edit metadata joins onto unopened files by path, so a frame is findable by its
        film stock without being in the session — and without being hashed.
        """
        query = (query or "").strip()
        if not query:
            self.set_status("Type a search first, e.g. film:portra", 3000)
            return
        roots = self.library_roots()
        if not roots:
            self.set_status("Add a library folder first", 4000)
            return
        self.set_status("SEARCHING LIBRARY...")
        self.library_search_requested.emit(
            LibrarySearchTask(
                roots=roots,
                query=query,
                configs_by_path=self.session.repo.load_settings_by_path(),
                marks_by_path=self.session.repo.load_file_marks_by_path(),
                rewalk=rewalk,
            )
        )

    def _on_library_walk_progress(self, walked: int) -> None:
        self.set_status(f"SEARCHING LIBRARY... {walked} files")

    def _on_library_search_finished(self, paths: List[str]) -> None:
        self.library_search_finished.emit(len(paths))
        if not paths:
            self.set_status("No frames in the library match that search", 4000)
            return
        self.set_status(f"{len(paths)} frame{'s' if len(paths) != 1 else ''} found", 3000)
        self.request_asset_discovery(paths, auto_open=True, replace_existing=True)

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
        self.request_asset_discovery(_component_paths(files), replace_existing=True, reselect_path=self.state.current_file_path)

    def apply_scan_setup(self, capture: str, light: str) -> None:
        """Apply the scanning-setup wizard's answer: Linear RAW and Narrowband are rig
        properties, so they land on the new-file defaults, the open frame and every
        already-edited frame at once."""
        linear_raw, narrowband = scan_setup_values(capture, light)
        self.session.repo.save_global_settings(
            {
                "scan_setup": {"capture": capture, "light": light},
                "last_linear_raw": linear_raw,
                "last_narrowband_scan": narrowband,
            }
        )

        reload_needed = False
        if self.state.current_file_hash:
            reload_needed = self.state.config.process.linear_raw != linear_raw
            new_config = replace(
                self.state.config,
                process=replace(
                    self.state.config.process,
                    linear_raw=linear_raw,
                    narrowband_scan=narrowband,
                    **invalidate_local_bounds(self.state.config.process),
                ),
            )
            # render=False when reloading: bounds must not be analysed on the stale decode.
            self.session.update_config(new_config, persist=True, render=not reload_needed)

        count = 0
        for asset in self.session.state.uploaded_files:
            file_hash = asset["hash"]
            if file_hash == self.state.current_file_hash:
                continue
            # Frames with no saved edits inherit the values from the sticky defaults when
            # they are first hydrated — writing them here would only churn the DB.
            saved = self.session.repo.load_file_settings(file_hash)
            if saved is None:
                continue
            updated = replace(
                saved,
                process=replace(
                    saved.process,
                    linear_raw=linear_raw,
                    narrowband_scan=narrowband,
                    **invalidate_local_bounds(saved.process),
                ),
            )
            self.session.push_external_history(file_hash, saved, updated)
            self.session.repo.save_file_settings(file_hash, updated, file_path=asset["path"])
            count += 1

        if reload_needed and self.state.current_file_path:
            self.load_file(self.state.current_file_path)
        if count:
            self.session.settings_synced.emit(f"Scanning setup applied to {count} other frame{'s' if count != 1 else ''}")
            self.session.settings_saved.emit()

    def set_half_frame_mode(self, enabled: bool) -> None:
        """Persist the half-frame toggle and re-discover already-loaded assets so the
        mode splits/collapses frames in place (not only on the next folder load)."""
        self.session.repo.save_global_setting("half_frame_mode", bool(enabled))
        files = self.session.state.uploaded_files
        if not files:
            return
        self.request_asset_discovery(_component_paths(files), replace_existing=True, reselect_path=self.state.current_file_path)

    # ── half-frame split & crop profile ─────────────────────────────────

    _HALF_FRAME_PROFILE_KEY = "half_frame_profile"

    def half_frame_profile(self) -> dict | None:
        """Saved ``(crop_rect, split_x, gutter_thickness)`` profile, shared across
        every half-frame split. Scanner-independent — the same crop/split applies
        whether the scans came from a SANE scanner, a camera copy-stand, or a
        folder import."""
        return self.session.repo.get_global_setting(self._HALF_FRAME_PROFILE_KEY, default=None)

    def save_half_frame_profile(self, crop_rect, split_x: float, gutter_thickness: float) -> None:
        self.session.repo.save_global_setting(
            self._HALF_FRAME_PROFILE_KEY,
            {
                "crop_rect": list(crop_rect),
                "split_x": float(split_x),
                "gutter_thickness": float(gutter_thickness),
            },
        )

    def open_half_frame_dialog(self, file_path: str) -> dict | None:
        """Open the half-frame split & crop editor on one scan; return the profile
        dict on Apply, None on cancel."""
        import numpy as np

        from negpy.desktop.view.widgets.half_frame_dialog import HalfFrameDialog
        from negpy.services.assets.half_frame import detect_split_x
        from negpy.services.assets.thumbnails import decode_source_image

        try:
            img = decode_source_image(file_path)
            if img is None:
                return None
            buf = np.asarray(img)
        except Exception as e:
            self.set_status(f"Could not load preview: {e}")
            return None

        saved = self.half_frame_profile()
        initial_rect = tuple(saved["crop_rect"]) if saved else None
        initial_split = saved["split_x"] if saved else detect_split_x(buf)
        initial_gutter = saved["gutter_thickness"] if saved else 0.0

        dialog = HalfFrameDialog(
            buf,
            initial_rect=initial_rect,
            initial_split=initial_split,
            initial_gutter=initial_gutter,
            parent=None,
        )
        if dialog.exec():
            profile = {
                "crop_rect": list(dialog.crop_rect()),
                "split_x": dialog.split_x(),
                "gutter_thickness": dialog.gutter_thickness(),
            }
            self.save_half_frame_profile(profile["crop_rect"], profile["split_x"], profile["gutter_thickness"])
            return profile
        return None

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
            self.session.state.rendered_thumbnails.clear()
            self.session.add_files([], validated_info=valid_assets)
            self.generate_missing_thumbnails()
            idx = None
            if reselect_path:
                # Guard on a set path: `None in (path, green_path, blue_path)` spuriously
                # matches any non-RGB frame (its green/blue paths are absent → None).
                idx = next(
                    (
                        i
                        for i, f in enumerate(self.session.state.uploaded_files)
                        if reselect_path in (f.get("path"), f.get("green_path"), f.get("blue_path"))
                    ),
                    None,
                )
            if idx is None:
                # No prior frame to restore (a fresh folder open): land on the first frame
                # in filmstrip (sorted/filtered) order, not discovery order.
                ordered = self.session.asset_model.visible_actual_indices_ordered()
                idx = ordered[0] if ordered else 0
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
                # Select the first newly-loaded frame in filmstrip (sorted/filtered) order,
                # not discovery order — otherwise the initial frame lands mid-strip.
                new_indices = set(range(first_new_idx, len(self.session.state.uploaded_files)))
                ordered = self.session.asset_model.visible_actual_indices_ordered()
                target = next((i for i in ordered if i in new_indices), first_new_idx)
                self.session.select_file(target)
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

    def _half_slice_for_asset(
        self, path: Optional[str], file_hash: Optional[str]
    ) -> Optional[tuple[int, float, tuple[float, float, float, float] | None, float]]:
        """(half, split_x, crop_rect, gutter_thickness) for the asset at path/hash, or None."""
        if not file_hash:
            return None
        for f in self.state.uploaded_files:
            if f.get("hash") == file_hash or (path and f.get("path") == path and f.get("hash") == file_hash):
                half = int(f.get("half") or 0)
                if not half:
                    return None
                cr = f.get("crop_rect")
                crop_rect: tuple[float, float, float, float] | None = None
                if isinstance(cr, (tuple, list)):
                    vals = tuple(float(v) for v in cr)
                    if len(vals) == 4:
                        crop_rect = vals  # type: ignore[assignment]
                return (
                    half,
                    float(f.get("split_x") or 0.5),
                    crop_rect,
                    float(f.get("gutter_thickness") or 0.0),
                )
        return None

    def _active_half(self) -> Optional[tuple[int, float, tuple[float, float, float, float] | None, float]]:
        """(half, split_x, crop_rect, gutter_thickness) of the active asset, or None for whole-frame."""
        return self._half_slice_for_asset(self.state.current_file_path, self.state.current_file_hash)

    def _render_memo_key(self, config: Optional[WorkspaceConfig] = None) -> str:
        """Identity of everything that shapes the displayed render of the current
        config: the edit itself plus every display-path input. Any mismatch is a
        memo miss, so navigate-back only skips straight to pixels that would be
        reproduced exactly."""
        import hashlib
        import json

        config = self.state.config if config is None else config
        proofing = self.state.soft_proof_enabled
        narrowband = self.state.config.process.narrowband_scan
        parts = (
            json.dumps(config.to_dict(), sort_keys=True, default=str),
            self.state.hq_preview,
            self.state.workspace_color_space,
            self.state.gpu_enabled,
            proofing,
            self.effective_input_icc() if (proofing or narrowband) else None,
            self.effective_output_icc() if proofing else None,
            hashlib.md5(self.state.monitor_icc_bytes).hexdigest() if self.state.monitor_icc_bytes else "",
        )
        return hashlib.md5(repr(parts).encode()).hexdigest()

    def _strip_memo_key(self, kind: str = "tone") -> str:
        """The render key for a proof mosaic, prefixed by kind so the two can't collide.

        Both ladders are absolute, so each proof supplies the fields it varies and its mosaic
        is invariant to whatever those currently are. Pinning them makes print/pick/print again
        a cache hit. Any other edit (crop, paper, toning...) lands on a different key.

        Rotation is not in the key: an entry holds all four orientations.
        """
        exposure = self.state.config.exposure
        if kind == "colour":
            exposure = replace(exposure, wb_magenta=0.0, wb_yellow=0.0)
        else:
            exposure = replace(exposure, density=1.0, grade=115.0)
        return f"{kind}:{self._render_memo_key(replace(self.state.config, exposure=exposure))}"

    def _retain_displayed_texture(self) -> Optional[GPUTexture]:
        """File the on-screen GPU render in the memo and return it for the cleanup to spare.

        Refused mid-render: that render paints into the same pooled texture, so the pixels
        would stop matching the key they are filed under.
        """
        identity = self._last_render_identity
        self._last_render_identity = None
        texture = self.state.last_metrics.get("base_positive")
        if identity is None or not isinstance(texture, GPUTexture):
            return None
        if self._is_rendering or self._pending_render_task is not None:
            return None
        source_hash, memo_key, content_rect = identity
        self._render_memo.store(source_hash, memo_key, {"base_positive": texture, "content_rect": content_rect})
        return texture

    def _on_file_selected_load(self, file_path: str) -> None:
        """``session.file_selected`` handler: navigation honors the sticky-zoom preference."""
        self.load_file(file_path, preserve_zoom=self.state.sticky_zoom)

    def load_file(self, file_path: str, preserve_zoom: bool = False, force_detect: bool = False) -> None:
        """
        Dispatches RAW decode to a background worker to keep the UI thread free.
        """
        self._prefetch_gen += 1
        self._preview_load_t0 = time.perf_counter()
        self._requested_file_path = file_path
        # A strip belongs to one frame, and the memo fast path below repaints without
        # going through request_render — so drop it here, not only there. Zone pins
        # froze their sample from this frame, so they go the same way.
        self._clear_test_strip()
        self._drop_zone_pins()

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

        retained = self._retain_displayed_texture()
        # A retained texture outlives the pool, so the canvas keeps sampling it; without
        # one it must let go before the engine frees what it is showing.
        if retained is None:
            self.gpu_textures_released.emit()
        self._render_cleanup_requested.emit(retained)
        # The cleanup destroys the GPU textures last_metrics still points at; drop the
        # densitometer's probe sources so hover readouts go quiet until the next render.
        self.state.last_metrics.pop("normalized_log", None)
        self.state.last_metrics.pop("base_positive", None)
        self.state.last_metrics.pop("thumbnail_source", None)

        if memo is not None:
            with self.state.metrics_lock:
                self.state.last_metrics["base_positive"] = memo["base_positive"]
                self.state.last_metrics["content_rect"] = memo.get("content_rect")
                self.state.last_metrics["splash"] = False
                self.state.last_metrics["compare"] = False
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
        if pending_import is not None and (pending_import.capture_roll or pending_import.capture_frame is not None):
            meta = self.state.config.metadata
            self.state.config = replace(
                self.state.config,
                metadata=replace(
                    meta,
                    capture_roll=pending_import.capture_roll or meta.capture_roll,
                    capture_frame=(pending_import.capture_frame if pending_import.capture_frame is not None else meta.capture_frame),
                ),
            )
            self.state.is_dirty = True

        rgbscan = self.state.config.rgbscan
        stitch = self.state.config.stitch
        hdr = self.state.config.hdr
        flatfield = self.state.config.flatfield
        half_info = self._active_half()
        self.preview_load_requested.emit(
            PreviewLoadTask(
                file_path=file_path,
                workspace_color_space=self.state.workspace_color_space,
                use_camera_wb=not effective_linear_raw(self.state.config.process, self.state.config.exposure.render_intent),
                full_resolution=self.state.hq_preview,
                # The half suffix distinguishes the two halves' preview caches now
                # that the slice happens pre-downsample (each half is its own buffer).
                file_hash=self._file_hash_for_path(file_path),
                # A memoized frame is already painted — the embedded-JPEG splash
                # would repaint stale pixels over it.
                use_splash=memo is None,
                detect_mode=(
                    pending_import.detect_mode
                    if pending_import is not None
                    else force_detect or (self.state.autodetect_enabled and self.state.current_file_is_new)
                ),
                # Whole, not flattened: the worker gates on the same predicates the decode
                # paths use (is_rgb_triplet / stitch_active / hdr_active), so a disabled
                # section needs no blanking here.
                rgbscan=rgbscan,
                stitch=stitch,
                hdr=hdr,
                flatfield_profile_id=flatfield.profile_id if (stitch.stitch_enabled and flatfield.apply) else "",
                half_slice=half_info,
            )
        )

    def _split_active_half(self, raw: Any, dims: Any) -> tuple[Any, Any]:
        """No-op: the half-frame slice now happens in PreviewManager before the
        preview downsample, so both splash and linear buffers arrive already
        sliced to the active half (and at the same pixels export analyzes).
        Kept as a passthrough for the splash/loaded handlers that still call it.
        """
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
            self.state.last_metrics["compare"] = False
        self.image_updated.emit()

    def _on_preview_load_failed(self, file_path: str, message: str) -> None:
        for f in self.state.uploaded_files:
            if f["path"] == file_path:
                f["decode_failed"] = message
                self.session.asset_model.refresh()
                return

    def _on_preview_loaded(
        self,
        file_path: str,
        raw: Any,
        dims: Any,
        source_cs: str,
        ir_preview: Any,
        detected_mode: str,
        cam_matrix: Any = None,
    ) -> None:
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
        self.state.preview_cam_xyz, self.state.preview_camera_wb = cam_matrix or (None, None)
        self.state.preview_proxy = _interactive_proxy(raw)
        self.state.preview_ir = ir_preview
        self.state.preview_ir_proxy = _interactive_ir_proxy(ir_preview, self.state.preview_proxy)
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
            if idx < 0 or not files:
                return
            display_order = self.session.asset_model.visible_actual_indices_ordered()
            for path, h in neighbor_paths_and_hashes(files, display_order, idx):
                # Match the cache key load_file will use for this neighbour: its own saved
                # linear_raw, not the current file's. Otherwise the warm buffer lands under
                # the wrong key and navigation re-decodes anyway.
                saved = self.session.repo.load_file_settings(h) if h else None
                # effective_, so the key matches what load_file will decode. A neighbour with no
                # saved edit still resolves False here: its mode is not known without hydrating it,
                # which is the same miss as before rather than a new one.
                linear_raw = effective_linear_raw(saved.process, saved.exposure.render_intent) if saved else False
                neighbour_half = self._half_slice_for_asset(path, h)
                self.preview_load_requested.emit(
                    PreviewLoadTask(
                        file_path=path,
                        workspace_color_space=self.state.workspace_color_space,
                        use_camera_wb=not linear_raw,
                        # Half-size only: a full-res HQ neighbour (~720MB) evicts the
                        # active buffer; the cache key separates resolutions.
                        full_resolution=False,
                        file_hash=h,
                        use_splash=False,
                        for_cache_warm=True,
                        half_slice=neighbour_half,
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
        # Resolution is in the memo key: every entry is now a permanent miss holding a
        # full-size texture.
        self._render_memo.clear()
        self.session.set_hq_preview(not self.state.hq_preview)
        self._render_memo.large_entries = self.state.hq_preview
        if self.state.current_file_path:
            self.load_file(self.state.current_file_path, preserve_zoom=True)

    def handle_canvas_clicked(self, nx: float, ny: float) -> None:
        if self.state.active_tool == ToolMode.WB_PICK:
            self._handle_wb_pick(nx, ny)
        elif self.state.active_tool == ToolMode.DUST_PICK:
            self._handle_dust_pick(nx, ny)
        elif self.state.active_tool == ToolMode.SCRATCH_LINE:
            self._handle_scratch_line_pick(nx, ny)
        elif self.state.active_tool == ToolMode.ZONE_PLACE:
            self._handle_zone_pin(nx, ny)

    def set_active_tool(self, mode: ToolMode) -> None:
        # Both the crop and analysis-region tools show the full uncropped frame, so
        # crossing into/out of that set must re-render to swap the preview.
        uncropped = {ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW}
        preview_mode_changed = (self.state.active_tool in uncropped) != (mode in uncropped)
        leaving_crop = self.state.active_tool == ToolMode.CROP_MANUAL and mode != ToolMode.CROP_MANUAL
        leaving_zone_place = self.state.active_tool == ToolMode.ZONE_PLACE and mode != ToolMode.ZONE_PLACE
        self.state.active_tool = mode
        self.tool_sync_requested.emit()
        if leaving_zone_place:
            self.clear_zone_pins()
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

    def toggle_zones_overlay(self, force: Optional[bool] = None) -> None:
        """Adams-zone box overlay. Repaint only — the boxes are carved from the frame
        the canvas already holds, so no re-render is needed."""
        self.state.zones_overlay = (not self.state.zones_overlay) if force is None else bool(force)
        self.zones_overlay_changed.emit(self.state.zones_overlay)

    def toggle_grain_focuser(self, force: Optional[bool] = None) -> None:
        """Grain focuser loupe. Repaint only — it magnifies the frame the canvas already
        holds, so no re-render is needed."""
        self.state.grain_focuser = (not self.state.grain_focuser) if force is None else bool(force)
        self.grain_focuser_changed.emit(self.state.grain_focuser)

    def toggle_printing_notes(self, force: Optional[bool] = None) -> None:
        """Printing-notes overlay (dodge/burn map + print recipe). Repaint only — every
        number it shows is already in the config, so no re-render is needed."""
        self.state.printing_notes = (not self.state.printing_notes) if force is None else bool(force)
        self.printing_notes_changed.emit(self.state.printing_notes)

    def _set_alt_process(self, target: AltProcess, force: Optional[bool] = None) -> None:
        """B&W only — the stage is a no-op in any other mode. The two processes are
        mutually exclusive, so selecting one clears the other."""
        cfg = self.state.config
        on = (cfg.altproc.alt_process != target) if force is None else bool(force)
        mode = target if on else AltProcess.NONE
        self.session.update_config(replace(cfg, altproc=replace(cfg.altproc, alt_process=mode)), persist=True)
        self.request_render()

    def toggle_lith(self, force: Optional[bool] = None) -> None:
        self._set_alt_process(AltProcess.LITH, force)

    def toggle_cyanotype(self, force: Optional[bool] = None) -> None:
        self._set_alt_process(AltProcess.CYANOTYPE, force)

    def request_printing_notes_export(self) -> None:
        """Save the marked-up work print as its own file. The annotated pixels live in the
        canvas, so the view answers the signal (the print itself is never touched)."""
        if not self.state.current_file_path:
            return
        self.printing_notes_requested.emit()

    def printing_notes_target_path(self) -> Optional[str]:
        """Next free `<stem>_notes.jpg` in the export folder."""
        export_path = self._ensure_valid_export_path()
        if not export_path or not self.state.current_file_path:
            return None
        if self.state.config.export.output_mode == ExportPresetOutputMode.SAME_AS_SOURCE:
            export_path = os.path.dirname(self.state.current_file_path)
        stem = os.path.splitext(os.path.basename(self.state.current_file_path))[0]
        os.makedirs(export_path, exist_ok=True)
        path = os.path.join(export_path, f"{stem}_notes.jpg")
        counter = 2
        while os.path.exists(path):
            path = os.path.join(export_path, f"{stem}_notes_{counter}.jpg")
            counter += 1
        return path

    def arm_zone_target(self, zone: float) -> None:
        """Zone picked on the strip: the next canvas click prints that spot there.
        Picking the armed zone again disarms."""
        if self.state.preview_raw is None:
            return
        if self.state.zone_arm_target == float(zone):
            self._disarm_zone_target()
            return
        # Same reason compare, the peek and the strip are exclusive: they all want the canvas.
        restore = self.state.compare_mode or self.state.flat_peek
        if self.state.compare_mode:
            self.state.compare_mode = False
            self.compare_changed.emit(False)
        if self.state.flat_peek:
            self.state.flat_peek = False
            self.flat_peek_changed.emit(False)
        self._clear_test_strip()
        if restore:
            self.request_render()
        self.state.zone_arm_target = float(zone)
        self.set_active_tool(ToolMode.ZONE_PLACE)
        self.zone_arm_changed.emit(self.state.zone_arm_target)

    def _disarm_zone_target(self) -> None:
        """Drop the armed zone, and the tool with it when no pins remain."""
        armed = self.state.zone_arm_target is not None
        self.state.zone_arm_target = None
        if armed:
            self.zone_arm_changed.emit(None)
        if not self.state.zone_pins and self.state.active_tool == ToolMode.ZONE_PLACE:
            self.set_active_tool(ToolMode.NONE)

    def _handle_zone_pin(self, nx: float, ny: float) -> None:
        """Armed: the pin takes the zone picked on the strip. Unarmed: it takes the zone
        it already reads, so a bare click meters without moving the print."""
        from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R
        from negpy.features.exposure.placement import MAX_PINS, ZonePin

        val = self._sample_normalized_log(nx, ny, radius=2)
        if val is None:
            return
        armed = self.state.zone_arm_target
        val_luma = LUMA_R * val[0] + LUMA_G * val[1] + LUMA_B * val[2]
        target = armed if armed is not None else round(self._pin_zone(val_luma) * 3.0) / 3.0
        pin = ZonePin(
            nx=nx,
            ny=ny,
            val_rgb=val,
            val_luma=val_luma,
            target_zone=target,
            retargeted=armed is not None,
        )
        pins = self.state.zone_pins
        if len(pins) < MAX_PINS:
            pins.append(pin)
        else:
            nearest = min(range(len(pins)), key=lambda i: (pins[i].nx - nx) ** 2 + (pins[i].ny - ny) ** 2)
            pins[nearest] = pin
        if armed is not None:
            self.state.zone_arm_target = None
            self.zone_arm_changed.emit(None)
        self._refresh_pin_labels()
        self.zone_pins_changed.emit()
        if armed is not None:
            self._preview_zone_solution()

    def move_zone_pin(self, index: int, nx: float, ny: float, final: bool = False) -> None:
        """Drag a pin: re-samples the tone under it. An untargeted pin re-snaps to the
        new reading, a retargeted one keeps its zone, and the solve waits for `final`."""
        from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R

        pins = self.state.zone_pins
        if not 0 <= index < len(pins):
            return
        self._pin_dragging = not final
        val = self._sample_normalized_log(nx, ny, radius=2)
        if val is not None:
            pin = pins[index]
            val_luma = LUMA_R * val[0] + LUMA_G * val[1] + LUMA_B * val[2]
            target = pin.target_zone if pin.retargeted else round(self._pin_zone(val_luma) * 3.0) / 3.0
            pins[index] = replace(pin, nx=nx, ny=ny, val_rgb=val, val_luma=val_luma, target_zone=target)
            self._refresh_pin_labels()
        self.zone_pins_changed.emit()
        if final and self._zone_preview_shown:
            self._preview_zone_solution()

    def _pin_zone(self, val_luma: float) -> float:
        from negpy.features.exposure.placement import predicted_zone

        return predicted_zone(
            self.state.config.exposure,
            self.state.config.process.process_mode,
            self.state.last_metrics,
            val_luma,
        )

    def _solve_zone_placement(self) -> Optional[Any]:
        from negpy.features.exposure.placement import solve_placement

        if not self.state.zone_pins:
            return None
        return solve_placement(
            self.state.config.exposure,
            self.state.config.process.process_mode,
            self.state.last_metrics,
            self.state.zone_pins,
        )

    def _refresh_pin_labels(self) -> None:
        """Re-read each pin's zone through the current curve. Called before every
        zone_pins_changed emit: the overlay and the sidebar repaint in connection
        order, so the label cannot be left to whichever runs first."""
        from negpy.features.exposure.densitometer import zone_roman

        pins = self.state.zone_pins
        for i, pin in enumerate(pins):
            label = zone_roman(self._pin_zone(pin.val_luma))
            if pin.label != label:
                pins[i] = replace(pin, label=label)

    def zone_pin_readouts(self) -> List[Tuple[int, str, float, Optional[str], bool]]:
        """Sidebar rows: (index, measured roman, target zone, achieved roman when the
        target is out of the paper's scale, solvable). Refreshes each pin's canvas label."""
        from negpy.features.exposure.densitometer import zone_roman

        pins = self.state.zone_pins
        if not pins:
            self._pin_solution = None
            return []
        self._refresh_pin_labels()
        # The two-pin nested bisection costs ~15 ms, too slow per mouse-move: mid-drag
        # the last solve stands in, and the drag's end recomputes it.
        if not self._pin_dragging:
            self._pin_solution = self._solve_zone_placement()
        sol = self._pin_solution
        rows = []
        for i, pin in enumerate(pins):
            achieved = zone_roman(sol.achieved[i]) if sol is not None and sol.clamped and i < len(sol.achieved) else None
            rows.append((i, pin.label, pin.target_zone, achieved, sol is not None))
        return rows

    def zone_solve_caption(self) -> str:
        """What the current pins are solving, named as the sliders name it. Reads the
        solution `zone_pin_readouts` cached — call it after, not before."""
        sol = self._pin_solution
        if sol is None or not self.state.zone_pins:
            return ""
        controls = ["Print Density"]
        if "grade" in sol.fields:
            controls.append("ISO-R Grade")
        if sol.knee:
            controls.append(_KNEE_LABELS[sol.knee])
        return "Solving " + " + ".join(controls)

    def set_zone_pin_target(self, index: int, zone: float) -> None:
        """Retarget one pin and preview the solved exposure without committing it."""
        pins = self.state.zone_pins
        if not 0 <= index < len(pins):
            return
        pins[index] = replace(pins[index], target_zone=min(max(float(zone), 0.0), 10.0), retargeted=True)
        self.zone_pins_changed.emit()
        self._preview_zone_solution()

    def _preview_zone_solution(self) -> None:
        sol = self._solve_zone_placement()
        if sol is None:
            return
        self._pin_solution = sol
        self._zone_preview_shown = True
        self.request_render(
            readback_metrics=False,
            config_override=replace(self.state.config, exposure=replace(self.state.config.exposure, **sol.fields)),
        )

    def apply_zone_placement(self) -> None:
        """Commit the solved Print Density (and Grade, and the knee control a third pin
        was solved on) and put the tool down. The autos it replaces go off: one left on
        would re-move the placed tones."""
        sol = self._solve_zone_placement()
        if sol is None:
            return
        self._zone_preview_shown = False
        self.session.update_config(
            replace(self.state.config, exposure=replace(self.state.config.exposure, **sol.fields)),
            persist=True,
        )
        self.set_active_tool(ToolMode.NONE)  # drops the pins; no preview left to restore
        self.request_render()

    def remove_zone_pin(self, index: int) -> None:
        """Drop one pin; what remains re-solves. Dropping the last one puts the committed
        print back and the tool down."""
        pins = self.state.zone_pins
        if not 0 <= index < len(pins):
            return
        pins.pop(index)
        self._pin_solution = None
        self._refresh_pin_labels()
        self.zone_pins_changed.emit()
        if not pins:
            self.set_active_tool(ToolMode.NONE)
        elif self._zone_preview_shown:
            self._preview_zone_solution()

    def clear_zone_pins(self) -> None:
        """Drop the pins; restores the committed edit if a preview was on the canvas."""
        restore = self._zone_preview_shown
        self._drop_zone_pins()
        if restore:
            self.request_render()

    def _drop_zone_pins(self) -> None:
        if self.state.zone_arm_target is not None:
            self.state.zone_arm_target = None
            self.zone_arm_changed.emit(None)
        if not self.state.zone_pins and not self._zone_preview_shown:
            return
        self.state.zone_pins.clear()
        self._zone_preview_shown = False
        self._pin_dragging = False
        self._pin_solution = None
        self.zone_pins_changed.emit()

    def toggle_ring_around(self, force: Optional[bool] = None) -> None:
        """Print (or clear) the colour ring-around — the M/Y filtration proof."""
        self.toggle_test_strip(force, kind="colour")

    def toggle_test_strip(self, force: Optional[bool] = None, kind: str = "tone") -> None:
        """Print (or clear) a proof mosaic: the density × grade strip, or the colour ring-around.
        Entering dispatches one job, since these need pixels the canvas doesn't have.

        Both share the one proof slot, so asking for the other kind swaps it.
        """
        showing = (self.state.test_strip or self.state.test_strip_pending) and self.state.test_strip_kind == kind
        target = (not showing) if force is None else bool(force)
        if not target:
            self._clear_test_strip()
            return
        if self.state.preview_raw is None:
            return

        # Unrotated: one print yields every orientation, so rotating never re-renders.
        grid = RING_GRID if kind == "colour" else STRIP_GRID
        overrides = ring_overrides() if kind == "colour" else strip_overrides()
        toast = "Printing the colour ring-around…" if kind == "colour" else "Printing test strip…"

        # Reprinting an unchanged proof is a pile of renders for pixels we already have.
        cached = self._strip_memo.get(self.state.current_file_hash or "", self._strip_memo_key(kind))
        if cached is not None:
            self.state.test_strip_kind = kind
            self.on_strip_finished(cached["mosaics"], cached["content_rect"], from_cache=True)
            return

        self.state.test_strip_kind = kind
        self.state.test_strip_pending = True
        self.test_strip_changed.emit(False)
        # A few seconds of renders: tick the HUD so it doesn't read as wedged.
        self.status_message_requested.emit(toast, 2500)
        self.status_progress_requested.emit(0, len(overrides))
        self.strip_requested.emit(
            TestStripTask(
                buffer=self.state.preview_raw,
                config=self.state.config,
                source_hash=self.state.current_file_hash or "preview",
                # Always preview res, never HQ: full-res renders per patch would take minutes,
                # and each patch is shown at a fraction of the frame's width anyway.
                preview_size=float(APP_CONFIG.preview_render_size),
                overrides=tuple(overrides),
                grid=grid,
                gpu_enabled=self.state.gpu_enabled,
                ir_buffer=self.state.preview_ir,
                cam_xyz=self.state.preview_cam_xyz,
                camera_wb=self.state.preview_camera_wb,
            )
        )

    def on_strip_progress(self, done: int, total: int) -> None:
        if self.state.test_strip_pending:
            self.status_progress_requested.emit(done, total)

    def _clear_test_strip(self) -> None:
        """Drop the strip and its mosaic; silent when there was nothing up."""
        if not (self.state.test_strip or self.state.test_strip_pending):
            return
        if self.state.test_strip_pending:
            self.status_progress_requested.emit(0, 0)  # total <= 0 hides the bar
        self.state.test_strip = False
        self.state.test_strip_pending = False
        self.state.test_strip_mosaic = None
        self.state.test_strip_mosaics = None
        self.state.test_strip_content_rect = None
        self.test_strip_changed.emit(False)

    def on_strip_finished(self, mosaics: Any, content_rect: Any, from_cache: bool = False) -> None:
        # A render that landed while the strip was building already cancelled it.
        if not (self.state.test_strip_pending or from_cache):
            return
        if not from_cache:
            # Keyed on the config as it stands now, not as it was at dispatch: measured
            # bounds are persisted after a render with render=False, so the config drifts
            # mid-print without invalidating anything (the same drift RenderMemo.rekey
            # exists for). Keying at dispatch made every reprint a miss. Safe because a
            # change that *did* matter would have gone through request_render, which
            # cancels the strip outright.
            self._strip_memo.store(
                self.state.current_file_hash or "",
                self._strip_memo_key(self.state.test_strip_kind),
                {"mosaics": mosaics, "content_rect": content_rect},
            )
        self.state.test_strip_pending = False
        self.state.test_strip = True
        self.state.test_strip_mosaics = mosaics
        self.state.test_strip_mosaic = mosaics[self.state.test_strip_rotation]
        self.state.test_strip_content_rect = content_rect
        self.test_strip_changed.emit(True)
        label = "Ring-around" if self.state.test_strip_kind == "colour" else "Test strip"
        self.status_message_requested.emit(f"{label} ready — click a patch to keep it", 4000)

    def rotate_test_strip(self, direction: int) -> bool:
        """Turn the ladder rather than the image while a proof is on the canvas; True = consumed.

        Gated on a mosaic being up, never on `pending`: the rotate controls are global, and a
        proof that is only expected must not swallow them. Mid-print the turn falls through to
        the image, dropping the print like any other edit.
        """
        if not (self.state.test_strip and self.state.test_strip_mosaics):
            return False
        self.state.test_strip_rotation = (self.state.test_strip_rotation + direction) % 4
        self.state.test_strip_mosaic = self.state.test_strip_mosaics[self.state.test_strip_rotation]
        self.test_strip_changed.emit(True)
        return True

    def _on_strip_error(self, _message: str) -> None:
        """A print that errors never reaches on_strip_finished, and the stuck `pending` flag holds
        the progress bar open and blocks a reprint."""
        if self.state.test_strip_pending:
            self._clear_test_strip()

    def apply_test_strip_pick(self, row: int, col: int) -> None:
        """Commit the clicked patch's settings, then drop the proof.

        Cast Removal and the Auto toggles are left alone: the patches were rendered under
        them, so flipping one would render something other than the patch that was clicked.
        """
        if not self.state.test_strip:
            return
        exposure = self.state.config.exposure
        colour = self.state.test_strip_kind == "colour"
        base_grid = RING_GRID if colour else STRIP_GRID
        rotation = self.state.test_strip_rotation
        cells = rotate_grid(ring_cells() if colour else strip_cells(), base_grid, rotation)
        _, _, first, second = cells[row * proof_grid(base_grid, rotation)[1] + col]
        if colour:
            new_exposure = replace(exposure, wb_magenta=first, wb_yellow=second)
        else:
            new_exposure = replace(exposure, density=first, grade=second)
        self._clear_test_strip()
        self.session.update_config(replace(self.state.config, exposure=new_exposure), persist=True)
        self.request_render()

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
            return resolve_asset_hdr(resolve_asset_stitch(resolve_asset_rgbscan(self.state.config, asset), asset), asset)
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
            self.set_status(f"Auto Crop All preserved {count_of(preflight_skipped, 'frame')}; nothing to analyze", 4000)
            return

        token = self._begin_batch("autocrop", "Auto cropping roll", abortable=True)
        if token is None:
            return
        self._autocrop_batch_token = token
        self._autocrop_dispatched = len(frames)
        self._autocrop_preflight_skipped = preflight_skipped
        self._autocrop_cancel_requested = False
        self.set_status(f"Auto cropping {count_of(len(frames), 'frame')}...")
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
            self._update_thumbnail_from_state()

    def clear_retouch(self) -> None:
        from negpy.desktop.view.confirm import confirm_clear_heals

        conf = self.state.config.retouch
        count = len(conf.manual_dust_spots) + len(conf.manual_heal_strokes) + len(conf.scratch_lines)
        if count == 0:
            return
        # Wiping every heal is not step-recoverable like single-heal undo — confirm.
        if not confirm_clear_heals(None, count):
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=[], manual_heal_strokes=[], scratch_lines=[]),
            ),
            persist=True,
        )
        self.request_render()

    def delete_heal(self, kind: str, index: int) -> None:
        """Removes one placed heal by identity ("stroke"/"spot", index) — lets the
        user pick off a bad patch directly instead of unwinding newer heals first."""
        strokes = list(self.state.config.retouch.manual_heal_strokes)
        spots = list(self.state.config.retouch.manual_dust_spots)
        lines = list(self.state.config.retouch.scratch_lines)
        if kind == "stroke" and 0 <= index < len(strokes):
            strokes.pop(index)
        elif kind == "spot" and 0 <= index < len(spots):
            spots.pop(index)
        elif kind == "line" and 0 <= index < len(lines):
            lines.pop(index)
        else:
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=spots, manual_heal_strokes=strokes, scratch_lines=lines),
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
        lines = list(self.state.config.retouch.scratch_lines)
        if lines:
            lines.pop()
        elif strokes:
            strokes.pop()
        elif spots:
            spots.pop()
        else:
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=spots, manual_heal_strokes=strokes, scratch_lines=lines),
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

    def _handle_scratch_line_pick(self, nx: float, ny: float) -> None:
        """One click near a transport scratch: trace the whole line and commit it.

        Traced on the source-frame preview, so the stored line is in raw coordinates and the
        render re-measures the scratch at its own resolution. A click that finds nothing says
        so rather than committing a line that would repair nothing.
        """
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        preview = self.state.preview_raw
        if uv_grid is None or preview is None:
            return
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)
        line = trace_scratch(preview, rx, ry, self.state.config.retouch.scratch_threshold)
        if line is None:
            self.status_message_requested.emit("No scratch found there — click directly on the line", 3000)
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, scratch_lines=list(self.state.config.retouch.scratch_lines) + [line]),
            ),
            persist=True,
        )
        self.request_render()

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
        # Brush size is a diameter at HEAL_SIZE_REF scale (same convention as the
        # pipeline radius and the overlay cursor). The trailing zeroes are the retired
        # clone-source offset: repairs are content-aware now, but the stroke keeps its
        # four-element shape so stored edits load unchanged.
        stroke = ([[rx, ry] for rx, ry in raw_pts], size, 0.0, 0.0)
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_heal_strokes=conf.manual_heal_strokes + [stroke]),
            ),
            persist=True,
        )
        self.request_render()

    def handle_local_mask_created(self, shape: str, viewport_vertices: list) -> None:
        from negpy.features.local.logic import min_points
        from negpy.features.local.models import LocalMask, MaskShape

        mask_shape = MaskShape(shape)
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None or len(viewport_vertices) < min_points(mask_shape):
            return

        raw_vertices = tuple(CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_vertices)

        mask = LocalMask(vertices=raw_vertices, shape=mask_shape)
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
        from negpy.features.local.logic import min_points

        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        local = self.state.config.local
        if uv_grid is None or not (0 <= index < len(local.masks)):
            return
        if len(viewport_vertices) < min_points(local.masks[index].shape):
            return
        raw_vertices = tuple(CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_vertices)
        masks = list(local.masks)
        masks[index] = replace(masks[index], vertices=raw_vertices)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.config_updated.emit()
        self.request_render()

    def delete_local_vertex(self, index: int, vertex_index: int) -> None:
        """Remove one vertex from a polygon mask. Keep a minimum of 3 vertices."""
        from negpy.features.local.models import MaskShape

        local = self.state.config.local
        if not (0 <= index < len(local.masks)):
            return
        mask = local.masks[index]
        if mask.shape != MaskShape.POLYGON:
            return
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

    def update_selected_local_mask(self, persist: bool = True, readback_metrics: bool = True, **changes) -> None:
        local = self.state.config.local
        idx = self.state.local_selected_mask
        if not (0 <= idx < len(local.masks)):
            return
        masks = list(local.masks)
        masks[idx] = replace(masks[idx], **changes)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=persist)
        self.request_render(readback_metrics=readback_metrics)

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
                f"Strongly recommended: crop the remaining {count_of(total - cropped, 'file')} "
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
            sheet_note = (
                f"Note: the Sheet filter is on — only the {count_of(total, 'visible frame')} {plural(total, 'is', 'are')} analyzed.\n\n"
            )

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
            p = self.session.repo.load_file_settings(f_info["hash"]) or self.session.config_for_asset(f_info)
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
                p = self.session.repo.load_file_settings(f_info["hash"]) or self.session.config_for_asset(f_info)
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

    def set_active_flatfield_profile(self, profile_id: str) -> None:
        """
        Selects the globally active flat-field reference profile (or clears it when
        ``profile_id`` is empty). Stamps its id + rig distortion onto the current
        image and re-renders.
        """
        from negpy.services.assets.flatfield import FlatFieldProfiles

        self.session.repo.save_global_setting("flatfield_active_profile", profile_id or "")
        prof = FlatFieldProfiles.get(profile_id) if profile_id else None
        pid = prof.id if prof else ""
        new_ff = replace(self.state.config.flatfield, profile_id=pid, apply=bool(pid), k1=prof.k1 if prof else 0.0)
        self.session.update_config(replace(self.state.config, flatfield=new_ff), persist=True)
        self.request_render()

    def save_flatfield_profile(self, name: str, path: str) -> None:
        """
        Bakes a reference image into a named flat-field profile and makes it active.
        """
        from negpy.services.assets.flatfield import FlatFieldProfiles

        profile_id = FlatFieldProfiles.create(name, path)
        if profile_id is None:
            self.set_status("Flat-field: could not read that reference image", 3000)
            return
        self.set_active_flatfield_profile(profile_id)
        self.set_status(f"Flat-field profile '{name}' saved", 2000)

    def delete_flatfield_profile(self, profile_id: str) -> None:
        """
        Removes a flat-field profile; clears the active correction if it was selected.
        """
        from negpy.features.flatfield.logic import invalidate_gain
        from negpy.services.assets.flatfield import FlatFieldProfiles

        if not profile_id:
            return
        FlatFieldProfiles.delete(profile_id)
        invalidate_gain(profile_id)
        if self.session.repo.get_global_setting("flatfield_active_profile") == profile_id:
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
            from negpy.services.assets.flatfield import FlatFieldProfiles

            FlatFieldProfiles.set_k1(active, k1)
        self.request_render()

    # ── Scanner integration ───────────────────────────────────────────

    def request_scan_devices(self) -> None:
        """Request device enumeration on the scan worker thread."""
        self.scan_devices_requested.emit()

    def set_scan_backend(self, backend_id: str) -> None:
        """Route the chosen scanner backend to the worker thread."""
        self.scan_backend_requested.emit(backend_id)

    def start_scan(self, req: ScanRequest) -> None:
        """Start a scan. The UI connects to scan signals for state updates."""
        self.scan_worker.prepare_scan()
        self.scan_started.emit()
        self.scan_requested.emit(req)

    def start_batch(self, req: BatchRequest) -> None:
        """Start a frame-range batch scan over a roll/strip feeder."""
        self.scan_worker.prepare_scan()
        self.scan_started.emit()
        self.scan_batch_requested.emit(req)

    def start_roll_preview(self, req: RollPreviewRequest) -> None:
        """Preview strip slots (results via scan_roll_preview_ready, then
        scan_roll_preview_finished). No scan_started — preview is dialog-local."""
        self.scan_worker.prepare_scan()
        self.scan_roll_preview_requested.emit(req)

    def eject_scanner(self, device_id: str) -> None:
        """Trigger the scanner's eject action on the worker thread."""
        self.scan_eject_requested.emit(device_id)

    def cancel_scan(self) -> None:
        self.scan_worker.cancel()

    def _on_scan_finished(self, path: str) -> None:
        """Auto-add scanned file to NegPy file list and select it."""
        self.scan_finished.emit(path)
        self._pending_scanned_file = path
        self.request_asset_discovery([path])

    def _on_scan_batch_finished(self, paths: list) -> None:
        """Import every frame a batch completed, including a stopped or failed run."""
        self.scan_batch_finished.emit(paths)
        if paths:
            self._pending_scanned_file = paths[-1]
            self.request_asset_discovery(list(paths))

    # ── Stitch (multi-part scan composite) ─────────────────────────────

    def request_stitch_selected(self) -> None:
        """Register the selected frames into one stitched composite asset."""
        if self._batch_busy("Stitch"):
            return
        files = [self.state.uploaded_files[i] for i in sorted(set(self.state.selected_indices)) if 0 <= i < len(self.state.uploaded_files)]
        by_path = {f["path"]: f for f in files}  # half-frame assets share a path
        ordered = sorted(by_path.values(), key=lambda f: os.path.basename(f["path"]).lower())
        if len(ordered) < 2:
            self.set_status("Select two or more frames to stitch", 4000)
            return
        if any(f.get("stitch_paths") for f in ordered):
            self.set_status("Stitching an already-stitched frame is not supported", 4000)
            return
        if self._begin_batch("stitch", "Stitching frames", abortable=True) is None:
            return
        self.stitch_requested.emit(
            StitchTask(
                files=tuple(dict(f) for f in ordered),
                params_by_path={f["path"]: self._batch_params_for(f) for f in ordered},
            )
        )

    def _on_stitch_registered(self, payload: dict) -> None:
        self._end_batch("stitch")
        files = payload["files"]
        part_paths = [f["path"] for f in files]
        triplets = tuple((f.get("green_path") or "", f.get("blue_path") or "") for f in files)
        composite = {
            "name": stitch_name(part_paths),
            "path": part_paths[0],
            "hash": stitch_hash([f["hash"] for f in files]),
            "stitch_paths": tuple(part_paths[1:]),
            "stitch_transforms": payload["transforms"],
            "stitch_canvas": payload["canvas"],
            "stitch_sizes": payload["sizes"],
            "stitch_triplets": triplets,
            "stitch_align": bool(files[0].get("align", True)),
            # Same inheritance as a merge: a composite's fresh hash would otherwise take
            # the stale sticky mode rather than the parts' own.
            "process_mode": self._composite_process_mode(files),
        }
        if all(triplets[0]):
            # Thumbnail decode and the sensor-unmix skip read the primary's pair from here.
            composite.update(green_path=triplets[0][0], blue_path=triplets[0][1], align=composite["stitch_align"])
        wanted = set(part_paths)
        indices = [i for i, f in enumerate(self.state.uploaded_files) if f["path"] in wanted]
        self.session.apply_composite(indices, composite)
        self.set_status(f"Stitched {count_of(len(files), 'frame')}", 4000)
        # The composite bypasses asset discovery, so nothing else queues its thumbnail.
        self.generate_missing_thumbnails()

    def _on_stitch_cancelled(self) -> None:
        self._on_batch_cancelled("stitch")

    def _on_stitch_error(self, message: str) -> None:
        self._end_batch("stitch")
        self.set_status(message, 6000)

    def request_unstitch(self) -> None:
        """Dissolve the active stitched composite back into its part frames.

        Part edits restore from the DB by content hash; the composite's edits stay
        keyed under its stitch hash for a future re-stitch of the same parts."""
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        parts = asset.get("stitch_paths")
        if not parts:
            return
        paths = [asset["path"], *parts]
        # Triplet parts must come back as triplet assets, not as loose exposures.
        align = bool(asset.get("stitch_align", True))
        triplets = {path: [green, blue, align] for path, (green, blue) in zip(paths, asset.get("stitch_triplets") or ()) if green and blue}
        for green, blue, _ in triplets.values():
            paths.extend((green, blue))
        self.state.uploaded_files.pop(idx)
        key = asset_thumbnail_key(asset)
        self.session.state.thumbnails.pop(key, None)
        self.session.state.rendered_thumbnails.discard(key)
        self.session.asset_model.refresh()
        self._pending_scanned_file = paths[0]
        self.request_asset_discovery(paths, restore_triplets=triplets or None)

    def _composite_process_mode(self, files: list) -> str:
        """The film process a composite should inherit from its source frames.

        The most common mode among them, ties going to the first (the reference frame /
        primary part). Majority rather than just the primary's: a bracket's extreme
        exposures can autodetect differently — the frame that blows 46% of its area is
        not a reliable vote — while the frames of one physical slide always agree in fact.
        """
        modes = [str(self.session.config_for_asset(f).process.process_mode) for f in files]
        if not modes:
            return str(self.state.config.process.process_mode)
        counts = Counter(modes)
        top = max(counts.values())
        return next(m for m in modes if counts[m] == top)

    # ── HDR (bracketed-exposure merge) ─────────────────────────────────

    def request_hdr_merge_selected(self) -> None:
        """Solve the selected frames into one merged bracket asset."""
        if self._batch_busy("HDR merge"):
            return
        files = [self.state.uploaded_files[i] for i in sorted(set(self.state.selected_indices)) if 0 <= i < len(self.state.uploaded_files)]
        by_path = {f["path"]: f for f in files}  # half-frame assets share a path
        ordered = sorted(by_path.values(), key=lambda f: os.path.basename(f["path"]).lower())
        if len(ordered) < 2:
            self.set_status("Select two or more exposures of the same frame to merge", 4000)
            return
        if any(f.get("hdr_paths") for f in ordered):
            self.set_status("Merging an already-merged frame is not supported", 4000)
            return
        # Both are multi-file source assembly and an asset carries one primary path; the
        # composition order is definable but is not wired, so refuse rather than guess.
        if any(f.get("stitch_paths") for f in ordered):
            self.set_status("HDR merge of a stitched frame is not supported", 4000)
            return
        if any(f.get("green_path") for f in ordered):
            self.set_status("HDR merge of an RGB-scan triplet is not supported", 4000)
            return
        # Halves share a path, so by_path already dropped one of each pair; merging them
        # would silently produce a whole-frame composite. Half-frame assets are left whole
        # by every other assembly (see _expand_half_frames) for the same reason.
        if any(f.get("half") for f in ordered):
            self.set_status("HDR merge of a half-frame asset is not supported", 4000)
            return
        if self._begin_batch("hdr", "Merging exposures", abortable=True) is None:
            return
        self.hdr_requested.emit(
            HdrTask(
                files=tuple(dict(f) for f in ordered),
                params_by_path={f["path"]: self._batch_params_for(f) for f in ordered},
            )
        )

    def _on_hdr_solved(self, payload: dict) -> None:
        self._end_batch("hdr")
        files = payload["files"]
        reference = payload["reference"]
        # The reference frame becomes the composite's primary: it is the asset's own path
        # everywhere downstream, and the merge expresses radiance in its units.
        ordered = [files[reference], *[f for i, f in enumerate(files) if i != reference]]
        ratios = payload["ratios"]
        ordered_ratios = (ratios[reference], *[r for i, r in enumerate(ratios) if i != reference])
        frame_paths = [f["path"] for f in ordered]
        composite = {
            "name": hdr_name(frame_paths),
            "path": frame_paths[0],
            "hash": hdr_hash([f["hash"] for f in ordered]),
            "hdr_paths": tuple(frame_paths[1:]),
            "hdr_ratios": tuple(float(r) for r in ordered_ratios),
            "hdr_align": True,
            "hdr_anchor": "",  # bracket middle until the user nominates an exposure
            "process_mode": self._composite_process_mode(ordered),
        }
        wanted = set(frame_paths)
        indices = [i for i, f in enumerate(self.state.uploaded_files) if f["path"] in wanted]
        self.session.apply_composite(indices, composite)
        stops = math.log2(max(ratios) / min(ratios)) if min(ratios) > 0 else 0.0
        self.set_status(f"Merged {len(files)} exposures spanning {stops:.1f} stops", 4000)
        # The composite bypasses asset discovery, so nothing else queues its thumbnail.
        self.generate_missing_thumbnails()

    def _on_hdr_cancelled(self) -> None:
        self._on_batch_cancelled("hdr")

    def _on_hdr_error(self, message: str) -> None:
        self._end_batch("hdr")
        self.set_status(message, 6000)

    def apply_config(self, config: WorkspaceConfig, persist: bool = False, readback_metrics: bool = True) -> None:
        """Adopt `config` and repaint by whichever route the change actually needs.

        A change to a *source* input — a bracket, a triplet, a stitch, Linear RAW — cannot
        be honoured by re-running the pipeline: assembly happens while the source is
        decoded, so the buffer the pipeline starts from is already the wrong one. Compare
        `source_token` and re-decode when it moves.

        The alternative is every such control remembering to reload for itself, which is
        how the HDR render exposure shipped writing a value that never reached the canvas.
        """
        needs_decode = source_token(config) != source_token(self.state.config)
        # render=False on the decode branch: state_changed would analyse bounds against
        # the stale pre-reload buffer.
        self.session.update_config(config, persist=persist, render=not needs_decode)
        if needs_decode and self.state.current_file_path:
            self.load_file(self.state.current_file_path, preserve_zoom=True)
        else:
            self.request_render(readback_metrics=readback_metrics)

    def set_hdr_anchor(self, path: str) -> None:
        """Render the active merge at `path`'s exposure ("" = the bracket's middle).

        Which frame looks right is intent, not a measurement: the exposure *reference* is
        the longest frame that does not clip, which on a slide is brighter than the capture
        the photographer metered — a slide's own brightest point is denser than clear film.
        Stored on the asset, like the rest of the bracket, so it survives re-hydration.
        """
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        if not asset.get("hdr_paths") or str(asset.get("hdr_anchor", "") or "") == path:
            return
        asset["hdr_anchor"] = path
        # The asset dict is authoritative for the bracket, and only the manifest carries it
        # across a restart; nothing else persists between here and quitting.
        self.session.persist_session()
        cfg = self.state.config
        self.set_status(f"Rendering the merge as {os.path.basename(path)}" if path else "Rendering the merge at the bracket middle", 4000)
        # apply_config re-decodes: the bracket is merged while the source is decoded, so
        # the scale lives in the buffer the pipeline starts from, and a render alone
        # would re-run the pipeline over the already-merged buffer and change nothing.
        self.apply_config(replace(cfg, hdr=replace(cfg.hdr, hdr_anchor=path)))

    def request_unmerge_hdr(self) -> None:
        """Dissolve the active merged frame back into its exposures.

        Frame edits restore from the DB by content hash; the composite's edits stay keyed
        under its HDR hash for a future re-merge of the same bracket."""
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        frames = asset.get("hdr_paths")
        if not frames:
            return
        paths = [asset["path"], *frames]
        self.state.uploaded_files.pop(idx)
        key = asset_thumbnail_key(asset)
        self.session.state.thumbnails.pop(key, None)
        self.session.state.rendered_thumbnails.discard(key)
        self.session.asset_model.refresh()
        self._pending_scanned_file = paths[0]
        self.request_asset_discovery(paths)

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
        # Ensure the worker thread runs: the sidebar counts these writes and gates Scan until
        # each one reports back, so a write queued to a never-started thread would gate forever.
        self._ensure_capture_thread()
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
        capture_roll = getattr(req, "roll_name", "") if req is not None else ""
        capture_frame = getattr(req, "frame_number", None) if req is not None else None
        if white:  # slides / B&W negatives force a positive process
            mode = WhiteCaptureMode(req.white_process_mode)
            target = {WhiteCaptureMode.E6: ProcessMode.E6, WhiteCaptureMode.BW: ProcessMode.BW}.get(mode)
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(
                process_mode=target,
                detect_mode=target is None,
                capture_roll=capture_roll,
                capture_frame=capture_frame,
            )
        elif rgb:
            # Independently exposed RGB channels have no broadband orange-mask signal for
            # the normal classifier. They are negative scans unless capture metadata says
            # otherwise, so carry C-41 through discovery instead of guessing from the merge.
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(
                process_mode=ProcessMode.C41,
                capture_roll=capture_roll,
                capture_frame=capture_frame,
            )
        elif req is not None:
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(
                capture_roll=capture_roll,
                capture_frame=capture_frame,
            )
        self._pending_scanned_file = paths[0]
        self.request_asset_discovery(list(paths))

    def effective_output_icc(self) -> Optional[str]:
        """Output profile the preview proofs through: a custom override, else the
        profile for the selected export color space. None means no proof (Same as Source)."""
        return self.state.icc_output_path or ColorSpaceRegistry.get_icc_path(self.state.config.export.export_color_space)

    def effective_input_icc(self, process: Optional[ProcessConfig] = None, render_intent: Optional[str] = None) -> Optional[str]:
        """Source profile for color management: an explicit Input ICC wins; else the
        bundled RGBScan profile when Narrowband Scan is on; else None.

        The transparency transfer suppresses the implicit RGBScan profile: it has already
        converted the buffer to the working space through the camera matrix, so applying an
        input characterisation on top of that is a second, competing transform. An explicit
        Input ICC still wins — that is a deliberate user choice about their own source.
        """
        p = process if process is not None else self.state.config.process
        if self.state.icc_input_path:
            return self.state.icc_input_path
        intent = render_intent if render_intent is not None else self.state.config.exposure.render_intent
        if p.narrowband_scan and not is_transparency_transfer(p.process_mode, p.e6_normalize, intent):
            return get_resource_path("icc/RGBScan.icc")
        return None

    def display_transform_params(self, splash: bool = False) -> tuple[str, Optional[bytes], Optional[tuple]]:
        """Everything the display transform needs for the current render, as
        ``(color_space, monitor_icc_bytes, proof)``.

        Single source of truth for every consumer of a rendered buffer — the canvas
        shader, the CPU overlay and the filmstrip thumbnail must agree, or the same
        frame shows two different colours. Renders arrive in the working space; a
        proof is *not* baked into them, it is folded into the display LUT here (see
        ``get_display_lut``), which is what lets a GPU texture go to the shader
        untouched. ``splash`` marks the embedded camera thumbnail, already sRGB.
        """
        if splash:
            return ColorSpace.SRGB.value, self.state.monitor_icc_bytes, None
        return self.state.workspace_color_space, self.state.monitor_icc_bytes, self.proof_profiles()

    def proof_profiles(self) -> Optional[tuple]:
        """``(input_icc, output_icc)`` for the preview proof, or None when off.

        Narrowband Scan supplies an implicit *input* profile whether or not the
        soft-proof toggle is on; the output profile only applies with the toggle.
        """
        proofing = self.state.soft_proof_enabled
        if not (proofing or self.state.config.process.narrowband_scan):
            return None
        icc_input = self.effective_input_icc()
        icc_output = self.effective_output_icc() if proofing else None
        if not (icc_input or icc_output):
            return None
        return icc_input, icc_output

    def proof_active(self) -> bool:
        """True when the preview should soft-proof: the toggle is on and an input or
        output profile is available, or Narrowband Scan supplies an implicit input
        profile. Off → preview is the edit on the monitor."""
        if self.effective_input_icc() and self.state.config.process.narrowband_scan:
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

        # The strip's patches were printed from the config as it stood; once the edit
        # moves they are a proof of something else, so drop them (this also cancels a
        # strip still building). Zone pins die the same way.
        if config_override is None:
            self._clear_test_strip()
            self._drop_zone_pins()

        if self.state.preview_raw is None:
            return

        preview_raw = self.state.preview_raw
        if preview_raw is None:
            return

        # A drag asks for no metrics; the release does. Interactive frames go through
        # the proxy, so full resolution is reached only once the gesture settles.
        interactive = not readback_metrics
        ir_buffer = self.state.preview_ir
        if interactive and self.state.preview_proxy is not None:
            preview_raw = self.state.preview_proxy
            # The IR plane must follow the image it is read against.
            ir_buffer = self.state.preview_ir_proxy

        target_size = float(APP_CONFIG.preview_render_size)
        if self.state.hq_preview:
            target_size = float(max(preview_raw.shape[:2]))

        crop_preview_full = self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW)
        # Only a plain render of the saved edit is reproducible on navigate-back;
        # overrides (compare/flat peek), splash and tool previews are not memoized.
        # Interactive frames are excluded: a proxy render filed under the full-resolution
        # config's key would be painted back as if it were the real one.
        memo_key = ""
        if config_override is None and not ephemeral and not crop_preview_full and not interactive:
            memo_key = self._render_memo_key()

        task = RenderTask(
            buffer=preview_raw,
            config=config_override if config_override is not None else self.state.config,
            source_hash=self.state.current_file_hash or "preview",
            preview_size=target_size,
            gpu_enabled=self.state.gpu_enabled,
            readback_metrics=readback_metrics,
            ir_buffer=ir_buffer,
            crop_preview_full=crop_preview_full,
            ephemeral=ephemeral,
            memo_key=memo_key,
            compare=self.state.compare_mode,
            interactive=interactive,
            # Mirrors should_update_thumb, minus its pending-task check.
            wants_thumbnail=(not interactive and not ephemeral and config_override is None and self.state.config is not self._thumb_config),
            cam_xyz=self.state.preview_cam_xyz,
            camera_wb=self.state.preview_camera_wb,
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
            # Mutually exclusive with flat-peek: drop it so its toggle can't linger
            # lit while the compare baseline is what's actually on screen (mirrors
            # toggle_flat_peek, which exits compare on the way in).
            if self.state.flat_peek:
                self.state.flat_peek = False
                self.flat_peek_changed.emit(False)
            # Same reason the strip and the peek are exclusive: both want the canvas.
            self._clear_test_strip()
            self.state.compare_mode = True
            self.compare_changed.emit(True)
            self.request_render(readback_metrics=False, config_override=self._baseline_compare_config())

    def rerender_active_view(self) -> None:
        """Re-render the canvas keeping whatever comparison overlay is active.

        Geometry ops (rotate/flip) change the config but shouldn't kick the user
        out of before/after or flat-peek; a plain request_render() would exit both.
        Passing the mode's config_override re-renders in place and leaves the mode on.
        """
        if self.state.compare_mode:
            self.request_render(readback_metrics=False, config_override=self._baseline_compare_config())
        elif self.state.flat_peek:
            self.request_render(readback_metrics=False, config_override=flat_master_config(self.state.config))
        else:
            self.request_render()

    # --- Flat ("for editing elsewhere") master output -----------------------

    def set_flat_output(self, enabled: bool) -> None:
        """Toggle the flat digital-intermediate output intent (export + peek)."""
        if self.state.flat_output == enabled:
            return
        self.state.flat_output = enabled
        if enabled:
            self.state.linear_output = False
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
        if enabled:
            self.linear_output_changed.emit(False)
        # If a peek is active and flat output was turned off, drop back to the edit.
        if not enabled and self.state.flat_peek:
            self.toggle_flat_peek(force=False)

    def set_linear_output(self, enabled: bool) -> None:
        """Toggle the linear output intent (raw loader dump, no pipeline)."""
        if self.state.linear_output == enabled:
            return
        self.state.linear_output = enabled
        if enabled:
            self.state.flat_output = False
            self.flat_output_changed.emit(False)
            if self.state.flat_peek:
                self.toggle_flat_peek(force=False)
        self.session.save_flat_output_prefs()
        self.linear_output_changed.emit(enabled)

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
        if target:
            self._clear_test_strip()

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
        same authoritative source individual export gets via select_file.

        For the currently active file, always use the live session config to ensure
        unsaved edits (e.g., crosstalk adjustments not yet persisted) are included
        in the export.

        Half-frame siblings share capture-side spectral-crosstalk calibration — if one
        half has it enabled and the other's DB entry defaults to 0, propagate the active
        session value so both frames get identical dye-unmixing during export.
        """
        # For the active file, use the live session config — it may have unsaved
        # changes that the user expects to see in the export. For other files, use
        # saved DB settings (or fall back to session config if none exist).
        if f.get("hash") == self.state.current_file_hash:
            params = self.state.config
        else:
            params = self.session.repo.load_file_settings(f["hash"]) or self.state.config

        # Propagate capture-side crosstalk between sibling half-frames. The dye-unmix
        # calibration is a property of the scanner-film pair, not individual frames —
        # if one half was calibrated and the other wasn't (still at default), both need
        # the same correction during export.
        base = f.get("hash", "")
        if "#" in base:
            half_val = int(base.split("#")[-1])
            sibling_hash = half_hash(base.rsplit("#", 1)[0], 3 - half_val)
            sibling_params = self.session.repo.load_file_settings(sibling_hash)
            session_ct = self.state.config.process.crosstalk_strength
            params_ct = params.process.crosstalk_strength
            sibling_ct = sibling_params.process.crosstalk_strength if sibling_params else session_ct
            # If this frame's crosstalk differs from the sibling but is at default (0),
            # inherit the sibling's value so both get identical correction.
            if abs(params_ct) < 1e-9 and abs(sibling_ct) > 1e-9:
                proc = params.process
                params = replace(
                    params,
                    process=replace(
                        proc,
                        crosstalk_strength=sibling_ct,
                        crosstalk_matrix=sibling_params.process.crosstalk_matrix if sibling_params else proc.crosstalk_matrix,
                    ),
                )

        return resolve_asset_hdr(resolve_asset_stitch(resolve_asset_rgbscan(params, f), f), f)

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
            export_settings.icc_input_path = self.effective_input_icc(task_params.process, task_params.exposure.render_intent)
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

    def export_work_print(self, name: str) -> None:
        """Make a named version live, then export it through the normal export path."""
        self.session.load_work_print(name)
        self.request_export()

    def _flush_export_ui(self) -> None:
        """Push pending Export-panel edits into state before any export path reads config."""
        flush = self.flush_export_settings
        if flush is not None:
            flush()

    def request_linear_output_export(self, files: list[dict] | None = None) -> None:
        """Export decoded linear buffers as untagged 16-bit TIFFs to the export folder."""
        from negpy.services.export.linear_output import export_linear_output, is_linear_output_supported

        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        if files is None:
            file_path = self.state.current_file_path
            if not file_path:
                return
            if not is_linear_output_supported(file_path):
                self.set_status("Linear Output is not supported for this file type", 4000)
                return
            # Reuse the asset dict from uploaded_files so RGB-scan triplet (green_path/
            # blue_path) and stitch fields reach _batch_params_for — a bare {path, name,
            # hash} dict makes resolve_asset_rgbscan/resolve_asset_stitch reset those
            # configs, silently exporting only the primary narrowband exposure.
            file_info = next(
                (f for f in self.state.uploaded_files if f.get("hash") == self.state.current_file_hash),
                None,
            )
            if file_info is None:
                file_info = {"path": file_path, "name": os.path.basename(file_path), "hash": self.state.current_file_hash}
            files = [file_info]

        supported = [f for f in files if is_linear_output_supported(f["path"])]
        if not supported:
            self.set_status("No files support Linear Output", 4000)
            return

        if len(supported) > 1 and not self._confirm_bulk_export(f"Linear-export {count_of(len(supported), 'frame')}?"):
            return

        exported = 0
        expansion = self.state.linear_expansion
        linear_fmt = self.state.linear_format
        out_ext = "jxl" if linear_fmt == "jxl" else "tiff"
        for f in supported:
            params = self._batch_params_for(f)
            stitch = params.stitch if params.stitch.stitch_enabled else None
            frames = hdr_frame_paths(f)
            # Same naming rule as a normal export: the bracket's first frame, suffixed so
            # the merge does not write over that frame's own linear output.
            stem = f"{hdr_stem(frames)}-HDR" if frames else os.path.splitext(os.path.basename(f["path"]))[0]
            out_path = os.path.join(export_path, f"{stem}_linear.{out_ext}")
            counter = 2
            while os.path.exists(out_path):
                out_path = os.path.join(export_path, f"{stem}_linear_{counter}.{out_ext}")
                counter += 1
            try:
                export_linear_output(
                    f["path"],
                    out_path,
                    geometry=params.geometry,
                    expansion=expansion,
                    rgbscan=params.rgbscan,
                    stitch=stitch,
                    hdr=params.hdr,
                    flatfield=params.flatfield,
                    process=params.process,
                    apply_wb=self.state.linear_apply_wb,
                    apply_flatfield=self.state.linear_apply_flatfield,
                    apply_sensor=self.state.linear_apply_sensor,
                    apply_ice=self.state.linear_apply_ice,
                    retouch=params.retouch,
                    gamma_key=self.state.linear_gamma_key,
                    output_format=linear_fmt,
                    jxl_effort=self.state.linear_jxl_effort,
                )
                exported += 1
            except Exception as e:
                logger.warning("Linear output failed for %s: %s", f.get("name"), e)
                self.set_status(f"Linear Output failed: {os.path.basename(f['path'])}: {e}", 4000)

        if exported:
            self.set_status(f"Linear Output: exported {count_of(exported, 'file')}", 4000)

    def request_export(self) -> None:
        """Exports the current file using the settings currently shown in the Export panel."""
        self._flush_export_ui()
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
            export_conf = flat_export_config(export_conf)
        source_exif = self.state.source_exif.get(self.state.current_file_hash or "")

        # Reuse the asset dict from uploaded_files so half-frame fields
        # (half, split_x, crop_rect, gutter_thickness) reach the exporter —
        # without them process_export gets half=0 and renders the whole scan
        # instead of the half, dropping the crop and shifting log bounds.
        file_info = next(
            (f for f in self.state.uploaded_files if f.get("hash") == self.state.current_file_hash),
            None,
        )
        if file_info is None:
            file_info = {
                "name": os.path.basename(self.state.current_file_path),
                "path": self.state.current_file_path,
                "hash": self.state.current_file_hash,
            }

        bounds_override = None
        if file_info.get("hash") == self.state.current_file_hash:
            with self.state.metrics_lock:
                bounds_override = self.state.last_metrics.get("log_bounds")

        self._run_export_tasks(
            [
                ExportTask(
                    file_info=file_info,
                    params=params,
                    export_settings=preset_from_export_config(export_conf),
                    gpu_enabled=self.state.gpu_enabled,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=self.state.config.metadata,
                    working_color_space=self.state.workspace_color_space,
                )
            ]
        )

    def request_export_selected(self) -> None:
        """Batch-exports the currently selected files using the current export settings."""
        selected = [self.state.uploaded_files[i] for i in self.state.selected_indices if 0 <= i < len(self.state.uploaded_files)]
        self.request_batch_export(files=[f for f in selected if not f.get("excluded")])

    def request_batch_export(self, files: list[dict] | None = None) -> None:
        """Batch-exports the given files (all visible by default) using the current export settings."""
        self._flush_export_ui()
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

        if len(files) > 1 and not self._confirm_bulk_export(f"Export {count_of(len(files), 'frame')}?"):
            return

        if self.state.config.export.export_sidecars_enabled:
            self._write_edit_sidecars(files)

        flat = self.state.flat_output

        tasks = []
        for f in files:
            # Delivery settings are session-level. A per-file config from the DB
            # bypasses _apply_sticky_settings and carries whatever export block was
            # current when that frame was last saved, so honouring it exports at a
            # size/format the panel never shows (#750).
            params = replace(self._batch_params_for(f), export=current_export)

            final_export = replace(
                params.export,
                icc_input_path=self.effective_input_icc(params.process, params.exposure.render_intent),
                icc_output_path=icc_output,
            )

            if flat:
                params = flat_master_config(params)
                final_export = flat_export_config(final_export)

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
        self._flush_export_ui()
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
            if not self._confirm_bulk_export(
                f"Export {count_of(n_frames, 'frame')} through {count_of(n_presets, 'preset')} ({count_of(n_files, 'file')})?"
            ):
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

        # Reuse the asset dict from uploaded_files so RGB-scan triplet (green_path/
        # blue_path) and stitch fields reach _batch_params_for — a bare {path, name,
        # hash} dict makes resolve_asset_rgbscan/resolve_asset_stitch reset those
        # configs, silently preset-exporting only the primary un-merged exposure.
        file_info = next(
            (f for f in self.state.uploaded_files if f.get("hash") == self.state.current_file_hash),
            None,
        )
        if file_info is None:
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
        self._flush_export_ui()
        if self._batch_busy("contact sheet"):
            return
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return

        out_dir = self._contact_sheet_output_dir(visible_files)
        if not out_dir:
            return

        if len(visible_files) > 1 and not self._confirm_bulk_export(
            f"Render a contact sheet from {count_of(len(visible_files), 'frame')}?"
        ):
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
            Q_ARG(bool, cs.contact_sheet_show_labels),
            Q_ARG(str, cs.contact_sheet_background_color),
            Q_ARG(str, cs.contact_sheet_label_color),
        )

    def _write_edit_sidecars(self, files: list[dict]) -> int:
        """Write a .negpy edit sidecar next to each source (each frame's own saved edits). Returns count written."""
        repo = self.session.repo
        written = 0
        for f in files:
            half = int(f.get("half") or 0)
            params = load_or_promote(
                repo, f["hash"], f["path"], half=half, composite=bool(f.get("hdr_paths") or f.get("stitch_paths"))
            ) or self.session.config_for_asset(f)
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
        self.set_status(f"Wrote {count_of(written, 'edit sidecar')}", 4000)

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
            box.setText(f"{count_of(n, 'file')} already {plural(n, 'exists', 'exist')} in the export destination.")
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

    def _on_render_busy(self, label: str) -> None:
        """Slow uncached render step (IR bake, inpaint) — hold a toast until the frame lands."""
        self._busy_toast = True
        self.set_status(label, _BUSY_TOAST_MS)

    def _clear_busy_toast(self) -> None:
        if self._busy_toast:
            self._busy_toast = False
            self.set_status("")

    def _on_render_finished(self, _result: Any, metrics: Dict[str, Any]) -> None:
        self._is_rendering = False
        self._clear_busy_toast()

        if self._first_render_t0 is not None and not metrics.get("ephemeral"):
            logger.info(
                "load-timing first_render %.0fms (buffer -> painted) %s",
                (time.perf_counter() - self._first_render_t0) * 1000,
                self.state.current_file_path,
            )
            self._first_render_t0 = None

        # Config is replaced wholesale on every edit, so identity detects any change.
        # Not mid-gesture: the filmstrip only has to be right once the drag settles.
        should_update_thumb = (
            self._pending_render_task is None
            and not metrics.get("ephemeral")
            and not metrics.get("interactive")
            and self.state.config is not self._thumb_config
        )

        with self.state.metrics_lock:
            self.state.last_metrics.update(metrics)
            self.state.last_metrics["splash"] = False

        result = metrics.get("base_positive")
        memoizable = bool(metrics.get("memo_key")) and metrics.get("source_hash") == self.state.current_file_hash
        # The pool overwrites a GPU texture on the next frame, so only its identity is
        # kept here; load_file files the texture itself on the way out.
        self._last_render_identity = (
            (metrics["source_hash"], metrics["memo_key"], metrics.get("content_rect"))
            if memoizable and isinstance(result, GPUTexture)
            else None
        )

        if metrics.get("gpu_fallback") and not self._gpu_fallback_notified:
            self._gpu_fallback_notified = True
            self.set_status("GPU acceleration failed — using CPU", 5000)

        self.image_updated.emit()

        # By reference — display buffers are read-only downstream. After the repaint:
        # overwriting an entry frees the texture the canvas has just stopped sampling.
        if memoizable and isinstance(result, np.ndarray):
            self._render_memo.store(
                metrics["source_hash"],
                metrics["memo_key"],
                {"base_positive": result, "content_rect": metrics.get("content_rect")},
            )

        if should_update_thumb:
            self._thumb_config = self.state.config
            # persist=False: refresh in-memory only; disk JPEG written on switch/save/export.
            self._update_thumbnail_from_state(persist=False)

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
        # file (late metric after a fast switch) — they aren't this frame's bounds. Nor
        # from a mid-gesture frame, which measured the proxy rather than the real buffer.
        if metrics.get("ephemeral") or metrics.get("interactive"):
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
                # so the first navigate-back after an initial render still hits. A GPU
                # render is not filed until navigate-away, so its identity follows too.
                self._render_memo.rekey(src or self.state.current_file_hash or "", self._render_memo_key())
                if self._last_render_identity is not None:
                    self._last_render_identity = (
                        self._last_render_identity[0],
                        self._render_memo_key(),
                        self._last_render_identity[2],
                    )

    def _on_render_error(self, message: str) -> None:
        self.state.is_processing = self._is_rendering = False
        self._busy_toast = False  # the failure message below replaces the toast
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
        self._update_thumbnail_from_state()

    def _asset_for_render(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The asset a finished render belongs to — not whichever one is selected now.

        A render carries the hash it was started for, and it can land after the user has
        moved on: select a frame, start its render, click the next one before the decode
        finishes. Keying those pixels by the current selection files one frame's picture
        under another frame's thumbnail, which then shows the wrong image until that frame
        is clicked and re-rendered. The render memo already guards this way.

        Falls back to the selection when the render carries no hash, which is the
        active_file_changing caller — there the outgoing file is still selected.
        """
        source_hash = metrics.get("source_hash")
        if not source_hash:
            return None
        for asset in self.state.uploaded_files:
            if asset.get("hash") == source_hash:
                return asset
        # No fallback to the selected frame. This runs on file switch, on save and after an
        # export as well as from the render itself, so last_metrics can hold a render whose
        # frame has since left the list — a different folder was opened, or a merge swapped
        # frames for a composite. Guessing files that buffer under whatever is selected now,
        # and persists it, so one frame wears another's picture until it is rendered again.
        # A skipped refresh costs nothing; the next render of that frame writes it.
        return None

    def _update_thumbnail_from_state(self, persist: bool = True) -> None:
        if not self.state.current_file_path or not self.state.current_file_hash:
            return
        with self.state.metrics_lock:
            metrics = dict(self.state.last_metrics)
        asset = self._asset_for_render(metrics)
        if asset is None:
            return
        buffer = metrics.get("base_positive")

        # The render worker supplies host pixels; reading back here would put a
        # full-frame copy on the UI thread.
        if isinstance(buffer, GPUTexture):
            buffer = metrics.get("thumbnail_source")

        if buffer is not None and not isinstance(buffer, np.ndarray):
            buffer = metrics.get("analysis_buffer")
        if buffer is None or not isinstance(buffer, np.ndarray):
            return

        # Same transform the canvas used for this buffer, so the filmstrip and the
        # canvas can't disagree about the frame's colour.
        display_cs, monitor_bytes, proof = self.display_transform_params(splash=bool(metrics.get("splash")))
        # The asset's own key, so the batch (source) path re-serves this rendered positive
        # instead of the uninverted source merge it would decode itself.
        self.thumbnail_update_requested.emit(
            ThumbnailUpdateTask(
                file_hash=asset_thumbnail_key(asset),
                buffer=buffer,
                color_space=display_cs,
                monitor_icc_bytes=monitor_bytes,
                proof=proof,
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
        # Memo-owned textures outlive the pool, so they must die before the device.
        self._render_memo.clear()
        self.render_worker.destroy_all()

        # All GPU-touching threads are now joined; release the wgpu device.
        GPUDevice.destroy_singleton()
