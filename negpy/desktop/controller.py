import os
import time
from dataclasses import fields, replace
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QMessageBox

from negpy.desktop.converters import ImageConverter
from negpy.desktop.session import AppState, DesktopSessionManager, ToolMode, resolve_asset_rgbscan
from negpy.desktop.workers.export import ExportTask, ExportWorker
from negpy.desktop.workers.render import (
    AssetDiscoveryTask,
    AssetDiscoveryWorker,
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
from negpy.domain.models import (
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
    WorkspaceConfig,
    export_blocked,
    flat_export_config,
    flat_master_config,
    preset_from_export_config,
    resolve_preset_export,
)
from negpy.services.assets.sidecar import load_or_promote, write_sidecar
from negpy.features.exposure.logic import (
    calculate_wb_shifts,
    calculate_wb_shifts_from_log,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.geometry.logic import apply_fine_rotation, detect_closest_aspect_ratio
from negpy.features.lab.models import LabConfig
from negpy.features.local.models import LocalAdjustmentsConfig
from negpy.features.process.models import ProcessMode, invalidate_local_bounds
from negpy.features.retouch.logic import fallback_source_offset, select_source_offset
from negpy.features.retouch.models import RetouchConfig
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
    analysis_buffer_preview_requested = pyqtSignal(float)
    rotation_guide_requested = pyqtSignal()
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
    tone_drag_changed = pyqtSignal(str)  # exposure field being slider-dragged; "" = drag ended
    scan_devices_requested = pyqtSignal()
    scan_requested = pyqtSignal(ScanRequest)
    scan_devices_ready = pyqtSignal(list)
    scan_progress = pyqtSignal(float)
    scan_finished = pyqtSignal(str)
    scan_error = pyqtSignal(str)
    scan_started = pyqtSignal()

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
        self._gpu_fallback_notified = False
        self._cleaned_up = False
        self._active_batch: Optional[str] = None

        self.preview_service = PreviewManager()
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

        self.canvas: Any = None
        self._is_rendering = False
        self._pending_render_task: Any = None

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
        self.export_worker.cancelled.connect(self._on_batch_cancelled)
        self.export_worker.error.connect(self._on_render_error)
        self.export_worker.error.connect(self._on_export_task_error)

        self.thumbnail_requested.connect(self.thumb_worker.generate)
        self.thumb_worker.progress.connect(self._on_thumbnail_progress)
        self.thumbnail_update_requested.connect(self.thumb_worker.update_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_worker.error.connect(self._on_render_error)

        self.normalization_requested.connect(self.norm_worker.process)
        self.norm_worker.progress.connect(self._on_normalization_progress)
        self.norm_worker.finished.connect(self._on_normalization_finished)
        self.norm_worker.cancelled.connect(self._on_batch_cancelled)
        self.norm_worker.error.connect(self._on_render_error)
        self.norm_worker.error.connect(self._on_batch_error)

        self.asset_discovery_requested.connect(self.discovery_worker.process)
        self.discovery_worker.progress.connect(self._on_discovery_progress)
        self.discovery_worker.finished.connect(self._on_discovery_finished)
        self.discovery_worker.error.connect(self._on_render_error)

        self.preview_load_requested.connect(self.preview_load_worker.process)
        self.preview_load_worker.splash.connect(self._on_splash_preview)
        self.preview_load_worker.finished.connect(self._on_preview_loaded)
        self.preview_load_worker.error.connect(self._on_render_error)

        self.scan_devices_requested.connect(self.scan_worker.list_devices)
        self.scan_worker.devices_ready.connect(self.scan_devices_ready.emit)
        self.scan_worker.progress.connect(self.scan_progress.emit)
        self.scan_worker.finished.connect(self._on_scan_finished)
        self.scan_worker.error.connect(self.scan_error.emit)
        self.scan_requested.connect(self.scan_worker.run_scan)

        self.session.active_file_changing.connect(lambda: self._update_thumbnail_from_state(force_readback=True))
        self.session.file_selected.connect(self.load_file)
        self.session.state_changed.connect(self.config_updated.emit)
        self.session.state_changed.connect(self._render_debounce.start)
        self.session.files_changed.connect(self._render_debounce.start)

    def generate_missing_thumbnails(self) -> None:
        missing = [f for f in self.state.uploaded_files if f["name"] not in self.state.thumbnails]
        if missing:
            self.set_status("GENERATING THUMBNAILS...")
            self._begin_batch("Generating thumbnails", abortable=False)
            self.thumbnail_requested.emit(missing)

    def _on_thumbnail_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"THUMBNAIL {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _on_thumbnails_finished(self, new_thumbs: Dict[str, Any]) -> None:
        self.set_status("GALLERIES UPDATED", 3000)
        self.status_progress_requested.emit(0, 0)
        self._end_batch()
        for name, pil_img in new_thumbs.items():
            if pil_img:
                u8_arr = np.array(pil_img.convert("RGB"))
                self.state.thumbnails[name] = QIcon(QPixmap.fromImage(ImageConverter.to_qimage(u8_arr)))
        self.session.asset_model.refresh()

    # --- Batch progress popup -------------------------------------------------

    def _begin_batch(self, title: str, abortable: bool) -> None:
        self._active_batch = title if abortable else None
        self.batch_started.emit(title, abortable)

    def _end_batch(self) -> None:
        self._active_batch = None
        self.batch_finished.emit()

    def _on_batch_progress(self, current: int, total: int, name: str) -> None:
        self.batch_progress.emit(current, total, name)

    def _on_batch_cancelled(self) -> None:
        self.set_status("Aborted", 3000)
        self._end_batch()

    def _on_batch_error(self, _message: str) -> None:
        self._end_batch()

    def abort_active_batch(self) -> None:
        """Requests cancellation of the running abortable batch (export or analysis)."""
        if self._active_batch in ("Exporting", "Contact sheet"):
            self.export_worker.cancel()
        elif self._active_batch == "Analyzing roll":
            self.norm_worker.cancel()

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
        Silently skips if a discovery task is already in progress.

        `replace_existing` rebuilds the asset list from the results (instead of
        appending) and reselects `reselect_path` — used when re-running discovery
        over already-loaded files (e.g. an RGB-scan mode toggle).
        """
        if self._discovery_running:
            return

        from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS

        self._discovery_running = True
        self._auto_open_after_discovery = auto_open
        self._replace_after_discovery = replace_existing
        self._reselect_after_discovery = reselect_path
        self.set_status("SCANNING FOR ASSETS...")
        self._begin_batch("Hashing files", abortable=False)
        rgb_scan = bool(self.session.repo.get_global_setting("rgbscan_mode", False))
        task = AssetDiscoveryTask(
            paths=paths,
            supported_extensions=tuple(SUPPORTED_RAW_EXTENSIONS),
            rgb_scan=rgb_scan,
            restore_triplets=restore_triplets,
        )
        self.asset_discovery_requested.emit(task)

    def set_rgb_scan_mode(self, enabled: bool) -> None:
        """Persist the RGB-scan toggle and re-discover already-loaded assets so the
        mode regroups/ungroups triplets in place (not only on the next folder load)."""
        self.session.repo.save_global_setting("rgbscan_mode", bool(enabled))
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
        self._end_batch()
        self.status_progress_requested.emit(0, 0)
        self._discovery_running = False
        auto_open = self._auto_open_after_discovery
        self._auto_open_after_discovery = False
        replace_existing = self._replace_after_discovery
        reselect_path = self._reselect_after_discovery
        self._replace_after_discovery = False
        self._reselect_after_discovery = None
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
            return

        if valid_assets:
            first_new_idx = len(self.session.state.uploaded_files)
            self.session.add_files([], validated_info=valid_assets)
            self.generate_missing_thumbnails()
            if pending_scan:
                self._select_file_by_path(pending_scan)
                self._pending_scanned_file = None
            elif auto_open and not self.state.current_file_path and len(self.session.state.uploaded_files) > first_new_idx:
                self.session.select_file(first_new_idx)
        else:
            self.set_status("NO SUPPORTED ASSETS FOUND", 3000)
            self.status_progress_requested.emit(0, 0)

    def _file_hash_for_path(self, file_path: str) -> Optional[str]:
        if self.state.current_file_path == file_path and self.state.current_file_hash:
            return self.state.current_file_hash
        for f in self.state.uploaded_files:
            if f.get("path") == file_path:
                return f.get("hash")
        return None

    def load_file(self, file_path: str, preserve_zoom: bool = False, force_detect: bool = False) -> None:
        """
        Dispatches RAW decode to a background worker to keep the UI thread free.
        """
        self._prefetch_gen += 1
        self._preview_load_t0 = time.perf_counter()
        self._requested_file_path = file_path
        if not preserve_zoom:
            self.zoom_requested.emit(1.0)
        self.set_status(f"Loading {os.path.basename(file_path)}...")
        self.loading_started.emit()
        self._thumb_config = None

        self._render_cleanup_requested.emit()

        self.state.preview_raw = None
        self.state.preview_ir = None
        self.state.has_ir = False
        self.state.original_res = (0, 0)

        rgbscan = self.state.config.rgbscan
        self.preview_load_requested.emit(
            PreviewLoadTask(
                file_path=file_path,
                workspace_color_space=self.state.workspace_color_space,
                use_camera_wb=not self.state.config.process.linear_raw,
                full_resolution=self.state.hq_preview,
                file_hash=self._file_hash_for_path(file_path),
                detect_mode=force_detect or (self.state.autodetect_enabled and self.state.current_file_is_new),
                green_path=rgbscan.green_path if rgbscan.enabled else "",
                blue_path=rgbscan.blue_path if rgbscan.enabled else "",
                align=rgbscan.align,
            )
        )

    def _on_splash_preview(self, file_path: str, raw: Any, dims: Any) -> None:
        if self._requested_file_path != file_path:
            return
        self.state.original_res = dims
        # Paint the embedded sRGB thumbnail directly — no pipeline; the real render replaces it.
        with self.state.metrics_lock:
            self.state.last_metrics["base_positive"] = raw
            self.state.last_metrics["splash"] = True
        self.image_updated.emit()

    def _on_preview_loaded(self, file_path: str, raw: Any, dims: Any, source_cs: str, ir_preview: Any, detected_mode: str) -> None:
        if self._requested_file_path != file_path:
            return
        logger.info(
            "load-timing preview_e2e %.0fms (load request -> decoded buffer) %s",
            (time.perf_counter() - self._preview_load_t0) * 1000,
            file_path,
        )
        self.state.preview_raw = raw
        self.state.preview_ir = ir_preview
        self.state.has_ir = ir_preview is not None
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
                        file_hash=h,
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
            self.request_render()

    def cancel_active_tool(self) -> None:
        if self.state.active_tool != ToolMode.NONE:
            self.set_active_tool(ToolMode.NONE)

    def show_rotation_guide(self) -> None:
        """Request the canvas show the fine-rotation alignment grid."""
        self.rotation_guide_requested.emit()

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

    def confirm_manual_crop(self) -> None:
        """Close the crop tool (committing the current rect) — invoked by a double-click
        inside the crop box so the user needn't return to the Crop button."""
        if self.state.active_tool == ToolMode.CROP_MANUAL:
            self.set_active_tool(ToolMode.NONE)

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
        self.request_render()

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

        new_ratio = detect_closest_aspect_ratio(transformed, fallback=geom.autocrop_ratio)
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
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=[], manual_heal_strokes=[]),
            )
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
            )
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
        # diameter at preview_render_size scale (same convention as the pipeline
        # radius size/2·scale_factor and the overlay cursor).
        offset = (0.0, 0.0)
        preview = self.state.preview_raw
        if preview is not None:
            scale = max(preview.shape[:2]) / float(APP_CONFIG.preview_render_size)
            offset = select_source_offset(preview, raw_pts, 0.5 * size * scale, index)
        else:
            offset = fallback_source_offset(index, size, (self.state.original_res[1], self.state.original_res[0]))

        stroke = ([[rx, ry] for rx, ry in raw_pts], size, float(offset[0]), float(offset[1]))
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_heal_strokes=conf.manual_heal_strokes + [stroke]),
            )
        )
        self.request_render()

    def handle_lasso_completed(self, viewport_vertices: list) -> None:
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None or len(viewport_vertices) < 3:
            return

        raw_vertices = tuple(CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_vertices)

        from negpy.features.local.models import PolygonMask

        mask = PolygonMask(vertices=raw_vertices, strength=0.3, feather=0.02)
        local = self.state.config.local
        new_masks = local.masks + (mask,)
        new_local = replace(local, masks=new_masks)
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.state.local_selected_mask = len(new_masks) - 1
        self.set_active_tool(ToolMode.NONE)  # auto-exit draw mode once the polygon closes
        self.config_updated.emit()
        self.request_render()

    def select_local_mask(self, index: int) -> None:
        self.state.local_selected_mask = index
        self.config_updated.emit()

    def delete_selected_local_mask(self) -> None:
        local = self.state.config.local
        idx = self.state.local_selected_mask
        if not (0 <= idx < len(local.masks)):
            return
        new_masks = local.masks[:idx] + local.masks[idx + 1 :]
        new_local = replace(local, masks=new_masks)
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.state.local_selected_mask = -1
        self.config_updated.emit()
        self.request_render()

    def clear_local(self) -> None:
        new_local = replace(self.state.config.local, masks=())
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.state.local_selected_mask = -1
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
        if is_log:
            # CPU stores "final_bounds", GPU stores "log_bounds".
            bounds = metrics.get("final_bounds") or metrics.get("log_bounds")
            new_m, new_y = calculate_wb_shifts_from_log(sampled[:3], bounds)
        else:
            delta_m, delta_y = calculate_wb_shifts(sampled[:3])
            damping = 0.4
            new_m = exp.wb_magenta + delta_m * damping
            new_y = exp.wb_yellow + delta_y * damping

        new_exp = replace(
            exp,
            wb_cyan=0.0,
            wb_magenta=float(np.clip(new_m, -1.0, 1.0)),
            wb_yellow=float(np.clip(new_y, -1.0, 1.0)),
        )
        self.session.update_config(replace(self.state.config, exposure=new_exp), persist=True, record_history=True)
        self.request_render()

    def request_batch_normalization(self) -> None:
        """
        Initiates background analysis for batch normalization.
        """
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
            crop_note = (
                "Crop status: none of the files have a crop set. Analysis samples the "
                "full frame, so any border or letterboxing around the negative will skew "
                "the average. For best results, crop each file to the negative area — or, "
                "if that's not practical, raise the Analysis Buffer to exclude the margin."
            )
        elif cropped < total:
            crop_note = (
                f"Crop status: {cropped} of {total} files have a crop set. The uncropped "
                "ones are analyzed on the full frame, so their borders may skew the "
                "average. Crop them, or raise the Analysis Buffer to exclude the margin."
            )
        else:
            crop_note = (
                "Crop status: all files are cropped — analysis runs on the negative area, "
                "ignoring borders. The Analysis Buffer still trims a margin inside the crop."
            )

        reply = QMessageBox.question(
            None,
            "Batch Analysis",
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
            f"{crop_note}\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.set_status("Starting Batch Normalization...")
        self._begin_batch("Analyzing roll", abortable=True)
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
        self._end_batch()
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
            self.session.repo.save_file_settings(f_info["hash"], new_p)

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
                self.session.repo.save_file_settings(f_info["hash"], new_p)

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

    def _select_file_by_path(self, path: str) -> None:
        """Find a file by path in uploaded_files and select it."""
        for i, f_info in enumerate(self.session.state.uploaded_files):
            if f_info.get("path") == path:
                self.session.select_file(i)
                return

    def effective_output_icc(self) -> Optional[str]:
        """Output profile the preview proofs through: a custom override, else the
        profile for the selected export color space. None means no proof (Same as Source)."""
        return self.state.icc_output_path or ColorSpaceRegistry.get_icc_path(self.state.config.export.export_color_space)

    def proof_active(self) -> bool:
        """True when the preview should soft-proof: the toggle is on and an input or
        output profile is available. Off → preview is the edit on the monitor."""
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

        self.set_status("Rendering...")

        preview_raw = self.state.preview_raw
        if preview_raw is None:
            return

        target_size = float(APP_CONFIG.preview_render_size)
        if self.state.hq_preview:
            target_size = float(max(preview_raw.shape[:2]))

        # Soft-proof gating: Output/Input ICC only touch the preview when the toggle is
        # on; otherwise the preview is the edit shown on the monitor (export unaffected).
        proofing = self.state.soft_proof_enabled
        icc_input = self.state.icc_input_path if proofing else None
        effective_output = self.effective_output_icc() if proofing else None

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
            crop_preview_full=self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW),
            ephemeral=ephemeral,
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

    def set_local_overlay_visible(self, visible: bool) -> None:
        """Show/hide the dodge/burn mask overlay (view-only; no re-render)."""
        self.state.show_local_overlay = visible
        if self.canvas:
            self.canvas.overlay.update()

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
            export_settings.icc_input_path = self.state.icc_input_path
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
        if not self.state.current_file_path:
            return

        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        export_conf = replace(
            self.state.config.export,
            export_path=export_path,
            icc_input_path=self.state.icc_input_path,
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
        self.request_batch_export(files=selected)

    def request_batch_export(self, override_settings: bool = False, files: list[dict] | None = None) -> None:
        """Batch-exports the given files (all visible by default) using current settings, optionally applied to all."""
        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        current_export = replace(self.state.config.export, export_path=export_path)
        icc_input = self.state.icc_input_path
        icc_output = self.state.icc_output_path
        sync_metadata = self.state.config.metadata.sync_to_batch

        if files is None:
            files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]

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

            final_export = replace(
                params.export,
                icc_input_path=icc_input,
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
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _dispatch_preset_export(self, files: list[dict]) -> None:
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
        self._begin_batch("Contact sheet", abortable=True)
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
            params = load_or_promote(repo, f["hash"], f["path"]) or self.state.config
            try:
                write_sidecar(f["path"], params)
                written += 1
            except Exception as exc:
                logger.warning("Sidecar write failed for %s: %s", f.get("path"), exc)
        return written

    def export_edit_sidecars(self) -> None:
        """Explicit batch sidecar export for all visible files (ignores the on-export toggle)."""
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return
        written = self._write_edit_sidecars(visible_files)
        self.set_status(f"Wrote {written} edit sidecar(s)", 4000)

    def _run_export_tasks(self, tasks: List[ExportTask]) -> None:
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

        self._export_start_time = time.time()
        self._export_failures = 0
        self._begin_batch("Exporting", abortable=True)
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_batch",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
        )

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

        if metrics.get("gpu_fallback") and not self._gpu_fallback_notified:
            self._gpu_fallback_notified = True
            self.set_status("GPU acceleration failed — using CPU", 5000)
        else:
            self.set_status("READY", 1000)

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
        self._end_batch()
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
        self.render_worker.destroy_all()

        # All GPU-touching threads are now joined; release the wgpu device.
        GPUDevice.destroy_singleton()
