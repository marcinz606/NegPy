import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QMessageBox

from negpy.desktop.converters import ImageConverter
from negpy.desktop.session import AppState, DesktopSessionManager, ToolMode
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
    ExportPreset,
    ExportPresetOutputMode,
    WorkspaceConfig,
    preset_from_export_config,
)
from negpy.features.exposure.logic import (
    calculate_wb_shifts,
    calculate_wb_shifts_from_log,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.geometry.logic import apply_fine_rotation, detect_closest_aspect_ratio
from negpy.features.lab.models import LabConfig
from negpy.features.process.models import ProcessMode, invalidate_local_bounds
from negpy.features.retouch.models import RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
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
        toning=ToningConfig(),
        finish=FinishConfig(),
        retouch=RetouchConfig(),
    )


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
    export_finished = pyqtSignal(float)
    render_requested = pyqtSignal(RenderTask)
    preview_load_requested = pyqtSignal(PreviewLoadTask)
    normalization_requested = pyqtSignal(NormalizationTask)
    analysis_buffer_preview_requested = pyqtSignal(float)
    asset_discovery_requested = pyqtSignal(AssetDiscoveryTask)
    thumbnail_requested = pyqtSignal(list)
    thumbnail_update_requested = pyqtSignal(ThumbnailUpdateTask)
    tool_sync_requested = pyqtSignal()
    config_updated = pyqtSignal()
    monitor_profile_changed = pyqtSignal()
    compare_changed = pyqtSignal(bool)
    zoom_requested = pyqtSignal(float)
    zoom_changed = pyqtSignal(float)
    _render_cleanup_requested = pyqtSignal()
    status_message_requested = pyqtSignal(str, int)
    status_progress_requested = pyqtSignal(int, int)
    pixel_readout = pyqtSignal(str, str)
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
        self._first_render_done = False
        self._export_start_time = 0.0
        self._discovery_running = False
        self._auto_open_after_discovery = False
        self._gpu_fallback_notified = False
        self._cleaned_up = False

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
        self.pixel_readout.emit("", "")

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
        import cv2

        rgb_u8 = np.array([[[r255, g255, b255]]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)[0, 0]
        L = int(round(int(lab[0]) * 100 / 255))
        a = int(lab[1]) - 128
        b_val = int(lab[2]) - 128
        self.pixel_readout.emit(
            f"RGB {r255:>3} {g255:>3} {b255:>3}",
            f"Lab {L:>3} {a:+4d} {b_val:+4d}",
        )

    def set_status(self, message: str, timeout: int = 0) -> None:
        self.status_message_requested.emit(message, timeout)

    def _connect_signals(self) -> None:
        self.render_requested.connect(self.render_worker.process)
        self._render_cleanup_requested.connect(self.render_worker.cleanup)
        self.render_worker.finished.connect(self._on_render_finished)
        self.render_worker.metrics_updated.connect(self._on_metrics_updated)
        self.render_worker.error.connect(self._on_render_error)

        self.export_worker.progress.connect(self.export_progress.emit)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.error.connect(self._on_render_error)

        self.thumbnail_requested.connect(self.thumb_worker.generate)
        self.thumb_worker.progress.connect(self._on_thumbnail_progress)
        self.thumbnail_update_requested.connect(self.thumb_worker.update_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_worker.error.connect(self._on_render_error)

        self.normalization_requested.connect(self.norm_worker.process)
        self.norm_worker.progress.connect(self._on_normalization_progress)
        self.norm_worker.finished.connect(self._on_normalization_finished)
        self.norm_worker.error.connect(self._on_render_error)

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

        self.session.file_selected.connect(self.load_file)
        self.session.state_changed.connect(self.config_updated.emit)
        self.session.state_changed.connect(self._render_debounce.start)
        self.session.files_changed.connect(self._render_debounce.start)

    def generate_missing_thumbnails(self) -> None:
        missing = [f for f in self.state.uploaded_files if f["name"] not in self.state.thumbnails]
        if missing:
            self.set_status("GENERATING THUMBNAILS...")
            self.thumbnail_requested.emit(missing)

    def _on_thumbnail_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"THUMBNAIL {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)

    def _on_thumbnails_finished(self, new_thumbs: Dict[str, Any]) -> None:
        self.set_status("GALLERIES UPDATED", 3000)
        self.status_progress_requested.emit(0, 0)
        for name, pil_img in new_thumbs.items():
            if pil_img:
                u8_arr = np.array(pil_img.convert("RGB"))
                self.state.thumbnails[name] = QIcon(QPixmap.fromImage(ImageConverter.to_qimage(u8_arr)))
        self.session.asset_model.refresh()

    def request_asset_discovery(self, paths: List[str], auto_open: bool = False) -> None:
        """
        Starts asynchronous discovery of supported assets.
        Silently skips if a discovery task is already in progress.
        """
        if self._discovery_running:
            return

        from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS

        self._discovery_running = True
        self._auto_open_after_discovery = auto_open
        self.set_status("SCANNING FOR ASSETS...")
        task = AssetDiscoveryTask(paths=paths, supported_extensions=tuple(SUPPORTED_RAW_EXTENSIONS))
        self.asset_discovery_requested.emit(task)

    def _on_discovery_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"HASHING {current}/{total}: {name}")

    def _on_discovery_finished(self, valid_assets: List[Dict]) -> None:
        """
        Adds discovered assets to the session and starts thumbnail generation.
        """
        self._discovery_running = False
        auto_open = self._auto_open_after_discovery
        self._auto_open_after_discovery = False
        pending_scan = getattr(self, "_pending_scanned_file", None)

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
        self._first_render_done = False

        self._render_cleanup_requested.emit()

        self.state.preview_raw = None
        self.state.preview_ir = None
        self.state.has_ir = False
        self.state.original_res = (0, 0)

        self.preview_load_requested.emit(
            PreviewLoadTask(
                file_path=file_path,
                workspace_color_space=self.state.workspace_color_space,
                use_camera_wb=not self.state.config.exposure.linear_raw,
                full_resolution=self.state.hq_preview,
                file_hash=self._file_hash_for_path(file_path),
                detect_mode=force_detect or (self.state.autodetect_enabled and self.state.current_file_is_new),
            )
        )

    def _on_splash_preview(self, file_path: str, raw: Any, dims: Any) -> None:
        if self._requested_file_path != file_path:
            return
        self.state.preview_raw = raw
        self.state.original_res = dims
        self.request_render()

    def _on_preview_loaded(self, file_path: str, raw: Any, dims: Any, source_cs: str, ir_preview: Any, detected_mode: str) -> None:
        logger.debug(
            "preview e2e (load request to decoded buffer) %.3fs for %s",
            time.perf_counter() - self._preview_load_t0,
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
                linear_raw = saved.exposure.linear_raw if saved else False
                self.preview_load_requested.emit(
                    PreviewLoadTask(
                        file_path=path,
                        workspace_color_space=self.state.workspace_color_space,
                        use_camera_wb=not linear_raw,
                        full_resolution=self.state.hq_preview,
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
        self.state.active_tool = mode
        self.tool_sync_requested.emit()

    def cancel_active_tool(self) -> None:
        if self.state.active_tool != ToolMode.NONE:
            self.set_active_tool(ToolMode.NONE)

    def handle_crop_completed(self, nx1: float, ny1: float, nx2: float, ny2: float) -> None:
        if self.state.active_tool != ToolMode.CROP_MANUAL:
            return
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return

        rx1, ry1 = CoordinateMapping.map_click_to_raw(nx1, ny1, uv_grid)
        rx2, ry2 = CoordinateMapping.map_click_to_raw(nx2, ny2, uv_grid)

        new_geo = replace(
            self.state.config.geometry,
            manual_crop_rect=(
                min(rx1, rx2),
                min(ry1, ry2),
                max(rx1, rx2),
                max(ry1, ry2),
            ),
            auto_crop_enabled=False,
        )
        new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
        self.session.update_config(replace(self.state.config, geometry=new_geo, process=new_proc))
        self.state.active_tool = ToolMode.NONE
        self.tool_sync_requested.emit()
        self.request_render()

    def handle_crop_translated(self, nx1: float, ny1: float, nx2: float, ny2: float) -> None:
        if self.state.config.geometry.manual_crop_rect is None:
            return
        new_geo = replace(self.state.config.geometry, manual_crop_rect=(nx1, ny1, nx2, ny2))
        self.session.update_config(replace(self.state.config, geometry=new_geo))
        self.request_render()

    def reset_crop(self) -> None:
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
                retouch=replace(self.state.config.retouch, manual_dust_spots=[]),
            )
        )
        self.request_render()

    def undo_last_retouch(self) -> None:
        """
        Removes the most recently added dust spot.
        """
        spots = list(self.state.config.retouch.manual_dust_spots)
        if spots:
            spots.pop()
            self.session.update_config(
                replace(
                    self.state.config,
                    retouch=replace(self.state.config.retouch, manual_dust_spots=spots),
                )
            )
            self.request_render()

    def _handle_dust_pick(self, nx: float, ny: float) -> None:
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)
        new_spots = self.state.config.retouch.manual_dust_spots + [(rx, ry, float(self.state.config.retouch.manual_dust_size))]
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=new_spots),
            )
        )
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
            new_m, new_y = calculate_wb_shifts_from_log(sampled[:3])
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
            "  • D-Range Clip — how aggressively the highlight/shadow tails are "
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
        task = NormalizationTask(
            files=visible_files,
            workspace_color_space=self.state.workspace_color_space,
            override_analysis_buffer=self.state.config.process.analysis_buffer,
            override_drange_clip=self.state.config.process.drange_clip,
        )
        self.normalization_requested.emit(task)

    def _on_normalization_progress(self, current: int, total: int, name: str, has_crop: bool) -> None:
        """
        Updates UI status during batch analysis.
        """
        marker = "cropped" if has_crop else "full frame"
        self.set_status(f"Analyzing {current}/{total}: {name} [{marker}]...")
        self.status_progress_requested.emit(current, total)

    def _on_normalization_finished(self, locked_floors: tuple, locked_ceils: tuple) -> None:
        """
        Applies averaged normalization baseline to all files.
        """
        for f_info in self.state.uploaded_files:
            p = self.session.repo.load_file_settings(f_info["hash"]) or replace(self.state.config)
            new_process = replace(
                p.process,
                use_roll_average=True,
                locked_floors=locked_floors,
                locked_ceils=locked_ceils,
                roll_name=None,
            )
            new_p = replace(p, process=new_process)
            self.session.repo.save_file_settings(f_info["hash"], new_p)

        # Update current state
        new_process = replace(
            self.state.config.process,
            use_roll_average=True,
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
                    use_roll_average=True,
                    locked_floors=locked_floors,
                    locked_ceils=locked_ceils,
                    roll_name=name,
                )
                new_p = replace(p, process=new_process)
                self.session.repo.save_file_settings(f_info["hash"], new_p)

            new_process = replace(
                self.state.config.process,
                use_roll_average=True,
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
        path = self.session.repo.get_flatfield_profile(name) if name else None
        new_ff = replace(self.state.config.flatfield, reference_path=path or "", enabled=bool(path))
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

    def set_flatfield_enabled(self, enabled: bool) -> None:
        """
        Per-image toggle to enable/disable flat-field correction for the current frame.
        """
        new_ff = replace(self.state.config.flatfield, enabled=enabled)
        self.session.update_config(replace(self.state.config, flatfield=new_ff), persist=True)
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

    def request_render(self, readback_metrics: bool = True, config_override: Optional[WorkspaceConfig] = None) -> None:
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
            from copy import copy

            p = copy(preset)
            p.icc_input_path = self.state.icc_input_path
            tasks.append(
                ExportTask(
                    file_info=file_info,
                    params=params,
                    export_settings=p,
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
        source_exif = self.state.source_exif.get(self.state.current_file_hash or "")

        self._run_export_tasks(
            [
                ExportTask(
                    file_info={
                        "name": os.path.basename(self.state.current_file_path),
                        "path": self.state.current_file_path,
                        "hash": self.state.current_file_hash,
                    },
                    params=self.state.config,
                    export_settings=preset_from_export_config(export_conf),
                    gpu_enabled=self.state.gpu_enabled,
                    source_exif=source_exif,
                    metadata_config=self.state.config.metadata,
                    working_color_space=self.state.workspace_color_space,
                )
            ]
        )

    def request_batch_export(self, override_settings: bool = False) -> None:
        """Batch-exports all visible files using current settings, optionally applied to all."""
        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        current_export = replace(self.state.config.export, export_path=export_path)
        icc_input = self.state.icc_input_path
        icc_output = self.state.icc_output_path
        sync_metadata = self.state.config.metadata.sync_to_batch

        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]

        tasks = []
        for f in visible_files:
            params = self.session.repo.load_file_settings(f["hash"]) or self.state.config

            if override_settings:
                params = replace(params, export=current_export)

            final_export = replace(
                params.export,
                icc_input_path=icc_input,
                icc_output_path=icc_output,
            )

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

    def request_preset_export(self) -> None:
        """Initiates high-resolution export for the current file using enabled presets."""
        if not self.state.current_file_path:
            return

        presets = self._enabled_presets()
        if not presets:
            QMessageBox.information(None, "No presets enabled", "Enable at least one export preset in the Export panel.")
            return

        if not self._validate_preset_paths(presets):
            return

        source_exif = self.state.source_exif.get(self.state.current_file_hash or "")
        file_info = {
            "name": os.path.basename(self.state.current_file_path),
            "path": self.state.current_file_path,
            "hash": self.state.current_file_hash,
        }
        tasks = self._tasks_for_file(
            file_info,
            self.state.config,
            presets,
            source_exif=source_exif,
            metadata_config=self.state.config.metadata,
        )
        if tasks:
            self._run_export_tasks(tasks)

    def request_preset_batch_export(self) -> None:
        """Initiates batch export for all visible files using enabled presets."""
        presets = self._enabled_presets()
        if not presets:
            QMessageBox.information(None, "No presets enabled", "Enable at least one export preset in the Export panel.")
            return

        if not self._validate_preset_paths(presets):
            return

        sync_metadata = self.state.config.metadata.sync_to_batch
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]

        tasks = []
        for f in visible_files:
            params = self.session.repo.load_file_settings(f["hash"]) or self.state.config

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

        if tasks:
            self._run_export_tasks(tasks)

    def request_contact_sheet(self) -> None:
        """Renders all visible files small and writes darkroom contact sheet(s)."""
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return

        if self.state.config.export.output_mode == ExportPresetOutputMode.SAME_AS_SOURCE:
            out_dir = os.path.dirname(visible_files[0]["path"])
        else:
            resolved = self._ensure_valid_export_path()
            if not resolved:
                return
            out_dir = resolved

        tasks = []
        for f in visible_files:
            params = self.session.repo.load_file_settings(f["hash"]) or self.state.config
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

    def _run_export_tasks(self, tasks: List[ExportTask]) -> None:
        self._export_start_time = time.time()
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_batch",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
        )

    def _on_render_finished(self, _result: Any, metrics: Dict[str, Any]) -> None:
        self._is_rendering = False

        # Snapshot the thumbnail once the render has converged (no newer render queued),
        # so we don't capture a premature/unconverted frame.
        should_update_thumb = not self._first_render_done and self._pending_render_task is None

        with self.state.metrics_lock:
            self.state.last_metrics.update(metrics)

        if metrics.get("gpu_fallback") and not self._gpu_fallback_notified:
            self._gpu_fallback_notified = True
            self.set_status("GPU acceleration failed — using CPU", 5000)
        else:
            self.set_status("READY", 1000)

        self.image_updated.emit()

        if should_update_thumb:
            self._first_render_done = True
            self._update_thumbnail_from_state(force_readback=True)

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

        # If render produced fresh log bounds, persist them locally
        if "log_bounds" in metrics and not self.state.config.process.use_roll_average:
            bounds = metrics.get("log_bounds")

            changes = {}
            if bounds and not self.state.config.process.lock_bounds:
                current = self.state.config.process
                if bounds.floors != current.local_floors or bounds.ceils != current.local_ceils:
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

    def _on_export_finished(self) -> None:
        elapsed = time.time() - self._export_start_time
        self.export_finished.emit(elapsed)
        self._update_thumbnail_from_state(force_readback=True)

    def _update_thumbnail_from_state(self, force_readback: bool = False) -> None:
        if not self.state.current_file_path or not self.state.current_file_hash:
            return
        with self.state.metrics_lock:
            metrics = dict(self.state.last_metrics)
        buffer = metrics.get("base_positive")

        if isinstance(buffer, GPUTexture):
            buffer = buffer.readback()

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
