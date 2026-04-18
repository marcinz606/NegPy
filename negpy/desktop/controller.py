import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap

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
from negpy.features.exposure.logic import (
    calculate_wb_shifts,
    calculate_wb_shifts_from_log,
)
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)


class AppController(QObject):
    """
    Main application orchestrator.
    Manages UI state synchronization, background workers, and render flow.
    """

    image_updated = pyqtSignal()
    metrics_available = pyqtSignal(dict)
    loading_started = pyqtSignal()
    export_progress = pyqtSignal(int, int, str)
    export_finished = pyqtSignal(float)
    render_requested = pyqtSignal(RenderTask)
    preview_load_requested = pyqtSignal(PreviewLoadTask)
    normalization_requested = pyqtSignal(NormalizationTask)
    asset_discovery_requested = pyqtSignal(AssetDiscoveryTask)
    thumbnail_requested = pyqtSignal(list)
    thumbnail_update_requested = pyqtSignal(ThumbnailUpdateTask)
    tool_sync_requested = pyqtSignal()
    config_updated = pyqtSignal()
    zoom_requested = pyqtSignal(float)
    zoom_changed = pyqtSignal(float)
    status_message_requested = pyqtSignal(str, int)
    status_progress_requested = pyqtSignal(int, int)
    pixel_readout = pyqtSignal(str, str)

    def __init__(self, session_manager: DesktopSessionManager):
        super().__init__()
        self.session = session_manager
        self.state: AppState = session_manager.state
        self._first_render_done = False
        self._export_start_time = 0.0
        self._discovery_running = False
        self._gpu_fallback_notified = False

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
        self.preview_load_worker.finished.connect(self._on_preview_loaded)
        self.preview_load_worker.error.connect(self._on_render_error)

        self.session.file_selected.connect(self.load_file)
        self.session.state_changed.connect(self.config_updated.emit)
        self.session.state_changed.connect(self._render_debounce.start)

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

    def request_asset_discovery(self, paths: List[str]) -> None:
        """
        Starts asynchronous discovery of supported assets.
        Silently skips if a discovery task is already in progress.
        """
        if self._discovery_running:
            return

        from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS

        self._discovery_running = True
        self.set_status("SCANNING FOR ASSETS...")
        task = AssetDiscoveryTask(paths=paths, supported_extensions=tuple(SUPPORTED_RAW_EXTENSIONS))
        self.asset_discovery_requested.emit(task)

    def _on_discovery_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"HASHING {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)

    def _on_discovery_finished(self, valid_assets: List[Dict]) -> None:
        """
        Adds discovered assets to the session and starts thumbnail generation.
        """
        self._discovery_running = False
        if valid_assets:
            self.session.add_files([], validated_info=valid_assets)
            self.generate_missing_thumbnails()
        else:
            self.set_status("NO SUPPORTED ASSETS FOUND", 3000)
            self.status_progress_requested.emit(0, 0)

    def load_file(self, file_path: str, preserve_zoom: bool = False) -> None:
        """
        Dispatches RAW decode to a background worker to keep the UI thread free.
        """
        if not preserve_zoom:
            self.zoom_requested.emit(1.0)
        self.set_status(f"Loading {os.path.basename(file_path)}...")
        self.loading_started.emit()
        self._first_render_done = False

        self.render_worker.cleanup()

        self.state.preview_raw = None
        self.state.original_res = (0, 0)

        self.preview_load_requested.emit(
            PreviewLoadTask(
                file_path=file_path,
                workspace_color_space=self.state.workspace_color_space,
                use_camera_wb=self.state.config.exposure.use_camera_wb,
                full_resolution=self.state.hq_preview,
            )
        )

    def _on_preview_loaded(self, file_path: str, raw: Any, dims: Any) -> None:
        self.state.preview_raw = raw
        self.state.original_res = dims
        self.state.current_file_path = file_path
        self.request_render()

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
        )
        new_proc = replace(self.state.config.process, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0))
        self.session.update_config(replace(self.state.config, geometry=new_geo, process=new_proc))
        self.state.active_tool = ToolMode.NONE
        self.tool_sync_requested.emit()
        self.request_render()

    def reset_crop(self) -> None:
        new_proc = replace(self.state.config.process, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0))
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(self.state.config.geometry, manual_crop_rect=None),
                process=new_proc,
            )
        )
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
        if not self.state.uploaded_files:
            return

        self.set_status("Starting Batch Normalization...")
        task = NormalizationTask(
            files=self.state.uploaded_files.copy(),
            workspace_color_space=self.state.workspace_color_space,
        )
        self.normalization_requested.emit(task)

    def _on_normalization_progress(self, current: int, total: int, name: str) -> None:
        """
        Updates UI status during batch analysis.
        """
        self.set_status(f"Analyzing {current}/{total}: {name}...")
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

        self.set_status("Batch Normalization Complete", 3000)
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
            local_floors=(0.0, 0.0, 0.0),
            local_ceils=(0.0, 0.0, 0.0),
        )
        self.session.update_config(replace(self.state.config, process=new_process))
        self.request_render()

    def request_render(self, readback_metrics: bool = True) -> None:
        """
        Dispatches a render task to the worker thread.
        Direct callers bypass the debounce; the timer is cancelled to avoid a duplicate.
        """
        self._render_debounce.stop()
        if self.state.preview_raw is None:
            return

        self.set_status("Rendering...")

        target_size = float(APP_CONFIG.preview_render_size)
        if self.state.hq_preview and self.state.preview_raw is not None:
            target_size = float(max(self.state.preview_raw.shape[:2]))

        task = RenderTask(
            buffer=self.state.preview_raw,
            config=self.state.config,
            source_hash=self.state.current_file_hash or "preview",
            preview_size=target_size,
            icc_profile_path=self.state.icc_profile_path,
            icc_invert=self.state.icc_invert,
            color_space=self.state.workspace_color_space,
            gpu_enabled=self.state.gpu_enabled,
            readback_metrics=readback_metrics,
        )

        if self._is_rendering:
            self._pending_render_task = task
            return

        self._is_rendering = True
        self.render_requested.emit(task)

    def _ensure_valid_export_path(self) -> Optional[str]:
        """
        Checks if the current export path is valid. If not, prompts the user.
        Returns the valid path or None if the user cancelled.
        """
        export_path = self.state.config.export.export_path
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
        """
        Initiates high-resolution export for the current file.
        """
        if not self.state.current_file_path:
            return

        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        export_conf = replace(
            self.state.config.export,
            export_path=export_path,
            apply_icc=self.state.apply_icc_to_export,
            icc_profile_path=self.state.icc_profile_path,
            icc_invert=self.state.icc_invert,
        )

        self._run_export_tasks(
            [
                ExportTask(
                    file_info={
                        "name": os.path.basename(self.state.current_file_path),
                        "path": self.state.current_file_path,
                        "hash": self.state.current_file_hash,
                    },
                    params=self.state.config,
                    export_settings=export_conf,
                    gpu_enabled=self.state.gpu_enabled,
                )
            ]
        )

    def request_batch_export(self, override_settings: bool = False) -> None:
        """
        Initiates batch export, optionally applying current export settings to all files.
        """
        export_path = self._ensure_valid_export_path()
        if not export_path:
            return

        current_export = replace(self.state.config.export, export_path=export_path)
        icc_path = self.state.icc_profile_path
        icc_invert = self.state.icc_invert
        apply_icc = self.state.apply_icc_to_export

        tasks = []
        for f in self.state.uploaded_files:
            params = self.session.repo.load_file_settings(f["hash"]) or self.state.config

            if override_settings:
                params = replace(params, export=current_export)

            final_export = replace(
                params.export,
                apply_icc=apply_icc,
                icc_profile_path=icc_path,
                icc_invert=icc_invert,
            )

            bounds_override = None
            if f["hash"] == self.state.current_file_hash:
                with self.state.metrics_lock:
                    bounds_override = self.state.last_metrics.get("log_bounds")

            tasks.append(
                ExportTask(
                    file_info=f,
                    params=params,
                    export_settings=final_export,
                    gpu_enabled=self.state.gpu_enabled,
                    bounds_override=bounds_override,
                )
            )

        if tasks:
            self._run_export_tasks(tasks)

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

        should_update_thumb = not self._first_render_done
        self._first_render_done = True

        with self.state.metrics_lock:
            self.state.last_metrics.update(metrics)

        if metrics.get("gpu_fallback") and not self._gpu_fallback_notified:
            self._gpu_fallback_notified = True
            self.set_status("GPU acceleration failed — using CPU", 5000)
        else:
            self.set_status("READY", 1000)

        self.image_updated.emit()

        if should_update_thumb:
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
            if bounds:
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
            )
        )

    def cleanup(self) -> None:
        """
        Total system evacuation on exit.
        """
        self.render_thread.quit()
        self.render_thread.wait()
        self.export_thread.quit()
        self.export_thread.wait()
        self.thumb_thread.quit()
        self.thumb_thread.wait()
        self.norm_thread.quit()
        self.norm_thread.wait()
        self.discovery_thread.quit()
        self.discovery_thread.wait()
        self.preview_load_thread.quit()
        self.preview_load_thread.wait()
        self.render_worker.destroy_all()
