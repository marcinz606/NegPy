import os
import time
from dataclasses import replace
from typing import List, Dict, Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QMetaObject, Q_ARG, Qt
from PyQt6.QtGui import QIcon, QPixmap

from src.desktop.session import DesktopSessionManager, AppState, ToolMode
from src.desktop.workers.render import (
    RenderWorker,
    RenderTask,
    ThumbnailWorker,
    ThumbnailUpdateTask,
)
from src.desktop.workers.export import ExportWorker, ExportTask
from src.services.rendering.preview_manager import PreviewManager
from src.infrastructure.filesystem.watcher import FolderWatchService
from src.infrastructure.storage.local_asset_store import LocalAssetStore
from src.services.view.coordinate_mapping import CoordinateMapping
from src.kernel.system.config import APP_CONFIG
from src.desktop.converters import ImageConverter
from src.features.exposure.logic import calculate_wb_shifts
from src.infrastructure.gpu.resources import GPUTexture
from src.kernel.system.logging import get_logger

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
    thumbnail_requested = pyqtSignal(list)
    thumbnail_update_requested = pyqtSignal(ThumbnailUpdateTask)
    tool_sync_requested = pyqtSignal()
    config_updated = pyqtSignal()
    status_message_requested = pyqtSignal(str, int)
    status_progress_requested = pyqtSignal(int, int)

    def __init__(self, session_manager: DesktopSessionManager):
        super().__init__()
        self.session = session_manager
        self.state: AppState = session_manager.state
        self._first_render_done = False
        self._export_start_time = 0.0

        self.preview_service = PreviewManager()
        self.watcher = FolderWatchService()
        self.asset_store = LocalAssetStore(
            APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir
        )
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

        self._is_rendering = False
        self._pending_render_task: Any = None

        self._connect_signals()

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
        self.thumbnail_update_requested.connect(self.thumb_worker.update_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)

        self.session.file_selected.connect(self.load_file)
        self.session.state_changed.connect(self.config_updated.emit)
        self.session.state_changed.connect(self.request_render)

    def generate_missing_thumbnails(self) -> None:
        missing = [
            f
            for f in self.state.uploaded_files
            if f["name"] not in self.state.thumbnails
        ]
        if missing:
            self.thumbnail_requested.emit(missing)

    def _on_thumbnails_finished(self, new_thumbs: Dict[str, Any]) -> None:
        self.set_status("GALLERIES UPDATED", 3000)
        for name, pil_img in new_thumbs.items():
            if pil_img:
                u8_arr = np.array(pil_img.convert("RGB"))
                self.state.thumbnails[name] = QIcon(
                    QPixmap.fromImage(ImageConverter.to_qimage(u8_arr))
                )
        self.session.asset_model.refresh()

    def load_file(self, file_path: str) -> None:
        """Loads a new RAW file into the linear preview workspace."""
        self.set_status(f"Loading {os.path.basename(file_path)}...")
        self.loading_started.emit()
        self._first_render_done = False

        # Evacuate VRAM before large allocation
        self.render_worker.cleanup()

        try:
            raw, dims, _ = self.preview_service.load_linear_preview(
                file_path,
                self.state.workspace_color_space,
                use_camera_wb=self.state.config.exposure.use_camera_wb,
            )
            self.state.preview_raw = raw
            self.state.original_res = dims
            self.state.current_file_path = file_path
            self.request_render()
        except Exception as e:
            logger.error(f"Asset load failed: {e}")

    def handle_canvas_clicked(self, nx: float, ny: float) -> None:
        if self.state.active_tool == ToolMode.WB_PICK:
            self._handle_wb_pick(nx, ny)
        elif self.state.active_tool == ToolMode.DUST_PICK:
            self._handle_dust_pick(nx, ny)

    def set_active_tool(self, mode: ToolMode) -> None:
        self.state.active_tool = mode
        self.tool_sync_requested.emit()

    def handle_crop_completed(
        self, nx1: float, ny1: float, nx2: float, ny2: float
    ) -> None:
        if self.state.active_tool != ToolMode.CROP_MANUAL:
            return
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
        self.session.update_config(replace(self.state.config, geometry=new_geo))
        self.state.active_tool = ToolMode.NONE
        self.tool_sync_requested.emit()
        self.request_render()

    def reset_crop(self) -> None:
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(self.state.config.geometry, manual_crop_rect=None),
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
        """Removes the most recently added dust spot."""
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
        uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)
        new_spots = self.state.config.retouch.manual_dust_spots + [
            (rx, ry, float(self.state.config.retouch.manual_dust_size))
        ]
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=new_spots),
            )
        )
        self.request_render()

    def _handle_wb_pick(self, nx: float, ny: float) -> None:
        metrics = self.state.last_metrics
        img = metrics.get("analysis_buffer")
        if img is None:
            img = metrics.get("base_positive")

        if isinstance(img, GPUTexture):
            # On-demand readback for picking (avoids frame-by-frame transfer)
            img = img.readback()

        if img is None or not isinstance(img, np.ndarray):
            return

        h, w = img.shape[:2]
        sampled = img[int(np.clip(ny * h, 0, h - 1)), int(np.clip(nx * w, 0, w - 1))]

        # Ensure we only use RGB channels (ignore Alpha if present)
        delta_m, delta_y = calculate_wb_shifts(sampled[:3])

        # Apply damping to account for high-contrast curve slopes
        damping = 0.4
        exp = self.state.config.exposure
        new_m = np.clip(exp.wb_magenta + delta_m * damping, -1.0, 1.0)
        new_y = np.clip(exp.wb_yellow + delta_y * damping, -1.0, 1.0)

        new_exp = replace(
            self.state.config.exposure,
            wb_magenta=float(new_m),
            wb_yellow=float(new_y),
        )
        self.session.update_config(replace(self.state.config, exposure=new_exp))
        self.request_render()

    def request_render(self, readback_metrics: bool = True) -> None:
        """Dispatches a render task to the worker thread."""
        if self.state.preview_raw is None:
            return

        self.set_status("Rendering...")
        task = RenderTask(
            buffer=self.state.preview_raw,
            config=self.state.config,
            source_hash=self.state.current_file_hash or "preview",
            preview_size=float(APP_CONFIG.preview_render_size),
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

    def request_export(self) -> None:
        if not self.state.current_file_path:
            return

        export_conf = replace(
            self.state.config.export,
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

    def request_batch_export(self) -> None:
        # Synchronize ICC state to export config
        export_conf = replace(
            self.state.config.export,
            apply_icc=self.state.apply_icc_to_export,
            icc_profile_path=self.state.icc_profile_path,
            icc_invert=self.state.icc_invert,
        )

        tasks = [
            ExportTask(
                file_info=f,
                params=self.session.repo.load_file_settings(f["hash"])
                or self.state.config,
                export_settings=export_conf,
                gpu_enabled=self.state.gpu_enabled,
            )
            for f in self.state.uploaded_files
        ]
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

    def _on_render_finished(self, result: Any, metrics: Dict[str, Any]) -> None:
        self._is_rendering = False

        # Only update thumbnail on the very first render of a file
        should_update_thumb = not self._first_render_done
        self._first_render_done = True

        self.state.last_metrics.update(metrics)
        self.set_status("READY", 1000)
        self.image_updated.emit()

        if should_update_thumb:
            self._update_thumbnail_from_state(force_readback=True)

    def _on_metrics_updated(self, metrics: Dict[str, Any]) -> None:
        self.state.last_metrics.update(metrics)
        self.metrics_available.emit(metrics)

    def _on_render_error(self, message: str) -> None:
        self.state.is_processing = self._is_rendering = False
        self._pending_render_task = None
        logger.error(f"Worker failure: {message}")

    def _on_export_finished(self) -> None:
        elapsed = time.time() - self._export_start_time
        self.export_finished.emit(elapsed)
        self._update_thumbnail_from_state(force_readback=True)

    def _update_thumbnail_from_state(self, force_readback: bool = False) -> None:
        # Handle signal arguments (e.g. file path string) getting passed as force_readback
        if not isinstance(force_readback, bool):
            force_readback = False

        if not self.state.current_file_path or not self.state.current_file_hash:
            return
        metrics = self.state.last_metrics
        buffer = metrics.get("base_positive")

        # GPU textures need to be read back to CPU memory for thumbnail processing.
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
                buffer=buffer.copy(),
            )
        )

    def cleanup(self) -> None:
        """Total system evacuation on exit."""
        self.render_thread.quit()
        self.render_thread.wait()
        self.export_thread.quit()
        self.export_thread.wait()
        self.thumb_thread.quit()
        self.thumb_thread.wait()
        self.render_worker.destroy_all()
