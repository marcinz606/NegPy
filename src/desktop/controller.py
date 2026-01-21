import os
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


class AppController(QObject):
    """
    Orchestrates application logic, threading, and state synchronization.
    """

    image_updated = pyqtSignal()
    metrics_available = pyqtSignal(dict)
    loading_started = pyqtSignal()
    export_progress = pyqtSignal(int, int, str)
    export_finished = pyqtSignal()
    render_requested = pyqtSignal(RenderTask)
    thumbnail_requested = pyqtSignal(list)
    thumbnail_update_requested = pyqtSignal(ThumbnailUpdateTask)
    tool_sync_requested = pyqtSignal()
    config_updated = pyqtSignal()

    def __init__(self, session_manager: DesktopSessionManager):
        super().__init__()
        self.session = session_manager
        self.state: AppState = session_manager.state
        self._first_render_done = False

        self.preview_service = PreviewManager()
        self.watcher = FolderWatchService()
        self.asset_store = LocalAssetStore(
            APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir
        )
        self.asset_store.initialize()

        # Render Thread
        self.render_thread = QThread()
        self.render_worker = RenderWorker()
        self.render_worker.moveToThread(self.render_thread)
        self.render_thread.start()

        # Export Thread
        self.export_thread = QThread()
        self.export_worker = ExportWorker()
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.start()

        # Thumbnail Thread
        self.thumb_thread = QThread()
        self.thumb_worker = ThumbnailWorker(self.asset_store)
        self.thumb_worker.moveToThread(self.thumb_thread)
        self.thumb_thread.start()

        # Render Flow Control
        self._is_rendering = False
        self._pending_render_task: Any = None

        self._connect_signals()

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

        # File navigation
        self.session.file_selected.connect(self.load_file)
        self.session.file_selected.connect(self._update_thumbnail_from_state)
        # Global UI sync
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
        """Updates state with new thumbnail icons and refreshes UI."""
        for name, pil_img in new_thumbs.items():
            if pil_img:
                u8_arr = np.array(pil_img.convert("RGB"))
                qimg = ImageConverter.to_qimage(u8_arr)
                pixmap = QPixmap.fromImage(qimg)
                self.state.thumbnails[name] = QIcon(pixmap)

        self.session.asset_model.refresh()

    def load_file(self, file_path: str) -> None:
        self.loading_started.emit()
        target_cs = self.state.workspace_color_space
        use_cam_wb = self.state.config.exposure.use_camera_wb
        self._first_render_done = False

        # Clear GPU cache before loading new high-res raw to free VRAM
        self.render_worker.cleanup()

        try:
            raw, dims, metadata = self.preview_service.load_linear_preview(
                file_path, target_cs, use_camera_wb=use_cam_wb
            )

            self.state.preview_raw = raw
            self.state.original_res = dims
            self.state.current_file_path = file_path
            self.request_render()

        except Exception as e:
            print(f"Loading error: {e}")

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

        ur1, ur2 = min(rx1, rx2), max(rx1, rx2)
        vr1, vr2 = min(ry1, ry2), max(ry1, ry2)

        new_geo = replace(
            self.state.config.geometry,
            manual_crop_rect=(ur1, vr1, ur2, vr2),
        )
        self.session.update_config(replace(self.state.config, geometry=new_geo))
        self.state.active_tool = ToolMode.NONE
        self.tool_sync_requested.emit()
        self.request_render()

    def reset_crop(self) -> None:
        new_geo = replace(self.state.config.geometry, manual_crop_rect=None)
        self.session.update_config(replace(self.state.config, geometry=new_geo))
        self.request_render()

    def save_current_edits(self) -> None:
        """Manually persists edits to database and updates thumbnail."""
        if self.state.current_file_hash:
            self.session.update_config(self.state.config, persist=True)
            self._update_thumbnail_from_state()

    def clear_retouch(self) -> None:
        new_ret = replace(self.state.config.retouch, manual_dust_spots=[])
        self.session.update_config(replace(self.state.config, retouch=new_ret))
        self.request_render()

    def _handle_dust_pick(self, nx: float, ny: float) -> None:
        uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return

        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)
        size = float(self.state.config.retouch.manual_dust_size)

        new_spots = self.state.config.retouch.manual_dust_spots + [(rx, ry, size)]
        new_retouch = replace(self.state.config.retouch, manual_dust_spots=new_spots)
        self.session.update_config(replace(self.state.config, retouch=new_retouch))
        self.request_render()

    def _handle_wb_pick(self, nx: float, ny: float) -> None:
        metrics = self.state.last_metrics
        # Use analysis_buffer (CPU side) if available, otherwise base_positive
        img = metrics.get("analysis_buffer")
        if img is None:
            img = metrics.get("base_positive")

        if img is None or not isinstance(img, np.ndarray):
            return

        h, w = img.shape[:2]
        px = int(np.clip(nx * w, 0, w - 1))
        py = int(np.clip(ny * h, 0, h - 1))
        sampled = img[py, px]

        dm, dy = calculate_wb_shifts(sampled)

        new_exp = replace(
            self.state.config.exposure,
            wb_cyan=0.0,
            wb_magenta=float(np.clip(dm, -1, 1)),
            wb_yellow=float(np.clip(dy, -1, 1)),
        )

        self.session.update_config(replace(self.state.config, exposure=new_exp))
        self.request_render()

    def request_render(self) -> None:
        if self.state.preview_raw is None:
            return

        task = RenderTask(
            buffer=self.state.preview_raw,
            config=self.state.config,
            source_hash=self.state.current_file_hash or "preview",
            preview_size=1200.0,
            icc_profile_path=self.state.icc_profile_path,
            icc_invert=self.state.icc_invert,
            color_space=self.state.workspace_color_space,
        )

        if self._is_rendering:
            self._pending_render_task = task
            return

        self._is_rendering = True
        self.render_requested.emit(task)

    def request_export(self) -> None:
        if not self.state.current_file_path:
            return

        task = ExportTask(
            file_info={
                "name": os.path.basename(self.state.current_file_path),
                "path": self.state.current_file_path,
                "hash": self.state.current_file_hash,
            },
            params=self.state.config,
            export_settings=self.state.config.export,
        )
        self._run_export_tasks([task])

    def request_batch_export(self) -> None:
        tasks = []
        for file_info in self.state.uploaded_files:
            f_hash = file_info["hash"]
            saved_config = self.session.repo.load_file_settings(f_hash)

            tasks.append(
                ExportTask(
                    file_info=file_info,
                    params=saved_config or self.state.config,
                    export_settings=self.state.config.export,
                )
            )

        if tasks:
            self._run_export_tasks(tasks)

    def _run_export_tasks(self, tasks: List[ExportTask]) -> None:
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_batch",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
        )

    def _on_render_finished(self, buffer: Any, metrics: Dict[str, Any]) -> None:
        self.state.last_metrics = metrics
        self.state.is_processing = False
        self.image_updated.emit()

        if not self._first_render_done:
            self._first_render_done = True
            self._update_thumbnail_from_state()

        self._is_rendering = False
        if self._pending_render_task:
            task = self._pending_render_task
            self._pending_render_task = None
            self._is_rendering = True
            self.render_requested.emit(task)

    def _on_metrics_updated(self, metrics: Dict[str, Any]) -> None:
        self.state.last_metrics.update(metrics)
        self.metrics_available.emit(metrics)

    def _on_render_error(self, message: str) -> None:
        self.state.is_processing = False
        self._is_rendering = False
        self._pending_render_task = None
        print(f"Render Error: {message}")

    def _on_export_finished(self) -> None:
        self.export_finished.emit()
        self._update_thumbnail_from_state()

    def _update_thumbnail_from_state(self) -> None:
        """Triggers a background thumbnail update from current render result."""
        if not self.state.current_file_path or not self.state.current_file_hash:
            return

        metrics = self.state.last_metrics
        buffer = metrics.get("base_positive")

        # If using GPU, base_positive is a texture view.
        # We use analysis_buffer which is a downsampled CPU ndarray.
        if buffer is not None and not isinstance(buffer, np.ndarray):
            buffer = metrics.get("analysis_buffer")

        if buffer is None or not isinstance(buffer, np.ndarray):
            return

        task = ThumbnailUpdateTask(
            filename=os.path.basename(self.state.current_file_path),
            file_hash=self.state.current_file_hash,
            buffer=buffer.copy(),
        )
        self.thumbnail_update_requested.emit(task)

    def cleanup(self) -> None:
        self.render_thread.quit()
        self.render_thread.wait()
        self.export_thread.quit()
        self.export_thread.wait()
        self.thumb_thread.quit()
        self.thumb_thread.wait()
        self.render_worker.destroy_all()
