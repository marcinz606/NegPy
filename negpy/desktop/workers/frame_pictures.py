"""Large pictures of a roll's frames, with their edits, for Assign Capture Times. Its own CPU-only
ImageProcessor never contends with the live render's GPU pool."""

from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from negpy.services.export.contact_sheet_roll import tile_params

PICTURE_PX = 1200


class FramePicturesWorker(QObject):
    """Renders the frames that wait, first one first. A failed render emits a null image."""

    ready = pyqtSignal(str, QImage)

    def __init__(self, sources: dict[str, tuple[dict, Any]], long_px: int) -> None:
        super().__init__()
        self._sources = sources
        self._long_px = long_px
        self._lock = threading.Lock()
        self._waiting: list[str] = []
        self._stopped = False
        self._processor: Any = None

    def wait_for(self, keys: list[str]) -> None:
        """Thread-safe: these frames replace the ones still waiting."""
        with self._lock:
            self._waiting = [k for k in keys if k in self._sources]

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._waiting = []

    def _next(self) -> Optional[str]:
        with self._lock:
            return None if self._stopped or not self._waiting else self._waiting.pop(0)

    @pyqtSlot()
    def run(self) -> None:
        while (key := self._next()) is not None:
            self.ready.emit(key, self._render(key))
        if self._processor is not None:
            self._processor.cleanup()

    def _render(self, key: str) -> QImage:
        asset, config = self._sources[key]
        try:
            if self._processor is None:
                from negpy.services.rendering.image_processor import ImageProcessor

                self._processor = ImageProcessor(use_gpu=False)
            tile = self._processor.render_display_array(
                asset["path"],
                tile_params(config),
                asset["hash"],
                target_long_px=self._long_px,
                prefer_gpu=False,
                fast_decode=True,
                half=int(asset.get("half") or 0),
                split_x=float(asset.get("split_x") or 0.5),
                crop_rect=tuple(asset["crop_rect"]) if asset.get("crop_rect") else None,
                gutter_thickness=float(asset.get("gutter_thickness") or 0.0),
            )
        except Exception:
            return QImage()
        if tile is None:
            return QImage()
        rgb = np.ascontiguousarray(tile[:, :, :3], dtype=np.uint8)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class FramePictures(QObject):
    """GUI-thread handle. request() renders the given frames in that order; a new request replaces
    the frames still waiting. `ready(key, image)` comes once per rendered frame."""

    ready = pyqtSignal(str, QImage)
    _kick = pyqtSignal()

    def __init__(self, sources: dict[str, tuple[dict, Any]], long_px: int = PICTURE_PX, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Started on first use: a running QThread aborts if destroyed without quit().
        self._thread = QThread()
        self._worker = FramePicturesWorker(sources, long_px)
        self._worker.moveToThread(self._thread)
        self._kick.connect(self._worker.run)
        self._worker.ready.connect(self.ready)

    def request(self, keys: list[str]) -> None:
        if not keys:
            return
        if not self._thread.isRunning():
            self._thread.start()
        self._worker.wait_for(keys)
        self._kick.emit()

    def stop(self) -> None:
        self._worker.stop()
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
