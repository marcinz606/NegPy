"""Contact Sheet background work. Its own CPU-only ImageProcessor never contends with the live render's GPU pool."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.services.export.contact_sheet_roll import FrameFacts, SheetFrame, read_frame_facts, tile_params
from negpy.services.rendering.image_processor import ImageProcessor

# Header reads are I/O-bound; a few in flight leave the CPU to the app.
_FACT_WORKERS = 4
PREVIEW_TILE_PX = 320


@dataclass(frozen=True)
class ContactSheetPrepTask:
    generation: int
    assets: tuple[dict, ...]


@dataclass(frozen=True)
class ContactSheetPreviewTask:
    generation: int
    frames: tuple[SheetFrame, ...]
    long_px: int = PREVIEW_TILE_PX
    working_color_space: str = WORKING_COLOR_SPACE


class ContactSheetPreviewWorker(QObject):
    prepared = pyqtSignal(int, object)  # generation, list[FrameFacts] in asset order
    tile_ready = pyqtSignal(int, int, object)  # generation, frame index, ndarray or None
    finished = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self._processor: Optional[ImageProcessor] = None
        self._lock = threading.Lock()
        self._cancelled: set[int] = set()

    def cancel(self, generation: int) -> None:
        """Thread-safe: called from the GUI thread while a generation runs here."""
        with self._lock:
            self._cancelled.add(int(generation))

    def _is_cancelled(self, generation: int) -> bool:
        with self._lock:
            return generation in self._cancelled

    def _forget(self, generation: int) -> None:
        with self._lock:
            self._cancelled.discard(generation)

    @pyqtSlot(object)
    def prepare(self, task: ContactSheetPrepTask) -> None:
        """Always emits `prepared` unless cancelled: the controller waits for it."""
        generation = int(task.generation)
        facts: list[FrameFacts] = [FrameFacts()] * len(task.assets)
        try:
            with ThreadPoolExecutor(max_workers=_FACT_WORKERS) as pool:
                futures = {pool.submit(read_frame_facts, asset): i for i, asset in enumerate(task.assets)}
                for future in as_completed(futures):
                    if self._is_cancelled(generation):
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    try:
                        facts[futures[future]] = future.result()
                    except Exception:
                        pass
        except Exception:
            pass
        if self._is_cancelled(generation):
            self._forget(generation)
            return
        self.prepared.emit(generation, facts)

    @pyqtSlot(object)
    def render(self, task: ContactSheetPreviewTask) -> None:
        generation = int(task.generation)
        if self._processor is None:
            self._processor = ImageProcessor(use_gpu=False)
        try:
            frames = task.frames
            for index, frame in enumerate(frames):
                if self._is_cancelled(generation):
                    return
                info = frame.asset
                next_path = frames[index + 1].asset.get("path") if index + 1 < len(frames) else None
                tile = self._processor.render_display_array(
                    info["path"],
                    tile_params(frame.config),
                    info["hash"],
                    target_long_px=task.long_px,
                    prefer_gpu=False,
                    working_color_space=task.working_color_space,
                    fast_decode=True,
                    half=int(info.get("half") or 0),
                    split_x=float(info.get("split_x") or 0.5),
                    crop_rect=tuple(info["crop_rect"]) if info.get("crop_rect") else None,
                    gutter_thickness=float(info.get("gutter_thickness") or 0.0),
                    keep_source=next_path == info.get("path"),
                )
                self.tile_ready.emit(generation, index, tile)
            if not self._is_cancelled(generation):
                self.finished.emit(generation)
        finally:
            self._forget(generation)
            if self._processor is not None:
                self._processor.cleanup()


class ContactSheetPreview(QObject):
    """GUI-thread handle on the preview worker."""

    prepared = pyqtSignal(int, object)
    tile_ready = pyqtSignal(int, int, object)
    _prepare_requested = pyqtSignal(object)
    _render_requested = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Started on first use: a running QThread aborts if destroyed without quit().
        self._thread = QThread()
        self._worker = ContactSheetPreviewWorker()
        self._worker.moveToThread(self._thread)
        self._prepare_requested.connect(self._worker.prepare)
        self._render_requested.connect(self._worker.render)
        self._worker.prepared.connect(self.prepared)
        self._worker.tile_ready.connect(self.tile_ready)
        self._generation = 0

    def _next_generation(self) -> int:
        if not self._thread.isRunning():
            self._thread.start()
        self._generation += 1
        return self._generation

    def prepare(self, assets: tuple[dict, ...]) -> int:
        generation = self._next_generation()
        self._prepare_requested.emit(ContactSheetPrepTask(generation, tuple(assets)))
        return generation

    def render(self, frames: tuple[SheetFrame, ...], working_color_space: str = WORKING_COLOR_SPACE) -> int:
        generation = self._next_generation()
        self._render_requested.emit(ContactSheetPreviewTask(generation, tuple(frames), working_color_space=working_color_space))
        return generation

    def cancel(self, generation: int) -> None:
        self._worker.cancel(generation)

    def shutdown(self) -> None:
        self._worker.cancel(self._generation)
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
