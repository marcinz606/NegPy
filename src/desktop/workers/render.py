from dataclasses import dataclass
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from src.domain.models import WorkspaceConfig
from src.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class RenderTask:
    """
    Request parameters for a single render pass.
    """

    buffer: np.ndarray
    config: WorkspaceConfig
    source_hash: str
    preview_size: float


class RenderWorker(QObject):
    """
    Executes the Darkroom Engine pipeline in a background thread.
    """

    finished = pyqtSignal(np.ndarray, dict)
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()

    @pyqtSlot(RenderTask)
    def process(self, task: RenderTask) -> None:
        """
        Runs the engine and emits the resulting buffer and metrics.
        """
        try:
            # We copy the buffer to avoid mutation issues across threads
            buffer_copy = task.buffer.copy()

            result, metrics = self._processor.run_pipeline(
                buffer_copy,
                task.config,
                task.source_hash,
                render_size_ref=task.preview_size,
            )

            self.finished.emit(result, metrics)
        except Exception as e:
            self.error.emit(str(e))


class ThumbnailWorker(QObject):
    """
    Generates asset thumbnails asynchronously.
    """

    finished = pyqtSignal(dict)  # filename -> thumb_path

    def __init__(self, asset_store) -> None:
        super().__init__()
        self._store = asset_store

    @pyqtSlot(list)
    def generate(self, files: list) -> None:
        """
        Runs the batch thumbnail generator.
        """
        import asyncio
        from src.services.assets import thumbnails as thumb_service

        # We run the existing async thumbnail logic inside a new loop
        # since this worker lives in its own thread.
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            new_thumbs = loop.run_until_complete(
                thumb_service.generate_batch_thumbnails(files, self._store)
            )
            self.finished.emit(new_thumbs)
        except Exception as e:
            print(f"DEBUG: Thumbnail generation error: {e}")
