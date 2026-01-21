from dataclasses import dataclass
from typing import Optional
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
    icc_profile_path: Optional[str] = None
    icc_invert: bool = False
    color_space: str = "Adobe RGB"


@dataclass(frozen=True)
class ThumbnailUpdateTask:
    """
    Request to update a thumbnail from a rendered buffer.
    """

    filename: str
    file_hash: str
    buffer: np.ndarray


class RenderWorker(QObject):
    """
    Executes the Darkroom Engine pipeline in a background thread.
    """

    finished = pyqtSignal(object, dict)  # buffer (ndarray or TextureView), metrics
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()

    def cleanup(self) -> None:
        """
        Releases processor resources.
        """
        self._processor.cleanup()

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

            # Apply Soft Proofing / ICC if requested (only for CPU results)
            if task.icc_profile_path and isinstance(result, np.ndarray):
                pil_img = self._processor.buffer_to_pil(result, task.config)
                pil_proof, _ = self._processor._apply_color_management(
                    pil_img,
                    task.color_space,
                    task.icc_profile_path,
                    task.icc_invert,
                )
                # Convert back to float32 buffer for display
                arr = np.array(pil_proof)
                if arr.dtype == np.uint8:
                    result = arr.astype(np.float32) / 255.0
                elif arr.dtype == np.uint16:
                    result = arr.astype(np.float32) / 65535.0

                # Update metrics to reflect the proofed image if needed
                metrics["base_positive"] = result
            elif not isinstance(result, np.ndarray):
                # Ensure base_positive in metrics points to the texture view for display
                metrics["base_positive"] = result

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

    @pyqtSlot(ThumbnailUpdateTask)
    def update_rendered(self, task: ThumbnailUpdateTask) -> None:
        """
        Generates a thumbnail from a rendered buffer.
        """
        from src.services.assets.thumbnails import get_rendered_thumbnail

        try:
            # Copy buffer to thread
            buf = task.buffer.copy()
            thumb = get_rendered_thumbnail(buf, task.file_hash, self._store)

            if thumb:
                self.finished.emit({task.filename: thumb})
        except Exception as e:
            print(f"DEBUG: Rendered thumbnail error: {e}")
