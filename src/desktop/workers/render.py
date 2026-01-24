from dataclasses import dataclass
from typing import Optional
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from src.domain.models import WorkspaceConfig
from src.services.rendering.image_processor import ImageProcessor
from src.kernel.system.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RenderTask:
    """Immutable rendering request payload."""

    buffer: np.ndarray
    config: WorkspaceConfig
    source_hash: str
    preview_size: float
    icc_profile_path: Optional[str] = None
    icc_invert: bool = False
    color_space: str = "Adobe RGB"
    gpu_enabled: bool = True
    readback_metrics: bool = True


@dataclass(frozen=True)
class ThumbnailUpdateTask:
    """Request to update persistent thumbnail cache."""

    filename: str
    file_hash: str
    buffer: np.ndarray


class RenderWorker(QObject):
    """
    Background rendering worker.
    Decouples engine execution from the UI thread to maintain 60FPS interaction.
    """

    finished = pyqtSignal(object, dict)  # (ndarray|GPUTexture, metrics)
    metrics_updated = pyqtSignal(dict)  # Late-arriving metrics (histogram, etc.)
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()

    @property
    def processor(self) -> ImageProcessor:
        return self._processor

    def cleanup(self) -> None:
        """Evacuates transient GPU resources."""
        self._processor.cleanup()

    def destroy_all(self) -> None:
        """Full teardown of processing resources."""
        self._processor.destroy_all()

    @pyqtSlot(RenderTask)
    def process(self, task: RenderTask) -> None:
        """Executes the rendering pipeline for a single frame."""
        try:
            img_src = task.buffer.copy()

            result, metrics = self._processor.run_pipeline(
                img_src,
                task.config,
                task.source_hash,
                render_size_ref=task.preview_size,
                prefer_gpu=task.gpu_enabled,
                readback_metrics=task.readback_metrics,
            )

            from src.infrastructure.gpu.resources import GPUTexture

            if task.icc_profile_path and isinstance(result, GPUTexture):
                result = result.readback()

            if task.icc_profile_path and isinstance(result, np.ndarray):
                pil_img = self._processor.buffer_to_pil(result, task.config)
                pil_proof, _ = self._processor._apply_color_management(
                    pil_img,
                    task.color_space,
                    task.icc_profile_path,
                    task.icc_invert,
                )
                arr = np.array(pil_proof)
                result = arr.astype(np.float32) / (
                    65535.0 if arr.dtype == np.uint16 else 255.0
                )

            # Ensure ground truth is stored in metrics for view consumption
            metrics["base_positive"] = result

            self.finished.emit(result, metrics)
            self.metrics_updated.emit(metrics)

        except Exception as e:
            self.error.emit(str(e))


class ThumbnailWorker(QObject):
    """Asynchronous thumbnail generation worker."""

    finished = pyqtSignal(dict)

    def __init__(self, asset_store) -> None:
        super().__init__()
        self._store = asset_store

    @pyqtSlot(list)
    def generate(self, files: list) -> None:
        """Generates thumbnails for a list of files."""
        import asyncio
        from src.services.assets import thumbnails as thumb_service

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_thumbs = loop.run_until_complete(
                thumb_service.generate_batch_thumbnails(files, self._store)
            )
            self.finished.emit(new_thumbs)
        except Exception as e:
            logger.error(f"Thumbnail generation failure: {e}")

    @pyqtSlot(ThumbnailUpdateTask)
    def update_rendered(self, task: ThumbnailUpdateTask) -> None:
        """Updates thumbnail from a rendered positive buffer."""
        from src.services.assets.thumbnails import get_rendered_thumbnail

        try:
            buf = task.buffer.copy()
            thumb = get_rendered_thumbnail(buf, task.file_hash, self._store)
            if thumb:
                self.finished.emit({task.filename: thumb})
        except Exception as e:
            logger.error(f"Thumbnail update failure: {e}")
