from dataclasses import dataclass
from typing import Optional
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from negpy.domain.models import WorkspaceConfig
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.features.exposure.normalization import analyze_log_exposure_bounds
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.kernel.system.logging import get_logger

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


@dataclass(frozen=True)
class NormalizationTask:
    """Request to analyze log bounds for a set of files."""

    files: list[dict]
    workspace_color_space: str


@dataclass(frozen=True)
class AssetDiscoveryTask:
    """Request to find and hash image files in paths."""

    paths: list[str]
    supported_extensions: tuple[str, ...]


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

            from negpy.infrastructure.gpu.resources import GPUTexture

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
                result = arr.astype(np.float32) / (65535.0 if arr.dtype == np.uint16 else 255.0)

            # Ensure ground truth is stored in metrics for view consumption
            metrics["base_positive"] = result

            self.finished.emit(result, metrics)
            self.metrics_updated.emit(metrics)

        except Exception as e:
            self.error.emit(str(e))


class ThumbnailWorker(QObject):
    """
    Asynchronous thumbnail generation worker.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)

    def __init__(self, asset_store) -> None:
        super().__init__()
        self._store = asset_store

    @pyqtSlot(list)
    def generate(self, files: list) -> None:
        """
        Generates thumbnails for a list of files with progress reporting.
        """
        import asyncio
        from negpy.services.assets import thumbnails as thumb_service

        try:
            total = len(files)

            async def _progress_callback(current: int, name: str):
                self.progress.emit(current, total, name)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_thumbs = loop.run_until_complete(
                thumb_service.generate_batch_thumbnails(files, self._store, progress_callback=_progress_callback)
            )
            self.finished.emit(new_thumbs)
        except Exception as e:
            logger.error(f"Thumbnail generation failure: {e}")

    @pyqtSlot(ThumbnailUpdateTask)
    def update_rendered(self, task: ThumbnailUpdateTask) -> None:
        """Updates thumbnail from a rendered positive buffer."""
        from negpy.services.assets.thumbnails import get_rendered_thumbnail

        try:
            buf = task.buffer.copy()
            thumb = get_rendered_thumbnail(buf, task.file_hash, self._store)
            if thumb:
                self.finished.emit({task.filename: thumb})
        except Exception as e:
            logger.error(f"Thumbnail update failure: {e}")


class AssetDiscoveryWorker(QObject):
    """
    Background worker for file system crawling and hashing.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    @pyqtSlot(AssetDiscoveryTask)
    def process(self, task: AssetDiscoveryTask) -> None:
        """
        Scans paths for supported images and calculates hashes.
        """
        import os
        from negpy.kernel.image.logic import calculate_file_hash

        discovered_paths = []
        for path in task.paths:
            try:
                if os.path.isdir(path):
                    for f in os.listdir(path):
                        if f.lower().endswith(task.supported_extensions):
                            discovered_paths.append(os.path.join(path, f))
                else:
                    if path.lower().endswith(task.supported_extensions):
                        discovered_paths.append(path)
            except Exception as e:
                logger.error(f"Discovery error for {path}: {e}")

        total = len(discovered_paths)
        valid_assets = []

        for i, path in enumerate(discovered_paths):
            name = os.path.basename(path)
            self.progress.emit(i + 1, total, name)

            try:
                f_hash = calculate_file_hash(path)
                if not f_hash.startswith("err_"):
                    valid_assets.append({"name": name, "path": path, "hash": f_hash})
            except Exception as e:
                logger.error(f"Skipping invalid file {path}: {e}")

        self.finished.emit(valid_assets)


class NormalizationWorker(QObject):
    """
    Asynchronous batch normalization worker.
    Analyzes multiple RAW files to find a consistent baseline.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(tuple, tuple, tuple)
    error = pyqtSignal(str)

    def __init__(self, preview_service, repo) -> None:
        super().__init__()
        self._preview_service = preview_service
        self._repo = repo

    @pyqtSlot(NormalizationTask)
    def process(self, task: NormalizationTask) -> None:
        """
        Executes analysis on a batch of files.
        """
        total = len(task.files)
        all_floors = []
        all_ceils = []
        all_casts = []

        try:
            for i, f_info in enumerate(task.files):
                self.progress.emit(i + 1, total, f_info["name"])

                params = self._repo.load_file_settings(f_info["hash"])
                use_camera_wb = params.exposure.use_camera_wb if params else False
                analysis_buffer = params.process.analysis_buffer if params else DEFAULT_WORKSPACE_CONFIG.process.analysis_buffer
                process_mode = params.process.process_mode if params else DEFAULT_WORKSPACE_CONFIG.process.process_mode
                e6_normalize = params.process.e6_normalize if params else DEFAULT_WORKSPACE_CONFIG.process.e6_normalize
                shadow_threshold = params.process.shadow_cast_threshold if params else DEFAULT_WORKSPACE_CONFIG.process.shadow_cast_threshold

                raw, _, _ = self._preview_service.load_linear_preview(
                    f_info["path"],
                    task.workspace_color_space,
                    use_camera_wb=use_camera_wb,
                )

                bounds = analyze_log_exposure_bounds(
                    raw,
                    analysis_buffer=analysis_buffer,
                    process_mode=process_mode,
                    e6_normalize=e6_normalize,
                )
                all_floors.append(bounds.floors)
                all_ceils.append(bounds.ceils)

                from negpy.features.exposure.normalization import normalize_log_image
                from negpy.features.exposure.shadows import analyze_shadow_cast
                epsilon = 1e-6
                img_log = np.log10(np.clip(raw, epsilon, 1.0))
                res_norm = normalize_log_image(img_log, bounds)
                cast = analyze_shadow_cast(res_norm, shadow_threshold)
                all_casts.append(cast)

            floors_arr = np.array(all_floors)
            ceils_arr = np.array(all_ceils)
            casts_arr = np.array(all_casts)

            def get_robust_mean(data: np.ndarray) -> np.ndarray:
                results = []
                for ch in range(3):
                    ch_data = data[:, ch]
                    if len(ch_data) < 5:
                        results.append(np.mean(ch_data))
                        continue

                    low, high = np.percentile(ch_data, [10, 90])
                    mask = (ch_data >= low) & (ch_data <= high)
                    valid = ch_data[mask]

                    if valid.size > 0:
                        results.append(np.mean(valid))
                    else:
                        results.append(np.mean(ch_data))
                return np.array(results)

            avg_floors = get_robust_mean(floors_arr)
            avg_ceils = get_robust_mean(ceils_arr)
            avg_casts = get_robust_mean(casts_arr)

            self.finished.emit(
                tuple(map(float, avg_floors)),
                tuple(map(float, avg_ceils)),
                tuple(map(float, avg_casts)),
            )

        except Exception as e:
            logger.error(f"Normalization analysis failure: {e}")
            self.error.emit(str(e))
