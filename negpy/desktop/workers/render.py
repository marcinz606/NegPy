import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.system.config import APP_CONFIG, DEFAULT_WORKSPACE_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.image_processor import ImageProcessor

logger = get_logger(__name__)


@dataclass(frozen=True)
class RenderTask:
    """Immutable rendering request payload."""

    buffer: np.ndarray
    config: WorkspaceConfig
    source_hash: str
    preview_size: float
    icc_input_path: Optional[str] = None
    icc_output_path: Optional[str] = None
    color_space: str = "Adobe RGB"
    gpu_enabled: bool = True
    readback_metrics: bool = True
    ir_buffer: Optional[np.ndarray] = None
    # Monitor ICC profile bytes (detected on the UI thread); soft proof is shown on
    # this display. None = sRGB display.
    monitor_icc_bytes: Optional[bytes] = None


@dataclass(frozen=True)
class ThumbnailUpdateTask:
    """Request to update persistent thumbnail cache."""

    filename: str
    file_hash: str
    buffer: np.ndarray
    color_space: str = "sRGB"


@dataclass(frozen=True)
class NormalizationTask:
    """Request to analyze log bounds for a set of files."""

    files: list[dict]
    workspace_color_space: str
    # Roll-wide overrides taken from the current image: applied to every file's
    # analysis before averaging so the whole roll shares one buffer / d-range.
    override_analysis_buffer: float
    override_drange_clip: float


@dataclass(frozen=True)
class AssetDiscoveryTask:
    """Request to find and hash image files in paths."""

    paths: list[str]
    supported_extensions: tuple[str, ...]


@dataclass(frozen=True)
class PreviewLoadTask:
    """Request to decode a RAW file into a linear preview buffer."""

    file_path: str
    workspace_color_space: str
    use_camera_wb: bool
    full_resolution: bool = False
    file_hash: str | None = None
    use_splash: bool = True
    for_cache_warm: bool = False
    detect_mode: bool = False  # run process-mode autodetect (new files only)


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

    @pyqtSlot()
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
            result, metrics = self._processor.run_pipeline(
                task.buffer,
                task.config,
                task.source_hash,
                render_size_ref=task.preview_size,
                prefer_gpu=task.gpu_enabled,
                readback_metrics=task.readback_metrics,
                ir_buffer=task.ir_buffer,
            )

            soft_proof = task.icc_input_path or task.icc_output_path

            if soft_proof and isinstance(result, GPUTexture):
                result = result.readback()

            if soft_proof and isinstance(result, np.ndarray):
                pil_img = self._processor.buffer_to_pil(result, task.config)
                pil_proof = self._processor.soft_proof_preview(
                    pil_img,
                    task.color_space,
                    task.icc_input_path,
                    task.icc_output_path,
                    task.monitor_icc_bytes,
                )
                arr = np.array(pil_proof)
                result = arr.astype(np.float32) / (65535.0 if arr.dtype == np.uint16 else 255.0)

            # Ensure ground truth is stored in metrics for view consumption
            metrics["base_positive"] = result

            self.finished.emit(result, metrics)
            self.metrics_updated.emit(metrics)

        except Exception as e:
            logger.exception("Render pipeline failed")
            self.error.emit(str(e))


class ThumbnailWorker(QObject):
    """
    Asynchronous thumbnail generation worker.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

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
            try:
                new_thumbs = loop.run_until_complete(
                    thumb_service.generate_batch_thumbnails(files, self._store, progress_callback=_progress_callback)
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)
            self.finished.emit(new_thumbs)
        except Exception as e:
            logger.error(f"Thumbnail generation failure: {e}")
            self.error.emit(str(e))

    @pyqtSlot(ThumbnailUpdateTask)
    def update_rendered(self, task: ThumbnailUpdateTask) -> None:
        """Updates thumbnail from a rendered positive buffer."""
        from negpy.services.assets.thumbnails import get_rendered_thumbnail

        try:
            buf = task.buffer.copy()
            thumb = get_rendered_thumbnail(buf, task.file_hash, self._store, color_space=task.color_space)
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


class PreviewLoadWorker(QObject):
    """
    Background worker for decoding RAW files into a linear preview buffer.
    Keeps the UI thread free during slow I/O and demosaicing.
    """

    # (file_path, raw, dims, source_cs, ir_preview, detected_mode)
    finished = pyqtSignal(str, object, object, str, object, str)
    splash = pyqtSignal(str, object, object)  # (file_path, buffer, dims) — first paint
    error = pyqtSignal(str)

    def __init__(self, preview_service) -> None:
        super().__init__()
        self._preview_service = preview_service

    @pyqtSlot(PreviewLoadTask)
    def process(self, task: PreviewLoadTask) -> None:
        if task.for_cache_warm:
            try:
                self._preview_service.load_linear_preview(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                )
            except Exception as e:
                logger.debug("Preview cache warm failed for %s: %s", task.file_path, e)
            return
        t0 = time.perf_counter()
        try:
            if task.use_splash and not task.full_resolution:
                # Open the file once; get splash + linear in a single pass.
                sp, (raw, dims, metadata) = self._preview_service.load_splash_and_linear(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                )
                if sp is not None:
                    sbuf, sdims = sp
                    self.splash.emit(task.file_path, sbuf, sdims)
            else:
                raw, dims, metadata = self._preview_service.load_linear_preview(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                )
            source_cs = metadata.get("color_space", "")
            ir_preview = metadata.get("ir_preview")
            detected_mode = self._detect_mode(task, raw) if task.detect_mode else ""
            logger.debug("PreviewLoadWorker load %.3fs for %s", time.perf_counter() - t0, task.file_path)
            self.finished.emit(task.file_path, raw, dims, source_cs, ir_preview, detected_mode)
        except Exception as e:
            logger.exception(f"Asset load failed: {task.file_path}")
            self.error.emit(str(e))

    def _detect_mode(self, task: PreviewLoadTask, raw) -> str:
        """Classify film process mode; re-decode no-WB since the C41 mask is hidden by camera WB."""
        from negpy.features.process.logic import detect_process_mode

        try:
            if not task.use_camera_wb:
                scan = raw
            else:
                scan, _, _ = self._preview_service.load_linear_preview(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=False,
                )
            return str(detect_process_mode(scan))
        except Exception:
            logger.exception(f"Process-mode detection failed: {task.file_path}")
            return ""


class NormalizationWorker(QObject):
    """
    Asynchronous batch normalization worker.
    Analyzes multiple RAW files to find a consistent baseline.
    """

    progress = pyqtSignal(int, int, str, bool)
    finished = pyqtSignal(tuple, tuple)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, preview_service, repo) -> None:
        super().__init__()
        self._preview_service = preview_service
        self._repo = repo
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        """Requests the running analysis stop; no baseline is applied."""
        self._cancel.set()

    @pyqtSlot(NormalizationTask)
    def process(self, task: NormalizationTask) -> None:
        """
        Executes analysis on a batch of files using parallel workers.
        """
        import asyncio

        import numpy as np

        from negpy.domain.interfaces import PipelineContext
        from negpy.features.exposure.normalization import analyze_log_exposure_bounds
        from negpy.features.geometry.processor import GeometryProcessor

        self._cancel.clear()
        total = len(task.files)
        limit = max(1, APP_CONFIG.max_workers // 2)
        semaphore = asyncio.Semaphore(limit)
        lock = asyncio.Lock()
        completed = 0

        async def _analyze_file(f_info: dict):
            nonlocal completed
            async with semaphore:
                if self._cancel.is_set():
                    return None
                try:
                    params = self._repo.load_file_settings(f_info["hash"])
                    # Roll-wide buffer / d-range from the current image — applied to every
                    # file so one slider setting drives the whole batch baseline.
                    analysis_buffer = task.override_analysis_buffer
                    drange_clip = task.override_drange_clip
                    process_mode = params.process.process_mode if params else DEFAULT_WORKSPACE_CONFIG.process.process_mode
                    e6_normalize = params.process.e6_normalize if params else DEFAULT_WORKSPACE_CONFIG.process.e6_normalize
                    geometry = params.geometry if params else DEFAULT_WORKSPACE_CONFIG.geometry
                    linear_raw = params.exposure.linear_raw if params else DEFAULT_WORKSPACE_CONFIG.exposure.linear_raw

                    # Use to_thread for blocking CPU/IO bound load and analysis.
                    # Decode with the SAME WB the render path uses (use_camera_wb =
                    # not linear_raw): the roll-average bounds are applied to the
                    # render-decoded image, so analysing in a different WB space
                    # shifts per-channel floors/ceils and produces a color cast.
                    raw, _, _ = await asyncio.to_thread(
                        self._preview_service.load_linear_preview,
                        f_info["path"],
                        task.workspace_color_space,
                        not linear_raw,  # use_camera_wb
                        False,  # full_resolution
                        f_info.get("hash"),
                    )

                    ctx = PipelineContext(
                        original_size=(raw.shape[1], raw.shape[0]),
                        scale_factor=1.0,
                        process_mode=process_mode,
                    )
                    transformed = await asyncio.to_thread(GeometryProcessor(geometry).process, raw, ctx)
                    has_crop = ctx.active_roi is not None

                    bounds = await asyncio.to_thread(
                        analyze_log_exposure_bounds,
                        transformed,
                        roi=ctx.active_roi,
                        analysis_buffer=analysis_buffer,
                        process_mode=process_mode,
                        e6_normalize=e6_normalize,
                        percentile_clip=drange_clip,
                    )

                    async with lock:
                        completed += 1
                        count = completed
                    self.progress.emit(count, total, f_info["name"], has_crop)
                    return bounds.floors, bounds.ceils, f_info["name"]
                except Exception as e:
                    logger.error(f"Failed to analyze {f_info['name']}: {e}")
                    async with lock:
                        completed += 1
                        count = completed
                    self.progress.emit(count, total, f_info["name"], False)
                    return None

        async def _run_batch():
            tasks = [_analyze_file(f) for f in task.files]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            batch_results = loop.run_until_complete(_run_batch())
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

            if self._cancel.is_set():
                self.cancelled.emit()
                return

            valid_results = [r for r in batch_results if r is not None]
            if not valid_results:
                raise RuntimeError("All files in batch failed analysis")

            floors_arr = np.array([r[0] for r in valid_results])
            ceils_arr = np.array([r[1] for r in valid_results])

            def get_robust_mean(data: np.ndarray) -> np.ndarray:
                results = []
                for ch in range(3):
                    ch_data = data[:, ch]
                    if len(ch_data) < 5:
                        results.append(np.mean(ch_data))
                        continue

                    low, high = np.percentile(ch_data, [25, 75])
                    mask = (ch_data >= low) & (ch_data <= high)
                    valid = ch_data[mask]

                    if valid.size > 0:
                        results.append(np.mean(valid))
                    else:
                        results.append(np.mean(ch_data))
                return np.array(results)

            avg_floors = get_robust_mean(floors_arr)
            avg_ceils = get_robust_mean(ceils_arr)

            self.finished.emit(
                tuple(map(float, avg_floors)),
                tuple(map(float, avg_ceils)),
            )

        except Exception as e:
            logger.error(f"Batch Normalization failure: {e}")
            self.error.emit(str(e))
