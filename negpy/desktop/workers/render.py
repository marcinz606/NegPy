import threading
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import output_histogram
from negpy.features.flatfield.logic import apply_flatfield
from negpy.features.geometry.batch_autocrop import detect_crop_candidate, resolve_roll_crops
from negpy.features.geometry.processor import GeometryProcessor
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
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
    # True while the crop tool is active: show the full uncropped frame instead of
    # the final crop.
    crop_preview_full: bool = False
    # Display-only first paint (embedded-JPEG splash): its analysis must not persist.
    ephemeral: bool = False
    # Identity of everything that shaped these pixels; non-empty makes the result
    # eligible for the navigate-back render memo (echoed in metrics).
    memo_key: str = ""


@dataclass(frozen=True)
class ThumbnailUpdateTask:
    """Request to update the filmstrip thumbnail from a rendered buffer."""

    filename: str
    file_hash: str
    buffer: np.ndarray
    color_space: str = WORKING_COLOR_SPACE
    persist: bool = True  # False = in-memory filmstrip only, skip the disk JPEG encode.


@dataclass(frozen=True)
class NormalizationTask:
    """Request to analyze log bounds for a set of files."""

    files: list[dict]
    workspace_color_space: str
    # Roll-wide overrides taken from the current image: applied to every file's
    # analysis before averaging so the whole roll shares one buffer / luma bounds.
    override_analysis_buffer: float
    override_luma_range_clip: float
    override_color_range_clip: float
    # Capture-side unmix must match the render path — bounds measured under a
    # different matrix are invalid for it.
    override_crosstalk_strength: float = 0.0
    override_crosstalk_matrix: tuple | None = None


@dataclass(frozen=True)
class BatchAutoCropInput:
    """One frame and its dispatch-time settings for roll-aware crop analysis."""

    file_info: dict
    config: WorkspaceConfig
    fingerprint: tuple


@dataclass(frozen=True)
class BatchAutoCropTask:
    """Request to detect and calibrate explicit crops across a visible roll."""

    frames: list[BatchAutoCropInput]
    workspace_color_space: str
    generation: int = 0


@dataclass(frozen=True)
class BatchAutoCropResult:
    """Resolved crop payload for controller-side conflict checks and persistence."""

    file_info: dict
    fingerprint: tuple
    manual_crop_rect: tuple[float, float, float, float]
    correction_angle: float
    confidence: float
    calibrated: bool


@dataclass(frozen=True)
class AssetDiscoveryTask:
    """Request to find and hash image files in paths."""

    paths: list[str]
    supported_extensions: tuple[str, ...]
    rgb_scan: bool = False  # Group discovered files into R/G/B triplets (one asset per frame).
    restore_triplets: dict | None = None  # {red_path: [green, blue]} — rebuild known triplets (session restore).
    half_frame: bool = False  # Expand each file into two half-frame assets (left/right).


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
    green_path: str = ""  # RGB-scan triplet: green/blue exposures merged with file_path (red).
    blue_path: str = ""
    align: bool = True  # sub-pixel registration of the triplet


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
            # Splash shares the file's source_hash but is the embedded JPEG, not the linear
            # decode — isolate its cache identity so it can't leak into the real render.
            pipeline_source_hash = task.source_hash + ("\x00splash" if task.ephemeral else "")
            result, metrics = self._processor.run_pipeline(
                task.buffer,
                task.config,
                pipeline_source_hash,
                render_size_ref=task.preview_size,
                prefer_gpu=task.gpu_enabled,
                readback_metrics=task.readback_metrics,
                ir_buffer=task.ir_buffer,
                crop_preview_full=task.crop_preview_full,
            )

            soft_proof = task.icc_input_path or task.icc_output_path

            # CPU renders have no in-shader histogram; bin the float output here,
            # before soft-proofing quantizes it to 8/16-bit (comb artifacts).
            if task.readback_metrics and "histogram_raw" not in metrics and isinstance(result, np.ndarray):
                metrics["histogram_raw"] = output_histogram(result)

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
            # Render identity, so the controller can reject stale/ephemeral bounds writeback.
            metrics["source_hash"] = task.source_hash
            metrics["ephemeral"] = task.ephemeral
            metrics["memo_key"] = task.memo_key

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
            store = self._store if task.persist else None
            thumb = get_rendered_thumbnail(buf, task.file_hash, store, color_space=task.color_space)
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
        # Half-frame re-discovery passes both halves' (identical) paths — hash once.
        discovered_paths = list(dict.fromkeys(discovered_paths))

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

        if task.restore_triplets:
            valid_assets = self._attach_restored_triplets(valid_assets, task.restore_triplets)
        elif task.rgb_scan and valid_assets:
            valid_assets = self._group_rgb_triplets(valid_assets)

        if task.half_frame and valid_assets:
            valid_assets = self._expand_half_frames(valid_assets)

        self.finished.emit(valid_assets)

    def _expand_half_frames(self, assets: list) -> list:
        """Expand each file into two half-frame assets sharing the path, with
        per-half hash/name identities. Triplet assets stay whole (unsupported combo)."""
        from negpy.services.assets.half_frame import detect_split_x_for_file, half_hash, half_name

        out = []
        for i, a in enumerate(assets):
            if a.get("green_path"):
                out.append(a)
                continue
            self.progress.emit(i + 1, len(assets), f"Split {a['name']}")
            split_x = detect_split_x_for_file(a["path"])
            for half in (1, 2):
                out.append({**a, "name": half_name(a["name"], half), "hash": half_hash(a["hash"], half), "half": half, "split_x": split_x})
        return out

    def _attach_restored_triplets(self, assets: list, triplets: dict) -> list:
        """Re-attach saved green/blue exposures to restored red assets (no reclassification)."""
        import os

        out = []
        for a in assets:
            gb = triplets.get(a["path"])
            if gb and gb[0] and gb[1] and os.path.exists(gb[0]) and os.path.exists(gb[1]):
                base = os.path.splitext(a["name"])[0]
                align = bool(gb[2]) if len(gb) > 2 else True
                out.append({**a, "name": f"{base} (RGB)", "green_path": gb[0], "blue_path": gb[1], "align": align})
            else:
                out.append(a)
        return out

    def _group_rgb_triplets(self, assets: list) -> list:
        """Classify each file by dominant channel and merge consecutive R/G/B triplets
        into one asset (red is primary; green/blue ride along). Unmatched files stay individual."""
        import os

        from negpy.features.rgbscan.logic import classify_channel, group_triplets, probe_channel_means

        by_path = {a["path"]: a for a in assets}
        ordered = sorted(by_path, key=lambda p: os.path.basename(p).lower())

        items = []
        for i, p in enumerate(ordered):
            self.progress.emit(i + 1, len(ordered), f"RGB {os.path.basename(p)}")
            try:
                items.append((p, classify_channel(probe_channel_means(p))))
            except Exception as e:
                logger.error(f"RGB-scan classification failed for {p}: {e}")

        result = []
        grouped = set()
        for t in group_triplets(items):
            if not t.ok:
                continue
            red = by_path[t.red]
            base = os.path.splitext(red["name"])[0]
            result.append({**red, "name": f"{base} (RGB)", "green_path": t.green, "blue_path": t.blue})
            grouped.update({t.red, t.green, t.blue})

        result.extend(by_path[p] for p in ordered if p not in grouped)
        return result


class PreviewLoadWorker(QObject):
    """
    Background worker for decoding RAW files into a linear preview buffer.
    Keeps the UI thread free during slow I/O and demosaicing.
    """

    # (file_path, raw, dims, source_cs, ir_preview, detected_mode)
    finished = pyqtSignal(str, object, object, str, object, str)
    splash = pyqtSignal(str, object, object)  # (file_path, buffer, dims) — first paint
    error = pyqtSignal(str)
    # (file_path, message) — error carries no path, so badge attribution needs this
    load_failed = pyqtSignal(str, str)

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
            if task.green_path and task.blue_path:
                # RGB-scan triplet: assemble the frame from the three exposures.
                # No splash — the red embedded JPEG would flash a red-cast preview.
                raw, dims, metadata = self._preview_service.load_linear_preview_rgb(
                    task.file_path,
                    task.green_path,
                    task.blue_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                    align=task.align,
                )
                source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
                ir_preview = metadata.get("ir_preview")
                detected_mode = self._detect_mode(task, raw) if task.detect_mode else ""
                logger.info(
                    "load-timing preview_worker_total %.0fms (rgb load->buffer) %s",
                    (time.perf_counter() - t0) * 1000,
                    task.file_path,
                )
                self.finished.emit(task.file_path, raw, dims, source_cs, ir_preview, detected_mode)
                return
            if task.use_splash and not task.full_resolution:
                # Open the file once; get splash + linear in a single pass.
                sp, (raw, dims, metadata) = self._preview_service.load_splash_and_linear(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                    log_timings=True,
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
                    log_timings=True,
                )
            source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
            ir_preview = metadata.get("ir_preview")
            detected_mode = self._detect_mode(task, raw) if task.detect_mode else ""
            logger.info(
                "load-timing preview_worker_total %.0fms (load->buffer) %s",
                (time.perf_counter() - t0) * 1000,
                task.file_path,
            )
            self.finished.emit(task.file_path, raw, dims, source_cs, ir_preview, detected_mode)
        except Exception as e:
            logger.exception(f"Asset load failed: {task.file_path}")
            self.error.emit(str(e))
            self.load_failed.emit(task.file_path, str(e))

    def _detect_mode(self, task: PreviewLoadTask, raw) -> str:
        """Classify film process mode; re-decode no-WB since the C41 mask is hidden by camera WB."""
        from negpy.features.process.logic import detect_process_mode

        t0 = time.perf_counter()
        try:
            if not task.use_camera_wb:
                scan = raw
            else:
                # Camera WB hides the C41 mask — re-decode no-WB (lean; detect downsamples).
                scan = self._preview_service.decode_for_detection(task.file_path)
            mode = str(detect_process_mode(scan))
            logger.info(
                "load-timing detect %.0fms mode=%s (re_decode=%s) %s",
                (time.perf_counter() - t0) * 1000,
                mode,
                task.use_camera_wb,
                task.file_path,
            )
            return mode
        except Exception:
            logger.exception(f"Process-mode detection failed: {task.file_path}")
            return ""


class BatchAutoCropWorker(QObject):
    """Decode, preprocess, and roll-calibrate visible frames off the UI thread."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)  # list[BatchAutoCropResult]
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, preview_service) -> None:
        super().__init__()
        self._preview_service = preview_service
        self._cancel_lock = threading.RLock()
        self._cancelled_generations: set[int] = set()
        self._active_generation: int | None = None

    def cancel(self, generation: int | None = None) -> None:
        """Cancel one queued/running generation without poisoning a later run."""
        with self._cancel_lock:
            target = self._active_generation if generation is None else generation
            self._cancelled_generations.add(0 if target is None else int(target))

    def _emit_cancelled_if_requested(self, generation: int) -> bool:
        with self._cancel_lock:
            if generation not in self._cancelled_generations:
                return False
            self._cancelled_generations.discard(generation)
            if self._active_generation == generation:
                self._active_generation = None
        self.cancelled.emit()
        return True

    def _emit_finished_unless_cancelled(self, generation: int, results: list[BatchAutoCropResult]) -> None:
        """Atomically choose the terminal signal for a generation."""
        with self._cancel_lock:
            if generation in self._cancelled_generations:
                self._cancelled_generations.discard(generation)
                if self._active_generation == generation:
                    self._active_generation = None
                cancelled = True
            else:
                if self._active_generation == generation:
                    self._active_generation = None
                cancelled = False
                self.finished.emit(results)
        if cancelled:
            self.cancelled.emit()

    def _decode(self, frame: BatchAutoCropInput, workspace_color_space: str) -> np.ndarray:
        from negpy.services.assets.half_frame import base_hash, slice_for_asset

        file_info = frame.file_info
        config = frame.config
        rgbscan = config.rgbscan
        common = {
            "use_camera_wb": not config.process.linear_raw,
            "full_resolution": False,
            "file_hash": base_hash(file_info.get("hash")),  # halves share one decode
        }
        if rgbscan.enabled and rgbscan.green_path and rgbscan.blue_path:
            raw, _, _ = self._preview_service.load_linear_preview_rgb(
                file_info["path"],
                rgbscan.green_path,
                rgbscan.blue_path,
                workspace_color_space,
                align=rgbscan.align,
                **common,
            )
        else:
            raw, _, _ = self._preview_service.load_linear_preview(
                file_info["path"],
                workspace_color_space,
                **common,
            )
        return slice_for_asset(raw, file_info)

    @pyqtSlot(BatchAutoCropTask)
    def process(self, task: BatchAutoCropTask) -> None:
        """Collect frame evidence sequentially, then resolve it as one roll."""
        generation = int(task.generation)
        with self._cancel_lock:
            self._active_generation = generation
        if self._emit_cancelled_if_requested(generation):
            return
        total = len(task.frames)
        evidence = []
        source_by_key: dict[str, BatchAutoCropInput] = {}

        try:
            for index, frame in enumerate(task.frames):
                if self._emit_cancelled_if_requested(generation):
                    return

                current = index + 1
                file_info = frame.file_info
                name = str(file_info.get("name") or file_info.get("path") or current)
                key = f"{index}:{file_info.get('hash', '')}"

                try:
                    raw = self._decode(frame, task.workspace_color_space)
                    if self._emit_cancelled_if_requested(generation):
                        return

                    config = frame.config
                    corrected = apply_flatfield(raw, config.flatfield)
                    if self._emit_cancelled_if_requested(generation):
                        return
                    detection_geometry = replace(
                        config.geometry,
                        manual_crop_rect=None,
                        auto_crop_enabled=False,
                        autocrop_offset=0,
                    )
                    context = PipelineContext(
                        original_size=(corrected.shape[1], corrected.shape[0]),
                        scale_factor=1.0,
                        process_mode=config.process.process_mode,
                    )
                    distortion_k1 = config.flatfield.k1 if config.flatfield.apply else 0.0
                    transformed = GeometryProcessor(detection_geometry, distortion_k1).process(corrected, context)
                    if self._emit_cancelled_if_requested(generation):
                        return
                    candidate = detect_crop_candidate(
                        key,
                        transformed,
                        target_ratio=config.geometry.autocrop_ratio,
                    )
                    if self._emit_cancelled_if_requested(generation):
                        return

                    evidence.append(candidate)
                    source_by_key[candidate.key] = frame
                except Exception:
                    if self._emit_cancelled_if_requested(generation):
                        return
                    logger.exception("Auto Crop All skipped failed frame %s", name)

                self.progress.emit(current, total, name)

            if self._emit_cancelled_if_requested(generation):
                return
            resolved = resolve_roll_crops(evidence)
            if self._emit_cancelled_if_requested(generation):
                return

            results: list[BatchAutoCropResult] = []
            for crop in resolved:
                source = source_by_key.get(crop.key)
                if source is None:
                    logger.warning("Auto Crop All ignored result with unknown key %s", crop.key)
                    continue
                results.append(
                    BatchAutoCropResult(
                        file_info=source.file_info,
                        fingerprint=source.fingerprint,
                        manual_crop_rect=crop.manual_crop_rect,
                        correction_angle=crop.correction_angle,
                        confidence=crop.confidence,
                        calibrated=crop.calibrated,
                    )
                )
            self._emit_finished_unless_cancelled(generation, results)
        except Exception as exc:
            if self._emit_cancelled_if_requested(generation):
                return
            with self._cancel_lock:
                if self._active_generation == generation:
                    self._active_generation = None
            logger.exception("Auto Crop All worker failure")
            self.error.emit(str(exc))


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
        from negpy.features.exposure.normalization import analyze_log_exposure_bounds, resolve_crosstalk_matrix
        from negpy.features.geometry.processor import GeometryProcessor
        from negpy.services.assets.half_frame import base_hash, slice_for_asset

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
                    # Roll-wide buffer / luma bounds from the current image — applied to every
                    # file so one slider setting drives the whole batch baseline.
                    analysis_buffer = task.override_analysis_buffer
                    luma_range_clip = task.override_luma_range_clip
                    color_range_clip = task.override_color_range_clip
                    process_mode = params.process.process_mode if params else DEFAULT_WORKSPACE_CONFIG.process.process_mode
                    e6_normalize = params.process.e6_normalize if params else DEFAULT_WORKSPACE_CONFIG.process.e6_normalize
                    geometry = params.geometry if params else DEFAULT_WORKSPACE_CONFIG.geometry
                    linear_raw = params.process.linear_raw if params else DEFAULT_WORKSPACE_CONFIG.process.linear_raw

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
                        base_hash(f_info.get("hash")),  # halves share one decode
                    )
                    raw = slice_for_asset(raw, f_info)

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
                        percentile_clip=luma_range_clip,
                        color_clip=color_range_clip,
                        unmix=resolve_crosstalk_matrix(task.override_crosstalk_strength, task.override_crosstalk_matrix),
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
