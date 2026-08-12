import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import output_histogram, proof_grid, rotate_grid, strip_mosaic
from negpy.features.flatfield.logic import apply_flatfield
from negpy.features.hdr.models import HdrConfig, hdr_active
from negpy.features.geometry.batch_autocrop import detect_crop_candidate, resolve_roll_crops
from negpy.features.process.sensor import apply_sensor_correction, effective_sensor_matrix
from negpy.features.process.logic import effective_linear_raw
from negpy.infrastructure.loaders.helpers import unsupported_raw_reason
from negpy.features.rgbscan.models import RgbScanConfig, is_rgb_triplet
from negpy.features.stitch.models import StitchConfig, stitch_active
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
    gpu_enabled: bool = True
    readback_metrics: bool = True
    ir_buffer: Optional[np.ndarray] = None
    # True while the crop tool is active: show the full uncropped frame instead of
    # the final crop.
    crop_preview_full: bool = False
    # Display-only first paint (embedded-JPEG splash): its analysis must not persist.
    ephemeral: bool = False
    # Identity of everything that shaped these pixels; non-empty makes the result
    # eligible for the navigate-back render memo (echoed in metrics).
    memo_key: str = ""
    # These pixels are the before/after baseline, not the edit. Echoed in metrics so
    # the BEFORE badge tracks what is painted, not the pending toggle.
    compare: bool = False
    # Produced while a gesture is live. Echoed in metrics so the UI thread can skip
    # what only has to be right once the gesture settles.
    interactive: bool = False
    # Only the controller knows whether the filmstrip is already current for this config.
    wants_thumbnail: bool = False
    # Decoder XYZ->camera matrix for this source; only the transparency transfer reads it.
    cam_xyz: Optional[list] = None
    # As-shot WB multipliers, needed only when the buffer was decoded without WB.
    camera_wb: Optional[list] = None


@dataclass(frozen=True)
class TestStripTask:
    """Request to print a proof mosaic off one frame: the density × grade strip or the color
    ring-around. `overrides` is one ExposureConfig field-override dict per patch, row-major
    over `grid`, unrotated — the worker assembles all four orientations."""

    buffer: np.ndarray
    config: WorkspaceConfig
    source_hash: str
    preview_size: float
    overrides: tuple
    grid: tuple
    gpu_enabled: bool = True
    ir_buffer: Optional[np.ndarray] = None
    # Decoder XYZ->camera matrix for this source; only the transparency transfer reads it.
    cam_xyz: Optional[list] = None
    # As-shot WB multipliers, needed only when the buffer was decoded without WB.
    camera_wb: Optional[list] = None


@dataclass(frozen=True)
class ThumbnailUpdateTask:
    """Request to update the filmstrip thumbnail from a rendered buffer."""

    file_hash: str  # asset_thumbnail_key — the filmstrip and the disk cache share it
    buffer: np.ndarray
    # Display-transform inputs, from AppController.display_transform_params — must be
    # the same triple the canvas used for this buffer or the thumbnail's color drifts.
    color_space: str = WORKING_COLOR_SPACE
    monitor_icc_bytes: Optional[bytes] = None
    proof: Optional[tuple] = None
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
    restore_stitches: dict | None = None  # {primary_path: {paths, transforms, canvas, sizes, hash}} (session restore).
    restore_hdr: dict | None = None  # {reference_path: {paths, ratios, align, hash}} (session restore).
    half_frame_profile: dict | None = None  # {crop_rect, split_x, gutter_thickness} override


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
    # The assembly configs travel whole rather than flattened into loose fields. They are
    # frozen and hashable, the worker rebuilt them from the pieces anyway, and a field
    # added to one of them then needs no change here or at the call site.
    rgbscan: RgbScanConfig = RgbScanConfig()  # triplet: green/blue exposures merged with file_path (red)
    stitch: StitchConfig = StitchConfig()  # composite: non-primary parts + stored registration
    hdr: HdrConfig = HdrConfig()  # bracket: the other exposures, merged with file_path (the reference)
    flatfield_profile_id: str = ""  # per-part flat-field profile for stitch previews
    half_slice: tuple[int, float, tuple[float, float, float, float] | None, float] | None = (
        None  # (half, split_x, crop_rect, gutter_thickness)
    )


class RenderWorker(QObject):
    """
    Background rendering worker.
    Decouples engine execution from the UI thread to maintain 60FPS interaction.
    """

    finished = pyqtSignal(object, dict)  # (ndarray|GPUTexture, metrics)
    metrics_updated = pyqtSignal(dict)  # Late-arriving metrics (histogram, etc.)
    strip_finished = pyqtSignal(object, object)  # (one mosaic ndarray per quarter-turn, content_rect|None)
    strip_progress = pyqtSignal(int, int)  # (patches printed, total)
    busy = pyqtSignal(str)  # a slow uncached pipeline step is starting
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()
        self._processor.on_slow_step = self.busy.emit

    @property
    def processor(self) -> ImageProcessor:
        return self._processor

    @pyqtSlot(object)
    def cleanup(self, retain: object = None) -> None:
        """Evacuates transient GPU resources; ``retain`` is handed to its new owner."""
        self._processor.cleanup(retain=retain)

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
                cam_xyz=task.cam_xyz,
                camera_wb=task.camera_wb,
            )

            # CPU renders have no in-shader histogram; bin the float output here.
            if task.readback_metrics and "histogram_raw" not in metrics and isinstance(result, np.ndarray):
                metrics["histogram_raw"] = output_histogram(result)

            # The soft proof is not baked in: it rides the display LUT, so a GPU
            # texture reaches the canvas shader without a readback.

            # Taken here because the engine recycles its stage textures next frame.
            # Always assigned: the controller merges metrics into a running dict, so a
            # stale entry would be filed under this asset's key.
            metrics["thumbnail_source"] = (
                np.ascontiguousarray(result.readback()[:, :, :3]) if task.wants_thumbnail and isinstance(result, GPUTexture) else None
            )

            # Ensure ground truth is stored in metrics for view consumption
            metrics["base_positive"] = result
            # Render identity, so the controller can reject stale/ephemeral bounds writeback.
            metrics["source_hash"] = task.source_hash
            metrics["ephemeral"] = task.ephemeral
            metrics["memo_key"] = task.memo_key
            metrics["compare"] = task.compare
            metrics["interactive"] = task.interactive

            self.finished.emit(result, metrics)
            self.metrics_updated.emit(metrics)

        except Exception as e:
            logger.exception("Render pipeline failed")
            self.error.emit(str(e))

    @pyqtSlot(TestStripTask)
    def build_strip(self, task: TestStripTask) -> None:
        """Render the frame once per patch and keep only that patch.

        Runs on the render thread with the canvas's own ImageProcessor, so the patches are the
        pixels the canvas would show. Every field a proof overrides (density/grade, or the
        color head's magenta/yellow) is absent from the analysis cache key, so the per-frame
        metering is reused and only the exposure stage onward re-dispatches. Metrics are
        dropped: a proof must not disturb the writeback the real render owns.
        """
        try:
            tiles = []
            content_rect = None
            for override in task.overrides:
                config = replace(task.config, exposure=replace(task.config.exposure, **override))
                result, metrics = self._processor.run_pipeline(
                    task.buffer,
                    config,
                    task.source_hash,
                    render_size_ref=task.preview_size,
                    prefer_gpu=task.gpu_enabled,
                    readback_metrics=False,
                    ir_buffer=task.ir_buffer,
                    wants_uv_grid=False,
                    cam_xyz=task.cam_xyz,
                    camera_wb=task.camera_wb,
                )
                if isinstance(result, GPUTexture):
                    result = result.readback()
                # GPU readback is rgba32float; ImageConverter.to_qimage assumes RGB888 (w*3).
                if isinstance(result, np.ndarray) and result.ndim == 3 and result.shape[2] >= 4:
                    result = np.ascontiguousarray(result[:, :, :3])
                tiles.append(result)
                if content_rect is None:
                    content_rect = metrics.get("content_rect")
                self.strip_progress.emit(len(tiles), len(task.overrides))

            # One mosaic per quarter-turn while the tiles are still in hand: a rotated ladder needs
            # a different slice of each render. Peak memory is unchanged, the tiles dominate.
            mosaics = tuple(strip_mosaic(rotate_grid(tiles, task.grid, k), proof_grid(task.grid, k)) for k in range(4))
            # Unproofed like every other rendered buffer; the overlay proofs the mosaic
            # through the same display LUT the canvas uses.
            self.strip_finished.emit(mosaics, content_rect)

        except Exception as e:
            logger.exception("Test strip render failed")
            self.error.emit(str(e))


_THUMB_CHUNK = 8


class ThumbnailWorker(QObject):
    """
    Asynchronous thumbnail generation worker.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    # Chunks of the running batch, so a large folder fills its filmstrip as it goes
    # instead of staying blank until the last file lands.
    partial = pyqtSignal(dict)
    # Rendered positives use their own signal so the batch's bulk overwrite can't
    # clobber a frame that already rendered on the canvas.
    rendered_finished = pyqtSignal(dict)
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

            # Chunked, not per-file: every emit costs the model a full relayout.
            pending: dict = {}

            def _ready_callback(key: str, thumb) -> None:
                pending[key] = thumb
                if len(pending) >= _THUMB_CHUNK:
                    self.partial.emit(dict(pending))
                    pending.clear()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                new_thumbs = loop.run_until_complete(
                    thumb_service.generate_batch_thumbnails(
                        files,
                        self._store,
                        progress_callback=_progress_callback,
                        ready_callback=_ready_callback,
                    )
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
            thumb = get_rendered_thumbnail(
                buf,
                task.file_hash,
                store,
                color_space=task.color_space,
                monitor_icc_bytes=task.monitor_icc_bytes,
                proof=task.proof,
            )
            if thumb:
                self.rendered_finished.emit({task.file_hash: thumb})
        except Exception as e:
            logger.error(f"Thumbnail update failure: {e}")


def _safe_call(fn: Callable[[str], Any], path: str) -> Any:
    """``fn(path)`` or None — a bad file is skipped, the pass keeps going."""
    try:
        return fn(path)
    except Exception as e:
        logger.error(f"Skipping invalid file {path}: {e}")
        return None


# Seek-bound: each hash costs 18 seeks, so cpu_count() concurrent readers thrash a spinning disk.
_HASH_WORKERS = min(8, APP_CONFIG.max_workers)
# Real decodes; halved for memory headroom, as NormalizationWorker does.
_DECODE_WORKERS = max(1, APP_CONFIG.max_workers // 2)


class AssetDiscoveryWorker(QObject):
    """
    Background worker for file system crawling and hashing.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def _map_files(
        self,
        paths: list[str],
        fn: Callable[[str], Any],
        label: Callable[[str], str],
        workers: int,
    ) -> list[Any]:
        """Run an expensive per-file pass in parallel, results in input order.

        Order is load-bearing: it becomes the filmstrip order. Progress counts
        completions, so it advances out of order — which is what a bar wants.
        """
        total = len(paths)
        if total < 2 or workers < 2:
            out = []
            for i, path in enumerate(paths):
                self.progress.emit(i + 1, total, label(path))
                out.append(_safe_call(fn, path))
            return out

        results: list[Any] = [None] * total
        with ThreadPoolExecutor(max_workers=min(workers, total)) as ex:
            futures = {ex.submit(_safe_call, fn, path): i for i, path in enumerate(paths)}
            for done, fut in enumerate(as_completed(futures), 1):
                i = futures[fut]
                results[i] = fut.result()
                self.progress.emit(done, total, label(paths[i]))
        return results

    @pyqtSlot(AssetDiscoveryTask)
    def process(self, task: AssetDiscoveryTask) -> None:
        """
        Scans paths for supported images and calculates hashes.
        """
        import os

        from negpy.infrastructure.loaders.constants import is_ir_sidecar_path
        from negpy.kernel.image.logic import file_hashes
        from negpy.services.assets.hash_migration import blank_ambiguous_legacy_hashes

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
        # IR companions ride along with their main TIFF; they are never assets of their own.
        discovered_paths = [p for p in discovered_paths if not is_ir_sidecar_path(p)]

        valid_assets = []
        digests = self._map_files(discovered_paths, file_hashes, os.path.basename, _HASH_WORKERS)

        for path, digest in zip(discovered_paths, digests):
            if digest is None:
                continue
            f_hash, legacy = digest
            if not f_hash.startswith("err_"):
                # Stamped once here so sorting and date search never stat per row.
                valid_assets.append(
                    {
                        "name": os.path.basename(path),
                        "path": path,
                        "hash": f_hash,
                        "legacy_hash": legacy,
                        "mtime": os.path.getmtime(path),
                    }
                )

        blank_ambiguous_legacy_hashes(valid_assets)

        if task.restore_triplets:
            valid_assets = self._attach_restored_triplets(valid_assets, task.restore_triplets)
        elif task.rgb_scan and valid_assets:
            valid_assets = self._group_rgb_triplets(valid_assets)

        if task.restore_stitches and valid_assets:
            valid_assets = self._attach_restored_stitches(valid_assets, task.restore_stitches)

        if task.restore_hdr and valid_assets:
            valid_assets = self._attach_restored_hdr(valid_assets, task.restore_hdr)

        if task.half_frame and valid_assets:
            valid_assets = self._expand_half_frames(valid_assets, profile=task.half_frame_profile)

        self.finished.emit(valid_assets)

    def _expand_half_frames(self, assets: list, profile: dict | None = None) -> list:
        """Expand each file into two half-frame assets sharing the path, with
        per-half hash/name identities. Composite assets (triplet, stitch, HDR) stay
        whole — an unsupported combination.

        When ``profile`` is set (a {crop_rect, split_x, gutter_thickness} dict saved
        from the half-frame rectangle editor), it overrides the auto-detected split
        and adds the crop rect + gutter to every expanded half.
        """
        import os

        from negpy.services.assets.half_frame import detect_split_x_for_file, half_hash, half_name

        def _splittable(a: dict) -> bool:
            return not (a.get("green_path") or a.get("stitch_paths") or a.get("hdr_paths"))

        if profile is None:
            paths = [a["path"] for a in assets if _splittable(a)]
            detected = self._map_files(paths, detect_split_x_for_file, lambda p: f"Split {os.path.basename(p)}", _DECODE_WORKERS)
            splits = dict(zip(paths, detected))
        else:
            splits = {}

        out = []
        for a in assets:
            if not _splittable(a):
                out.append(a)
                continue
            if profile is not None:
                split_x = float(profile.get("split_x") or 0.5)
            else:
                detected_x = splits.get(a["path"])
                split_x = 0.5 if detected_x is None else float(detected_x)
            legacy = a.get("legacy_hash")
            for half in (1, 2):
                entry = {
                    **a,
                    "name": half_name(a["name"], half),
                    "hash": half_hash(a["hash"], half),
                    "legacy_hash": half_hash(legacy, half) if legacy else "",
                    "half": half,
                    "split_x": split_x,
                }
                if profile is not None:
                    cr = profile.get("crop_rect")
                    if cr is not None:
                        entry["crop_rect"] = tuple(cr)
                    entry["gutter_thickness"] = float(profile.get("gutter_thickness") or 0.0)
                out.append(entry)
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

    def _attach_restored_stitches(self, assets: list, stitches: dict) -> list:
        """Re-attach saved stitch registrations to restored primary assets (no re-registration).
        A composite whose parts vanished from disk restores as a plain asset."""
        import os

        from negpy.features.stitch.models import stitch_name

        out = []
        for a in assets:
            entry = stitches.get(a["path"])
            # Triplet exposures count as parts — one missing decodes that part red-only.
            needed = [*(entry.get("paths") or ()), *(p for t in entry.get("triplets") or () for p in t if p)] if entry else []
            if entry and entry.get("paths") and all(os.path.exists(p) for p in needed):
                out.append(
                    {
                        **a,
                        "name": stitch_name([a["path"], *entry["paths"]]),
                        "hash": entry["hash"],
                        # The composite hash is the parts' — inheriting the primary part's
                        # legacy digest would rehome that part's edit onto the composite.
                        "legacy_hash": "",
                        "stitch_paths": tuple(entry["paths"]),
                        "stitch_transforms": tuple(tuple(float(v) for v in t) for t in entry["transforms"]),
                        "stitch_canvas": (int(entry["canvas"][0]), int(entry["canvas"][1])),
                        "stitch_sizes": tuple((int(s[0]), int(s[1])) for s in entry["sizes"]),
                        "stitch_triplets": tuple((str(t[0]), str(t[1])) for t in entry.get("triplets") or ()),
                        "stitch_align": bool(entry.get("align", True)),
                        "process_mode": entry.get("process_mode", ""),
                    }
                )
            else:
                out.append(a)
        return out

    def _attach_restored_hdr(self, assets: list, merges: dict) -> list:
        """Re-attach saved brackets to restored reference assets (no re-solve). A merge whose
        exposures vanished from disk restores as a plain asset."""
        import os

        from negpy.features.hdr.models import hdr_name

        out = []
        for a in assets:
            entry = merges.get(a["path"])
            if entry and entry.get("paths") and all(os.path.exists(p) for p in entry["paths"]):
                out.append(
                    {
                        **a,
                        "name": hdr_name([a["path"], *entry["paths"]]),
                        "hash": entry["hash"],
                        # The composite hash is the bracket's — inheriting the reference
                        # frame's legacy digest would rehome that frame's edit onto it.
                        "legacy_hash": "",
                        "hdr_paths": tuple(entry["paths"]),
                        "hdr_ratios": tuple(float(r) for r in entry.get("ratios") or ()),
                        "hdr_align": bool(entry.get("align", True)),
                        "hdr_anchor": str(entry.get("anchor", "") or ""),
                        "hdr_anchor_ev": float(entry.get("anchor_ev", 1.0)),
                    }
                )
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

        means = self._map_files(ordered, probe_channel_means, lambda p: f"RGB {os.path.basename(p)}", _DECODE_WORKERS)
        items = [(p, classify_channel(m)) for p, m in zip(ordered, means) if m is not None]

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

    # (file_path, raw, dims, source_cs, ir_preview, detected_mode, (cam_xyz, camera_wb))
    finished = pyqtSignal(str, object, object, str, object, str, object)
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
                    half_slice=task.half_slice,
                )
            except Exception as e:
                logger.debug("Preview cache warm failed for %s: %s", task.file_path, e)
            return
        t0 = time.perf_counter()
        try:
            if stitch_active(task.stitch):
                # Stitch composite: replay the stored registration at preview scale.
                # No splash — the primary's embedded JPEG would flash a half frame.
                raw, dims, metadata = self._preview_service.load_linear_preview_stitch(
                    task.file_path,
                    task.stitch,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                    flatfield_profile_id=task.flatfield_profile_id,
                )
                source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
                ir_preview = metadata.get("ir_preview")
                detected_mode = self._detect_mode(task, raw) if task.detect_mode else ""
                logger.info(
                    "load-timing preview_worker_total %.0fms (stitch load->buffer) %s",
                    (time.perf_counter() - t0) * 1000,
                    task.file_path,
                )
                self.finished.emit(
                    task.file_path, raw, dims, source_cs, ir_preview, detected_mode, (metadata.get("cam_xyz"), metadata.get("camera_wb"))
                )
                return
            if hdr_active(task.hdr):
                # Bracketed capture: merge the exposures into one linear source. No splash —
                # the reference frame's embedded JPEG would flash the unmerged exposure.
                raw, dims, metadata = self._preview_service.load_linear_preview_hdr(
                    task.file_path,
                    task.hdr,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                )
                source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
                ir_preview = metadata.get("ir_preview")
                detected_mode = self._detect_mode(task, raw) if task.detect_mode else ""
                logger.info(
                    "load-timing preview_worker_total %.0fms (hdr load->buffer) %s",
                    (time.perf_counter() - t0) * 1000,
                    task.file_path,
                )
                self.finished.emit(
                    task.file_path, raw, dims, source_cs, ir_preview, detected_mode, (metadata.get("cam_xyz"), metadata.get("camera_wb"))
                )
                return
            if is_rgb_triplet(task.rgbscan):
                # RGB-scan triplet: assemble the frame from the three exposures.
                # No splash — the red embedded JPEG would flash a red-cast preview.
                raw, dims, metadata = self._preview_service.load_linear_preview_rgb(
                    task.file_path,
                    task.rgbscan,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                )
                source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
                ir_preview = metadata.get("ir_preview")
                detected_mode = self._detect_mode(task, raw) if task.detect_mode else ""
                logger.info(
                    "load-timing preview_worker_total %.0fms (rgb load->buffer) %s",
                    (time.perf_counter() - t0) * 1000,
                    task.file_path,
                )
                self.finished.emit(
                    task.file_path, raw, dims, source_cs, ir_preview, detected_mode, (metadata.get("cam_xyz"), metadata.get("camera_wb"))
                )
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
                    half_slice=task.half_slice,
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
                    half_slice=task.half_slice,
                )
            source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
            ir_preview = metadata.get("ir_preview")
            detected_mode = self._detect_mode(task, raw) if task.detect_mode else ""
            logger.info(
                "load-timing preview_worker_total %.0fms (load->buffer) %s",
                (time.perf_counter() - t0) * 1000,
                task.file_path,
            )
            self.finished.emit(
                task.file_path, raw, dims, source_cs, ir_preview, detected_mode, (metadata.get("cam_xyz"), metadata.get("camera_wb"))
            )
        except Exception as e:
            logger.exception(f"Asset load failed: {task.file_path}")
            # libraw reports "Unsupported file format or not RAW file" for a file whose
            # tags it parsed perfectly and whose payload it cannot decode, which reads as
            # "your NEF is broken". Ask why only once the decode has actually failed, so
            # the check costs nothing on the files that work.
            message = unsupported_raw_reason(task.file_path) or str(e)
            self.error.emit(message)
            self.load_failed.emit(task.file_path, message)

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
            "use_camera_wb": not effective_linear_raw(config.process, config.exposure.render_intent),
            "full_resolution": False,
            "file_hash": base_hash(file_info.get("hash")),  # halves share one decode
        }
        hdr = config.hdr
        if hdr.hdr_enabled and hdr.hdr_paths:
            raw, _, _ = self._preview_service.load_linear_preview_hdr(file_info["path"], hdr, workspace_color_space, **common)
        elif rgbscan.enabled and rgbscan.green_path and rgbscan.blue_path:
            raw, _, _ = self._preview_service.load_linear_preview_rgb(file_info["path"], rgbscan, workspace_color_space, **common)
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
                        rebate_trim=config.geometry.autocrop_rebate_trim,
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
                    # effective_, not the stored flag: the transfer path decodes neutral whatever
                    # the flag says, and the comment below is the reason this has to match it.
                    _an = params if params else DEFAULT_WORKSPACE_CONFIG
                    linear_raw = effective_linear_raw(_an.process, _an.exposure.render_intent)

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
                    # Bounds must be measured on the same channel mix the render path
                    # normalizes (triplet composites are never sensor-corrected there).
                    sensor_matrix = effective_sensor_matrix(params.process) if params is not None else None
                    if sensor_matrix is not None and not is_rgb_triplet(params.rgbscan):
                        raw = await asyncio.to_thread(apply_sensor_correction, raw, sensor_matrix)

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
