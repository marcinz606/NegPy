import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Optional

import negpy_core
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import (
    color_histogram,
    output_histogram,
    proof_grid,
    rotate_grid,
    strip_mosaic,
    vectorscope_histogram,
    waveform,
    waveform_color,
    waveform_parade,
)
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.lens.models import LensCorrections
from negpy.services.rendering.lens import flatfield_source, lens_decode_token, metadata_lens_corrections
from negpy.features.hdr.models import HdrConfig, hdr_active
from negpy.features.geometry.batch_autocrop import CropEvidence, detect_crop_candidate, resolve_roll_crops
from negpy.features.process.capture_color import wb_only_cam_xyz
from negpy.features.process.sensor import apply_sensor_correction, effective_sensor_matrix
from negpy.features.process.logic import effective_highlight_reconstruction, effective_linear_raw, highlight_reconstruction_bakes_wb
from negpy.features.process.path import positive_decode
from negpy.features.process.models import DemosaicMode
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import unsupported_raw_reason
from negpy.infrastructure.loaders.raw_shots import thumbnail_shot
from negpy.features.rgbscan.models import RgbScanConfig, is_rgb_triplet
from negpy.features.stitch.models import StitchConfig, stitch_active
from negpy.features.geometry.models import squares_on_arm
from negpy.features.geometry.processor import GeometryProcessor
from negpy.features.geometry.skew import squared_geometry
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.memory import available_system_memory_bytes
from negpy.kernel.system import timing
from negpy.services.rendering.image_processor import ImageProcessor, preview_takes_unmix, read_source_base
from negpy.services.rendering.prefetch_policy import MIN_RAM_RESERVE_BYTES

logger = get_logger(__name__)


class _DecodeGate:
    """One native decode at a time, so codec buffers never overlap. A foreground preview may run
    beside a prefetch of another file: LibRaw's unpack cannot stop mid-read. Each kind is one thread."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._holders: dict[str, str] = {}

    def _may_enter(self, kind: str, path: str) -> bool:
        if not self._holders:
            return True
        if kind == "foreground" and set(self._holders) == {"prefetch"}:
            return self._holders["prefetch"] != path
        return False

    @contextmanager
    def hold(self, kind: str, path: str = "", abandoned: Optional[Callable[[], bool]] = None) -> Iterator[bool]:
        """Yields whether the gate was taken; a wait gives up and yields False once *abandoned* is true."""
        acquired = False
        with self._cond:
            while not self._may_enter(kind, path):
                if abandoned is not None and abandoned():
                    break
                self._cond.wait(0.1 if abandoned is not None else None)
            else:
                self._holders[kind] = path
                acquired = True
        try:
            yield acquired
        finally:
            if acquired:
                with self._cond:
                    self._holders.pop(kind, None)
                    self._cond.notify_all()


_DECODE_MEMORY_GATE = _DecodeGate()


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
    detect_buffer: Optional[np.ndarray] = None
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
    # `config` is a view of the edit, not the edit, so the pixels get no render identity.
    config_override: bool = False
    # Decoder XYZ->camera matrix for this source; only the transparency transfer reads it.
    cam_xyz: Optional[list] = None
    # As-shot WB multipliers, needed only when the buffer was decoded without WB.
    camera_wb: Optional[list] = None
    # Half-frame diptych: `buffer` is the whole cropped scan and these two configs render
    # its halves, which are then joined. `config` is unused then — the halves own the edit.
    diptych: Optional[tuple[WorkspaceConfig, WorkspaceConfig]] = None
    split_x: float = 0.5
    split_axis: str = "x"
    gutter_thickness: float = 0.0
    # The slice_half cuts that took `buffer` out of the decoded frame (a half, or a
    # diptych's crop), for the flat-field gain to take too.
    gain_slices: tuple = ()
    # EXIF orientation the decode baked into `buffer`, for the flat-field gain to take first.
    source_orientation: int = 1
    # The roll's clear film base set with the Film Base tool, per-channel log10 transmittance;
    # None lets the rebate reader stand in.
    film_base: Optional[tuple] = None


@dataclass(frozen=True)
class TestStripTask:
    """Request to print a proof mosaic off one frame: the density × grade strip, the color
    ring-around or the paper strip. `overrides` is one ExposureConfig field-override dict per
    patch, row-major over `grid`, unrotated — the worker assembles all four orientations.
    `keep_tiles` also hands back every whole-frame render, which the paper strip shows full frame."""

    buffer: np.ndarray
    config: WorkspaceConfig
    source_hash: str
    preview_size: float
    overrides: tuple
    grid: tuple
    gpu_enabled: bool = True
    ir_buffer: Optional[np.ndarray] = None
    detect_buffer: Optional[np.ndarray] = None
    # Decoder XYZ->camera matrix for this source; only the transparency transfer reads it.
    cam_xyz: Optional[list] = None
    # As-shot WB multipliers, needed only when the buffer was decoded without WB.
    camera_wb: Optional[list] = None
    source_orientation: int = 1
    keep_tiles: bool = False


@dataclass(frozen=True)
class PickBetterTask:
    """Request to render the Pick Better proof as three frames: the committed current edit
    plus the two candidates the user is choosing between. `left_config` is the stronger
    candidate, `right_config` the weaker; `current_config` is the live committed edit shown as
    the reference above the pair. Each is rendered as a full frame, not joined."""

    buffer: np.ndarray
    source_hash: str
    preview_size: float
    current_config: WorkspaceConfig
    left_config: WorkspaceConfig
    right_config: WorkspaceConfig
    gpu_enabled: bool = True
    ir_buffer: Optional[np.ndarray] = None
    detect_buffer: Optional[np.ndarray] = None
    # Decoder XYZ->camera matrix for this source; only the transparency transfer reads it.
    cam_xyz: Optional[list] = None
    # As-shot WB multipliers, needed only when the buffer was decoded without WB.
    camera_wb: Optional[list] = None


@dataclass(frozen=True)
class ThumbnailUpdateTask:
    """Request to update the filmstrip thumbnail from a rendered buffer."""

    file_hash: str  # asset_thumbnail_key — the filmstrip and the disk cache share it
    buffer: np.ndarray
    # Display-transform inputs from AppController.display_transform_params. Must be the same
    # triple the canvas used for this buffer, or the thumbnail's color drifts.
    color_space: str = WORKING_COLOR_SPACE
    monitor_icc_bytes: Optional[bytes] = None
    proof: Optional[tuple] = None
    persist: bool = True  # False = in-memory filmstrip only, skip the disk JPEG encode.
    # The render fingerprint stored with the JPEG; None reads as stale.
    fingerprint: Optional[str] = None


@dataclass(frozen=True)
class NormalizationInput:
    """One frame and its dispatch-time settings for roll-wide bounds analysis."""

    file_info: dict
    config: WorkspaceConfig


@dataclass(frozen=True)
class NormalizationTask:
    """Request to analyze log bounds for a set of files."""

    frames: list[NormalizationInput]
    workspace_color_space: str
    # Roll-wide overrides taken from the current image, applied to every file's analysis
    # before averaging, so the whole roll shares one buffer and luma bounds. None reads each frame's own.
    override_analysis_buffer: Optional[float]
    override_luma_range_clip: Optional[float]
    override_color_range_clip: Optional[float]
    # Detect Scenes: the same per-frame pass, grouped by light instead of pooled into a baseline.
    detect: bool = False
    # One frame at a time on the background core pool, so the live preview keeps the CPU.
    background: bool = False


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
    crop_rect: tuple[float, float, float, float]
    correction_angle: float
    confidence: float
    calibrated: bool
    # Tilt and Swing Square Before Crop set before detection; None leaves the frame's own.
    keystone: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class ThumbnailRenderInput:
    """One frame and its post-write settings for a background thumbnail refresh."""

    file_info: dict
    config: WorkspaceConfig
    thumbnail_key: str
    # Whether an Input ICC applies to this frame — mirrors AppController.effective_input_icc,
    # resolved at dispatch since the worker has no session state to check it itself.
    icc_input_active: bool = False


@dataclass(frozen=True)
class ThumbnailRenderTask:
    """Request to re-render a set of frames' thumbnails off the live render path."""

    frames: list[ThumbnailRenderInput]
    workspace_color_space: str
    generation: int = 0


@dataclass(frozen=True)
class AssetDiscoveryTask:
    """Request to find and hash image files in paths."""

    paths: list[str]
    supported_extensions: tuple[str, ...]
    rgb_scan: bool = False  # Group discovered files into R/G/B triplets (one asset per frame).
    restore_triplets: dict | None = None  # {red_path: [green, blue, align, hashes?]}
    half_frame: bool = False  # Split each file into two half-frame assets.
    # Without a roll the choice is per request path (a file, or a folder for every file in it),
    # in os.path.normcase(os.path.normpath()) form; half_frame=True splits every file instead.
    half_frame_paths: frozenset = frozenset()
    restore_stitches: dict | None = None  # {primary_path: {paths, transforms, canvas, sizes, hash}} (session restore).
    restore_hdr: dict | None = None  # {reference_path: {paths, ratios, align, hash}} (session restore).
    half_frame_profile: dict | None = None  # {crop_rect, split_x, gutter_thickness, split_axis} override
    half_frame_overrides: dict | None = None  # {base_hash: {crop_rect, split_x, gutter_thickness, split_axis}} per-file overrides
    skip_paths: frozenset = frozenset()  # Files a folder walk leaves out, normcased: those removed from the roll.


@dataclass(frozen=True)
class AutoDetectAllSplitsTask:
    """Request to re-find the gutter on every given scan, off the GUI thread."""

    paths: list[str]


@dataclass(frozen=True)
class PreviewLoadTask:
    """Request to decode a RAW file into a linear preview buffer."""

    file_path: str
    workspace_color_space: str
    use_camera_wb: bool
    generation: int = 0
    positive_decode: str = ""
    highlight_mode: int = 0
    bake_camera_wb: bool = False
    full_resolution: bool = False
    file_hash: str | None = None
    use_splash: bool = True
    for_cache_warm: bool = False
    integrated_gpu: bool = False
    protected_file_hashes: tuple[str, ...] = ()
    detect_mode: bool = False  # run process-mode detection (a white-light capture of unknown film)
    detect_source: bool = False  # measure whether the scan is of color film (ProcessConfig.source_color)
    # The assembly configs travel whole rather than flattened into loose fields. They are
    # frozen and hashable, the worker rebuilt them from the pieces anyway, and a new field on
    # one of them then needs no change here or at the call site.
    rgbscan: RgbScanConfig = RgbScanConfig()  # triplet: green/blue exposures merged with file_path (red)
    stitch: StitchConfig = StitchConfig()  # composite: non-primary parts + stored registration
    hdr: HdrConfig = HdrConfig()  # bracket: the other exposures, merged with file_path (the reference)
    flatfield_profile_id: str = ""  # per-part flat-field profile for stitch previews
    lens_corrections: LensCorrections = LensCorrections()
    lens_flatfield: FlatFieldConfig = FlatFieldConfig()
    demosaic: str = DemosaicMode.AUTO  # CFA interpolation for the preview decode
    raw_shot: int = 0  # shot of a multi-shot RAW, 0 Auto (see loaders/raw_shots)
    half_slice: tuple[int, float, tuple[float, float, float, float] | None, float, str] | None = (
        None  # (half, split_x, crop_rect, gutter_thickness, split_axis)
    )
    # A cache warm also reads the frame's clear base with this config; None skips the read.
    base_settings: Optional[WorkspaceConfig] = None


class RenderWorker(QObject):
    """
    Background rendering worker.
    Decouples engine execution from the UI thread to maintain 60FPS interaction.
    """

    finished = pyqtSignal(object, dict)  # (ndarray|GPUTexture, metrics)
    metrics_updated = pyqtSignal(dict)  # Late-arriving metrics (histogram, etc.)
    strip_finished = pyqtSignal(object, object, object)  # (one mosaic ndarray per quarter-turn, content_rect|None, tiles|None)
    strip_progress = pyqtSignal(int, int)  # (patches printed, total)
    pick_better_finished = pyqtSignal(object, object, object)  # (current, left, right ndarrays)
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
        """Evacuates transient GPU resources; ``retain`` is handed to its new owner.
        No gc.collect: the pool eviction frees the textures and a full collection per file
        switch is dead time on the navigation path."""
        self._processor.cleanup(collect=False, retain=retain)

    def destroy_all(self) -> None:
        """Full teardown of processing resources."""
        self._processor.destroy_all()

    def _render_diptych(self, task: RenderTask, pipeline_source_hash: str) -> tuple[np.ndarray, dict]:
        """Render the scan's two halves with their own configs and join them.

        Each half is sliced before the pipeline, so its normalization sees the pixels it
        was edited on — the same reason the half-frame preview slices pre-downsample. The
        halves get distinct pipeline hashes or they share the stage cache and the second
        render comes back as the first. Metrics are half 1's, marked so the controller
        does not write those bounds back to the whole-frame edit.
        """
        from negpy.services.assets.half_frame import gap_px, half_hash, join_halves, slice_half

        assert task.diptych is not None
        rendered = []
        for n, config in ((1, task.diptych[0]), (2, task.diptych[1])):
            buffer = np.ascontiguousarray(
                slice_half(task.buffer, n, task.split_x, gutter_thickness=task.gutter_thickness, split_axis=task.split_axis)
            )
            ir = None
            if task.ir_buffer is not None:
                ir = np.ascontiguousarray(
                    slice_half(task.ir_buffer, n, task.split_x, gutter_thickness=task.gutter_thickness, split_axis=task.split_axis)
                )
            detect = None
            if task.detect_buffer is not None:
                detect = np.ascontiguousarray(
                    slice_half(task.detect_buffer, n, task.split_x, gutter_thickness=task.gutter_thickness, split_axis=task.split_axis)
                )
            out, metrics = self._processor.run_pipeline(
                buffer,
                config,
                half_hash(pipeline_source_hash, n),
                render_size_ref=task.preview_size,
                prefer_gpu=task.gpu_enabled,
                readback_metrics=task.readback_metrics and n == 1,
                ir_buffer=ir,
                detect_buffer=detect,
                crop_preview_full=task.crop_preview_full,
                cam_xyz=task.cam_xyz,
                camera_wb=task.camera_wb,
                film_base=task.film_base,
                gain_slices=task.gain_slices + ((n, task.split_x, None, task.gutter_thickness, task.split_axis),),
                source_orientation=task.source_orientation,
            )
            if isinstance(out, GPUTexture):
                out = np.ascontiguousarray(out.readback()[:, :, :3])
            rendered.append((out, metrics))

        (left, metrics), (right, _) = rendered
        metrics["diptych"] = True
        # Half 1's GPU histogram describes half 1; let `process` bin the joined image instead.
        metrics.pop("histogram_raw", None)
        metrics.pop("waveform", None)
        metrics.pop("waveform_color", None)
        metrics.pop("waveform_parade", None)
        metrics.pop("vectorscope", None)
        # No joined scan to rebin; None, not a pop, so the merged metrics drop the last frame's.
        metrics["density_parade"] = None
        metrics["scan_full_scale"] = None
        along = 1 if task.split_axis == "x" else 0
        gap = gap_px(left.shape[along], right.shape[along], task.gutter_thickness)
        return join_halves(left, right, gap, axis=task.split_axis), metrics

    @pyqtSlot(RenderTask)
    def process(self, task: RenderTask) -> None:
        """Executes the rendering pipeline for a single frame."""
        try:
            # The splash shares the file's source_hash but is the embedded JPEG, not the linear
            # decode, so isolate its cache identity and it cannot leak into the real render.
            with timing.collect() as times:
                pipeline_source_hash = task.source_hash + ("\x00splash" if task.ephemeral else "")
                if task.diptych is not None:
                    result, metrics = self._render_diptych(task, pipeline_source_hash)
                else:
                    result, metrics = self._processor.run_pipeline(
                        task.buffer,
                        task.config,
                        pipeline_source_hash,
                        render_size_ref=task.preview_size,
                        prefer_gpu=task.gpu_enabled,
                        readback_metrics=task.readback_metrics,
                        ir_buffer=task.ir_buffer,
                        detect_buffer=task.detect_buffer,
                        crop_preview_full=task.crop_preview_full,
                        cam_xyz=task.cam_xyz,
                        camera_wb=task.camera_wb,
                        gain_slices=task.gain_slices,
                        source_orientation=task.source_orientation,
                        film_base=task.film_base,
                    )

                # CPU renders have no in-shader histogram; bin the float output here.
                if task.readback_metrics and "histogram_raw" not in metrics and isinstance(result, np.ndarray):
                    metrics["histogram_raw"] = output_histogram(result)
                    metrics["histogram_color"] = color_histogram(result)
                    metrics["waveform"] = waveform(result)
                    metrics["waveform_color"] = waveform_color(result)
                    metrics["waveform_parade"] = waveform_parade(result)
                    metrics["vectorscope"] = vectorscope_histogram(result)
                timing.lap("histograms")

                # The soft proof is not baked in: it rides the display LUT, so a GPU texture reaches
                # the canvas shader without a readback.

                # Taken here because the engine recycles its stage textures next frame. Always
                # assigned: the controller merges metrics into a running dict, so a stale entry
                # would be filed under this asset's key.
                metrics["thumbnail_source"] = (
                    np.ascontiguousarray(result.readback()[:, :, :3]) if task.wants_thumbnail and isinstance(result, GPUTexture) else None
                )
                timing.lap("thumbnail readback")
            h, w = task.buffer.shape[:2] if isinstance(task.buffer, np.ndarray) else (0, 0)
            # A drag renders many frames a second, so only a settled render logs at info.
            log = logger.info if task.readback_metrics and not task.interactive else logger.debug
            log("render-timing %s | %dx%d %s", timing.summary(times), w, h, task.source_hash[:8])

            # Ensure ground truth is stored in metrics for view consumption
            metrics["base_positive"] = result
            # Buffer resolution this frame rendered at, so the canvas reports zoom against
            # source pixels rather than against whichever proxy the pipeline was handed.
            metrics["render_long_edge"] = int(max(task.buffer.shape[:2])) if isinstance(task.buffer, np.ndarray) else 0
            # Render identity, so the controller can reject stale/ephemeral bounds writeback.
            metrics["source_hash"] = task.source_hash
            # The config these pixels ran on, so the thumbnail fingerprints what ran, not a later edit.
            # A crop-tool render shows the uncropped frame, which no config describes.
            metrics["render_identity"] = None if task.crop_preview_full or task.config_override else (task.source_hash, task.config)
            metrics["ephemeral"] = task.ephemeral
            metrics["memo_key"] = task.memo_key
            metrics["compare"] = task.compare
            metrics["interactive"] = task.interactive
            metrics["crop_preview_full"] = task.crop_preview_full
            metrics["config_override"] = task.config_override

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
                    detect_buffer=task.detect_buffer,
                    wants_uv_grid=False,
                    cam_xyz=task.cam_xyz,
                    camera_wb=task.camera_wb,
                    source_orientation=task.source_orientation,
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

            # One mosaic per quarter-turn while the tiles are still in hand: a rotated ladder
            # needs a different slice of each render. Peak memory is unchanged, since the tiles
            # dominate.
            mosaics = tuple(strip_mosaic(rotate_grid(tiles, task.grid, k), proof_grid(task.grid, k)) for k in range(4))
            # Unproofed like every other rendered buffer. The overlay proofs the mosaic through
            # the same display LUT the canvas uses.
            self.strip_finished.emit(mosaics, content_rect, tuple(tiles) if task.keep_tiles else None)

        except Exception as e:
            logger.exception("Test strip render failed")
            self.error.emit(str(e))

    @pyqtSlot(PickBetterTask)
    def build_pick_better(self, task: PickBetterTask) -> None:
        """Render the Pick Better proof as three frames: the committed current edit plus the
        two candidates, each a full frame at one ExposureConfig field's value. The three
        configs differ only in the exposure stage, which the analysis cache keys on, so the
        per-frame metering is reused. Metrics are dropped: these are proofs, and the real
        render owns the writeback, like the test strip."""
        try:
            rendered = []
            for config in (task.current_config, task.left_config, task.right_config):
                result, _ = self._processor.run_pipeline(
                    task.buffer,
                    config,
                    task.source_hash,
                    render_size_ref=task.preview_size,
                    prefer_gpu=task.gpu_enabled,
                    readback_metrics=False,
                    ir_buffer=task.ir_buffer,
                    detect_buffer=task.detect_buffer,
                    wants_uv_grid=False,
                    cam_xyz=task.cam_xyz,
                    camera_wb=task.camera_wb,
                )
                if isinstance(result, GPUTexture):
                    result = result.readback()
                if isinstance(result, np.ndarray) and result.ndim == 3 and result.shape[2] >= 4:
                    result = np.ascontiguousarray(result[:, :, :3])
                rendered.append(result)

            current, left, right = rendered
            self.pick_better_finished.emit(current, left, right)

        except Exception as e:
            logger.exception("Pick Better render failed")
            self.error.emit(str(e))


class ThumbnailWorker(QObject):
    """
    Asynchronous thumbnail generation worker.
    """

    activity = pyqtSignal(str)
    # A completed frame enters the filmstrip before the next source starts.
    partial = pyqtSignal(dict)
    # Rendered positives use their own signal, so the batch's bulk overwrite cannot clobber a
    # frame that already rendered on the canvas.
    rendered_finished = pyqtSignal(dict)

    def __init__(self, asset_store) -> None:
        super().__init__()
        self._store = asset_store
        self._files: list[dict] = []
        self._slow_files: list[dict] = []
        self._current = 0
        self._active = False
        self._slow_phase = False
        self._cancel_requested = threading.Event()
        self._next_timer: Optional[QTimer] = None

    def _timer(self) -> QTimer:
        """Create the scheduler in the worker's current Qt thread."""
        if self._next_timer is None:
            self._next_timer = QTimer(self)
            self._next_timer.setSingleShot(True)
            self._next_timer.timeout.connect(self._process_next)
        return self._next_timer

    def cancel_pending(self) -> None:
        """Stop after the native call in progress returns."""
        self._cancel_requested.set()

    @pyqtSlot()
    def cancel(self) -> None:
        self.cancel_pending()
        if self._active:
            self._finish()

    @pyqtSlot(list)
    def generate(self, files: list) -> None:
        """Replace the low-priority queue and yield between every source file."""
        timer = self._timer()
        timer.stop()
        self._cancel_requested.clear()
        self._files = list(files)
        self._slow_files = []
        self._current = 0
        self._active = bool(self._files)
        self._slow_phase = False
        if not self._active:
            self.activity.emit("")
            return
        timer.start(0)

    @pyqtSlot()
    def _process_next(self) -> None:
        if not self._active or self._cancel_requested.is_set():
            self._finish()
            return

        from negpy.services.assets.thumbnails import asset_thumbnail_key, get_thumbnail_worker

        f_info = self._files[self._current]
        key = asset_thumbnail_key(f_info)
        self.activity.emit(key)
        try:
            with _DECODE_MEMORY_GATE.hold("thumbnail"):
                thumb = get_thumbnail_worker(
                    f_info["path"],
                    f_info["hash"],
                    self._store,
                    int(f_info.get("half") or 0),
                    float(f_info.get("split_x") or 0.5),
                    f_info.get("green_path") or "",
                    f_info.get("blue_path") or "",
                    tuple(f_info["crop_rect"]) if f_info.get("crop_rect") else None,
                    float(f_info.get("gutter_thickness") or 0.0),
                    str(f_info.get("process_mode") or ""),
                    split_axis=str(f_info.get("split_axis") or "x"),
                    fast_only=not self._slow_phase,
                    should_cancel=self._cancel_requested.is_set,
                )
            if thumb is not None:
                self.partial.emit({key: thumb})
            elif not self._slow_phase:
                self._slow_files.append(f_info)
        except Exception as e:
            logger.error(f"Thumbnail generation failure: {e}")

        self._current += 1
        if self._cancel_requested.is_set():
            self._finish()
            return
        if self._current >= len(self._files) and not self._slow_phase and self._slow_files:
            self._files = self._slow_files
            self._slow_files = []
            self._current = 0
            self._slow_phase = True
            self._timer().start(0)
            return
        if self._current >= len(self._files):
            self._finish()
            return
        self._timer().start(0)

    def _finish(self) -> None:
        if not self._active:
            return
        if self._next_timer is not None:
            self._next_timer.stop()
        self._active = False
        self._files = []
        self._slow_files = []
        self._slow_phase = False
        self.activity.emit("")

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
                fingerprint=task.fingerprint,
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


# Seek-bound: each hash costs many seeks, so cpu_count() concurrent readers thrash a
# spinning disk.
_HASH_WORKERS = min(8, APP_CONFIG.max_workers)
# Real decodes; halved for memory headroom, as NormalizationWorker does.
_DECODE_WORKERS = max(1, APP_CONFIG.max_workers // 2)


def rgb_grouping_notice(made: int, loose: int, incomplete: int, mismatched: int, by_time: bool) -> str:
    """What RGB Scan did with a folder, when the answer is not "all of it".

    Names the reason, because the two failures call for different things: sets that
    are not one of each color mean the folder does not hold whole triplets, while
    sets that do not match mean the shots could not be put in the order they were
    taken. Silent on a clean folder — a status line nobody needs is noise.
    """
    if not loose:
        return ""
    from negpy.kernel.system.text import count_of

    reasons = []
    if incomplete:
        reasons.append(f"{count_of(incomplete, 'set')} not one of each color")
    if mismatched:
        reasons.append(f"{count_of(mismatched, 'set')} showing different frames")
    detail = f" — {', '.join(reasons)}" if reasons else ""
    order = "" if by_time else "; grouped by filename, as the files state no capture time"
    made_text = f"{count_of(made, 'frame')} assembled, " if made else ""
    return (
        f"Trichrome Mode: {made_text}{count_of(loose, 'file')} left separate{detail}{order}. "
        "Select a frame's three files, right-click and choose Merge Exposures (Trichrome) to join them."
    )


def rgb_nothing_matched_message(summary: dict) -> tuple[str, str]:
    """Title and body for a folder where RGB Scan assembled nothing at all.

    Two situations, opposite answers. A folder lit one color at a time is trichrome
    that could not be ordered, and the user needs the requirements. A folder lit the
    same way throughout is not trichrome at all, and the user needs the mode off.
    """
    from negpy.kernel.system.text import count_of

    files = count_of(summary.get("loose", 0), "file")
    if not summary.get("narrowband"):
        return (
            "Nothing to assemble",
            f"Trichrome Mode is on, but this folder does not look like trichrome captures: its {files} were all "
            "lit the same way, so there are no red, green and blue sets to combine.\n\n"
            "Turn Trichrome Mode off to work with them as ordinary frames.",
        )
    # Filenames only stop mattering once the files date themselves; without that they
    # carry the capture order and the claim would contradict the fallback.
    if summary.get("by_time"):
        naming = "Which of the three colors you shoot first does not matter, and filenames do not matter."
    else:
        naming = (
            "Which of the three colors you shoot first does not matter. These files record no capture "
            "time, so they were put in filename order — which means their names have to sort into the "
            "order the shots were taken."
        )
    return (
        "No triplets found",
        f"None of the {files} in this folder could be assembled into RGB triplets.\n\n"
        "Each frame needs three captures — one under red light, one under green, one under blue — "
        "taken back to back before you move on to the next frame, and the folder should hold "
        f"nothing else. {naming}\n\n"
        "You can also join files by hand: select a frame's three files, right-click and choose Merge Exposures (Trichrome).",
    )


def _without_parts(assets: list, parts: set) -> list:
    """Drop the files a re-attached composite is built from. A folder walk finds them
    beside the primary, and they belong to the composite, not to the roll."""
    return [a for a in assets if a["path"] not in parts] if parts else assets


class AssetDiscoveryWorker(QObject):
    """
    Background worker for file system crawling and hashing.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    rgb_grouped = pyqtSignal(dict)  # RGB-scan grouping outcome; the controller decides how loudly to say it
    splits_detected = pyqtSignal(dict)  # {path: (split, thickness, crop_rect|None, axis)}, for AutoDetectAllSplitsTask

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

    @pyqtSlot(AutoDetectAllSplitsTask)
    def process_auto_detect_all_splits(self, task: AutoDetectAllSplitsTask) -> None:
        import os

        from negpy.services.assets.half_frame import detect_split_and_crop_for_file

        detected = self._map_files(task.paths, detect_split_and_crop_for_file, lambda p: f"Split {os.path.basename(p)}", _DECODE_WORKERS)
        self.splits_detected.emit(dict(zip(task.paths, detected)))

    @pyqtSlot(AssetDiscoveryTask)
    def process(self, task: AssetDiscoveryTask) -> None:
        """Scans paths for supported images and calculates hashes. Emits `finished` or
        `error`: either one releases the import lane."""
        try:
            self._discover(task)
        except Exception as e:
            logger.exception("Asset discovery failed")
            self.error.emit(str(e))

    def _discover(self, task: AssetDiscoveryTask) -> None:
        import os

        from negpy.infrastructure.loaders.constants import is_hidden_path, is_ir_sidecar_path
        from negpy.kernel.image.logic import file_hashes
        from negpy.services.assets.migrations.hash import blank_ambiguous_legacy_hashes

        discovered_paths = []
        split_paths = set()
        for path in task.paths:
            split = task.half_frame or os.path.normcase(os.path.normpath(path)) in task.half_frame_paths
            try:
                if os.path.isdir(path):
                    for f in os.listdir(path):
                        if is_hidden_path(f):
                            continue
                        found = os.path.join(path, f)
                        if (
                            f.lower().endswith(task.supported_extensions)
                            and os.path.normcase(os.path.normpath(found)) not in task.skip_paths
                        ):
                            discovered_paths.append(found)
                            if split:
                                split_paths.add(found)
                else:
                    if not is_hidden_path(path) and path.lower().endswith(task.supported_extensions):
                        discovered_paths.append(path)
                        if split:
                            split_paths.add(path)
            except Exception as e:
                logger.error(f"Discovery error for {path}: {e}")
        # Half-frame re-discovery passes both halves' identical paths, so hash once.
        discovered_paths = list(dict.fromkeys(discovered_paths))
        # IR companions ride along with their main TIFF; they are never assets of their own.
        discovered_paths = [p for p in discovered_paths if not is_ir_sidecar_path(p)]

        valid_assets = []
        digests = self._map_files(discovered_paths, file_hashes, os.path.basename, _HASH_WORKERS)

        for path, digest in zip(discovered_paths, digests):
            if digest is None:
                continue
            f_hash, legacy = digest
            if f_hash.startswith("err_"):
                continue
            # Stamped once here so sorting and date search never stat per row.
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue  # Removed or renamed since it was hashed.
            valid_assets.append({"name": os.path.basename(path), "path": path, "hash": f_hash, "legacy_hash": legacy, "mtime": mtime})

        blank_ambiguous_legacy_hashes(valid_assets)

        if task.restore_triplets:
            valid_assets = self._attach_restored_triplets(valid_assets, task.restore_triplets)
        if task.rgb_scan and valid_assets:
            valid_assets = self._group_rgb_triplets(valid_assets)

        if task.restore_stitches and valid_assets:
            valid_assets = self._attach_restored_stitches(valid_assets, task.restore_stitches)

        if task.restore_hdr and valid_assets:
            valid_assets = self._attach_restored_hdr(valid_assets, task.restore_hdr)

        if split_paths and valid_assets:
            to_split = [a for a in valid_assets if a["path"] in split_paths]
            halves: dict = {}
            for half in self._expand_half_frames(to_split, profile=task.half_frame_profile, overrides=task.half_frame_overrides):
                halves.setdefault(half["path"], []).append(half)
            valid_assets = [h for a in valid_assets for h in (halves.get(a["path"]) or [a] if a["path"] in split_paths else [a])]

        self.finished.emit(valid_assets)

    def _expand_half_frames(
        self,
        assets: list,
        profile: dict | None = None,
        overrides: dict | None = None,
    ) -> list:
        """Two half-frame assets per file, sharing the path; composites stay whole.

        The split comes from ``overrides[base_hash]``, else the roll's ``profile``, else auto-detection.
        """
        import os

        from negpy.services.assets.half_frame import (
            base_hash,
            detect_split_axis_for_file,
            half_hash,
            half_name,
            is_composite,
            saved_crop_rect,
        )

        def _splittable(a: dict) -> bool:
            return not is_composite(a)

        overrides = overrides or {}
        auto_split = profile is None
        if auto_split:
            paths = [a["path"] for a in assets if _splittable(a) and base_hash(a["hash"]) not in overrides]
            detected = self._map_files(paths, detect_split_axis_for_file, lambda p: f"Split {os.path.basename(p)}", _DECODE_WORKERS)
            splits = dict(zip(paths, detected))
        else:
            splits = {}

        out = []
        for a in assets:
            if not _splittable(a):
                out.append(a)
                continue
            override = overrides.get(base_hash(a["hash"]))
            if override is not None:
                split_x = float(override.get("split_x") or 0.5)
                split_axis = str(override.get("split_axis") or "x")
            elif auto_split:
                # (0.5, "x") is the detector's "nothing found" value; with no profile there is nothing better.
                found = splits.get(a["path"])
                split_x, split_axis = (float(found[0]), str(found[1])) if found is not None else (0.5, "x")
            else:
                split_x = float(profile.get("split_x") or 0.5)
                split_axis = str(profile.get("split_axis") or "x")
            legacy = a.get("legacy_hash")
            for half in (1, 2):
                entry = {
                    **a,
                    "name": half_name(a["name"], half),
                    "hash": half_hash(a["hash"], half),
                    "legacy_hash": half_hash(legacy, half) if legacy else "",
                    "half": half,
                    "split_x": split_x,
                    "split_axis": split_axis,
                }
                source = override if override is not None else profile
                if source is not None:
                    cr = saved_crop_rect(source.get("crop_rect"))
                    if cr is not None:
                        entry["crop_rect"] = cr
                    entry["gutter_thickness"] = float(source.get("gutter_thickness") or 0.0)
                out.append(entry)
        return out

    def _attach_restored_triplets(self, assets: list, triplets: dict) -> list:
        """Re-attach known green/blue exposures to their red asset (no reclassification).

        A session manifest holds the red path alone, but a capture hands over all three,
        so the two exposures that became part of a frame are dropped from the roll.
        """
        import os

        from negpy.kernel.image.logic import calculate_file_hash
        from negpy.services.assets.triplets import made_by_hand

        hashes = {a["path"]: a["hash"] for a in assets}
        out = []
        parts: set = set()
        for a in assets:
            gb = triplets.get(a["path"])
            stored = list(gb[3]) if gb and len(gb) > 3 else ["", "", ""]
            if gb:
                for path, want in zip(gb[:2], stored[1:]):
                    if want and path and path not in hashes and os.path.exists(path):
                        hashes[path] = calculate_file_hash(path)
            # A stated member hash re-attaches only while the file still has it.
            changed = gb and any(want and path in hashes and hashes[path] != want for path, want in zip((a["path"], gb[0], gb[1]), stored))
            if gb and not changed and gb[0] and gb[1] and os.path.exists(gb[0]) and os.path.exists(gb[1]):
                base = os.path.splitext(a["name"])[0]
                align = bool(gb[2]) if len(gb) > 2 else True
                out.append(
                    {
                        **a,
                        "name": f"{base} (RGB)",
                        "green_path": gb[0],
                        "blue_path": gb[1],
                        "align": align,
                        "green_hash": hashes.get(gb[0], stored[1]),
                        "blue_hash": hashes.get(gb[1], stored[2]),
                        "triplet_by_hand": made_by_hand(gb),
                    }
                )
                parts.update({gb[0], gb[1]})
            else:
                out.append(a)
        return _without_parts(out, parts)

    def _attach_restored_stitches(self, assets: list, stitches: dict) -> list:
        """Re-attach saved stitch registrations to restored primary assets (no re-registration).
        A composite whose parts vanished from disk restores as a plain asset."""
        import os

        from negpy.features.stitch.models import stitch_name
        from negpy.services.assets.composites import part_paths

        out = []
        attached = []
        for a in assets:
            entry = stitches.get(a["path"])
            # Triplet exposures count as parts: one missing decodes that part red-only.
            needed = [*(entry.get("paths") or ()), *(p for t in entry.get("triplets") or () for p in t if p)] if entry else []
            if entry and entry.get("paths") and all(os.path.exists(p) for p in needed):
                out.append(
                    {
                        **a,
                        "name": stitch_name([a["path"], *entry["paths"]]),
                        "hash": entry["hash"],
                        # The composite hash is the parts' own: inheriting the primary part's
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
                attached.append(entry)
            else:
                out.append(a)
        return _without_parts(out, part_paths(attached))

    def _attach_restored_hdr(self, assets: list, merges: dict) -> list:
        """Re-attach saved brackets to restored reference assets (no re-solve). A merge whose
        exposures vanished from disk restores as a plain asset."""
        import os

        from negpy.features.hdr.models import hdr_name
        from negpy.services.assets.composites import part_paths

        out = []
        attached = []
        for a in assets:
            entry = merges.get(a["path"])
            if entry and entry.get("paths") and all(os.path.exists(p) for p in entry["paths"]):
                out.append(
                    {
                        **a,
                        "name": hdr_name([a["path"], *entry["paths"]]),
                        "hash": entry["hash"],
                        # The composite hash is the bracket's own: inheriting the reference
                        # frame's legacy digest would rehome that frame's edit onto it.
                        "legacy_hash": "",
                        "hdr_paths": tuple(entry["paths"]),
                        "hdr_ratios": tuple(float(r) for r in entry.get("ratios") or ()),
                        "hdr_align": bool(entry.get("align", True)),
                        "hdr_anchor": str(entry.get("anchor", "") or ""),
                        "hdr_anchor_ev": float(entry.get("anchor_ev", 1.0)),
                    }
                )
                attached.append(entry)
            else:
                out.append(a)
        return _without_parts(out, part_paths(attached))

    def _group_rgb_triplets(self, assets: list) -> list:
        """Classify each file by dominant channel and merge consecutive R/G/B triplets
        into one asset (red is primary; green/blue ride along). A chunk that does not
        hold one of each channel, or whose three exposures do not show the same frame,
        is left alone: its files stay individual and can be paired by hand.

        An asset that already carries its green and blue exposures keeps them and is
        not re-examined. Grouping runs beside a restore, not instead of it, so files
        the restore knew nothing about still get their chance.
        """
        import os

        from negpy.features.rgbscan.logic import (
            capture_ordered,
            capture_timestamp,
            classify_channel,
            group_triplets,
            looks_narrowband,
            probe_frame,
        )

        from negpy.services.export.frame_merge import is_merged_source

        # A file NegPy merged from an assembled frame is already one frame.
        done = {a["path"] for a in assets if (a.get("green_path") and a.get("blue_path")) or is_merged_source(a["path"])}
        assembled = [a for a in assets if a["path"] in done]
        assets = [a for a in assets if a["path"] not in done]
        if not assets:
            return assembled

        by_path = {a["path"]: a for a in assets}
        ordered = sorted(by_path, key=lambda p: os.path.basename(p).lower())

        stamps = self._map_files(ordered, capture_timestamp, lambda p: f"Time {os.path.basename(p)}", _HASH_WORKERS)
        times = {p: t for p, t in zip(ordered, stamps) if t}
        by_time = len(times) == len(ordered)
        ordered = capture_ordered(ordered, times)

        # An infrared exposure shot beside a triplet (`_IR`) stays its own frame: gray, it would
        # classify as a channel and shift every chunk after it.
        candidates = [p for p in ordered if not os.path.splitext(os.path.basename(p))[0].lower().endswith("_ir")]
        probes = self._map_files(candidates, probe_frame, lambda p: f"RGB {os.path.basename(p)}", _DECODE_WORKERS)
        items = [(p, classify_channel(pr.means)) for p, pr in zip(candidates, probes) if pr is not None]
        signatures = {p: pr.signature for p, pr in zip(candidates, probes) if pr is not None}

        result = list(assembled)
        grouped = set()
        incomplete = mismatched = 0
        for t in group_triplets(items, signatures):
            if not t.ok:
                # Which test it failed separates "this folder does not hold whole
                # triplets" from "these could not be put into the order they were shot".
                chunk = [(p, ch) for p, ch in items if p in (t.red, t.green, t.blue)]
                if group_triplets(chunk)[0].ok:
                    mismatched += 1
                else:
                    incomplete += 1
                continue
            red = by_path[t.red]
            base = os.path.splitext(red["name"])[0]
            result.append(
                {
                    **red,
                    "name": f"{base} (RGB)",
                    "green_path": t.green,
                    "blue_path": t.blue,
                    "green_hash": by_path[t.green]["hash"],
                    "blue_hash": by_path[t.blue]["hash"],
                }
            )
            grouped.update({t.red, t.green, t.blue})

        result.extend(by_path[p] for p in ordered if p not in grouped)
        loose = len(candidates) - len(grouped)
        if loose:
            summary = {
                "made": len(grouped) // 3,
                "loose": loose,
                "incomplete": incomplete,
                "mismatched": mismatched,
                "by_time": by_time,
                "narrowband": looks_narrowband([pr.means for pr in probes if pr is not None]),
            }
            logger.warning("RGB scan: %s", summary)
            self.rgb_grouped.emit(summary)
        return result


class PreviewLoadState:
    """Generations shared by the foreground and prefetch workers: a click on one obsoletes work queued on the other."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_generation = 0
        self._cancelled_prefetch_generations: set[int] = set()
        self._foreground_path: str | None = None

    def expect_generation(self, generation: int, file_path: str | None = None) -> None:
        """Obsoletes older preview work, except a running prefetch of *file_path*: the foreground load waits for it."""
        with self._lock:
            self._latest_generation = generation
            self._foreground_path = file_path
            self._cancelled_prefetch_generations = {
                cancelled for cancelled in self._cancelled_prefetch_generations if cancelled >= generation
            }

    def cancel_prefetch(self, generation: int) -> None:
        with self._lock:
            self._cancelled_prefetch_generations.add(generation)

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._latest_generation

    def prefetch_is_current(self, generation: int, file_path: str) -> bool:
        with self._lock:
            if file_path == self._foreground_path:
                return True
            return generation == self._latest_generation and generation not in self._cancelled_prefetch_generations


class PreviewLoadWorker(QObject):
    """
    Background worker for decoding RAW files into a linear preview buffer.
    Keeps the UI thread free during slow I/O and demosaicing.
    """

    # (file_path, raw, dims, source_cs, ir_preview, detected_mode, (cam_xyz, camera_wb), detect_preview, generation,
    # source_color or None)
    finished = pyqtSignal(str, object, object, str, object, str, object, object, int, object)
    splash = pyqtSignal(str, object, object, int)  # (file_path, buffer, dims, generation) — first paint
    error = pyqtSignal(str)
    # (file_path, applied long-edge cap px): an HQ load exceeded the GPU's VRAM budget
    # and was downsampled instead of crashing. Emitted alongside `finished`.
    vram_capped = pyqtSignal(str, int)
    # (file_path, ShotsInfo or None): what a multi-shot RAW decoded to, before finished for the same load.
    raw_shots = pyqtSignal(str, object)
    # (file_path, message): the error carries no path, so badge attribution needs this
    load_failed = pyqtSignal(str, str)
    prefetch_finished = pyqtSignal(int, str)

    def __init__(self, preview_service, state: PreviewLoadState | None = None) -> None:
        super().__init__()
        self._preview_service = preview_service
        self._state = state or PreviewLoadState()

    def _is_current(self, task: PreviewLoadTask) -> bool:
        return self._state.is_current(task.generation)

    def _prefetch_is_current(self, task: PreviewLoadTask) -> bool:
        return self._state.prefetch_is_current(task.generation, task.file_path)

    @pyqtSlot(PreviewLoadTask)
    def process(self, task: PreviewLoadTask) -> None:
        if task.for_cache_warm:
            self._process_prefetch(task)
            return
        if not self._is_current(task):
            return
        with _DECODE_MEMORY_GATE.hold("foreground", task.file_path, abandoned=lambda: not self._is_current(task)) as entered:
            if entered and self._is_current(task):
                self._process_locked(task)

    def _process_prefetch(self, task: PreviewLoadTask) -> None:
        try:
            if not self._prefetch_is_current(task):
                return
            with _DECODE_MEMORY_GATE.hold("prefetch", task.file_path):
                if not self._prefetch_is_current(task):
                    return
                self._preview_service.prefetch_linear_preview(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    file_hash=task.file_hash,
                    half_slice=task.half_slice,
                    demosaic=task.demosaic,
                    raw_shot=task.raw_shot,
                    positive_decode=task.positive_decode,
                    integrated_gpu=task.integrated_gpu,
                    protected_file_hashes=task.protected_file_hashes,
                    should_cancel=lambda: not self._prefetch_is_current(task),
                    highlight_mode=task.highlight_mode,
                    bake_camera_wb=task.bake_camera_wb,
                    lens_corrections=task.lens_corrections,
                    lens_flatfield=task.lens_flatfield,
                )
            # Outside the memory gate: the read holds no decode memory, and a foreground decode
            # of this file must not wait on it.
            if task.base_settings is not None and self._prefetch_is_current(task):
                cached = self._preview_service.peek_linear_preview(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    file_hash=task.file_hash,
                    demosaic=task.demosaic,
                    raw_shot=task.raw_shot,
                    positive_decode=task.positive_decode,
                    highlight_mode=task.highlight_mode,
                    bake_camera_wb=task.bake_camera_wb,
                    lens_corrections=task.lens_corrections,
                    lens_flatfield=task.lens_flatfield,
                )
                if cached is not None:
                    buffer, _dims, metadata = cached
                    read_source_base(buffer, task.base_settings, task.file_hash, metadata.get("orientation", 1))
        except InterruptedError:
            pass
        except Exception as error:
            logger.debug("Preview cache warm failed for %s: %s", task.file_path, error)
        finally:
            self.prefetch_finished.emit(task.generation, task.file_path)

    def _process_locked(self, task: PreviewLoadTask) -> None:
        if not self._is_current(task):
            return

        def cancelled() -> bool:
            return not self._is_current(task)

        t0 = time.perf_counter()
        try:
            if stitch_active(task.stitch):
                # Stitch composite: replay the stored registration at preview scale. No splash,
                # because the primary's embedded JPEG would flash a half frame.
                raw, dims, metadata = self._preview_service.load_linear_preview_stitch(
                    task.file_path,
                    task.stitch,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                    flatfield_profile_id=task.flatfield_profile_id,
                    demosaic=task.demosaic,
                    raw_shot=task.raw_shot,
                    should_cancel=cancelled,
                    highlight_mode=task.highlight_mode,
                    bake_camera_wb=task.bake_camera_wb,
                )
                if not self._is_current(task):
                    return
                source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
                ir_preview = metadata.get("ir_preview")
                detected_mode, source_color = self._detect(task, raw)
                logger.info(
                    "load-timing preview_worker_total %.0fms (stitch load->buffer) %s",
                    (time.perf_counter() - t0) * 1000,
                    task.file_path,
                )
                capped = metadata.get("vram_capped_long_edge")
                if capped:
                    self.vram_capped.emit(task.file_path, int(capped))
                self.raw_shots.emit(task.file_path, metadata.get("raw_shots"))
                self.finished.emit(
                    task.file_path,
                    raw,
                    dims,
                    source_cs,
                    ir_preview,
                    detected_mode,
                    (
                        metadata.get("cam_xyz"),
                        metadata.get("camera_wb"),
                        metadata.get("lens_correction"),
                        lens_decode_token(task.lens_corrections, task.lens_flatfield),
                        metadata.get("orientation", 1),
                    ),
                    metadata.get("detect_preview"),
                    task.generation,
                    source_color,
                )
                return
            if hdr_active(task.hdr):
                # Bracketed capture: merge the exposures into one linear source. No splash,
                # because the reference frame's embedded JPEG would flash the unmerged exposure.
                raw, dims, metadata = self._preview_service.load_linear_preview_hdr(
                    task.file_path,
                    task.hdr,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                    demosaic=task.demosaic,
                    raw_shot=task.raw_shot,
                    should_cancel=cancelled,
                    highlight_mode=task.highlight_mode,
                    bake_camera_wb=task.bake_camera_wb,
                )
                if not self._is_current(task):
                    return
                source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
                ir_preview = metadata.get("ir_preview")
                detected_mode, source_color = self._detect(task, raw)
                logger.info(
                    "load-timing preview_worker_total %.0fms (hdr load->buffer) %s",
                    (time.perf_counter() - t0) * 1000,
                    task.file_path,
                )
                capped = metadata.get("vram_capped_long_edge")
                if capped:
                    self.vram_capped.emit(task.file_path, int(capped))
                self.raw_shots.emit(task.file_path, metadata.get("raw_shots"))
                self.finished.emit(
                    task.file_path,
                    raw,
                    dims,
                    source_cs,
                    ir_preview,
                    detected_mode,
                    (
                        metadata.get("cam_xyz"),
                        metadata.get("camera_wb"),
                        metadata.get("lens_correction"),
                        lens_decode_token(task.lens_corrections, task.lens_flatfield),
                        metadata.get("orientation", 1),
                    ),
                    metadata.get("detect_preview"),
                    task.generation,
                    source_color,
                )
                return
            if is_rgb_triplet(task.rgbscan):
                # RGB-scan triplet: assemble the frame from the three exposures. No splash,
                # because the red embedded JPEG would flash a red-cast preview.
                raw, dims, metadata = self._preview_service.load_linear_preview_rgb(
                    task.file_path,
                    task.rgbscan,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                    demosaic=task.demosaic,
                    raw_shot=task.raw_shot,
                    should_cancel=cancelled,
                )
                if not self._is_current(task):
                    return
                source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
                ir_preview = metadata.get("ir_preview")
                detected_mode, source_color = self._detect(task, raw)
                logger.info(
                    "load-timing preview_worker_total %.0fms (rgb load->buffer) %s",
                    (time.perf_counter() - t0) * 1000,
                    task.file_path,
                )
                capped = metadata.get("vram_capped_long_edge")
                if capped:
                    self.vram_capped.emit(task.file_path, int(capped))
                self.raw_shots.emit(task.file_path, metadata.get("raw_shots"))
                self.finished.emit(
                    task.file_path,
                    raw,
                    dims,
                    source_cs,
                    ir_preview,
                    detected_mode,
                    (
                        metadata.get("cam_xyz"),
                        metadata.get("camera_wb"),
                        metadata.get("lens_correction"),
                        lens_decode_token(task.lens_corrections, task.lens_flatfield),
                        metadata.get("orientation", 1),
                    ),
                    metadata.get("detect_preview"),
                    task.generation,
                    source_color,
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
                    demosaic=task.demosaic,
                    raw_shot=task.raw_shot,
                    positive_decode=task.positive_decode,
                    should_cancel=cancelled,
                    highlight_mode=task.highlight_mode,
                    bake_camera_wb=task.bake_camera_wb,
                    lens_corrections=task.lens_corrections,
                    lens_flatfield=task.lens_flatfield,
                )
                if not self._is_current(task):
                    return
                if sp is not None:
                    sbuf, sdims = sp
                    self.splash.emit(task.file_path, sbuf, sdims, task.generation)
            else:
                raw, dims, metadata = self._preview_service.load_linear_preview(
                    task.file_path,
                    task.workspace_color_space,
                    use_camera_wb=task.use_camera_wb,
                    full_resolution=task.full_resolution,
                    file_hash=task.file_hash,
                    log_timings=True,
                    half_slice=task.half_slice,
                    demosaic=task.demosaic,
                    raw_shot=task.raw_shot,
                    positive_decode=task.positive_decode,
                    should_cancel=cancelled,
                    highlight_mode=task.highlight_mode,
                    bake_camera_wb=task.bake_camera_wb,
                    lens_corrections=task.lens_corrections,
                    lens_flatfield=task.lens_flatfield,
                )
                if not self._is_current(task):
                    return
            source_cs = metadata.get("color_space") or WORKING_COLOR_SPACE
            ir_preview = metadata.get("ir_preview")
            detected_mode, source_color = self._detect(task, raw)
            logger.info(
                "load-timing preview_worker_total %.0fms (load->buffer) %s",
                (time.perf_counter() - t0) * 1000,
                task.file_path,
            )
            capped = metadata.get("vram_capped_long_edge")
            if capped:
                self.vram_capped.emit(task.file_path, int(capped))
            self.raw_shots.emit(task.file_path, metadata.get("raw_shots"))
            self.finished.emit(
                task.file_path,
                raw,
                dims,
                source_cs,
                ir_preview,
                detected_mode,
                (
                    metadata.get("cam_xyz"),
                    metadata.get("camera_wb"),
                    metadata.get("lens_correction"),
                    lens_decode_token(task.lens_corrections, task.lens_flatfield),
                    metadata.get("orientation", 1),
                ),
                metadata.get("detect_preview"),
                task.generation,
                source_color,
            )
        except InterruptedError:
            return
        except Exception as e:
            if not self._is_current(task):
                return
            logger.exception(f"Asset load failed: {task.file_path}")
            # libraw reports "Unsupported file format or not RAW file" for a file whose tags it
            # parsed perfectly and whose payload it cannot decode, which reads as "your NEF is
            # broken". Ask why only once the decode has failed, so the check costs nothing on
            # the files that work.
            message = unsupported_raw_reason(task.file_path) or str(e)
            self.error.emit(message)
            self.load_failed.emit(task.file_path, message)

    def _detect(self, task: PreviewLoadTask, raw) -> tuple[str, Optional[bool]]:
        """(detected mode or "", source is color film or None), each only when the task asks for it."""
        if not (task.detect_mode or task.detect_source):
            return "", None
        from negpy.features.process.logic import detect_process_and_source

        t0 = time.perf_counter()
        try:
            if not task.use_camera_wb:
                scan = raw
            else:
                # Camera WB hides the C41 mask, so re-decode without WB. Lean: detect downsamples.
                scan = self._preview_service.decode_for_detection(task.file_path)
            mode, source_color = detect_process_and_source(scan)
            logger.info(
                "load-timing detect %.0fms mode=%s source_color=%s (re_decode=%s) %s",
                (time.perf_counter() - t0) * 1000,
                mode,
                source_color,
                task.use_camera_wb,
                task.file_path,
            )
            return (str(mode) if task.detect_mode else ""), (source_color if task.detect_source else None)
        except Exception:
            logger.exception(f"Process-mode detection failed: {task.file_path}")
            return "", None


def gain_slices_for_asset(file_info: dict) -> tuple:
    """The slice_half cuts a thumbnail's buffer took, for run_pipeline's gain_slices."""
    from negpy.services.assets.half_frame import asset_slice

    args = asset_slice(file_info)
    return () if args is None else (args,)


def _slice_meta_planes(meta: dict, file_info: dict) -> dict:
    """The IR and dust-detect planes cut like the buffer they ride with."""
    from negpy.services.assets.half_frame import slice_for_asset

    return {
        k: np.ascontiguousarray(slice_for_asset(v, file_info)) if k in ("ir_preview", "detect_preview") and v is not None else v
        for k, v in meta.items()
    }


def _decode_asset_preview_with_meta(
    preview_service,
    file_info: dict,
    config: WorkspaceConfig,
    workspace_color_space: str,
    *,
    raw_shot: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """Decode one asset the way the render path does, with its loader metadata: a stitch
    through its stored registration, a composite through the merge that assembles it, a
    plain frame direct — the same branch order `PreviewLoadWorker.process` uses.

    `raw_shot` overrides the edit's own shot choice; a thumbnail passes `thumbnail_shot`
    so that it never combines, while meters keep the shots the render combines.

    A batch that decodes only ``file_info["path"]`` sees one member of the composite. For a
    triplet that member holds real signal in the red channel alone, so anything measured off
    it is invalid for the assembled three-band source.
    """
    from negpy.services.assets.half_frame import base_hash, slice_for_asset

    rgbscan = config.rgbscan
    stitch = config.stitch
    common = {
        "use_camera_wb": not effective_linear_raw(config.process),
        "full_resolution": False,
        "file_hash": base_hash(file_info.get("hash")),  # halves share one decode
        "demosaic": config.process.demosaic_preview,
        "raw_shot": config.process.raw_shot if raw_shot is None else raw_shot,
    }
    hdr = config.hdr
    if stitch_active(stitch):
        raw, _, meta = preview_service.load_linear_preview_stitch(
            file_info["path"],
            stitch,
            workspace_color_space,
            flatfield_profile_id=config.flatfield.profile_id if (stitch.stitch_enabled and config.flatfield.apply) else "",
            **common,
        )
    elif hdr.hdr_enabled and hdr.hdr_paths:
        raw, _, meta = preview_service.load_linear_preview_hdr(
            file_info["path"],
            hdr,
            workspace_color_space,
            highlight_mode=effective_highlight_reconstruction(config.process),
            bake_camera_wb=highlight_reconstruction_bakes_wb(config.process),
            **common,
        )
    elif rgbscan.enabled and rgbscan.green_path and rgbscan.blue_path:
        # Narrowband triplet: no highlight_mode param, same reasoning as the decode path —
        # a single raw channel per exposure has no "highlight color" to reconstruct.
        raw, _, meta = preview_service.load_linear_preview_rgb(file_info["path"], rgbscan, workspace_color_space, **common)
    else:
        raw, _, meta = preview_service.load_linear_preview(
            file_info["path"],
            workspace_color_space,
            positive_decode=positive_decode(config.process),
            highlight_mode=effective_highlight_reconstruction(config.process),
            bake_camera_wb=highlight_reconstruction_bakes_wb(config.process),
            lens_corrections=metadata_lens_corrections(config),
            lens_flatfield=config.flatfield,
            **common,
        )
    return slice_for_asset(raw, file_info), _slice_meta_planes(meta, file_info)


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

    def _cancel_requested(self, generation: int) -> bool:
        """Peek at the cancel flag without consuming it: only the coordinating thread
        may emit the terminal signal, so a pool worker just stops."""
        with self._cancel_lock:
            return generation in self._cancelled_generations

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

    def _decode(self, frame: BatchAutoCropInput, workspace_color_space: str) -> tuple[np.ndarray, dict]:
        return _decode_asset_preview_with_meta(self._preview_service, frame.file_info, frame.config, workspace_color_space)

    def _frame_evidence(
        self, index: int, frame: BatchAutoCropInput, task: BatchAutoCropTask, generation: int
    ) -> tuple[Optional[CropEvidence], Optional[tuple[float, float]]]:
        """Decode and detect one frame, with the Tilt and Swing of Square Before Crop applied
        first. (None, None) when it failed or the run was cancelled.

        Only the keystone: the roll pools rotations, and the run's own deskew sets them."""
        file_info = frame.file_info
        if self._cancel_requested(generation):
            return None, None
        try:
            raw, meta = self._decode(frame, task.workspace_color_space)
            if self._cancel_requested(generation):
                return None, None

            config = frame.config
            corrected = flatfield_source(raw, config, gain_slices_for_asset(file_info), orientation=meta.get("orientation", 1))
            geometry = config.geometry
            keystone = None
            if squares_on_arm(geometry):
                fit = squared_geometry(corrected, geometry, config.process.process_mode)
                if fit is not None:
                    keystone = (fit[0].converge_v, fit[0].converge_h)
                    geometry = replace(geometry, converge_v=keystone[0], converge_h=keystone[1])
            detection_geometry = replace(
                geometry,
                crop_rect=None,
                crop_from_auto=False,
                autocrop_offset=0,
            )
            context = PipelineContext(
                original_size=(corrected.shape[1], corrected.shape[0]),
                scale_factor=1.0,
                process_mode=config.process.process_mode,
            )
            transformed = GeometryProcessor(detection_geometry).process(corrected, context)
            if self._cancel_requested(generation):
                return None, None
            evidence = detect_crop_candidate(
                f"{index}:{file_info.get('hash', '')}",
                transformed,
                target_ratio=config.geometry.autocrop_ratio,
                rebate_trim=config.geometry.autocrop_rebate_trim,
            )
            return evidence, keystone
        except Exception:
            if self._cancel_requested(generation):
                return None, None
            logger.exception("Auto-crop all frames skipped failed frame %s", file_info.get("name") or file_info.get("path") or index)
            return None, None

    @pyqtSlot(BatchAutoCropTask)
    def process(self, task: BatchAutoCropTask) -> None:
        """Collect frame evidence over a thread pool, then resolve it as one roll.

        Frames are independent until `resolve_roll_crops`, and each is a decode the GIL is
        not held for. Evidence is kept in frame order — it is what the roll medians and the
        template fit see — while progress counts completions, so the bar advances out of order.
        """
        generation = int(task.generation)
        with self._cancel_lock:
            self._active_generation = generation
        if self._emit_cancelled_if_requested(generation):
            return
        total = len(task.frames)

        def _name(index: int) -> str:
            info = task.frames[index].file_info
            return str(info.get("name") or info.get("path") or index + 1)

        try:
            # Half the cores: each worker holds a preview-scale frame, and the detector
            # inside it is already multi-core.
            workers = max(1, min(APP_CONFIG.max_workers // 2, total))
            candidates: list[Optional[CropEvidence]] = [None] * total
            keystone_by_key: dict[str, tuple[float, float]] = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self._frame_evidence, i, frame, task, generation): i for i, frame in enumerate(task.frames)}
                for done, future in enumerate(as_completed(futures), 1):
                    index = futures[future]
                    evidence, keystone = future.result()
                    candidates[index] = evidence
                    if evidence is not None and keystone is not None:
                        keystone_by_key[evidence.key] = keystone
                    self.progress.emit(done, total, _name(index))

            if self._emit_cancelled_if_requested(generation):
                return
            evidence = [item for item in candidates if item is not None]
            source_by_key = {item.key: task.frames[i] for i, item in enumerate(candidates) if item is not None}
            resolved = resolve_roll_crops(evidence)
            if self._emit_cancelled_if_requested(generation):
                return

            results: list[BatchAutoCropResult] = []
            for crop in resolved:
                source = source_by_key.get(crop.key)
                if source is None:
                    logger.warning("Auto-crop all frames ignored result with unknown key %s", crop.key)
                    continue
                results.append(
                    BatchAutoCropResult(
                        file_info=source.file_info,
                        fingerprint=source.fingerprint,
                        crop_rect=crop.crop_rect,
                        correction_angle=crop.correction_angle,
                        confidence=crop.confidence,
                        calibrated=crop.calibrated,
                        keystone=keystone_by_key.get(crop.key),
                    )
                )
            self._emit_finished_unless_cancelled(generation, results)
        except Exception as exc:
            if self._emit_cancelled_if_requested(generation):
                return
            with self._cancel_lock:
                if self._active_generation == generation:
                    self._active_generation = None
            logger.exception("Auto-crop all frames worker failure")
            self.error.emit(str(exc))


_THUMBNAIL_WORKERS = 2


def _room_for_another_frame(frame: ThumbnailRenderInput) -> bool:
    """True when RAM holds this frame's preview decode beside the one already in flight. A composite
    reads several files, and an unknown RAM size gives no margin, so both wait their turn."""
    config = frame.config
    if stitch_active(config.stitch) or hdr_active(config.hdr) or is_rgb_triplet(config.rgbscan):
        return False
    available = available_system_memory_bytes()
    if available is None:
        return False
    estimate = loader_factory.estimate_linear_preview_prefetch_memory(frame.file_info["path"], APP_CONFIG.preview_render_size)
    return available >= MIN_RAM_RESERVE_BYTES + estimate.temporary_bytes + estimate.cached_bytes


class ThumbnailRenderWorker(QObject):
    """Re-render frames a bulk settings write touched but did not open, off the live render
    path. Its own CPU-only ImageProcessor never contends with the live render's GPU texture
    pool, and its own PreviewManager cache never evicts the navigation cache. Generation-
    scoped cancel mirrors BatchAutoCropWorker."""

    rendered = pyqtSignal(object, object)  # ThumbnailRenderInput, ndarray — the frame rides
    # along so the controller can re-check the asset is still current before persisting.
    frame_started = pyqtSignal(int, int, str)  # index, total, name; emitted before the decode
    progress = pyqtSignal(int, int, str, float, float)  # done, total, name, decode s, render s
    finished = pyqtSignal(int)  # frames rendered
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, preview_service, live_preview_service=None) -> None:
        super().__init__()
        self._preview_service = preview_service
        self._live_preview_service = live_preview_service
        # One processor per pool thread: its caches hold one frame each.
        self._all_processors = [ImageProcessor(use_gpu=False) for _ in range(_THUMBNAIL_WORKERS)]
        self._processors: queue.SimpleQueue[ImageProcessor] = queue.SimpleQueue()
        for processor in self._all_processors:
            self._processors.put(processor)
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

    def _cancel_requested(self, generation: int) -> bool:
        with self._cancel_lock:
            return generation in self._cancelled_generations

    def _peek_live_preview(self, frame: ThumbnailRenderInput, workspace_color_space: str) -> Optional[tuple[np.ndarray, dict]]:
        """Copy of a plain frame's decode from the navigation cache, or None; never writes or reorders it.

        Keyed on the whole scan, as `_decode_asset_preview_with_meta` decodes it.
        """
        from negpy.services.assets.half_frame import base_hash, slice_for_asset

        config = frame.config
        if self._live_preview_service is None or stitch_active(config.stitch) or hdr_active(config.hdr) or is_rgb_triplet(config.rgbscan):
            return None
        hit = self._live_preview_service.peek_linear_preview(
            frame.file_info["path"],
            workspace_color_space,
            use_camera_wb=not effective_linear_raw(config.process),
            file_hash=base_hash(frame.file_info.get("hash")),
            demosaic=config.process.demosaic_preview,
            raw_shot=thumbnail_shot(frame.file_info["path"], config.process.raw_shot),
            positive_decode=positive_decode(config.process),
            highlight_mode=effective_highlight_reconstruction(config.process),
            bake_camera_wb=highlight_reconstruction_bakes_wb(config.process),
            lens_corrections=metadata_lens_corrections(config),
            lens_flatfield=config.flatfield,
        )
        logger.debug("thumbnail refresh live cache %s: %s", "miss" if hit is None else "hit", frame.file_info["path"])
        if hit is None:
            return None
        raw, _dims, meta = hit
        # The navigation cache still serves these arrays, so the pipeline gets its own.
        meta = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in meta.items()}
        return slice_for_asset(np.copy(raw), frame.file_info), _slice_meta_planes(meta, frame.file_info)

    def _decode(self, frame: ThumbnailRenderInput, workspace_color_space: str) -> tuple[np.ndarray, dict]:
        hit = self._peek_live_preview(frame, workspace_color_space)
        if hit is not None:
            return hit
        return _decode_asset_preview_with_meta(
            self._preview_service,
            frame.file_info,
            frame.config,
            workspace_color_space,
            raw_shot=thumbnail_shot(frame.file_info["path"], frame.config.process.raw_shot),
        )

    def _emit_finished_unless_cancelled(self, generation: int, rendered_count: int) -> None:
        """Atomically choose the terminal signal for a generation, so a `cancel()` racing
        the last frame's completion cannot land between the check and the emit."""
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
                self.finished.emit(rendered_count)
        if cancelled:
            self.cancelled.emit()

    @pyqtSlot(ThumbnailRenderTask)
    def process(self, task: ThumbnailRenderTask) -> None:
        """Render up to `_THUMBNAIL_WORKERS` frames at once on negpy_core's background pool, so the
        live preview keeps the other cores. Signals leave this thread in frame order."""
        generation = int(task.generation)
        with self._cancel_lock:
            self._active_generation = generation
        if self._emit_cancelled_if_requested(generation):
            return
        total = len(task.frames)
        rendered_count = 0
        pending: deque[tuple[int, str, ThumbnailRenderInput, Future]] = deque()
        try:
            with ThreadPoolExecutor(_THUMBNAIL_WORKERS, initializer=negpy_core.set_background_thread, initargs=(True,)) as pool:
                frames = iter(enumerate(task.frames, 1))
                nxt = next(frames, None)
                while nxt is not None or pending:
                    if (
                        nxt is not None
                        and not self._cancel_requested(generation)
                        and (not pending or (len(pending) < _THUMBNAIL_WORKERS and _room_for_another_frame(nxt[1])))
                    ):
                        done, frame = nxt
                        name = str(frame.file_info.get("name") or frame.file_info.get("path") or done)
                        self.frame_started.emit(done, total, name)
                        future = pool.submit(self._render_one, frame, task.workspace_color_space, generation)
                        pending.append((done, name, frame, future))
                        nxt = next(frames, None)
                        continue
                    if not pending:
                        break
                    done, name, frame, future = pending.popleft()
                    result, decode_s, render_s = future.result()
                    if self._cancel_requested(generation):
                        continue
                    if result is not None:
                        self.rendered.emit(frame, result)
                        rendered_count += 1
                    self.progress.emit(done, total, name, decode_s, render_s)

            for processor in self._all_processors:
                processor.cleanup(release_source_cache=True, collect=False)
            self._emit_finished_unless_cancelled(generation, rendered_count)
        except Exception as exc:
            if self._emit_cancelled_if_requested(generation):
                return
            with self._cancel_lock:
                if self._active_generation == generation:
                    self._active_generation = None
            logger.exception("Background thumbnail refresh worker failure")
            self.error.emit(str(exc))

    def _render_one(self, frame: ThumbnailRenderInput, color_space: str, generation: int) -> tuple[Optional[np.ndarray], float, float]:
        """Decode and render one frame on a pool thread: the RGB result (None when skipped or
        failed), decode seconds and render seconds. A failed frame counts its time too."""
        decode_s = render_s = 0.0
        decoding = True
        started = time.perf_counter()
        processor = self._processors.get()
        try:
            buffer, meta = self._decode(frame, color_space)
            decode_s = time.perf_counter() - started
            decoding = False
            cam_xyz = meta.get("cam_xyz")
            if frame.icc_input_active:
                cam_xyz = wb_only_cam_xyz(cam_xyz)
            if self._cancel_requested(generation):
                return None, decode_s, render_s
            started = time.perf_counter()
            result, _metrics = processor.run_pipeline(
                buffer,
                frame.config,
                frame.file_info["hash"],
                render_size_ref=float(APP_CONFIG.preview_render_size),
                prefer_gpu=False,
                readback_metrics=False,
                wants_uv_grid=False,
                cache_stages=False,
                ir_buffer=meta.get("ir_preview"),
                detect_buffer=meta.get("detect_preview"),
                cam_xyz=cam_xyz,
                camera_wb=meta.get("camera_wb"),
                gain_slices=gain_slices_for_asset(frame.file_info),
                source_orientation=meta.get("orientation", 1),
            )
            render_s = time.perf_counter() - started
            return (np.ascontiguousarray(result[:, :, :3]) if isinstance(result, np.ndarray) else None), decode_s, render_s
        except Exception:
            if decoding:
                decode_s = time.perf_counter() - started
            else:
                render_s = time.perf_counter() - started
            if not self._cancel_requested(generation):
                logger.exception("Background thumbnail refresh skipped failed frame %s", frame.file_info.get("name"))
            return None, decode_s, render_s
        finally:
            self._processors.put(processor)


class NormalizationWorker(QObject):
    """
    Asynchronous batch normalization worker.
    Analyzes multiple RAW files to find a consistent baseline.
    """

    progress = pyqtSignal(int, int, str, bool)
    # floors, ceils, outlier hashes, pooled neutral axis (None when no frame has one).
    finished = pyqtSignal(tuple, tuple, list, object)
    # Detect Scenes: the hashes of each group of frames under another light.
    detected = pyqtSignal(list)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, preview_service) -> None:
        super().__init__()
        self._preview_service = preview_service
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        """Requests the running analysis stop; no baseline is applied."""
        self._cancel.set()

    def arm(self) -> None:
        """Clears a Stop left from an earlier run. Called on the GUI thread before dispatch,
        so a Stop pressed before the queued run starts is kept."""
        self._cancel.clear()

    @pyqtSlot(NormalizationTask)
    def process(self, task: NormalizationTask) -> None:
        """
        Executes analysis on a batch of files using parallel workers.
        """
        import asyncio

        import numpy as np

        from negpy.domain.interfaces import PipelineContext
        from negpy.features.exposure.normalization import (
            analyze_log_exposure_bounds,
            detect_light_groups,
            effective_crosstalk_matrix,
            measure_neutral_axis_from_log,
            pool_frame_bounds,
            pool_neutral_axis,
            prefilter_log_grid,
            resolve_analysis_region,
            unmix_log_image,
        )
        from negpy.features.process.models import ProcessMode
        from negpy.features.geometry.logic import meter_detection_key, resolve_meter_rect
        from negpy.features.geometry.processor import GeometryProcessor

        total = len(task.frames)
        limit = 1 if task.background else max(1, APP_CONFIG.max_workers // 2)
        semaphore = asyncio.Semaphore(limit)
        lock = asyncio.Lock()
        completed = 0

        async def _analyze_file(frame: NormalizationInput):
            nonlocal completed
            f_info = frame.file_info
            async with semaphore:
                if self._cancel.is_set():
                    return None
                try:
                    params = frame.config
                    # Roll-wide buffer and luma bounds from the current image, applied to every
                    # file, so one slider setting drives the whole batch baseline.
                    p = params.process
                    analysis_buffer = p.analysis_buffer if task.override_analysis_buffer is None else task.override_analysis_buffer
                    luma_range_clip = p.luma_range_clip if task.override_luma_range_clip is None else task.override_luma_range_clip
                    color_range_clip = p.color_range_clip if task.override_color_range_clip is None else task.override_color_range_clip
                    process_mode = params.process.process_mode
                    geometry = params.geometry

                    # Decode with the render path's WB (use_camera_wb = not effective linear RAW):
                    # the roll bounds apply to the render-decoded image, and another WB casts it.
                    raw, decode_meta = await asyncio.to_thread(
                        _decode_asset_preview_with_meta,
                        self._preview_service,
                        f_info,
                        params,
                        task.workspace_color_space,
                    )
                    # The flat field the render applies.
                    raw = await asyncio.to_thread(
                        flatfield_source,
                        raw,
                        params,
                        gain_slices_for_asset(f_info),
                        orientation=decode_meta.get("orientation", 1),
                    )
                    # Bounds must be measured on the same channel mix the preview normalizes.
                    sensor_matrix = effective_sensor_matrix(params.process)
                    if sensor_matrix is not None and preview_takes_unmix(params):
                        raw = await asyncio.to_thread(apply_sensor_correction, raw, sensor_matrix, params.process.sensor_unmix)

                    ctx = PipelineContext(
                        original_size=(raw.shape[1], raw.shape[0]),
                        scale_factor=1.0,
                        process_mode=process_mode,
                    )
                    transformed = await asyncio.to_thread(GeometryProcessor(geometry).process, raw, ctx)
                    has_crop = ctx.active_roi is not None
                    # A frame's own freehand region or meter rect is where it meters, the same as in its render.
                    meter = None
                    if not has_crop and not geometry.crop_from_auto and params.process.analysis_rect is None:
                        meter = params.process.meter_rect
                        if params.process.meter_key != meter_detection_key(geometry, raw.shape):
                            meter = await asyncio.to_thread(resolve_meter_rect, raw, geometry, APP_CONFIG.preview_render_size)
                    roi, buffer = resolve_analysis_region(
                        transformed.shape, ctx.active_roi, analysis_buffer, params.process.analysis_rect, meter
                    )

                    # Each frame's own unmix, as its render applies it.
                    unmix = effective_crosstalk_matrix(params.process, process_mode)
                    bounds = await asyncio.to_thread(
                        analyze_log_exposure_bounds,
                        transformed,
                        roi=roi,
                        analysis_buffer=buffer,
                        percentile_clip=luma_range_clip,
                        color_clip=color_range_clip,
                        unmix=unmix,
                    )
                    axis = None
                    if process_mode == ProcessMode.C41:
                        grid = unmix_log_image(prefilter_log_grid(transformed, roi, buffer), unmix)
                        axis = await asyncio.to_thread(measure_neutral_axis_from_log, grid, bounds, None, 0.0)

                    async with lock:
                        completed += 1
                        count = completed
                    self.progress.emit(count, total, f_info["name"], has_crop)
                    return bounds.floors, bounds.ceils, f_info["hash"], axis
                except Exception as e:
                    logger.error(f"Failed to analyze {f_info['name']}: {e}")
                    async with lock:
                        completed += 1
                        count = completed
                    self.progress.emit(count, total, f_info["name"], False)
                    return None

        async def _run_batch():
            tasks = [_analyze_file(f) for f in task.frames]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            if task.background:
                loop.set_default_executor(ThreadPoolExecutor(1, initializer=negpy_core.set_background_thread, initargs=(True,)))
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
            if task.detect:
                groups = detect_light_groups([r[3] for r in valid_results])
                self.detected.emit([[valid_results[i][2] for i in g] for g in groups])
                return

            pooled, outliers = pool_frame_bounds(
                np.array([r[0] for r in valid_results]),
                np.array([r[1] for r in valid_results]),
            )
            axis = pool_neutral_axis([r[3] for r in valid_results], outliers)
            self.finished.emit(pooled.floors, pooled.ceils, [r[2] for r, out in zip(valid_results, outliers) if out], axis)

        except Exception as e:
            logger.error(f"Batch Normalization failure: {e}")
            self.error.emit(str(e))
