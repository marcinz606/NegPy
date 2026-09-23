import math
import os
import time
from collections import Counter
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QTransform
from PyQt6.QtWidgets import QCheckBox, QMessageBox

from negpy.kernel.system.memory import available_system_memory_bytes
from negpy.kernel.system.text import count_of, plural
from negpy.kernel.image.logic import working_oetf_encode
from negpy.desktop.converters import ImageConverter
from negpy.desktop.render_memo import RenderMemo
from negpy.desktop.session import (
    AppState,
    DesktopSessionManager,
    ToolMode,
    _source_effective_bounds,
    resolve_asset_hdr,
    resolve_asset_rgbscan,
    resolve_asset_stitch,
)
from negpy.desktop.workers.export import ExportTask, ExportWorker, LinearOutputTask, find_export_conflicts, resolve_output_dir
from negpy.desktop.workers.render import (
    AssetDiscoveryTask,
    AssetDiscoveryWorker,
    AutoDetectAllSplitsTask,
    rgb_grouping_notice,
    rgb_nothing_matched_message,
    BatchAutoCropInput,
    BatchAutoCropResult,
    BatchAutoCropTask,
    BatchAutoCropWorker,
    NormalizationInput,
    NormalizationTask,
    NormalizationWorker,
    PreviewLoadTask,
    PreviewLoadWorker,
    RenderTask,
    RenderWorker,
    TestStripTask,
    ThumbnailRenderInput,
    ThumbnailRenderTask,
    ThumbnailRenderWorker,
    ThumbnailUpdateTask,
    ThumbnailWorker,
)
from negpy.desktop.workers.embedding import EmbeddingWorker
from negpy.desktop.workers.scan_worker import BatchRequest, PrescanRequest, RollPreviewRequest, ScanRequest, ScanWorker
from negpy.desktop.workers.library import LibrarySearchTask, LibrarySearchWorker
from negpy.desktop.workers.hdr import HdrTask, HdrWorker
from negpy.desktop.workers.stitch import StitchTask, StitchWorker
from negpy.features.hdr.models import ANCHOR_EV_UNSET, hdr_frame_paths, hdr_hash, hdr_name
from negpy.features.process.capture_color import apply_camera_matrix, camera_to_working_matrix, lightbox_level, wb_only_cam_xyz
from negpy.features.process.logic import (
    effective_highlight_reconstruction,
    effective_linear_raw,
    highlight_reconstruction_bakes_wb,
    narrowband_profile_active,
)
from negpy.features.stitch.models import stitch_hash, stitch_name
from negpy.desktop.workers.capture_worker import (
    CalibrationRequest,
    CaptureRequest,
    CaptureWorker,
    LiveViewRequest,
)
from negpy.domain.models import (
    PROOF_INTENT_LABELS,
    ColorSpace,
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
    ProofCondition,
    ProofIntent,
    WorkspaceConfig,
    canonical_crop_ratio,
    export_blocked,
    flat_export_config,
    flat_master_config,
    preset_from_export_config,
    resolve_preset_export,
)
from negpy.services.assets.composites import forget_composite, restore_maps
from negpy.services.assets import rolls
from negpy.services.assets.half_frame import (
    HalfGeometry,
    base_hash,
    diptych_configs,
    forget_split_scan,
    half_hash,
    half_of,
    is_composite,
    remap_workspace_config,
    remember_split_scans,
    split_scans,
)
from negpy.services.export.templating import path_safe, render_export_filename
from negpy.services.assets.sidecar import load_or_promote, write_sidecar
from negpy.features.exposure.analysis import (
    RING_GRID,
    STRIP_GRID,
    proof_grid,
    ring_cells,
    ring_overrides,
    rotate_grid,
    strip_cells,
    strip_overrides,
)
from negpy.features.exposure.logic import (
    calculate_wb_shifts,
    calculate_wb_shifts_from_log,
)
from negpy.features.altprocess.models import AltProcess
from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.geometry.logic import (
    apply_fine_rotation,
    autocrop_detection_key,
    detect_closest_aspect_ratio,
    enforce_roi_aspect_ratio,
    has_manual_crop,
)
from negpy.features.geometry.models import FINE_ROTATION_LIMIT, AutocropMode
from negpy.features.geometry.processor import CropProcessor, GeometryProcessor
from negpy.domain.interfaces import PipelineContext
from negpy.features.lab.models import LabConfig
from negpy.features.local.models import LocalAdjustmentsConfig
from negpy.features.process.models import (
    ProcessConfig,
    ProcessMode,
    auto_meter_for_positive_source,
    cast_removal_for_mode,
    invalidate_local_bounds,
    scan_setup_values,
)
from negpy.desktop.settings_catalog import BOUNDS_INPUT_FIELDS, section_of_field
from negpy.services.assets.thumbnails import asset_thumbnail_key
from negpy.services.assets import semantic_model
from negpy.kernel.system.paths import get_default_user_dir, get_resource_path
from negpy.features.retouch.logic import downsample_ir, trace_scratch
from negpy.features.retouch.models import HEAL_SIZE_MAX, HEAL_SIZE_MIN, RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.infrastructure.capture.settings import WhiteCaptureMode
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.prefetch_policy import MIN_RAM_RESERVE_BYTES
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.services.rendering.source_identity import source_token
from negpy.services.rendering.lens import lens_decode_token, metadata_lens_corrections
from negpy.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)

# Busy toasts are cleared when the frame lands; the timeout is only a backstop for a
# render that dies without reaching _on_render_finished.
_BUSY_TOAST_MS = 30000
# Batch owners that share norm_thread (and its CPU) with the background thumbnail
# refresh — the only ones a running refresh actually needs to get out of the way of.
_NORM_THREAD_BATCH_OWNERS = frozenset({"autocrop", "normalization"})
# A keep_preview reload (same file, e.g. a mode switch) skips the spinner so a fast
# lens-correction toggle doesn't flicker — but then shows nothing while a slow decode
# runs. This backstop arms it late, only if that decode is still in flight by then.
_KEEP_PREVIEW_SPINNER_DELAY_MS = 400


@dataclass(frozen=True)
class _PendingCaptureImport:
    """Capture intent carried across asynchronous discovery and session hydration."""

    process_mode: Optional[ProcessMode] = None
    detect_mode: bool = False
    capture_roll: str = ""
    capture_frame: Optional[int] = None


def _interactive_proxy(raw: Optional[Any]) -> Optional[Any]:
    """Preview-resolution stand-in for an HQ buffer; None when one is not needed.

    Every downstream cost scales with this buffer, so interactive frames render
    against it rather than the full-resolution original.
    """
    if not isinstance(raw, np.ndarray) or raw.ndim < 2:
        return None
    long_edge = max(raw.shape[:2])
    if long_edge <= APP_CONFIG.preview_render_size:
        return None
    scale = APP_CONFIG.preview_render_size / float(long_edge)
    w, h = max(1, round(raw.shape[1] * scale)), max(1, round(raw.shape[0] * scale))
    return cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)


def _interactive_ir_proxy(ir: Optional[Any], proxy: Optional[Any]) -> Optional[Any]:
    """IR plane matched to ``proxy``'s shape, or None when no proxy is in use.

    Not a plain resize: a defect is a *minimum* in IR transmittance, which area
    averaging removes.
    """
    if proxy is None or not isinstance(ir, np.ndarray):
        return None
    h, w = proxy.shape[:2]
    if ir.shape[:2] == (h, w):
        return ir
    return downsample_ir(ir, max(h, w), dims=(w, h))


def _capture_import_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _component_paths(files: List[Dict]) -> List[str]:
    """Every source file behind the loaded assets, composites decomposed into their parts.

    Re-discovery over an asset list that only saw primaries would drop the rest."""
    paths: List[str] = []
    for f in files:
        paths.append(f["path"])
        paths.extend(f[k] for k in ("green_path", "blue_path") if f.get(k))
        paths.extend(f.get("stitch_paths") or ())
        paths.extend(p for t in f.get("stitch_triplets") or () for p in t if p)
    return list(dict.fromkeys(paths))


def _autocrop_fingerprint(config: WorkspaceConfig, workspace_color_space: str) -> tuple:
    """Identity of every setting that changes detection pixels or crop coordinates."""
    geometry = config.geometry
    flatfield = config.flatfield
    rgbscan = config.rgbscan
    return (
        int(geometry.rotation),
        round(float(geometry.fine_rotation), 7),
        bool(geometry.flip_horizontal),
        bool(geometry.flip_vertical),
        str(geometry.autocrop_mode),
        str(geometry.autocrop_ratio),
        int(geometry.autocrop_offset),
        round(float(geometry.autocrop_rebate_trim), 4),
        round(float(geometry.distortion_k1), 9),
        bool(geometry.lens_distortion_from_metadata),
        bool(geometry.lens_ca_from_metadata),
        bool(flatfield.apply),
        str(flatfield.profile_id),
        bool(config.process.linear_raw),
        bool(rgbscan.enabled),
        str(rgbscan.green_path),
        str(rgbscan.blue_path),
        bool(rgbscan.align),
        str(workspace_color_space),
    )


@dataclass(frozen=True)
class _DiscoveryRequest:
    paths: tuple[str, ...]
    auto_open: bool
    restore_triplets: Optional[dict]
    replace_existing: bool
    reselect_path: Optional[str]
    rgb_scan: bool
    half_frame: bool
    half_frame_profile: Optional[dict] = None  # {crop_rect, split_x, gutter_thickness}
    half_frame_overrides: Optional[dict] = None  # {base_hash: {crop_rect, split_x, gutter_thickness}}
    hot_folder: bool = False


def baseline_compare_config(config: WorkspaceConfig) -> WorkspaceConfig:
    """
    The 'before' config for the before/after view: reset the creative sections to defaults
    while keeping process (mode + normalization bounds), geometry/crop, export and metadata,
    so it shows the un-graded auto conversion of the same framed image.

    Cast Removal's default is mode-dependent (cast_removal_for_mode), not the bare
    ExposureConfig default, or a transparency's 'before' would gray-balance a color the
    live render never applies.
    """
    baseline_exposure = ExposureConfig()
    baseline_exposure = replace(
        baseline_exposure,
        cast_removal_strength=cast_removal_for_mode(config.process.process_mode, baseline_exposure.cast_removal_strength),
    )
    return replace(
        config,
        exposure=baseline_exposure,
        lab=LabConfig(),
        local=LocalAdjustmentsConfig(),
        toning=ToningConfig(),
        finish=FinishConfig(),
        retouch=RetouchConfig(),
    )


# Solved knee fields under their slider names (sidebar/tone.py).
_KNEE_LABELS = {
    "shadow_grade": "Shadows Grade",
    "highlight_grade": "Highlights Grade",
    "midtone_gamma": "Snap",
}

# A background thumbnail refresh grows its own preview cache on top of whatever the
# navigation and Auto Crop All caches already hold; deferred and retried rather than
# started under memory pressure.
_THUMBNAIL_REFRESH_MEMORY_RETRY_MS = 5000


def history_step_label(prev: Optional[WorkspaceConfig], config: WorkspaceConfig, index: int) -> str:
    """List label for a history step: index + which config sections changed vs. the previous step."""
    if prev is None:
        return f"{index} · base"
    changed = [f.name for f in fields(config) if getattr(prev, f.name) != getattr(config, f.name)]
    return f"{index} · {', '.join(changed)}" if changed else f"{index} · —"


_NOTHING_TO_APPLY = "Nothing to apply — every card already follows the roll"


class AppController(QObject):
    """
    Main application orchestrator.
    Manages UI state synchronization, background workers, and render flow.
    """

    image_updated = pyqtSignal()
    preview_loaded = pyqtSignal()
    metrics_available = pyqtSignal(dict)
    loading_started = pyqtSignal()
    load_failed = pyqtSignal()
    # Emitted before the GPU engine frees its texture pool; the canvas samples a
    # pooled texture directly and must drop it first.
    gpu_textures_released = pyqtSignal()
    export_progress = pyqtSignal(int, int, str)
    export_finished = pyqtSignal(float, int)
    render_requested = pyqtSignal(RenderTask)
    preview_load_requested = pyqtSignal(PreviewLoadTask)
    normalization_requested = pyqtSignal(NormalizationTask)
    batch_autocrop_requested = pyqtSignal(BatchAutoCropTask)
    thumbnail_render_requested = pyqtSignal(ThumbnailRenderTask)
    thumbnail_refresh_state_changed = pyqtSignal(bool)  # a background refresh started/stopped running
    analysis_buffer_preview_requested = pyqtSignal(float)
    analysis_buffer_drag_changed = pyqtSignal(bool)
    rotation_guide_requested = pyqtSignal()
    crop_guide_changed = pyqtSignal()
    dust_overlay_changed = pyqtSignal()
    zones_overlay_changed = pyqtSignal(bool)
    grain_focuser_changed = pyqtSignal(bool)
    printing_notes_changed = pyqtSignal(bool)
    printing_notes_requested = pyqtSignal()  # the canvas holds the annotated pixels
    strip_requested = pyqtSignal(TestStripTask)
    test_strip_changed = pyqtSignal(bool)  # True = mosaic is up, False = cleared or building
    zone_pins_changed = pyqtSignal()
    rgb_scan_mode_changed = pyqtSignal(bool)  # the mode changed from somewhere other than its button
    half_frame_mode_changed = pyqtSignal(bool)  # a roll became active; each remembers its own toggle
    zone_arm_changed = pyqtSignal(object)  # armed zone, or None
    asset_discovery_requested = pyqtSignal(AssetDiscoveryTask)
    auto_detect_all_splits_requested = pyqtSignal(AutoDetectAllSplitsTask)
    library_search_requested = pyqtSignal(LibrarySearchTask)
    library_index_scan_requested = pyqtSignal(list)  # library_roots(), for whole-library indexing
    library_search_finished = pyqtSignal(int)  # frames found (0 = nothing matched)
    library_cleared = pyqtSignal()  # roots forgotten elsewhere — the panel must re-read them
    stitch_requested = pyqtSignal(object)
    hdr_requested = pyqtSignal(object)
    thumbnail_requested = pyqtSignal(list)
    thumbnail_cancel_requested = pyqtSignal()
    thumbnail_update_requested = pyqtSignal(ThumbnailUpdateTask)
    embedding_requested = pyqtSignal(list)
    thumbnail_activity_changed = pyqtSignal(str)
    tool_sync_requested = pyqtSignal()
    config_updated = pyqtSignal()
    monitor_profile_changed = pyqtSignal()
    compare_changed = pyqtSignal(bool)
    compare_frame_ready = pyqtSignal()
    flat_output_changed = pyqtSignal(bool)
    linear_output_changed = pyqtSignal(bool)
    flat_peek_changed = pyqtSignal(bool)
    negative_peek_changed = pyqtSignal(bool)
    embedded_peek_changed = pyqtSignal(bool)
    zoom_requested = pyqtSignal(float)
    zoom_changed = pyqtSignal(float)
    _render_cleanup_requested = pyqtSignal(object)  # texture to spare, or None
    status_message_requested = pyqtSignal(str, int, str)
    status_progress_requested = pyqtSignal(int, int)
    batch_started = pyqtSignal(str, bool)  # title, abortable
    batch_progress = pyqtSignal(int, int, str)  # current, total, label
    batch_finished = pyqtSignal()
    pixel_readout_rgb = pyqtSignal(object)  # (r255, g255, b255) tuple or None
    densitometer_readout = pyqtSignal(object)  # DensitometerReading or None
    tone_drag_changed = pyqtSignal(str)  # exposure field being slider-dragged; "" = drag ended
    local_drag_changed = pyqtSignal(bool)  # a selected-mask slider is under the mouse
    scan_devices_requested = pyqtSignal()
    scan_backend_requested = pyqtSignal(str)
    scan_requested = pyqtSignal(ScanRequest)
    scan_devices_ready = pyqtSignal(list)
    scan_progress = pyqtSignal(float, str)  # progress, phase name
    scan_finished = pyqtSignal(str)
    scan_error = pyqtSignal(str)
    scan_started = pyqtSignal()
    scan_cancelled = pyqtSignal()
    scan_ejected = pyqtSignal(bool)
    scan_eject_error = pyqtSignal(str)
    scan_frame_done = pyqtSignal(int, str)  # batch: frame number, rgb path
    scan_batch_finished = pyqtSignal(list)  # batch: all completed rgb paths
    scan_batch_requested = pyqtSignal(BatchRequest)
    scan_eject_requested = pyqtSignal(str)
    scan_roll_preview_requested = pyqtSignal(RollPreviewRequest)
    scan_roll_preview_ready = pyqtSignal(object)  # one RollPreview per strip slot
    scan_roll_preview_finished = pyqtSignal()
    scan_prescan_requested = pyqtSignal(PrescanRequest)
    scan_prescan_ready = pyqtSignal(object)  # ScanResult from a low-DPI full-window preview
    scan_prescan_error = pyqtSignal(str)
    capture_light_requested = pyqtSignal(int, int, int, int, str)
    capture_requested = pyqtSignal(CaptureRequest)
    capture_light_set = pyqtSignal(int, int, int, int)
    capture_progress = pyqtSignal(float)
    capture_channel = pyqtSignal(str)  # "R"/"G"/"B" as each triplet channel starts
    capture_camera_setting_applied = pyqtSignal(str)  # a set_camera_setting call ran to completion
    capture_live_view_failed = pyqtSignal(str)  # preview thread died after retries; session dropped
    capture_live_view_unsupported = pyqtSignal(str)  # body advertises no preview; none was attempted
    capture_finished = pyqtSignal(list)
    capture_cancelled = pyqtSignal()
    capture_error = pyqtSignal(str)
    capture_status = pyqtSignal(str)
    live_view_requested = pyqtSignal(LiveViewRequest)
    live_view_stop_requested = pyqtSignal()
    camera_session_close_requested = pyqtSignal()
    live_view_focus_magnifier_requested = pyqtSignal(bool)
    live_view_focus_magnifier_pos_requested = pyqtSignal(int, int)
    live_view_camera_setting_requested = pyqtSignal(str, int)
    capture_live_view_started = pyqtSignal(str)
    calibration_requested = pyqtSignal(CalibrationRequest)
    capture_calibration_progress = pyqtSignal(float, str)
    capture_calibration_finished = pyqtSignal(object)
    capture_calibration_exposure = pyqtSignal(str)  # "over"/"under": target unreachable, aborted, no preset
    poll_connection_requested = pyqtSignal(str)  # light port (auto-poll)
    connection_polled = pyqtSignal(dict)  # {usb_ok, usb_model, light_ok, light_detail}
    poll_light_temp_requested = pyqtSignal(str)  # light port (temp-only poll, runs even mid-live-view)
    light_temp_polled = pyqtSignal(object)  # Scanlight LED temperature °C, or None

    def __init__(self, session_manager: DesktopSessionManager):
        super().__init__()
        self.session = session_manager
        self.state: AppState = session_manager.state
        self._thumb_config: Optional[WorkspaceConfig] = None
        self._active_diptych_memo: tuple[str, Optional[tuple[dict, tuple[WorkspaceConfig, WorkspaceConfig]]]] = ("", None)
        # Halves already known to hold a real edit; spares _may_persist_measured_bounds a
        # repeat lookup per render. Only ever grows, since a row is never deleted mid-session.
        self._measured_half_rows: set[str] = set()
        self._first_render_t0: Optional[float] = None
        self._export_start_time = 0.0
        self._export_failures = 0
        self._discovery_running = False
        self._auto_open_after_discovery = False
        self._replace_after_discovery = False
        self._reselect_after_discovery: Optional[str] = None
        self._announce_rgb = False
        self._pending_capture_imports: Dict[str, _PendingCaptureImport] = {}
        self._pending_asset_discoveries: List[_DiscoveryRequest] = []
        self._active_discovery_keys: frozenset[str] = frozenset()
        self._pending_scanned_file: Optional[str] = None
        self._gpu_fallback_notified = False
        self._cleaned_up = False
        self._active_batch: Optional[str] = None
        self._active_batch_title = ""
        self._active_batch_abortable = False
        self._batch_serial = 0
        self._active_batch_token: Optional[int] = None
        self._normalization_scope: Optional[str] = None
        self._scene_queue: List[str] = []
        # True while a hot-folder-triggered discovery owns the batch lane. Thumbnail
        # loading is an independent background queue and never owns this state.
        self._hot_folder_sequence_active = False
        self._autocrop_batch_token: Optional[int] = None
        self._autocrop_dispatched = 0
        self._autocrop_preflight_skipped = 0
        self._autocrop_cancel_requested = False
        # Background thumbnail refresh runs off the shared batch lane entirely — it must
        # never block Export or another user-triggered batch — so it tracks its own
        # generation instead of `_batch_serial`/`_active_batch_token`.
        self._thumbnail_render_generation = 0
        self._thumbnail_render_running = False
        # Hashes dispatched in the current generation with no `rendered` signal yet.
        self._thumbnail_render_pending: set[str] = set()
        # Hashes a pre-emption cut short, retried once the pre-empting batch is done.
        self._thumbnail_render_resume: set[str] = set()
        # True between cancel_thumbnail_refresh() and the worker's cancelled signal
        # landing — marks that cancellation as a user stop, not a pre-emption, so the
        # cancelled handler discards the backlog instead of resuming it.
        self._thumbnail_render_user_cancelled = False
        self.flush_export_settings: Optional[Callable[[], None]] = None
        # A rotate/flip on a frame with no cached thumbnail yet (generate_missing_thumbnails
        # is still decoding it) has nothing to turn; the pending turn recorded here is applied
        # to that decode's own result instead, once it lands — the disk cache it writes is
        # otherwise permanent, since get_thumbnail_worker serves it straight back on every
        # later request without re-deriving orientation.
        self._thumbnail_pending_correction: Dict[str, List[Any]] = {}

        # Which caller owns the embedding batch currently running on the shared
        # EmbeddingWorker -- "embeddings" (in-session) or "library_index" (whole
        # library). Set right before each embedding_requested.emit.
        self._embedding_batch_owner = "embeddings"
        # Requested while index_library() was still walking/hashing -- before there is
        # any embedding batch for EmbeddingWorker.cancel() to actually stop.
        self._library_index_cancelled = False

        self.preview_service = PreviewManager()
        self.batch_autocrop_preview_service = PreviewManager()
        self.thumbnail_render_preview_service = PreviewManager()
        self.watcher = FolderWatchService()
        self.asset_store = LocalAssetStore(APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir)
        self.asset_store.initialize()
        # Lazy, UI-thread-only CLIP text tower for embedding search queries -- a
        # separate instance from EmbeddingWorker's, which runs on its own thread.
        self._search_clip_model: Optional[semantic_model.ClipModel] = None

        # Thread management
        self.render_thread = QThread()
        self.render_worker = RenderWorker()
        self.render_worker.moveToThread(self.render_thread)
        self.render_thread.start()

        self.export_thread = QThread()
        self.export_worker = ExportWorker()
        self.export_worker.moveToThread(self.export_thread)
        # Shares the export thread: the batch lane serializes them anyway.
        self.stitch_worker = StitchWorker()
        self.stitch_worker.moveToThread(self.export_thread)
        self.hdr_worker = HdrWorker()
        self.hdr_worker.moveToThread(self.export_thread)
        self.export_thread.start()

        self.thumb_thread = QThread()
        self.thumb_worker = ThumbnailWorker(self.asset_store)
        self.thumb_worker.moveToThread(self.thumb_thread)
        # Shares the thumbnail thread: same I/O/CPU profile, and embedding indexing
        # runs right after a thumbnail batch finishes, never alongside it.
        self.embedding_worker = EmbeddingWorker(self.asset_store, self.session.repo)
        self.embedding_worker.moveToThread(self.thumb_thread)
        self.thumb_thread.start()

        self.norm_thread = QThread()
        self.norm_worker = NormalizationWorker(self.preview_service)
        self.norm_worker.moveToThread(self.norm_thread)
        self.batch_autocrop_worker = BatchAutoCropWorker(self.batch_autocrop_preview_service)
        self.batch_autocrop_worker.moveToThread(self.norm_thread)
        self.thumbnail_render_worker = ThumbnailRenderWorker(self.thumbnail_render_preview_service)
        self.thumbnail_render_worker.moveToThread(self.norm_thread)
        self.norm_thread.start()

        self.discovery_thread = QThread()
        self.discovery_worker = AssetDiscoveryWorker()
        self.discovery_worker.moveToThread(self.discovery_thread)
        # Shares the discovery thread: a library search ends in a discovery anyway,
        # and neither should ever run while the other is walking the disk.
        self.library_worker = LibrarySearchWorker()
        self.library_worker.moveToThread(self.discovery_thread)
        self.discovery_thread.start()

        self.preview_load_thread = QThread()
        self.preview_load_worker = PreviewLoadWorker(self.preview_service)
        self.preview_load_worker.moveToThread(self.preview_load_thread)
        self.preview_load_thread.start()

        self.scan_thread = QThread()
        self.scan_worker = ScanWorker()
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.start()

        self.capture_thread = QThread()
        self.capture_worker = CaptureWorker()
        self.capture_worker.moveToThread(self.capture_thread)
        # Started lazily on first capture use (_ensure_capture_thread). A *running* QThread
        # aborts if destroyed without quit(), and controller unit tests never scan, so an
        # unstarted thread stays invisible to their teardown loops. The app starts it as
        # soon as the Camera Scanning tab polls or the user acts.
        self._capture_thread_started = False

        self.canvas: Any = None
        self._is_rendering = False
        self._busy_toast = False
        self._pending_render_task: Any = None

        # Correction states share the bounded cache, so toggling back can paint immediately.
        self._render_memo = RenderMemo(keep_variants=True)
        # (source_hash, memo_key, content_rect) of the on-screen GPU render; load_file
        # files its texture under this on the way out.
        self._last_render_identity: Optional[tuple] = None
        self._expected_render_key = ""
        self._render_memo.large_entries = self.state.hq_preview
        # Test strips, keyed density/grade-blind (see _strip_memo_key). Four mosaics per
        # entry, hence the conservative budget.
        self._strip_memo = RenderMemo()
        self._strip_memo.large_entries = True

        self._render_debounce = QTimer()
        self._render_debounce.setSingleShot(True)
        self._render_debounce.setInterval(50)
        self._render_debounce.timeout.connect(self.request_render)

        self._crop_bounds_dirty = False
        self._zone_preview_shown = False
        self._pin_dragging = False
        self._pin_solution: Optional[Any] = None

        self._cursor_readout_timer = QTimer()
        self._cursor_readout_timer.setSingleShot(True)
        self._cursor_readout_timer.setInterval(33)
        self._cursor_readout_timer.timeout.connect(self._emit_pixel_readout)
        self._pending_cursor_nx: Optional[float] = None
        self._pending_cursor_ny: Optional[float] = None
        self._prefetch_gen = 0
        #: The texture the canvas is displaying, kept alive across back-to-back reloads.
        self._spared_texture: Optional[GPUTexture] = None
        self._preview_load_t0 = 0.0
        self._requested_file_path: str = ""
        self._foreground_preview_generation: Optional[int] = None
        self._neighbor_prefetch_generation: Optional[int] = None
        self._neighbor_prefetch_queue: list[PreviewLoadTask] = []
        self._prefetch_in_flight_generation: Optional[int] = None
        self._thumbnail_queue_active = False
        self._thumbnails_paused_for_foreground = False

        self._connect_signals()

    def register_canvas(self, canvas: Any) -> None:
        """
        Registers the canvas and connects its signals.
        """
        self.canvas = canvas
        self.zoom_requested.connect(self.canvas.set_zoom)
        self.canvas.zoom_changed.connect(self.zoom_changed.emit)
        self.canvas.cursor_position_changed.connect(self.on_cursor_moved)
        self.canvas.cursor_left_canvas.connect(self.on_cursor_left)

        from negpy.desktop.view.canvas.toolbar import CANVAS_COLORS

        idx = self.state.canvas_bg_index
        _, (r, g, b), _ = CANVAS_COLORS[idx]
        self.canvas.set_background_color(r, g, b)

    def on_cursor_moved(self, nx: float, ny: float) -> None:
        self._pending_cursor_nx = nx
        self._pending_cursor_ny = ny
        if not self._cursor_readout_timer.isActive():
            self._cursor_readout_timer.start()

    def on_cursor_left(self) -> None:
        self._pending_cursor_nx = None
        self._pending_cursor_ny = None
        self.pixel_readout_rgb.emit(None)
        self.densitometer_readout.emit(None)

    def _emit_pixel_readout(self) -> None:
        nx, ny = self._pending_cursor_nx, self._pending_cursor_ny
        if nx is None or ny is None or self.canvas is None:
            return
        rgb = self.canvas.get_pixel_rgb(nx, ny)
        if rgb is None:
            return
        r, g, b = rgb
        r255 = int(round(max(0.0, min(1.0, r)) * 255))
        g255 = int(round(max(0.0, min(1.0, g)) * 255))
        b255 = int(round(max(0.0, min(1.0, b)) * 255))
        self.pixel_readout_rgb.emit((r255, g255, b255))
        self.densitometer_readout.emit(self._compute_densitometer_reading(nx, ny, rgb))

    def _compute_densitometer_reading(self, nx: float, ny: float, display_rgb: tuple) -> Optional[Any]:
        """Probe the normalized-log frame under the cursor; None when unavailable."""
        from negpy.features.exposure.densitometer import compute_reading

        bounds = self.state.last_metrics.get("final_bounds") or self.state.last_metrics.get("log_bounds")
        if bounds is None:
            return None
        val = self._sample_normalized_log(nx, ny)
        if val is None:
            return None
        return compute_reading(val, bounds, display_rgb)

    def _sample_normalized_log(self, nx: float, ny: float, radius: int = 0) -> Optional[Tuple[float, float, float]]:
        """Mean of the (2·radius+1)² normalized-log patch at content-normalized nx,ny;
        None when no frame is probed. Shared by the hover probe (1×1) and zone pins."""
        from negpy.features.exposure.densitometer import map_display_to_norm

        metrics = self.state.last_metrics
        nl = metrics.get("normalized_log")
        if nl is None or self.canvas is None:
            return None
        disp = self.canvas.display_size()
        if disp is None:
            return None
        if isinstance(nl, np.ndarray):
            norm_h, norm_w = nl.shape[:2]
        else:
            norm_w, norm_h = nl.width, nl.height
        # nx,ny arrive content-normalized, because the overlay subtracts the border.
        # Passing content_rect here would compensate twice.
        pos = map_display_to_norm(
            nx,
            ny,
            disp[0],
            disp[1],
            None,
            metrics.get("active_roi"),
            self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW),
            norm_w,
            norm_h,
        )
        if pos is None:
            return None
        x, y = pos
        x0, x1 = max(0, x - radius), min(norm_w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(norm_h, y + radius + 1)
        try:
            if isinstance(nl, np.ndarray):
                val = nl[y0:y1, x0:x1].reshape(-1, nl.shape[2]).mean(axis=0)
            else:
                region = np.asarray(nl.readback_region(x0, y0, x1 - x0, y1 - y0), dtype=np.float32)
                val = region[..., :3].reshape(-1, 3).mean(axis=0)
        except Exception:
            return None
        return (float(val[0]), float(val[1]), float(val[2]))

    def set_status(self, message: str, timeout: int = 0, kind: str = "info") -> None:
        """kind: "info" | "warning" | "error" — the HUD colours the toast by it."""
        self.status_message_requested.emit(message, timeout, kind)

    def adjust_brush_size(self, delta: float) -> None:
        """Nudge the shared heal/scratch/exclusion brush diameter, clamped to its slider's
        range. The canvas routes here rather than to the widget: the slider is out of reach
        while an exclusion is painted with no tool active."""
        conf = self.state.config.retouch
        size = int(round(min(HEAL_SIZE_MAX, max(HEAL_SIZE_MIN, conf.manual_dust_size + delta))))
        if size == conf.manual_dust_size:
            return
        self.session.update_config(
            replace(self.state.config, retouch=replace(conf, manual_dust_size=size)),
            persist=True,
            render=False,
        )
        self.set_status(f"Brush Size {size} px", 1500)

    def _connect_signals(self) -> None:
        self.render_requested.connect(self.render_worker.process)
        self.strip_requested.connect(self.render_worker.build_strip)
        self._render_cleanup_requested.connect(self.render_worker.cleanup)
        self.render_worker.strip_finished.connect(self.on_strip_finished)
        self.render_worker.strip_progress.connect(self.on_strip_progress)
        self.render_worker.busy.connect(self._on_render_busy)
        self.render_worker.finished.connect(self._on_render_finished)
        self.render_worker.metrics_updated.connect(self._on_metrics_updated)
        self.render_worker.error.connect(self._on_render_error)
        self.render_worker.error.connect(self._on_strip_error)

        self.export_worker.progress.connect(self.export_progress.emit)
        self.export_worker.progress.connect(self._on_batch_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.cancelled.connect(self._on_export_batch_cancelled)
        self.export_worker.error.connect(self._on_export_task_error)

        self.stitch_requested.connect(self.stitch_worker.run)
        self.stitch_worker.progress.connect(self._on_batch_progress)
        self.stitch_worker.registered.connect(self._on_stitch_registered)
        self.stitch_worker.cancelled.connect(self._on_stitch_cancelled)
        self.stitch_worker.error.connect(self._on_stitch_error)

        self.hdr_requested.connect(self.hdr_worker.run)
        self.hdr_worker.progress.connect(self._on_batch_progress)
        self.hdr_worker.solved.connect(self._on_hdr_solved)
        self.hdr_worker.cancelled.connect(self._on_hdr_cancelled)
        self.hdr_worker.error.connect(self._on_hdr_error)

        self.thumbnail_requested.connect(self.thumb_worker.generate)
        self.thumbnail_cancel_requested.connect(self.thumb_worker.cancel)
        self.thumb_worker.activity.connect(self._on_thumbnail_activity)
        self.thumbnail_update_requested.connect(self.thumb_worker.update_rendered)
        self.thumb_worker.partial.connect(self._apply_thumbnails)
        self.thumb_worker.rendered_finished.connect(self._on_rendered_thumbnail)

        self.embedding_requested.connect(self.embedding_worker.generate)
        self.embedding_worker.progress.connect(self._on_embedding_progress)
        self.embedding_worker.partial.connect(self._apply_embeddings)
        self.embedding_worker.finished.connect(self._on_embeddings_finished)
        self.embedding_worker.error.connect(self._on_embedding_batch_error)

        self.normalization_requested.connect(self.norm_worker.process)
        self.norm_worker.progress.connect(self._on_normalization_progress)
        self.norm_worker.finished.connect(self._on_normalization_finished)
        self.norm_worker.cancelled.connect(self._on_normalization_cancelled)
        self.norm_worker.error.connect(self._on_normalization_error)

        self.batch_autocrop_requested.connect(self.batch_autocrop_worker.process)
        self.batch_autocrop_worker.progress.connect(self._on_batch_autocrop_progress)
        self.batch_autocrop_worker.finished.connect(self._on_batch_autocrop_finished)
        self.batch_autocrop_worker.cancelled.connect(self._on_batch_autocrop_cancelled)
        self.batch_autocrop_worker.error.connect(self._on_batch_autocrop_error)

        self.thumbnail_render_requested.connect(self.thumbnail_render_worker.process)
        self.thumbnail_render_worker.progress.connect(self._on_thumbnail_render_progress)
        self.thumbnail_render_worker.rendered.connect(self._on_thumbnail_rendered)
        self.thumbnail_render_worker.finished.connect(self._on_thumbnail_render_finished)
        self.thumbnail_render_worker.cancelled.connect(self._on_thumbnail_render_cancelled)
        self.thumbnail_render_worker.error.connect(self._on_thumbnail_render_error)
        self.session.frames_edited_offscreen.connect(self.refresh_thumbnails_for)

        self.asset_discovery_requested.connect(self.discovery_worker.process)
        self.discovery_worker.progress.connect(self._on_discovery_progress)
        self.discovery_worker.finished.connect(self._on_discovery_finished)
        self.discovery_worker.error.connect(self._on_discovery_batch_error)
        self.discovery_worker.rgb_grouped.connect(self._on_rgb_grouped)
        self.auto_detect_all_splits_requested.connect(self.discovery_worker.process_auto_detect_all_splits)
        self.discovery_worker.splits_detected.connect(self._on_splits_detected)
        self.library_search_requested.connect(self.library_worker.search)
        self.library_worker.progress.connect(self._on_library_walk_progress)
        self.library_worker.finished.connect(self._on_library_search_finished)
        self.library_worker.error.connect(self._on_library_search_error)
        self.library_index_scan_requested.connect(self.library_worker.scan_for_indexing)
        self.library_worker.indexing_scanned.connect(self._on_library_index_scanned)

        self.preview_load_requested.connect(self.preview_load_worker.process)
        self.preview_load_worker.splash.connect(self._on_splash_preview)
        self.preview_load_worker.finished.connect(self._on_preview_loaded)
        self.preview_load_worker.vram_capped.connect(self._on_hq_preview_vram_capped)
        self.preview_load_worker.error.connect(self._on_preview_load_error)
        self.preview_load_worker.load_failed.connect(self._on_preview_load_failed)
        self.preview_load_worker.prefetch_finished.connect(self._on_neighbor_prefetch_finished)

        self.scan_devices_requested.connect(self.scan_worker.list_devices)
        self.scan_backend_requested.connect(self.scan_worker.set_backend)
        self.scan_worker.devices_ready.connect(self.scan_devices_ready.emit)
        self.scan_worker.progress.connect(self.scan_progress.emit)
        self.scan_worker.finished.connect(self._on_scan_finished)
        self.scan_worker.error.connect(self.scan_error.emit)
        self.scan_requested.connect(self.scan_worker.run_scan)
        self.scan_batch_requested.connect(self.scan_worker.run_batch)
        self.scan_eject_requested.connect(self.scan_worker.eject)
        self.scan_worker.cancelled.connect(self.scan_cancelled.emit)
        self.scan_worker.frame_done.connect(self.scan_frame_done.emit)
        self.scan_worker.batch_finished.connect(self._on_scan_batch_finished)
        self.scan_worker.ejected.connect(self.scan_ejected.emit)
        self.scan_worker.eject_error.connect(self.scan_eject_error.emit)
        self.scan_roll_preview_requested.connect(self.scan_worker.run_roll_preview)
        self.scan_worker.roll_preview_ready.connect(self.scan_roll_preview_ready.emit)
        self.scan_worker.roll_preview_finished.connect(self.scan_roll_preview_finished.emit)
        self.scan_prescan_requested.connect(self.scan_worker.run_prescan)
        self.scan_worker.prescan_ready.connect(self.scan_prescan_ready.emit)
        self.scan_worker.prescan_error.connect(self.scan_prescan_error.emit)
        self.capture_light_requested.connect(self.capture_worker.set_light)
        self.capture_requested.connect(self.capture_worker.run_capture)
        self.capture_worker.light_set.connect(self.capture_light_set.emit)
        self.capture_worker.progress.connect(self.capture_progress.emit)
        self.capture_worker.channel.connect(self.capture_channel.emit)
        self.capture_worker.camera_setting_applied.connect(self.capture_camera_setting_applied.emit)
        self.capture_worker.live_view_failed.connect(self.capture_live_view_failed.emit)
        self.capture_worker.live_view_unsupported.connect(self.capture_live_view_unsupported.emit)
        self.capture_worker.finished.connect(self._on_capture_finished)
        self.capture_worker.cancelled.connect(self.capture_cancelled.emit)
        self.capture_worker.error.connect(self.capture_error.emit)
        self.capture_worker.status.connect(self.capture_status.emit)
        self.live_view_requested.connect(self.capture_worker.start_live_view)
        self.live_view_stop_requested.connect(self.capture_worker.stop_live_view)
        self.camera_session_close_requested.connect(self.capture_worker.close_camera_session)
        self.live_view_focus_magnifier_requested.connect(self.capture_worker.set_focus_magnifier)
        self.live_view_focus_magnifier_pos_requested.connect(self.capture_worker.set_focus_magnifier_pos)
        self.live_view_camera_setting_requested.connect(self.capture_worker.set_camera_setting)
        self.capture_worker.live_view_started.connect(self.capture_live_view_started.emit)
        self.calibration_requested.connect(self.capture_worker.run_calibration)
        self.capture_worker.calibration_progress.connect(self.capture_calibration_progress.emit)
        self.capture_worker.calibration_finished.connect(self.capture_calibration_finished.emit)
        self.capture_worker.calibration_exposure.connect(self.capture_calibration_exposure.emit)
        self.poll_connection_requested.connect(self.capture_worker.poll_connection)
        self.capture_worker.poll_status.connect(self.connection_polled.emit)
        self.poll_light_temp_requested.connect(self.capture_worker.poll_light_temp)
        self.capture_worker.light_temp_polled.connect(self.light_temp_polled.emit)

        self.session.active_file_changing.connect(self._update_thumbnail_from_state)
        self.session.session_emptied.connect(self._render_memo.clear)
        self.session.session_emptied.connect(self._strip_memo.clear)
        self.session.file_selected.connect(self._on_file_selected_load)
        self.session.state_changed.connect(self.config_updated.emit)
        self.session.state_changed.connect(self._render_debounce.start)
        self.session.files_changed.connect(self._render_debounce.start)

    def generate_missing_thumbnails(self) -> None:
        missing = [f for f in self.state.uploaded_files if asset_thumbnail_key(f) not in self.state.thumbnails]
        if missing:
            self._thumbnail_queue_active = True
            self.thumb_worker.cancel_pending()
            # Copies, carrying each frame's stored film process. The source decode cannot
            # tell a slide from a negative reliably, and inverting a positive is what put
            # negatives in the filmstrip. They are copies because these dicts cross to a
            # worker thread and uploaded_files must not grow a stale mode.
            self.thumbnail_requested.emit([{**f, "process_mode": self.session.stored_process_mode(f)} for f in missing])

    def _turn_thumbnails(self, keys: list, qt_transform: QTransform, pil_transpose: Any) -> bool:
        """Turns each cached thumbnail in place by one step. Memory and disk turn from
        their OWN current content, never from each other: a frame that rendered on
        the canvas has a memory icon ahead of its disk JPEG (persisted lazily), and
        deriving one from the other would clobber whichever is more current. A frame
        with no disk-cached thumbnail yet is still mid-decode in generate_missing_
        thumbnails, so the turn is queued instead and replayed onto that decode's own
        result in _apply_thumbnails."""
        changed = False
        for key in keys:
            icon = self.state.thumbnails.get(key)
            sizes = icon.availableSizes() if icon is not None else []
            if sizes:
                self.state.thumbnails[key] = QIcon(icon.pixmap(sizes[0]).transformed(qt_transform))
                changed = True
            cached = self.asset_store.get_thumbnail(key)
            if cached is not None:
                self.asset_store.save_thumbnail(key, cached.transpose(pil_transpose))
            else:
                self._thumbnail_pending_correction.setdefault(key, []).append(pil_transpose)
        return changed

    def rotate_thumbnails(self, keys: list, direction: int) -> None:
        """Turns each cached thumbnail in place by a quarter-turn, for a batch
        rotate on frames that are not the active one."""
        from PIL import Image

        turns = direction % 4
        if not turns or not keys:
            return
        pil_transpose = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}[turns]
        qt_transform = QTransform().rotate(-90 * direction)
        changed = self._turn_thumbnails(keys, qt_transform, pil_transpose)
        # push_external_history flagged these stale for the bulk geometry write; the turn
        # above already brings the cached bitmap into agreement with it, so no render is owed.
        self.state.stale_thumbnails.difference_update(keys)
        if changed:
            self.session.asset_model.refresh()

    def flip_thumbnails(self, keys: list, horizontal: bool) -> None:
        """Mirrors each cached thumbnail in place. See _turn_thumbnails for why
        memory and disk turn independently rather than one deriving from the other."""
        from PIL import Image

        if not keys:
            return
        pil_transpose = Image.Transpose.FLIP_LEFT_RIGHT if horizontal else Image.Transpose.FLIP_TOP_BOTTOM
        qt_transform = QTransform().scale(-1, 1) if horizontal else QTransform().scale(1, -1)
        changed = self._turn_thumbnails(keys, qt_transform, pil_transpose)
        self.state.stale_thumbnails.difference_update(keys)
        if changed:
            self.session.asset_model.refresh()

    def clear_thumbnail_cache(self) -> None:
        """Drops cached thumbnails on disk and in memory, then regenerates loaded ones."""
        self.thumb_worker.cancel_pending()
        self.thumbnail_cancel_requested.emit()
        self.asset_store.clear_thumbnails()
        # Must precede generate_missing_thumbnails: it only enqueues names absent here.
        self.state.thumbnails.clear()
        self.state.rendered_thumbnails.clear()
        self.state.stale_thumbnails.clear()
        self.session.asset_model.refresh()
        self.generate_missing_thumbnails()

    def _set_thumbnail(self, key: str, pil_img: Any) -> bool:
        """False when the image will not decode. PIL decodes lazily, so a truncated
        file raises here — on the UI thread — not in the worker that supplied it."""
        try:
            u8_arr = np.array(pil_img.convert("RGB"))
        except Exception as e:
            logger.warning(f"Unreadable thumbnail for {key}: {e}")
            return False
        self.state.thumbnails[key] = QIcon(QPixmap.fromImage(ImageConverter.to_qimage(u8_arr)))
        return True

    def _apply_thumbnails(self, new_thumbs: Dict[str, Any]) -> Set[str]:
        """Commit completed thumbnails to the filmstrip. Returns the
        keys whose image would not decode."""
        broken = set()
        loaded = {asset_thumbnail_key(f) for f in self.state.uploaded_files}
        # Stale beyond this roll's own lifetime: drop it rather than replay it onto an
        # unrelated future frame that happens to share the same content hash.
        for key in set(self._thumbnail_pending_correction) - loaded:
            del self._thumbnail_pending_correction[key]
        for key, pil_img in new_thumbs.items():
            # A frame that already rendered on the canvas has the correct inverted
            # thumbnail, so keep this batch from overwriting it with the placeholder.
            if pil_img and key in loaded and key not in self.state.rendered_thumbnails:
                pending = self._thumbnail_pending_correction.pop(key, None)
                if pending:
                    for pil_transpose in pending:
                        pil_img = pil_img.transpose(pil_transpose)
                    # This decode's own disk write (inside get_thumbnail_worker) already
                    # landed in the old orientation and is served back verbatim from then
                    # on, so it needs the same correction, not just the in-memory icon.
                    self.asset_store.save_thumbnail(key, pil_img)
                if not self._set_thumbnail(key, pil_img):
                    broken.add(key)
        self.session.asset_model.refresh()
        return broken

    def _on_thumbnail_activity(self, key: str) -> None:
        was_active = self._thumbnail_queue_active
        self._thumbnail_queue_active = bool(key)
        self.thumbnail_activity_changed.emit(key)
        if was_active and not key:
            # Thumbnails first: each file's preview is on disk by now, so the embedding
            # pass reuses it instead of decoding the RAW a second time.
            self.generate_missing_embeddings()

    def embed_search_query(self, text: str) -> Optional[np.ndarray]:
        """Embeds free text for search-by-meaning ranking. None if the model is not
        downloaded yet, so the caller falls back to the plain filter."""
        if not text or not semantic_model.clip_model_ready():
            return None
        if self._search_clip_model is None:
            self._search_clip_model = semantic_model.ClipModel()
        return self._search_clip_model.embed_text(text)

    def generate_missing_embeddings(self) -> None:
        """Indexes uploaded_files for search by meaning. Runs after thumbnails so each
        file's preview is already on disk (get_thumbnail_worker's own cache), avoiding a
        second RAW decode. No-op with the feature off or the model not yet downloaded."""
        if not self.state.semantic_search_enabled or not semantic_model.clip_model_ready():
            return
        hashes = [f["hash"] for f in self.state.uploaded_files]
        self.state.embeddings.update(self.session.repo.load_embeddings_for(hashes, semantic_model.MODEL_VERSION))
        # A live semantic filter excludes any file absent from state.embeddings, and an
        # already-indexed batch leaves `missing` empty, so nothing below refreshes it.
        self.session.asset_model.refresh()
        missing = [f for f in self.state.uploaded_files if f["hash"] not in self.state.embeddings]
        if not missing:
            return
        if self._begin_batch("embeddings", "Indexing for search by meaning", abortable=False) is None:
            return
        self._embedding_batch_owner = "embeddings"
        self.set_status("Indexing for search by meaning…")
        self.embedding_requested.emit([{**f, "process_mode": self.session.stored_process_mode(f)} for f in missing])

    def index_library(self) -> None:
        """Indexes every file under library_roots() for search by meaning, not just
        the currently open roll -- an explicit, cancellable action (never automatic),
        since it means decoding and embedding every photo in the library once."""
        if not self.state.semantic_search_enabled or not semantic_model.clip_model_ready():
            self.set_status("Turn on Search by meaning in Preferences first", 4000, kind="warning")
            return
        roots = self.library_roots()
        if not roots:
            self.set_status("Add a library folder first", 4000)
            return
        if self._begin_batch("library_index", "Indexing library for search by meaning", abortable=True) is None:
            return
        self._embedding_batch_owner = "library_index"
        self._library_index_cancelled = False
        self.set_status("Scanning library…")
        self.library_index_scan_requested.emit(roots)

    def _on_library_index_scanned(self, files: List[Dict[str, Any]]) -> None:
        """Every library file, hashed. Only the ones missing from image_embeddings
        under the current model go on to the expensive decode+embed pass -- indexing
        again after a cancel or a later import redoes none of the finished work."""
        if self._active_batch != "library_index":
            return  # a keyword search's own walk landed here; nothing to index
        if self._library_index_cancelled:
            self._end_batch("library_index")
            self.set_status("Indexing cancelled", 3000)
            return
        hashes = [f["hash"] for f in files]
        cached = self.session.repo.load_embeddings_for(hashes, semantic_model.MODEL_VERSION)
        missing = [f for f in files if f["hash"] not in cached]
        if not missing:
            self._end_batch("library_index")
            self.set_status("Library already indexed for search by meaning", 4000)
            return
        self.embedding_requested.emit(missing)

    def request_library_semantic_search(self, query: str) -> None:
        """The whole-library counterpart to request_library_search: ranks every
        indexed file (image_embeddings, not just the open session) by meaning and
        opens the matches, exactly like the keyword search's own hand-off."""
        embedding = self.embed_search_query(query)
        if embedding is None:
            self.set_status("Type a search first" if not query.strip() else "Search by meaning is not ready yet", 3000)
            return
        candidates = self.session.repo.load_all_embeddings(semantic_model.MODEL_VERSION)
        # A whole-scan embedding holds both subjects unrotated, so it ranks worse than
        # either half and is never the frame shown. Both half hashes being embedded is
        # what proves it superseded; split_scans() is a roll-wide toggle, not a per-file
        # fact, so it cannot answer this.
        vectors = {
            file_hash: vec
            for file_hash, (path, vec) in candidates.items()
            if path and half_hash(file_hash, 1) not in candidates and half_hash(file_hash, 2) not in candidates
        }
        ranked_hashes = semantic_model.rank_by_similarity(embedding, vectors)
        paths = [candidates[file_hash][0] for file_hash in ranked_hashes]
        self.library_search_finished.emit(len(paths))
        if not paths:
            self.set_status("No frames in the library match that search", 4000)
            return
        self.set_status(f"{len(paths)} frame{'s' if len(paths) != 1 else ''} found", 3000)
        self.state.active_roll_id = None
        self.half_frame_mode_changed.emit(self.half_frame_mode_for_roll(None))
        # These paths are the filtered result. Re-running the outlier check over this
        # small, mutually-similar set has no background to stand out from and can exclude
        # every frame, and a stale text filter can empty the batch just as easily.
        self.session.asset_model.clear_filters()
        self.request_asset_discovery(paths, auto_open=True, replace_existing=True)

    def _on_embedding_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"Indexing {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _apply_embeddings(self, new_embeddings: Dict[str, Any]) -> None:
        """Commit a batch (or a chunk of a running one) so an active search-by-meaning
        ranking improves mid-batch instead of waiting for the whole session to finish.
        Only writes into the live session cache for files actually loaded right now --
        a library-wide pass's results belong in the DB (already saved per file as they
        land), not in this in-memory dict."""
        live_hashes = {f["hash"] for f in self.state.uploaded_files}
        self.state.embeddings.update({h: v for h, v in new_embeddings.items() if h in live_hashes})
        self.session.asset_model.refresh()

    def _on_embeddings_finished(self, new_embeddings: Dict[str, Any]) -> None:
        self.status_progress_requested.emit(0, 0)
        self._end_batch(self._embedding_batch_owner)
        self._apply_embeddings(new_embeddings)

    def _on_embedding_batch_error(self, message: str) -> None:
        self._end_batch(self._embedding_batch_owner)
        self.status_progress_requested.emit(0, 0)
        logger.error(f"Embedding batch failed: {message}")
        self.set_status("Indexing for search by meaning failed", 4000, kind="warning")

    def _on_rendered_thumbnail(self, new_thumbs: Dict[str, Any]) -> None:
        """A canvas render produced a thumbnail — it supersedes any batch placeholder."""
        for key, pil_img in new_thumbs.items():
            if pil_img and self._set_thumbnail(key, pil_img):
                self.state.rendered_thumbnails.add(key)
                self.state.stale_thumbnails.discard(key)
        self.session.asset_model.refresh()
        self._resume_background_thumbnails()

    def _pause_background_thumbnails(self) -> None:
        """Give the selected frame exclusive access to native decode memory."""
        if not self._thumbnail_queue_active:
            return
        self._thumbnails_paused_for_foreground = True
        self.thumb_worker.cancel_pending()
        self.thumbnail_cancel_requested.emit()

    def _resume_background_thumbnails(self) -> None:
        if not getattr(self, "_thumbnails_paused_for_foreground", False):
            return
        if (
            getattr(self, "_foreground_preview_generation", None) is not None
            or getattr(self, "_is_rendering", False)
            or getattr(self, "_pending_render_task", None) is not None
        ):
            return
        self._thumbnails_paused_for_foreground = False
        self.generate_missing_thumbnails()

    def _continue_background_work(self) -> None:
        """Resume deferred work after the selected frame becomes idle."""
        if AppController._foreground_work_active(self):
            return
        prefetch_generation = getattr(self, "_neighbor_prefetch_generation", None)
        if prefetch_generation is not None and prefetch_generation == getattr(self, "_prefetch_gen", None):
            self._neighbor_prefetch_generation = None
            self._schedule_prefetch_neighbors()
            return
        if getattr(self, "_prefetch_in_flight_generation", None) is not None or getattr(self, "_neighbor_prefetch_queue", []):
            return
        AppController._resume_background_thumbnails(self)

    def _foreground_work_active(self) -> bool:
        return bool(
            getattr(self, "_foreground_preview_generation", None) is not None
            or getattr(self, "_is_rendering", False)
            or getattr(self, "_pending_render_task", None) is not None
            or getattr(self, "_active_batch", None) is not None
        )

    def _cancel_neighbor_prefetch(self) -> bool:
        self._neighbor_prefetch_generation = None
        self._neighbor_prefetch_queue.clear()
        generation = self._prefetch_in_flight_generation
        if generation is None:
            return False
        self.preview_load_worker.cancel_prefetch(generation)
        return True

    # --- Batch progress popup -------------------------------------------------

    @property
    def hot_folder_sequence_active(self) -> bool:
        """True while the active discovery came from the Hot Folder poll."""
        return self._hot_folder_sequence_active

    def _begin_batch(self, owner: str, title: str, abortable: bool) -> Optional[int]:
        """Claim the shared batch lane and return its generation token."""
        if self._active_batch is not None:
            self.set_status(f"{self._active_batch_title} is already running", 3000)
            return None
        if owner in _NORM_THREAD_BATCH_OWNERS:
            self._preempt_background_thumbnail_refresh()
        self._batch_serial += 1
        self._active_batch = owner
        self._active_batch_title = title
        self._active_batch_abortable = abortable
        self._active_batch_token = self._batch_serial
        self.batch_started.emit(title, abortable)
        return self._active_batch_token

    def _batch_busy(self, requested: str) -> bool:
        if self._active_batch is None:
            return False
        self.set_status(f"Cannot start {requested} while {self._active_batch_title} is running", 3000, kind="warning")
        return True

    def _end_batch(self, owner: str, token: Optional[int] = None) -> bool:
        """Release only the batch generation that owns the progress lane."""
        if self._active_batch != owner:
            return False
        if token is not None and token != self._active_batch_token:
            return False
        self._active_batch = None
        self._active_batch_title = ""
        self._active_batch_abortable = False
        self._active_batch_token = None
        self.batch_finished.emit()
        if self._pending_asset_discoveries and not self._discovery_running:
            QTimer.singleShot(0, self._start_next_asset_discovery)
        return True

    def _on_batch_progress(self, current: int, total: int, name: str) -> None:
        self.batch_progress.emit(current, total, name)

    def _on_batch_cancelled(self, owner: str) -> None:
        self.set_status("Aborted", 3000)
        self._end_batch(owner)

    def _on_export_batch_cancelled(self) -> None:
        owner = self._active_batch if self._active_batch in ("export", "contact_sheet") else "export"
        self._on_batch_cancelled(owner)

    def _on_discovery_batch_error(self, message: str) -> None:
        self._discovery_running = False
        self._hot_folder_sequence_active = False
        self._end_batch("discovery")
        self._report_worker_error("Import", message)

    def _on_normalization_cancelled(self) -> None:
        self._scene_queue = []
        self._on_batch_cancelled("normalization")

    def _on_normalization_error(self, message: str) -> None:
        self._scene_queue = []
        self._on_batch_error("normalization")
        self._report_worker_error("Analysis", message)

    def _on_batch_error(self, owner: str) -> None:
        self._end_batch(owner)

    def abort_active_batch(self) -> None:
        """Requests cancellation of the running abortable batch (export or analysis)."""
        if self._active_batch in ("export", "contact_sheet"):
            self.export_worker.cancel()
        elif self._active_batch == "normalization":
            self.norm_worker.cancel()
        elif self._active_batch == "autocrop":
            self._autocrop_cancel_requested = True
            self.batch_autocrop_worker.cancel(self._autocrop_batch_token)
        elif self._active_batch == "stitch":
            self.stitch_worker.cancel()
        elif self._active_batch == "hdr":
            self.hdr_worker.cancel()
        elif self._active_batch == "library_index":
            self._library_index_cancelled = True
            self.embedding_worker.cancel()

    def saved_session_paths(self) -> List[str]:
        """Returns last session's file paths that still exist on disk."""
        paths = self.session.repo.get_global_setting("session_files", []) or []
        return [p for p in paths if os.path.exists(p)]

    def restore_session(self) -> None:
        """Re-loads the previous session's files and reselects the active one."""
        paths = self.saved_session_paths()
        if not paths:
            return
        active = self.session.repo.get_global_setting("session_active_path")
        self._pending_scanned_file = active if active in paths else paths[0]
        triplets = self.session.repo.get_global_setting("session_triplets", {}) or {}
        self.state.active_roll_id = self._roll_id_for_restored_paths(paths)
        self.half_frame_mode_changed.emit(self.half_frame_mode_for_roll(self.state.active_roll_id))
        self.request_asset_discovery(paths, auto_open=True, restore_triplets=triplets)

    def _roll_id_for_restored_paths(self, paths: List[str]) -> Optional[str]:
        """The one roll every restored path agrees on, or None -- the roll a fresh
        process would otherwise forget it had open, the same "only when unambiguous"
        rule open_library_folders applies when several folders are opened at once."""
        candidates = set(rolls.rolls_containing_path(self.session.repo, paths[0]))
        for path in paths[1:]:
            candidates &= set(rolls.rolls_containing_path(self.session.repo, path))
            if not candidates:
                return None
        return next(iter(candidates)) if len(candidates) == 1 else None

    def request_asset_discovery(
        self,
        paths: List[str],
        auto_open: bool = False,
        restore_triplets: Optional[dict] = None,
        replace_existing: bool = False,
        reselect_path: Optional[str] = None,
        announce_rgb: bool = False,
        hot_folder: bool = False,
    ) -> None:
        """
        Starts asynchronous discovery of supported assets.
        Requests arriving while hashing is in progress are queued in order.

        `replace_existing` rebuilds the asset list from the results (instead of
        appending) and reselects `reselect_path` — used when re-running discovery
        over already-loaded files (e.g. an RGB-scan mode toggle).

        `announce_rgb` allows the modal report when RGB Scan assembles nothing. Set it
        where the user just asked for this folder or just turned the mode on; leaving it
        off is what keeps a restored session from opening a dialog nobody asked for.

        `hot_folder` marks this request as coming from the Hot Folder poll, not from a
        user action — it drives `hot_folder_sequence_active`, which is what the batch
        popup checks, rather than whether the toggle happens to be on.
        """
        self.thumb_worker.cancel_pending()
        self.thumbnail_cancel_requested.emit()
        self._announce_rgb = announce_rgb
        active_roll_id = self.state.active_roll_id
        request = _DiscoveryRequest(
            paths=tuple(paths),
            auto_open=auto_open,
            restore_triplets=restore_triplets,
            replace_existing=replace_existing,
            reselect_path=reselect_path,
            rgb_scan=bool(self.session.repo.get_global_setting("rgbscan_mode", False)),
            # A batch spanning several rolls has no roll-wide toggle to apply, and a
            # "confirmed diptych" hash records whichever roll's toggle was on at the
            # time rather than a per-file fact. Splitting waits for a roll that says so.
            half_frame=self.half_frame_mode_for_roll(active_roll_id) if active_roll_id else False,
            half_frame_profile=self.half_frame_profile(),
            half_frame_overrides=self.half_frame_overrides(),
            hot_folder=hot_folder,
        )
        if self._discovery_running:
            self._pending_asset_discoveries.append(request)
            return

        if self._active_batch is not None:
            self._pending_asset_discoveries.append(request)
            self.set_status(f"Queued asset discovery until {self._active_batch_title} finishes", 3000)
            return

        self._start_asset_discovery(request)

    def _start_asset_discovery(self, request: _DiscoveryRequest) -> None:
        """Start one request; callers ensure only one discovery is active."""

        from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS

        # Must be set before _begin_batch: it emits batch_started synchronously, and
        # the view reads this to decide whether to suppress the popup for it.
        previous_hot_folder_sequence = self._hot_folder_sequence_active
        self._hot_folder_sequence_active = request.hot_folder
        if self._begin_batch("discovery", "Hashing files", abortable=False) is None:
            self._hot_folder_sequence_active = previous_hot_folder_sequence
            self._pending_asset_discoveries.insert(0, request)
            return
        self._discovery_running = True
        self._auto_open_after_discovery = request.auto_open
        self._replace_after_discovery = request.replace_existing
        self._reselect_after_discovery = request.reselect_path
        self._active_discovery_keys = frozenset(_capture_import_key(path) for path in request.paths)
        self.set_status("Scanning for assets…")
        stitches, merges = restore_maps(self.session.repo)
        task = AssetDiscoveryTask(
            paths=list(request.paths),
            supported_extensions=tuple(SUPPORTED_RAW_EXTENSIONS),
            rgb_scan=request.rgb_scan,
            restore_triplets=request.restore_triplets,
            half_frame=request.half_frame,
            # Read as the request starts, not as it was queued: a composite made while
            # a discovery waits its turn must still be re-attached when the queue gets to it.
            restore_stitches=stitches,
            restore_hdr=merges,
            half_frame_profile=request.half_frame_profile,
            half_frame_overrides=request.half_frame_overrides,
        )
        self.asset_discovery_requested.emit(task)

    def _start_next_asset_discovery(self) -> None:
        if self._pending_asset_discoveries and not self._discovery_running and self._active_batch is None:
            self._start_asset_discovery(self._pending_asset_discoveries.pop(0))

    # --- Library (a library of Rolls) ------------------------------------------

    def library_roots(self) -> List[str]:
        """Top-level directories a library search walks. Maintained automatically by
        importing a roll (or a parent full of them) — not a user-visible list."""
        saved = self.session.repo.get_global_setting("library_roots", []) or []
        return [p for p in saved if isinstance(p, str)] if isinstance(saved, list) else []

    def _register_library_roots(self, paths: List[str]) -> None:
        roots = self.library_roots()
        new = [p for p in paths if p not in roots]
        if new:
            self.session.repo.save_global_setting("library_roots", [*roots, *new])

    def has_rolls(self) -> bool:
        return bool(rolls.saved_rolls(self.session.repo))

    def import_subfolders_as_rolls(self, parent_path: str) -> List[str]:
        """Recognize every immediate subfolder of *parent_path* as its own roll, and
        register it as a search root -- nothing is opened or loaded."""
        roll_ids = rolls.import_subfolders_as_rolls(self.session.repo, parent_path)
        if roll_ids:
            self._register_library_roots([parent_path])
        return roll_ids

    def open_library_folder(self, folder: str, add_to_session: bool = False) -> None:
        self.open_library_folders([folder], add_to_session=add_to_session)

    def open_library_folders(self, folders: List[str], add_to_session: bool = False) -> None:
        """Recognize and load one or several folders as rolls. Replacing the session
        costs nothing — every edit lives in the database under its own content hash,
        not in the file list."""
        present = [f for f in folders if os.path.isdir(f)]
        if not present:
            self.set_status("Folder is no longer on disk", 3000)
            return
        if not add_to_session:
            # Recognizing every opened folder is independent of which one, if any,
            # becomes the active roll -- that only makes sense for a single one.
            recognized = [rolls.recognize_folder(self.session.repo, f) for f in present]
            self.state.active_roll_id = recognized[0] if len(recognized) == 1 else None
            self.half_frame_mode_changed.emit(self.half_frame_mode_for_roll(self.state.active_roll_id))
            self._register_library_roots(present)
        self.request_asset_discovery(
            present,
            auto_open=True,
            replace_existing=not add_to_session,
            reselect_path=self.state.current_file_path if add_to_session else None,
        )

    def open_roll(self, roll_id: str) -> None:
        """Open a roll (folder or virtual) by id. A folder roll's own contents are
        (re)walked as usual, the same as opening it from the tree; its extra_paths --
        files added by hand that are not physically in the folder -- ride along in the
        same discovery pass, since request_asset_discovery already accepts a mix of
        folder and file paths."""
        entry = rolls.roll_for_id(self.session.repo, roll_id)
        if entry is None:
            self.set_status("That roll no longer exists", 3000)
            return
        if entry["kind"] == "folder":
            paths = [entry["folder_path"], *entry.get("extra_paths", [])]
        else:
            paths = list(entry.get("member_paths", []))
        if not paths:
            self.set_status("This roll has no frames", 3000)
            return
        self.state.active_roll_id = roll_id
        self.half_frame_mode_changed.emit(self.half_frame_mode_for_roll(roll_id))
        self.request_asset_discovery(paths, auto_open=True, replace_existing=True)

    def create_roll_from_session(self, name: str) -> Optional[str]:
        """Save the frames currently in the Film Strip as a new virtual roll: a roll
        that is not a folder, e.g. a library search's results, kept and named."""
        paths = [f["path"] for f in self.state.uploaded_files if f.get("path")]
        if not paths:
            self.set_status("Nothing loaded to save as a roll", 3000)
            return None
        roll_id = rolls.create_virtual_roll(self.session.repo, name, paths)
        # Seed the new roll's own half-frame toggle from the ad hoc session's, so saving as
        # a roll doesn't silently reset it to off the next time this roll is opened.
        by_roll = dict(self.session.repo.get_global_setting(self._HALF_FRAME_MODE_BY_ROLL_KEY, default=None) or {})
        by_roll[roll_id] = self.half_frame_mode_for_roll(None)
        self.session.repo.save_global_setting(self._HALF_FRAME_MODE_BY_ROLL_KEY, by_roll)
        self.state.active_roll_id = roll_id
        self.set_status(f"Saved as roll “{name}”", 3000)
        return roll_id

    def request_rename_roll(self, roll_id: str, new_name: str, rename_folder: bool) -> bool:
        """Rename a roll's display name, and -- only if asked -- its backing folder on
        disk too. All-or-nothing: if the disk rename fails (missing folder, a sibling
        already named that, no permission), the display name is left alone as well,
        so the two names can never end up telling different stories.
        """
        if rename_folder:
            entry = rolls.roll_for_id(self.session.repo, roll_id)
            old_path = entry.get("folder_path", "") if entry else ""
            new_path = rolls.rename_folder_roll_disk(self.session.repo, roll_id, new_name)
            if new_path is None:
                return False
            if old_path and roll_id == self.state.active_roll_id:
                self.session.rehome_folder_paths(old_path, new_path)
        rolls.rename_roll(self.session.repo, roll_id, new_name)
        return True

    def invalidate_library_walk(self) -> None:
        """Drop the cached traversal so the next search re-reads the folders."""
        QMetaObject.invokeMethod(self.library_worker, "invalidate", Qt.ConnectionType.QueuedConnection)

    def request_library_search(self, query: str, rewalk: bool = False) -> None:
        """Search every library root, not just the loaded frames, and open the matches.

        Edit metadata joins onto unopened files by path, so a frame is findable by its
        film stock without being in the session — and without being hashed.
        """
        query = (query or "").strip()
        if not query:
            self.set_status("Type a search first, e.g. film:portra", 3000)
            return
        roots = self.library_roots()
        if not roots:
            self.set_status("Add a library folder first", 4000)
            return
        self.set_status("Searching library…")
        self.library_search_requested.emit(
            LibrarySearchTask(
                roots=roots,
                query=query,
                configs_by_path=self.session.repo.load_settings_by_path(),
                marks_by_path=self.session.repo.load_file_marks_by_path(),
                rewalk=rewalk,
            )
        )

    def _on_library_walk_progress(self, walked: int) -> None:
        verb = "Scanning" if self._active_batch == "library_index" else "Searching"
        self.set_status(f"{verb} library… {walked} files")

    def _on_library_search_finished(self, paths: List[str]) -> None:
        self.library_search_finished.emit(len(paths))
        if not paths:
            self.set_status("No frames in the library match that search", 4000)
            return
        self.set_status(f"{len(paths)} frame{'s' if len(paths) != 1 else ''} found", 3000)
        # An ad hoc result, not (yet) any roll -- Save as Roll in the Film Strip turns it
        # into one.
        self.state.active_roll_id = None
        self.half_frame_mode_changed.emit(self.half_frame_mode_for_roll(None))
        # The hand-off already is the filtered result -- a semantic query left over from
        # an earlier, unrelated search would otherwise re-rank this batch by an embedding
        # that has nothing to do with it, dropping every file with no cached vector yet.
        self.session.asset_model.clear_filters()
        self.request_asset_discovery(paths, auto_open=True, replace_existing=True)

    def set_rgb_scan_mode(self, enabled: bool) -> None:
        """Persist the RGB-scan toggle and re-discover already-loaded assets so the
        mode regroups/ungroups triplets in place (not only on the next folder load)."""
        self.session.repo.save_global_setting("rgbscan_mode", bool(enabled))
        if enabled:
            # RGB-scan triplets are captured with narrowband LEDs, and correcting for them
            # is the point of the toggle, so switch it on together.
            self.session.repo.save_global_setting("last_narrowband_scan", True)
        files = self.session.state.uploaded_files
        if not files:
            return
        if enabled and not self.state.config.process.narrowband_scan:
            self.session.update_config(
                replace(self.state.config, process=replace(self.state.config.process, narrowband_scan=True)), persist=True
            )
            self.request_render()
        self.request_asset_discovery(
            _component_paths(files), replace_existing=True, reselect_path=self.state.current_file_path, announce_rgb=enabled
        )

    def apply_scan_setup(self, capture: str, light: str) -> None:
        """Apply the scanning-setup wizard's answer: Linear RAW and Narrowband are rig
        properties, so they land on the new-file defaults, the open frame and every
        already-edited frame at once."""
        linear_raw, narrowband = scan_setup_values(capture, light)
        self.session.repo.save_global_settings(
            {
                "scan_setup": {"capture": capture, "light": light},
                "last_linear_raw": linear_raw,
                "last_narrowband_scan": narrowband,
            }
        )

        reload_needed = False
        if self.state.current_file_hash:
            reload_needed = self.state.config.process.linear_raw != linear_raw
            new_config = replace(
                self.state.config,
                process=replace(
                    self.state.config.process,
                    linear_raw=linear_raw,
                    narrowband_scan=narrowband,
                    **invalidate_local_bounds(self.state.config.process),
                ),
            )
            # render=False when reloading: bounds must not be analysed on the stale decode.
            self.session.update_config(new_config, persist=True, render=not reload_needed)

        count = 0
        changed_hashes: list[str] = []
        for asset in self.session.state.uploaded_files:
            file_hash = asset["hash"]
            if file_hash == self.state.current_file_hash:
                continue
            # Frames with no saved edits inherit the sticky defaults when first hydrated,
            # so writing them here would only churn the DB.
            saved = self.session.repo.load_file_settings(file_hash)
            if saved is None:
                continue
            updated = replace(
                saved,
                process=replace(
                    saved.process,
                    linear_raw=linear_raw,
                    narrowband_scan=narrowband,
                    **invalidate_local_bounds(saved.process),
                ),
            )
            self.session.push_external_history(file_hash, saved, updated)
            self.session.repo.save_file_settings(file_hash, updated, file_path=asset["path"])
            changed_hashes.append(file_hash)
            count += 1

        if reload_needed and self.state.current_file_path:
            self.load_file(self.state.current_file_path)
        if count:
            self.session.settings_synced.emit(f"Scanning setup applied to {count} other frame{'s' if count != 1 else ''}")
            self.session.settings_saved.emit()
            self.session.frames_edited_offscreen.emit(changed_hashes)

    _HALF_FRAME_MODE_BY_ROLL_KEY = "half_frame_mode_by_roll"

    def half_frame_mode_for_roll(self, roll_id: Optional[str]) -> bool:
        """The half-frame toggle's state for *roll_id* -- each roll remembers its own,
        so switching rolls switches the toggle with it. An ad hoc session (no
        recognized roll, e.g. a library search result) reads the one sticky flag it
        always had."""
        if roll_id:
            by_roll = self.session.repo.get_global_setting(self._HALF_FRAME_MODE_BY_ROLL_KEY, default=None) or {}
            return bool(by_roll.get(roll_id, False))
        return bool(self.session.repo.get_global_setting("half_frame_mode", False))

    def set_half_frame_mode(self, enabled: bool) -> None:
        """Persist the half-frame toggle -- per roll when one is active, else the
        single sticky flag an ad hoc session always had -- and re-discover
        already-loaded assets so the mode splits/collapses frames in place (not only
        on the next folder load)."""
        roll_id = self.state.active_roll_id
        if roll_id:
            by_roll = dict(self.session.repo.get_global_setting(self._HALF_FRAME_MODE_BY_ROLL_KEY, default=None) or {})
            by_roll[roll_id] = bool(enabled)
            self.session.repo.save_global_setting(self._HALF_FRAME_MODE_BY_ROLL_KEY, by_roll)
        else:
            self.session.repo.save_global_setting("half_frame_mode", bool(enabled))
        self._active_diptych_memo = ("", None)
        files = self.session.state.uploaded_files
        if not files:
            return
        self.request_asset_discovery(_component_paths(files), replace_existing=True, reselect_path=self.state.current_file_path)

    # ── half-frame split & crop profile ─────────────────────────────────

    _HALF_FRAME_PROFILE_KEY = "half_frame_profile"
    _HALF_FRAME_OVERRIDES_KEY = "half_frame_overrides"

    def half_frame_profile(self) -> dict | None:
        """Saved ``(crop_rect, split_x, gutter_thickness)`` profile, shared across
        every half-frame split that has no override of its own. Scanner-independent —
        the same crop/split applies whether the scans came from a SANE scanner, a
        camera copy-stand, or a folder import."""
        return self.session.repo.get_global_setting(self._HALF_FRAME_PROFILE_KEY, default=None)

    def save_half_frame_profile(self, crop_rect, split_x: float, gutter_thickness: float) -> None:
        self.session.repo.save_global_setting(
            self._HALF_FRAME_PROFILE_KEY,
            {"crop_rect": list(crop_rect), "split_x": float(split_x), "gutter_thickness": float(gutter_thickness)},
        )

    def half_frame_overrides(self) -> dict:
        """Per-file ``(crop_rect, split_x, gutter_thickness)`` overrides, keyed by
        base file hash — for the odd frame the roll-wide profile (auto-detected or
        fixed) still gets wrong."""
        return dict(self.session.repo.get_global_setting(self._HALF_FRAME_OVERRIDES_KEY, default=None) or {})

    def half_frame_override(self, file_hash: str) -> dict | None:
        return self.half_frame_overrides().get(file_hash)

    def save_half_frame_override(self, file_hash: str, crop_rect, split_x: float, gutter_thickness: float) -> None:
        overrides = self.half_frame_overrides()
        overrides[file_hash] = {"crop_rect": list(crop_rect), "split_x": float(split_x), "gutter_thickness": float(gutter_thickness)}
        self.session.repo.save_global_setting(self._HALF_FRAME_OVERRIDES_KEY, overrides)

    def clear_half_frame_override(self, file_hash: str) -> None:
        overrides = self.half_frame_overrides()
        if file_hash in overrides:
            del overrides[file_hash]
            self.session.repo.save_global_setting(self._HALF_FRAME_OVERRIDES_KEY, overrides)

    def _path_for_base_hash(self, file_hash: str) -> str:
        return next((a["path"] for a in self.session.state.uploaded_files if base_hash(a.get("hash", "")) == file_hash), "")

    def current_base_file(self) -> tuple[Optional[str], Optional[str]]:
        """The current frame's (path, base hash), falling back to the first loaded file.

        Both halves of a half-frame asset share one path, so matching by path alone
        would always return whichever half comes first in the list — never the one
        actually active — and its own suffixed hash, which save_half_frame_override
        does not key by. base_hash() makes either mistake harmless.
        """
        current = self.state.current_file_path
        for f in self.session.state.uploaded_files:
            if f.get("path") == current:
                return f.get("path"), base_hash(f.get("hash"))
        if self.session.state.uploaded_files:
            f = self.session.state.uploaded_files[0]
            return f.get("path"), base_hash(f.get("hash"))
        return None, None

    def selected_base_hashes(self) -> list[str]:
        """Base hashes of the filmstrip selection, deduped (a half-frame asset's two
        halves can both be selected) and composites excluded."""
        files = self.session.state.uploaded_files
        seen: dict[str, None] = {}
        for i in self.session.state.selected_indices:
            if 0 <= i < len(files) and not is_composite(files[i]):
                h = base_hash(files[i]["hash"])
                if h:
                    seen.setdefault(h, None)
        return list(seen)

    def reload_after_half_frame_change(self) -> None:
        """Re-discover so a profile/override change takes effect immediately."""
        files = self.session.state.uploaded_files
        self.request_asset_discovery(
            [f["path"] for f in files if "path" in f],
            replace_existing=True,
            reselect_path=self.state.current_file_path,
        )

    def _half_frame_geometry_for(self, file_hash: str, file_path: str = "") -> HalfGeometry:
        """This file's currently effective half geometry: its own override, else the
        roll's saved profile, else — with no profile yet — the same per-file
        auto-detect discovery falls back to."""
        saved = self.half_frame_override(file_hash) or self.half_frame_profile()
        if saved is not None:
            cr = saved.get("crop_rect")
            return HalfGeometry(
                crop_rect=tuple(cr) if cr is not None else None,
                split_x=float(saved.get("split_x") or 0.5),
                gutter_thickness=float(saved.get("gutter_thickness") or 0.0),
            )
        if file_path:
            from negpy.services.assets.half_frame import detect_split_x_for_file

            return HalfGeometry(split_x=detect_split_x_for_file(file_path))
        return HalfGeometry()

    def _remap_half_frame_edits(self, file_hash: str, old_geom: HalfGeometry, new_geom: HalfGeometry) -> None:
        """Re-anchor both halves' saved manual edits from ``old_geom`` to ``new_geom``,
        so a heal stroke, dust spot, scratch line or dodge/burn mask stays on the same
        physical film location when the split or crop moves."""
        if old_geom == new_geom:
            return
        path = self._path_for_base_hash(file_hash)
        for half in (1, 2):
            h = half_hash(file_hash, half)
            saved = self.session.repo.load_file_settings(h)
            if saved is None:
                continue
            updated = remap_workspace_config(saved, half, old_geom, new_geom)
            if updated == saved:
                continue
            self.session.push_external_history(h, saved, updated)
            self.session.repo.save_file_settings(h, updated, file_path=path)

    _HALF_FRAME_APPLY_SCOPE_KEY = "half_frame_apply_scope"

    def open_half_frame_dialog(
        self,
        file_path: str,
        file_hash: str,
        selected_hashes: Optional[List[str]] = None,
        initial_scope: Optional[str] = None,
    ) -> dict | None:
        """Open the half-frame split & crop editor on one scan, seeded from
        ``file_hash``'s own effective geometry; on Apply, save the result and
        return it, or None on cancel.

        The dialog's own Apply split-button picks what gets written: its current
        choice — ``"current"`` (``file_hash``'s own override), ``"selected"`` (the
        same override on every hash in ``selected_hashes``) or ``"all"`` (the
        roll-wide profile, which every file without its own override inherits) —
        is read back after Apply and remembered as the next default, unless
        ``initial_scope`` pins one (the per-frame context menu always starts at
        ``"current"``, regardless of what was last used elsewhere). Either way,
        each affected file's manual edits are re-anchored from its old effective
        geometry to the new one first, so they stay put across the change.
        """
        import numpy as np

        from negpy.desktop.view.widgets.half_frame_dialog import HalfFrameDialog
        from negpy.services.assets.thumbnails import decode_source_image

        try:
            img = decode_source_image(file_path)
            if img is None:
                return None
            buf = np.asarray(img)
        except Exception as e:
            self.set_status(f"Could not load preview: {e}", kind="error")
            return None

        old_geom = self._half_frame_geometry_for(file_hash, file_path)
        saved_scope = initial_scope or self.session.repo.get_global_setting(self._HALF_FRAME_APPLY_SCOPE_KEY, "current")
        dialog = HalfFrameDialog(
            buf,
            initial_rect=old_geom.crop_rect,
            initial_split=old_geom.split_x,
            initial_gutter=old_geom.gutter_thickness,
            initial_scope=saved_scope,
            parent=None,
        )
        if not dialog.exec():
            return None

        scope = dialog.scope()
        self.session.repo.save_global_setting(self._HALF_FRAME_APPLY_SCOPE_KEY, scope)
        cx1, cy1, cx2, cy2 = dialog.crop_rect()
        result = {
            "crop_rect": [cx1, cy1, cx2, cy2],
            "split_x": dialog.split_x(),
            "gutter_thickness": dialog.gutter_thickness(),
        }
        new_geom = HalfGeometry((cx1, cy1, cx2, cy2), result["split_x"], result["gutter_thickness"])

        if scope == "all":
            overrides = self.half_frame_overrides()
            targets: set[str] = set()
            for a in self.session.state.uploaded_files:
                h = None if is_composite(a) else base_hash(a["hash"])
                if h and h not in overrides:
                    targets.add(h)
            for h in targets:
                self._remap_half_frame_edits(h, self._half_frame_geometry_for(h, self._path_for_base_hash(h)), new_geom)
            self.save_half_frame_profile(result["crop_rect"], result["split_x"], result["gutter_thickness"])
        else:
            scoped_targets = selected_hashes if scope == "selected" and selected_hashes else [file_hash]
            for h in scoped_targets:
                self._remap_half_frame_edits(h, self._half_frame_geometry_for(h, self._path_for_base_hash(h)), new_geom)
                self.save_half_frame_override(h, result["crop_rect"], result["split_x"], result["gutter_thickness"])
        return result

    def auto_detect_all_half_frame_splits(self) -> None:
        """Re-find the gutter on every loaded scan, off the GUI thread — a one-shot
        batch instead of adjusting each odd frame by hand. ``_on_splits_detected``
        saves the results once detection finishes."""
        targets: dict[str, str] = {}
        for a in self.session.state.uploaded_files:
            h = None if is_composite(a) else base_hash(a["hash"])
            if h:
                targets[h] = a["path"]
        paths = list(targets.values())
        if not paths:
            return
        self.set_status(f"Auto-detecting the split on {len(paths)} frame{'s' if len(paths) != 1 else ''}…")
        self.status_progress_requested.emit(0, len(paths))
        self.auto_detect_all_splits_requested.emit(AutoDetectAllSplitsTask(paths=paths))

    def _on_splits_detected(self, detected: dict[str, tuple[float, float, Optional[tuple[float, float, float, float]]]]) -> None:
        """AutoDetectAllSplitsTask finished: save each file's own detected split,
        gutter thickness and outer film crop as its override, re-anchoring its manual
        edits from whatever geometry it used before. A file whose crop detection
        failed keeps the crop it already had -- only the split and thickness move
        for it."""
        self.status_progress_requested.emit(0, 0)
        seen: set[str] = set()
        for a in self.session.state.uploaded_files:
            if is_composite(a) or a.get("path") not in detected:
                continue
            file_hash = base_hash(a["hash"])
            if not file_hash or file_hash in seen:
                continue
            seen.add(file_hash)
            old_geom = self._half_frame_geometry_for(file_hash, a["path"])
            split_x, gutter_thickness, crop_rect = detected[a["path"]]
            new_geom = replace(
                old_geom,
                split_x=split_x,
                gutter_thickness=gutter_thickness,
                crop_rect=old_geom.crop_rect if crop_rect is None else crop_rect,
            )
            self._remap_half_frame_edits(file_hash, old_geom, new_geom)
            self.save_half_frame_override(
                file_hash, new_geom.crop_rect or (0.0, 0.0, 1.0, 1.0), new_geom.split_x, new_geom.gutter_thickness
            )
        if not seen:
            return
        self.set_status(f"Auto-detected the split on {len(seen)} frame{'s' if len(seen) != 1 else ''}")
        files = self.session.state.uploaded_files
        self.request_asset_discovery(
            [f["path"] for f in files if "path" in f], replace_existing=True, reselect_path=self.state.current_file_path
        )

    def _on_discovery_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"Hashing {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _mark_diptychs(self, assets: List[Dict]) -> None:
        """Flag whole-frame scans the user split that already carry the two halves' edits.

        One query for the whole roll, at discovery, so every later reader — the filmstrip
        badge, the read-only panel, the exporter — finds the answer on the asset dict.
        """
        split = split_scans(self.session.repo)
        whole = []
        for a in assets:
            if a.get("half") or not a.get("hash"):
                continue
            if is_composite(a) or "#" in a["hash"] or a["hash"] not in split:
                a["diptych"] = False
                continue
            whole.append(a)
        if not whole:
            return
        found = self.session.repo.load_file_settings_many([half_hash(a["hash"], n) for a in whole for n in (1, 2)])
        for a in whole:
            a["diptych"] = half_hash(a["hash"], 1) in found or half_hash(a["hash"], 2) in found

    def _apply_roll_forks(self, assets: List[Dict]) -> None:
        """Rewrite an asset's hash to its roll-forked identity when the active roll has
        one for it, so every hash-keyed store (edits, history, thumbnails) resolves the
        fork automatically from here on -- the same trick half-frame splitting uses.

        Checked against the asset's own (pre-fork) hash, not its path: a half-frame
        scan's two halves have different hashes, so forking one never drags the other.
        """
        roll_id = self.state.active_roll_id
        if not roll_id:
            return
        entry = rolls.roll_for_id(self.session.repo, roll_id)
        forked = set(entry.get("forked_hashes", [])) if entry else set()
        for a in assets:
            if a.get("hash") in forked:
                a["hash"] = rolls.roll_edit_hash(a["hash"], roll_id)

    def _on_rgb_grouped(self, summary: dict) -> None:
        """Report what RGB Scan did with a folder it could not fully assemble.

        A partial result is a status line. Assembling nothing is a dead end the user
        cannot diagnose from a filmstrip of loose frames, so that is modal — but only
        when they just opened the folder or turned the mode on, never on a restore.
        """
        notice = rgb_grouping_notice(summary["made"], summary["loose"], summary["incomplete"], summary["mismatched"], summary["by_time"])
        if notice:
            self.set_status(notice, 12000)
        if summary["made"] or not self._announce_rgb:
            return
        if self.session.repo.get_global_setting("rgbscan_hide_empty_warning", False):
            return

        title, body = rgb_nothing_matched_message(summary)
        box = QMessageBox(QMessageBox.Icon.Information, title, body, QMessageBox.StandardButton.NoButton)
        if summary["narrowband"]:
            remember = QCheckBox("Do not show this again")
            box.setCheckBox(remember)
            close_btn = box.addButton(QMessageBox.StandardButton.Ok)
            box.setDefaultButton(close_btn)
            box.exec()
            if remember.isChecked():
                self.session.repo.save_global_setting("rgbscan_hide_empty_warning", True)
            return

        turn_off = box.addButton("Turn Off Trichrome Mode", QMessageBox.ButtonRole.AcceptRole)
        keep = box.addButton("Keep It On", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(turn_off)
        box.exec()
        if box.clickedButton() is turn_off:
            self.set_rgb_scan_mode(False)
            self.rgb_scan_mode_changed.emit(False)
        _ = keep

    def _on_discovery_finished(self, valid_assets: List[Dict]) -> None:
        """
        Adds discovered assets to the session and starts thumbnail generation.
        """
        remember_split_scans(self.session.repo, {base_hash(a["hash"]) for a in valid_assets if a.get("half")})
        self._mark_diptychs(valid_assets)
        self._apply_roll_forks(valid_assets)
        self._active_diptych_memo = ("", None)
        ended_batch = self._end_batch("discovery")
        self._hot_folder_sequence_active = False
        if not ended_batch and self._active_batch is None:
            # Preserve the completion signal for direct invocations and late
            # delivery without releasing a newer batch owner.
            self.batch_finished.emit()
        self.status_progress_requested.emit(0, 0)
        self._discovery_running = False
        auto_open = self._auto_open_after_discovery
        self._auto_open_after_discovery = False
        replace_existing = self._replace_after_discovery
        reselect_path = self._reselect_after_discovery
        self._replace_after_discovery = False
        self._reselect_after_discovery = None
        active_discovery_keys = self._active_discovery_keys
        self._active_discovery_keys = frozenset()

        # Files appended (not replaced) while a roll is active join its membership, so
        # reopening that roll later still shows what was added by hand.
        if not replace_existing and self.state.active_roll_id:
            rolls.add_extra_members(self.session.repo, self.state.active_roll_id, [a["path"] for a in valid_assets if a.get("path")])
        pending_scan = getattr(self, "_pending_scanned_file", None)

        if replace_existing and valid_assets:
            # Re-run over already-loaded files (e.g. RGB-scan toggle): rebuild the list
            # so dedup-by-hash doesn't drop a regrouped red, then reselect the active frame.
            self.session.state.uploaded_files.clear()
            self.session.state.rendered_thumbnails.clear()
            self.session.add_files([], validated_info=valid_assets)
            self.generate_missing_thumbnails()
            if not self._thumbnail_queue_active:
                # Nothing queued, so no idle transition will arrive to start the
                # embeddings pass (already-cached thumbnails are exactly what a
                # whole-library search's own matches already have).
                self.generate_missing_embeddings()
            idx = None
            if reselect_path:
                # Guard on a set path: `None in (path, green_path, blue_path)` matches any
                # non-RGB frame, whose green/blue paths are absent.
                idx = next(
                    (
                        i
                        for i, f in enumerate(self.session.state.uploaded_files)
                        if reselect_path in (f.get("path"), f.get("green_path"), f.get("blue_path"))
                    ),
                    None,
                )
            if idx is None:
                # No prior frame to restore (a fresh folder open): land on the first frame
                # in filmstrip (sorted/filtered) order, not discovery order.
                ordered = self.session.asset_model.visible_actual_indices_ordered()
                idx = ordered[0] if ordered else 0
            self.session.select_file(idx)
            self._start_next_asset_discovery()
            return

        selected_pending_scan = False
        if valid_assets:
            first_new_idx = len(self.session.state.uploaded_files)
            self.session.add_files([], validated_info=valid_assets)
            self.generate_missing_thumbnails()
            if not self._thumbnail_queue_active:
                # Nothing queued, so no idle transition will arrive to start the
                # embeddings pass (already-cached thumbnails are exactly what a
                # whole-library search's own matches already have).
                self.generate_missing_embeddings()
            if pending_scan and self._select_file_by_path(pending_scan):
                selected_pending_scan = True
            elif auto_open and not self.state.current_file_path and len(self.session.state.uploaded_files) > first_new_idx:
                # Select the first newly-loaded frame in filmstrip order, not discovery
                # order, or the initial frame lands mid-strip.
                new_indices = set(range(first_new_idx, len(self.session.state.uploaded_files)))
                ordered = self.session.asset_model.visible_actual_indices_ordered()
                target = next((i for i in ordered if i in new_indices), first_new_idx)
                self.session.select_file(target)
        else:
            self.set_status("No supported assets found", 3000, kind="warning")
            self.status_progress_requested.emit(0, 0)
            self._hot_folder_sequence_active = False

        if pending_scan:
            pending_key = _capture_import_key(pending_scan)
            if selected_pending_scan:
                # select_file emits load_file synchronously in the real session. Pop again
                # as a fallback for alternate session implementations and tests.
                self._pending_capture_imports.pop(pending_key, None)
                self._pending_scanned_file = None
            elif pending_key in active_discovery_keys:
                # This request finished without the intended primary asset. Drop only its
                # metadata; a later capture may already be waiting in the FIFO queue.
                self._pending_capture_imports.pop(pending_key, None)
                self._pending_scanned_file = None
        self._start_next_asset_discovery()

    def _file_hash_for_path(self, file_path: str) -> Optional[str]:
        if self.state.current_file_path == file_path and self.state.current_file_hash:
            return self.state.current_file_hash
        for f in self.state.uploaded_files:
            if f.get("path") == file_path:
                return f.get("hash")
        return None

    def _half_slice_for_asset(
        self, path: Optional[str], file_hash: Optional[str]
    ) -> Optional[tuple[int, float, tuple[float, float, float, float] | None, float]]:
        """(half, split_x, crop_rect, gutter_thickness) for the asset at path/hash, or None."""
        if not file_hash:
            return None
        for f in self.state.uploaded_files:
            if f.get("hash") == file_hash or (path and f.get("path") == path and f.get("hash") == file_hash):
                half = int(f.get("half") or 0)
                if not half:
                    return None
                cr = f.get("crop_rect")
                crop_rect: tuple[float, float, float, float] | None = None
                if isinstance(cr, (tuple, list)):
                    vals = tuple(float(v) for v in cr)
                    if len(vals) == 4:
                        crop_rect = vals  # type: ignore[assignment]
                return (
                    half,
                    float(f.get("split_x") or 0.5),
                    crop_rect,
                    float(f.get("gutter_thickness") or 0.0),
                )
        return None

    def _active_half(self) -> Optional[tuple[int, float, tuple[float, float, float, float] | None, float]]:
        """(half, split_x, crop_rect, gutter_thickness) of the active asset, or None for whole-frame."""
        return self._half_slice_for_asset(self.state.current_file_path, self.state.current_file_hash)

    def active_diptych(self) -> Optional[tuple[dict, tuple[WorkspaceConfig, WorkspaceConfig]]]:
        """(asset with the split geometry, half configs) for the active scan, or None.

        Memoized per hash: it is read on every render, and the halves' edits can only
        change while half-frame mode is on, where the active asset is a half instead.
        """
        file_hash = self.state.current_file_hash or ""
        if self._active_diptych_memo[0] != file_hash:
            asset = next((a for a in self.state.uploaded_files if a.get("hash") == file_hash), None)
            resolved = None
            if asset is not None:
                info, pair = self._diptych_task(asset)
                resolved = (info, pair) if pair is not None else None
            self._active_diptych_memo = (file_hash, resolved)
        return self._active_diptych_memo[1]

    def diptych_pair(self, file_info: dict) -> Optional[tuple[WorkspaceConfig, WorkspaceConfig]]:
        """The two halves' saved edits for a whole-frame scan, or None.

        Half-frame mode being off is implied: with it on the assets already *are* halves,
        which `half` on the asset dict reports.
        """
        if file_info.get("half") or file_info.get("diptych") is False or is_composite(file_info):
            return None
        return diptych_configs(self.session.repo, file_info.get("hash"))

    def _diptych_task(self, file_info: dict) -> tuple[dict, Optional[tuple[WorkspaceConfig, WorkspaceConfig]]]:
        """(asset dict with the split geometry stamped on, half configs) for a diptych.

        A whole-frame asset never went through `_expand_half_frames`, so the split comes
        from this file's own effective geometry — its override if it has one, else the
        saved profile, else auto-detected — the same resolution the halves were cut with.
        """
        pair = self.diptych_pair(file_info)
        if pair is None:
            return file_info, None
        geom = self._half_frame_geometry_for(file_info.get("hash") or "", file_info.get("path", ""))
        return (
            {**file_info, "split_x": geom.split_x, "crop_rect": geom.crop_rect, "gutter_thickness": geom.gutter_thickness},
            pair,
        )

    def _render_memo_key(self, config: Optional[WorkspaceConfig] = None) -> str:
        """Identity of everything that shapes the displayed render of the current
        config: the edit itself plus every display-path input. Any mismatch is a
        memo miss, so navigate-back only skips straight to pixels that would be
        reproduced exactly."""
        import hashlib
        import json

        config = self.state.config if config is None else config
        parts = (
            json.dumps(config.to_dict(), sort_keys=True, default=str),
            self.state.hq_preview,
            self.state.workspace_color_space,
            self.state.gpu_enabled,
            self.state.soft_proof_enabled,
            # The whole condition, so every proof control is part of the identity. Naming
            # the profiles alone would memo-hit across a change of intent or paper white.
            self.proof_profiles(),
            hashlib.md5(self.state.monitor_icc_bytes).hexdigest() if self.state.monitor_icc_bytes else "",
        )
        return hashlib.md5(repr(parts).encode()).hexdigest()

    def _strip_memo_key(self, kind: str = "tone") -> str:
        """The render key for a proof mosaic, prefixed by kind so the two can't collide.

        Both ladders are absolute, so each proof supplies the fields it varies and its mosaic
        is invariant to whatever those currently are. Pinning them makes print/pick/print again
        a cache hit. Any other edit (crop, paper, toning...) lands on a different key.

        Rotation is not in the key: an entry holds all four orientations.
        """
        exposure = self.state.config.exposure
        if kind == "color":
            exposure = replace(exposure, wb_magenta=0.0, wb_yellow=0.0)
        else:
            exposure = replace(exposure, density=1.0, grade=115.0)
        return f"{kind}:{self._render_memo_key(replace(self.state.config, exposure=exposure))}"

    def _retain_displayed_texture(self) -> Optional[GPUTexture]:
        """Spare the on-screen GPU render from the cleanup, and file it in the memo if it can be.

        Two separate questions. Filing is refused mid-render: that render paints into the
        same pooled texture, so the pixels would stop matching the key they are filed under.
        Sparing is not — the canvas must go on showing what it has until a new render
        replaces it, or a reload with no splash behind it paints nothing at all.
        """
        identity = self._last_render_identity
        self._last_render_identity = None
        texture = self.state.last_metrics.get("base_positive")
        if not isinstance(texture, GPUTexture):
            # load_file pops base_positive, so a second reload arriving before a render
            # completes finds nothing here while the canvas still shows the texture spared
            # on the previous pass. Spare that one again: reporting nothing would tell the
            # canvas to let go of what it is displaying.
            texture = self._spared_texture
        if not isinstance(texture, GPUTexture):
            self._spared_texture = None
            return None
        self._spared_texture = texture
        # Sparing the texture and filing it in the memo are separate questions, and
        # conflating them blanks the canvas. Filing is refused mid-render: that render
        # paints into the same pooled texture, so the pixels would stop matching their key.
        # Sparing is still right, because the canvas keeps sampling what it already shows
        # until the new render replaces it. A reload with no splash behind it has nothing
        # else to show meanwhile.
        if identity is not None and not self._is_rendering and self._pending_render_task is None:
            source_hash, memo_key, content_rect = identity
            self._render_memo.store(
                source_hash,
                memo_key,
                {
                    "base_positive": texture,
                    "content_rect": content_rect,
                    "render_long_edge": self.state.last_metrics.get("render_long_edge", 0),
                },
            )
        return texture

    def _on_file_selected_load(self, file_path: str) -> None:
        """``session.file_selected`` handler: navigation honors the sticky-zoom preference."""
        self.load_file(file_path, preserve_zoom=self.state.sticky_zoom)

    def _arm_delayed_spinner(self, generation: int) -> None:
        QTimer.singleShot(_KEEP_PREVIEW_SPINNER_DELAY_MS, lambda: self._maybe_start_delayed_spinner(generation))

    def _maybe_start_delayed_spinner(self, generation: int) -> None:
        if self._foreground_preview_generation == generation:
            self.loading_started.emit()

    def load_file(self, file_path: str, preserve_zoom: bool = False, force_detect: bool = False) -> None:
        """
        Dispatches RAW decode to a background worker to keep the UI thread free.
        """
        self._prefetch_gen += 1
        self.preview_load_worker.expect_generation(self._prefetch_gen, file_path)
        self._cancel_neighbor_prefetch()
        self._foreground_preview_generation = self._prefetch_gen
        self._pause_background_thumbnails()
        self._preview_load_t0 = time.perf_counter()
        keep_preview = preserve_zoom and self._requested_file_path == file_path
        self._requested_file_path = file_path
        # A strip belongs to one frame, and the memo fast path below repaints without
        # going through request_render, so drop it here too. Zone pins froze their sample
        # from this frame and go the same way. The compare split holds the frame the user
        # is leaving, so it goes too.
        self._clear_test_strip()
        self._drop_zone_pins()
        self.exit_compare()

        # Navigate-back fast path: the frame's last render is memoized and nothing that
        # shaped it has changed, since select_file already hydrated its config. Paint it
        # now, with no spinner and no toasts, and let the real render refresh the metrics.
        target_hash = self._file_hash_for_path(file_path)
        self._expected_render_key = self._render_memo_key()
        memo = self._render_memo.get(target_hash, self._expected_render_key) if target_hash else None

        if not preserve_zoom:
            self.zoom_requested.emit(1.0)
        if memo is None:
            if keep_preview:
                self._arm_delayed_spinner(self._prefetch_gen)
            else:
                self.loading_started.emit()
        self._thumb_config = None

        retained = self._retain_displayed_texture()
        # A retained texture outlives the pool, so the canvas keeps sampling it. Without
        # one it must let go before the engine frees what it is showing.
        if retained is None:
            self.gpu_textures_released.emit()
        self._render_cleanup_requested.emit(retained)
        # The cleanup destroys the GPU textures last_metrics still points at, so drop the
        # densitometer's probe sources. Hover readouts go quiet until the next render.
        self.state.last_metrics.pop("normalized_log", None)
        self.state.last_metrics.pop("base_positive", None)
        self.state.last_metrics.pop("thumbnail_source", None)

        if memo is not None:
            with self.state.metrics_lock:
                self.state.last_metrics["base_positive"] = memo["base_positive"]
                self.state.last_metrics["content_rect"] = memo.get("content_rect")
                self.state.last_metrics["render_long_edge"] = memo.get("render_long_edge", 0)
                self.state.last_metrics["splash"] = False
                self.state.last_metrics["proof"] = True
                # These pixels are this frame's own last render. Leaving the outgoing
                # frame's hash next to them would file them under it on the next
                # thumbnail refresh, which reads whatever last_metrics holds.
                self.state.last_metrics["source_hash"] = target_hash
            self.image_updated.emit()

        self.state.preview_raw = None
        self.state.preview_ir = None
        self.state.preview_detect = None
        self.state.preview_embedded = None
        self.state.has_ir = False
        self.state.original_res = (0, 0)
        if self.state.negative_peek:
            self.state.negative_peek = False
            self.negative_peek_changed.emit(False)
        if self.state.embedded_peek:
            self.state.embedded_peek = False
            self.embedded_peek_changed.emit(False)

        pending_import = self._pending_capture_imports.pop(_capture_import_key(file_path), None)
        if pending_import is not None and pending_import.process_mode is not None:
            process = self.state.config.process
            process = replace(
                process,
                process_mode=pending_import.process_mode,
                **invalidate_local_bounds(process),
            )
            self.state.config = replace(self.state.config, process=process)
            self.state.is_dirty = True
        if pending_import is not None and (pending_import.capture_roll or pending_import.capture_frame is not None):
            meta = self.state.config.metadata
            self.state.config = replace(
                self.state.config,
                metadata=replace(
                    meta,
                    capture_roll=pending_import.capture_roll or meta.capture_roll,
                    capture_frame=(pending_import.capture_frame if pending_import.capture_frame is not None else meta.capture_frame),
                ),
            )
            self.state.is_dirty = True

        rgbscan = self.state.config.rgbscan
        stitch = self.state.config.stitch
        hdr = self.state.config.hdr
        flatfield = self.state.config.flatfield
        half_info = self._active_half()
        if half_info is None:
            dip = self.active_diptych()
            if dip is not None:
                # half 0: cropped to the rect, still whole. The render worker splits it, so
                # both halves come off one decode.
                info = dip[0]
                half_info = (0, info["split_x"], info["crop_rect"], info["gutter_thickness"])
        self.preview_load_requested.emit(
            PreviewLoadTask(
                file_path=file_path,
                workspace_color_space=self.state.workspace_color_space,
                use_camera_wb=not effective_linear_raw(self.state.config.process, self.state.config.exposure.render_intent),
                generation=self._prefetch_gen,
                positive_source=self.state.config.process.positive_source,
                highlight_mode=effective_highlight_reconstruction(self.state.config.process),
                bake_camera_wb=highlight_reconstruction_bakes_wb(self.state.config.process, self.state.config.exposure.render_intent),
                full_resolution=self.state.hq_preview,
                # The half suffix distinguishes the two halves' preview caches now
                # that the slice happens pre-downsample (each half is its own buffer).
                file_hash=self._file_hash_for_path(file_path),
                # A reload or memo hit keeps the rendered frame until its replacement is ready.
                use_splash=memo is None and not keep_preview,
                detect_mode=(
                    pending_import.detect_mode
                    if pending_import is not None
                    else force_detect or (self.state.autodetect_enabled and self.state.current_file_is_new)
                ),
                # Whole, not flattened: the worker gates on the same predicates the decode
                # paths use, so a disabled section needs no blanking here.
                rgbscan=rgbscan,
                stitch=stitch,
                hdr=hdr,
                flatfield_profile_id=flatfield.profile_id if (stitch.stitch_enabled and flatfield.apply) else "",
                half_slice=half_info,
                demosaic=self.state.config.process.demosaic_preview,
                lens_corrections=metadata_lens_corrections(self.state.config),
                lens_flatfield=self.state.config.flatfield,
            )
        )

    def _split_active_half(self, raw: Any, dims: Any) -> tuple[Any, Any]:
        """No-op: the half-frame slice now happens in PreviewManager before the
        preview downsample, so both splash and linear buffers arrive already
        sliced to the active half (and at the same pixels export analyzes).
        Kept as a passthrough for the splash/loaded handlers that still call it.
        """
        return raw, dims

    def _on_splash_preview(self, file_path: str, raw: Any, dims: Any) -> None:
        if self._requested_file_path != file_path:
            return
        # A backlogged splash-decode worker can land after the real render for this same
        # file already has. Splash only ever bridges the gap before the real render
        # arrives, never replaces it once it has.
        target_hash = self._file_hash_for_path(file_path)
        with self.state.metrics_lock:
            if (
                target_hash is not None
                and self.state.last_metrics.get("splash") is False
                and self.state.last_metrics.get("source_hash") == target_hash
            ):
                return
        raw, dims = self._split_active_half(raw, dims)
        self.state.original_res = dims
        # Paint the embedded sRGB thumbnail directly, with no pipeline. The real render
        # replaces it.
        with self.state.metrics_lock:
            self.state.last_metrics["base_positive"] = raw
            self.state.last_metrics["render_long_edge"] = int(max(raw.shape[:2])) if isinstance(raw, np.ndarray) else 0
            self.state.last_metrics["splash"] = True
        self.image_updated.emit()

    def _on_preview_load_failed(self, file_path: str, message: str) -> None:
        for f in self.state.uploaded_files:
            if f["path"] == file_path:
                f["decode_failed"] = message
                self.session.asset_model.refresh()
                return

    def _on_hq_preview_vram_capped(self, file_path: str, capped_long_edge: int) -> None:
        """An HQ load exceeded the GPU's VRAM budget and was downsampled instead of
        crashing (see preview_manager._load_from_open_raw). Non-blocking — the user can
        keep working at the reduced resolution or raise max_texture_size in Preferences."""
        if self._requested_file_path != file_path:
            return
        self.set_status(
            f"Scan too large for available GPU memory — showing a {capped_long_edge}px preview instead of full resolution.",
            5000,
        )

    def _on_preview_loaded(
        self,
        file_path: str,
        raw: Any,
        dims: Any,
        source_cs: str,
        ir_preview: Any,
        detected_mode: str,
        cam_matrix: Any = None,
        detect_preview: Any = None,
    ) -> None:
        for f in self.state.uploaded_files:
            if f["path"] == file_path and f.pop("decode_failed", None) is not None:
                self.session.asset_model.refresh()
        if self._requested_file_path != file_path:
            return
        self._foreground_preview_generation = None
        decoded_lens_token = cam_matrix[3] if cam_matrix and len(cam_matrix) > 3 else ""
        if decoded_lens_token != lens_decode_token(metadata_lens_corrections(self.state.config), self.state.config.flatfield):
            return
        logger.info(
            "load-timing preview_e2e %.0fms (load request -> decoded buffer) %s",
            (time.perf_counter() - self._preview_load_t0) * 1000,
            file_path,
        )
        raw, dims = self._split_active_half(raw, dims)
        if ir_preview is not None:
            ir_preview, _ = self._split_active_half(ir_preview, None)
        self.state.preview_raw = raw
        self.state.preview_cam_xyz, self.state.preview_camera_wb = cam_matrix[:2] if cam_matrix else (None, None)
        self.state.preview_lens = cam_matrix[2] if cam_matrix and len(cam_matrix) > 2 else None
        self.state.preview_lens_path = file_path
        self.state.preview_lens_token = decoded_lens_token
        self.state.preview_proxy = _interactive_proxy(raw)
        self.state.preview_ir = ir_preview
        self.state.preview_ir_proxy = _interactive_ir_proxy(ir_preview, self.state.preview_proxy)
        self.state.preview_detect = detect_preview
        self.state.preview_detect_proxy = _interactive_ir_proxy(detect_preview, self.state.preview_proxy)
        self.state.has_ir = ir_preview is not None
        if not self.state.has_ir and self.state.dust_overlay_mode == "ir":
            self.state.dust_overlay_mode = "off"
        self.state.original_res = dims
        self.state.current_file_path = file_path
        self.state.source_cs = source_cs
        self._apply_detected_mode(detected_mode)
        self.preview_loaded.emit()
        self.config_updated.emit()
        self._first_render_t0 = time.perf_counter()
        self.request_render()
        self._neighbor_prefetch_generation = self._prefetch_gen

    def _schedule_prefetch_neighbors(self) -> None:
        generation = self._prefetch_gen
        QTimer.singleShot(50, lambda: self._prepare_neighbor_prefetch(generation))

    def _prepare_neighbor_prefetch(self, generation: int) -> None:
        from negpy.desktop.prefetch_logic import neighbor_assets

        if generation != self._prefetch_gen or self._foreground_work_active():
            return
        index = self.state.selected_file_idx
        files = self.state.uploaded_files
        if index < 0 or not files:
            AppController._resume_background_thumbnails(self)
            return

        display_order = self.session.asset_model.visible_actual_indices_ordered()
        selected_hash = files[index].get("hash")
        protected_file_hashes = [selected_hash] if selected_hash else []
        tasks = []
        for asset in neighbor_assets(files, display_order, index):
            task = self._neighbor_prefetch_task(asset, generation, tuple(protected_file_hashes))
            if task is None:
                continue
            tasks.append(task)
            if task.file_hash:
                protected_file_hashes.append(task.file_hash)
        self._neighbor_prefetch_queue = [task for task in tasks if task is not None]
        self._start_next_neighbor_prefetch()

    def _neighbor_prefetch_task(
        self,
        asset: dict,
        generation: int,
        protected_file_hashes: tuple[str, ...],
    ) -> Optional[PreviewLoadTask]:
        if is_composite(asset):
            return None
        file_hash = asset.get("hash")
        if not file_hash:
            return None
        saved = self.session.repo.load_file_settings(file_hash)
        linear_raw = effective_linear_raw(saved.process, saved.exposure.render_intent) if saved else False
        try:
            integrated_gpu = bool(self.state.gpu_enabled and GPUDevice.get().is_integrated)
        except Exception:
            integrated_gpu = False
        return PreviewLoadTask(
            file_path=asset["path"],
            workspace_color_space=self.state.workspace_color_space,
            use_camera_wb=not linear_raw,
            generation=generation,
            positive_source=saved.process.positive_source if saved else False,
            highlight_mode=effective_highlight_reconstruction(saved.process) if saved else 0,
            bake_camera_wb=(highlight_reconstruction_bakes_wb(saved.process, saved.exposure.render_intent) if saved else False),
            full_resolution=False,
            file_hash=file_hash,
            use_splash=False,
            for_cache_warm=True,
            integrated_gpu=integrated_gpu,
            protected_file_hashes=protected_file_hashes,
            half_slice=self._half_slice_for_asset(asset["path"], file_hash),
            demosaic=saved.process.demosaic_preview if saved else self.state.config.process.demosaic_preview,
            lens_corrections=metadata_lens_corrections(saved or self.state.config),
            lens_flatfield=(saved or self.state.config).flatfield,
        )

    def _start_next_neighbor_prefetch(self) -> None:
        if self._foreground_work_active():
            return
        if self._prefetch_in_flight_generation is not None:
            return
        if not self._neighbor_prefetch_queue:
            AppController._resume_background_thumbnails(self)
            return
        task = self._neighbor_prefetch_queue.pop(0)
        self._prefetch_in_flight_generation = task.generation
        self.preview_load_requested.emit(task)

    def _on_neighbor_prefetch_finished(self, generation: int, _file_path: str) -> None:
        if self._prefetch_in_flight_generation == generation:
            self._prefetch_in_flight_generation = None
        if generation != self._prefetch_gen:
            return
        if self._pending_render_task is not None and not self._is_rendering and self._foreground_preview_generation is None:
            self._dispatch_pending_render()
            return
        if self._foreground_work_active():
            return
        self._start_next_neighbor_prefetch()

    def _apply_detected_mode(self, detected_mode: str) -> None:
        """
        Silently apply the autodetected process mode for a new file. Never overrides
        a saved or user-edited mode (the worker only runs detection on new files).
        """
        if not detected_mode or detected_mode == self.state.config.process.process_mode:
            return
        new_proc = replace(
            self.state.config.process,
            process_mode=ProcessMode(detected_mode),
            **invalidate_local_bounds(self.state.config.process),
        )
        exp = self.state.config.exposure
        self.state.config = replace(
            self.state.config,
            process=new_proc,
            exposure=replace(
                exp,
                cast_removal_strength=cast_removal_for_mode(ProcessMode(detected_mode), exp.cast_removal_strength),
            ),
        )
        self.state.is_dirty = True

    def toggle_autodetect(self, enabled: bool) -> None:
        self.session.set_autodetect_enabled(enabled)
        if enabled and self.state.current_file_path:
            self.load_file(self.state.current_file_path, preserve_zoom=True, force_detect=True)

    def toggle_hq_preview(self) -> None:
        # Resolution is in the memo key: every entry is now a permanent miss holding a
        # full-size texture.
        self._render_memo.clear()
        self.session.set_hq_preview(not self.state.hq_preview)
        self._render_memo.large_entries = self.state.hq_preview
        if self.state.current_file_path:
            self.load_file(self.state.current_file_path, preserve_zoom=True)

    def handle_canvas_clicked(self, nx: float, ny: float) -> None:
        if self.state.active_tool == ToolMode.WB_PICK:
            self._handle_wb_pick(nx, ny)
        elif self.state.active_tool == ToolMode.DUST_PICK:
            self._handle_dust_pick(nx, ny)
        elif self.state.active_tool == ToolMode.SCRATCH_LINE:
            self._handle_scratch_line_pick(nx, ny)
        elif self.state.active_tool == ToolMode.ZONE_PLACE:
            self._handle_zone_pin(nx, ny)

    def set_active_tool(self, mode: ToolMode) -> None:
        # Both the crop and analysis-region tools show the full uncropped frame, so
        # entering or leaving that set must re-render to swap the preview.
        uncropped = {ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW}
        preview_mode_changed = (self.state.active_tool in uncropped) != (mode in uncropped)
        leaving_crop = self.state.active_tool == ToolMode.CROP_MANUAL and mode != ToolMode.CROP_MANUAL
        leaving_zone_place = self.state.active_tool == ToolMode.ZONE_PLACE and mode != ToolMode.ZONE_PLACE
        self.state.active_tool = mode
        self.tool_sync_requested.emit()
        if leaving_zone_place:
            self.clear_zone_pins()
        if leaving_crop and self._crop_bounds_dirty:
            # Recompute bounds once now the final crop is committed.
            new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
            self.session.update_config(replace(self.state.config, process=new_proc), render=False)
            self._crop_bounds_dirty = False
        if preview_mode_changed:
            if leaving_crop:
                # Same spinner treatment as an initial file load: the bounds recompute and
                # this render take a moment on a large HQ frame. image_updated dismisses it
                # when the render lands, which is guaranteed by the request_render() below.
                self.loading_started.emit()
            self.request_render()

    def cancel_active_tool(self) -> None:
        if self.state.active_tool != ToolMode.NONE:
            self.set_active_tool(ToolMode.NONE)

    def show_rotation_guide(self) -> None:
        """Request the canvas show the fine-rotation alignment grid."""
        self.rotation_guide_requested.emit()

    def set_crop_guide(self, guide: str) -> None:
        self.session.set_crop_guide(guide)
        self.crop_guide_changed.emit()

    def cycle_crop_guide_orientation(self) -> None:
        self.session.set_crop_guide_orientation((self.state.crop_guide_orientation + 1) % 8)
        self.crop_guide_changed.emit()

    def cycle_dust_overlay(self) -> None:
        """Advance the dust-detection overlay: Off → Marked → IR → Off
        (IR skipped when the scan has no IR channel). Repaint only — the data is
        already in state.last_metrics / state.preview_ir, no re-render needed."""
        seq = ["off", "marked", "ir"]
        if not self.state.has_ir:
            seq.remove("ir")
        cur = self.state.dust_overlay_mode if self.state.dust_overlay_mode in seq else "off"
        self.state.dust_overlay_mode = seq[(seq.index(cur) + 1) % len(seq)]
        self.dust_overlay_changed.emit()

    def toggle_zones_overlay(self, force: Optional[bool] = None) -> None:
        """Adams-zone box overlay. Repaint only — the boxes are carved from the frame
        the canvas already holds, so no re-render is needed."""
        self.state.zones_overlay = (not self.state.zones_overlay) if force is None else bool(force)
        self.zones_overlay_changed.emit(self.state.zones_overlay)

    def toggle_grain_focuser(self, force: Optional[bool] = None) -> None:
        """Grain focuser loupe. Repaint only — it magnifies the frame the canvas already
        holds, so no re-render is needed."""
        self.state.grain_focuser = (not self.state.grain_focuser) if force is None else bool(force)
        self.grain_focuser_changed.emit(self.state.grain_focuser)

    def toggle_printing_notes(self, force: Optional[bool] = None) -> None:
        """Printing-notes overlay (dodge/burn map + print recipe). Repaint only — every
        number it shows is already in the config, so no re-render is needed."""
        self.state.printing_notes = (not self.state.printing_notes) if force is None else bool(force)
        self.printing_notes_changed.emit(self.state.printing_notes)

    def _set_alt_process(self, target: AltProcess, force: Optional[bool] = None) -> None:
        """B&W only — the stage is a no-op in any other mode. The two processes are
        mutually exclusive, so selecting one clears the other."""
        cfg = self.state.config
        on = (cfg.altproc.alt_process != target) if force is None else bool(force)
        mode = target if on else AltProcess.NONE
        self.session.update_config(replace(cfg, altproc=replace(cfg.altproc, alt_process=mode)), persist=True)
        self.request_render()

    def toggle_lith(self, force: Optional[bool] = None) -> None:
        self._set_alt_process(AltProcess.LITH, force)

    def toggle_cyanotype(self, force: Optional[bool] = None) -> None:
        self._set_alt_process(AltProcess.CYANOTYPE, force)

    def request_printing_notes_export(self) -> None:
        """Save the marked-up work print as its own file. The annotated pixels live in the
        canvas, so the view answers the signal (the print itself is never touched)."""
        if not self.state.current_file_path:
            return
        self.printing_notes_requested.emit()

    def printing_notes_target_path(self) -> Optional[str]:
        """Next free `<stem>_notes.jpg` in the export folder."""
        export_path = self._ensure_valid_export_path()
        if export_path is None or not self.state.current_file_path:
            return None
        export_conf = replace(self.state.config.export, export_path=export_path)
        roll_root = self._roll_export_root(export_conf.output_mode, export_conf.output_subfolder)
        export_path = resolve_output_dir(
            self.state.current_file_path,
            preset_from_export_config(export_conf),
            roll_root,
        )
        stem = os.path.splitext(os.path.basename(self.state.current_file_path))[0]
        os.makedirs(export_path, exist_ok=True)
        path = os.path.join(export_path, f"{stem}_notes.jpg")
        counter = 2
        while os.path.exists(path):
            path = os.path.join(export_path, f"{stem}_notes_{counter}.jpg")
            counter += 1
        return path

    def arm_zone_target(self, zone: float) -> None:
        """Zone picked on the strip: the next canvas click prints that spot there.
        Picking the armed zone again disarms."""
        if self.state.preview_raw is None:
            return
        if self.state.zone_arm_target == float(zone):
            self._disarm_zone_target()
            return
        # Same reason compare, the peek and the strip are exclusive: they all want the canvas.
        restore = self.state.flat_peek
        self.exit_compare()
        if self.state.flat_peek:
            self.state.flat_peek = False
            self.flat_peek_changed.emit(False)
        self._clear_test_strip()
        if restore:
            self.request_render()
        self.state.zone_arm_target = float(zone)
        self.set_active_tool(ToolMode.ZONE_PLACE)
        self.zone_arm_changed.emit(self.state.zone_arm_target)

    def _disarm_zone_target(self) -> None:
        """Drop the armed zone, and the tool with it when no pins remain."""
        armed = self.state.zone_arm_target is not None
        self.state.zone_arm_target = None
        if armed:
            self.zone_arm_changed.emit(None)
        if not self.state.zone_pins and self.state.active_tool == ToolMode.ZONE_PLACE:
            self.set_active_tool(ToolMode.NONE)

    def _handle_zone_pin(self, nx: float, ny: float) -> None:
        """Armed: the pin takes the zone picked on the strip. Unarmed: it takes the zone
        it already reads, so a bare click meters without moving the print."""
        from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R
        from negpy.features.exposure.placement import MAX_PINS, ZonePin

        val = self._sample_normalized_log(nx, ny, radius=2)
        if val is None:
            return
        armed = self.state.zone_arm_target
        val_luma = LUMA_R * val[0] + LUMA_G * val[1] + LUMA_B * val[2]
        target = armed if armed is not None else round(self._pin_zone(val_luma) * 3.0) / 3.0
        pin = ZonePin(
            nx=nx,
            ny=ny,
            val_rgb=val,
            val_luma=val_luma,
            target_zone=target,
            retargeted=armed is not None,
        )
        pins = self.state.zone_pins
        if len(pins) < MAX_PINS:
            pins.append(pin)
        else:
            nearest = min(range(len(pins)), key=lambda i: (pins[i].nx - nx) ** 2 + (pins[i].ny - ny) ** 2)
            pins[nearest] = pin
        if armed is not None:
            self.state.zone_arm_target = None
            self.zone_arm_changed.emit(None)
        self._refresh_pin_labels()
        self.zone_pins_changed.emit()
        if armed is not None:
            self._preview_zone_solution()

    def move_zone_pin(self, index: int, nx: float, ny: float, final: bool = False) -> None:
        """Drag a pin: re-samples the tone under it. An untargeted pin re-snaps to the
        new reading, a retargeted one keeps its zone, and the solve waits for `final`."""
        from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R

        pins = self.state.zone_pins
        if not 0 <= index < len(pins):
            return
        self._pin_dragging = not final
        val = self._sample_normalized_log(nx, ny, radius=2)
        if val is not None:
            pin = pins[index]
            val_luma = LUMA_R * val[0] + LUMA_G * val[1] + LUMA_B * val[2]
            target = pin.target_zone if pin.retargeted else round(self._pin_zone(val_luma) * 3.0) / 3.0
            pins[index] = replace(pin, nx=nx, ny=ny, val_rgb=val, val_luma=val_luma, target_zone=target)
            self._refresh_pin_labels()
        self.zone_pins_changed.emit()
        if final and self._zone_preview_shown:
            self._preview_zone_solution()

    def _pin_zone(self, val_luma: float) -> float:
        from negpy.features.exposure.placement import predicted_zone

        return predicted_zone(
            self.state.config.exposure,
            self.state.config.process.process_mode,
            self.state.last_metrics,
            val_luma,
        )

    def _solve_zone_placement(self) -> Optional[Any]:
        from negpy.features.exposure.placement import solve_placement

        if not self.state.zone_pins:
            return None
        return solve_placement(
            self.state.config.exposure,
            self.state.config.process.process_mode,
            self.state.last_metrics,
            self.state.zone_pins,
        )

    def _refresh_pin_labels(self) -> None:
        """Re-read each pin's zone through the current curve. Called before every
        zone_pins_changed emit: the overlay and the sidebar repaint in connection
        order, so the label cannot be left to whichever runs first."""
        from negpy.features.exposure.densitometer import zone_roman

        pins = self.state.zone_pins
        for i, pin in enumerate(pins):
            label = zone_roman(self._pin_zone(pin.val_luma))
            if pin.label != label:
                pins[i] = replace(pin, label=label)

    def zone_pin_readouts(self) -> List[Tuple[int, str, float, Optional[str], bool]]:
        """Sidebar rows: (index, measured roman, target zone, achieved roman when the
        target is out of the paper's scale, solvable). Refreshes each pin's canvas label."""
        from negpy.features.exposure.densitometer import zone_roman

        pins = self.state.zone_pins
        if not pins:
            self._pin_solution = None
            return []
        self._refresh_pin_labels()
        # The two-pin nested bisection is too slow per mouse-move, so the last solve stands
        # in mid-drag and the drag's end recomputes it.
        if not self._pin_dragging:
            self._pin_solution = self._solve_zone_placement()
        sol = self._pin_solution
        rows = []
        for i, pin in enumerate(pins):
            achieved = zone_roman(sol.achieved[i]) if sol is not None and sol.clamped and i < len(sol.achieved) else None
            rows.append((i, pin.label, pin.target_zone, achieved, sol is not None))
        return rows

    def zone_solve_caption(self) -> str:
        """What the current pins are solving, named as the sliders name it. Reads the
        solution `zone_pin_readouts` cached — call it after, not before."""
        sol = self._pin_solution
        if sol is None or not self.state.zone_pins:
            return ""
        controls = ["Print Density"]
        if "grade" in sol.fields:
            controls.append("ISO-R Grade")
        if sol.knee:
            controls.append(_KNEE_LABELS[sol.knee])
        return "Solving " + " + ".join(controls)

    def set_zone_pin_target(self, index: int, zone: float) -> None:
        """Retarget one pin and preview the solved exposure without committing it."""
        pins = self.state.zone_pins
        if not 0 <= index < len(pins):
            return
        pins[index] = replace(pins[index], target_zone=min(max(float(zone), 0.0), 10.0), retargeted=True)
        self.zone_pins_changed.emit()
        self._preview_zone_solution()

    def _preview_zone_solution(self) -> None:
        sol = self._solve_zone_placement()
        if sol is None:
            return
        self._pin_solution = sol
        self._zone_preview_shown = True
        self.request_render(
            readback_metrics=False,
            config_override=replace(self.state.config, exposure=replace(self.state.config.exposure, **sol.fields)),
        )

    def apply_zone_placement(self) -> None:
        """Commit the solved Print Density (and Grade, and the knee control a third pin
        was solved on) and put the tool down. The autos it replaces go off: one left on
        would re-move the placed tones."""
        sol = self._solve_zone_placement()
        if sol is None:
            return
        self._zone_preview_shown = False
        self.session.update_config(
            replace(self.state.config, exposure=replace(self.state.config.exposure, **sol.fields)),
            persist=True,
        )
        self.set_active_tool(ToolMode.NONE)  # drops the pins; no preview left to restore
        self.request_render()

    def remove_zone_pin(self, index: int) -> None:
        """Drop one pin; what remains re-solves. Dropping the last one puts the committed
        print back and the tool down."""
        pins = self.state.zone_pins
        if not 0 <= index < len(pins):
            return
        pins.pop(index)
        self._pin_solution = None
        self._refresh_pin_labels()
        self.zone_pins_changed.emit()
        if not pins:
            self.set_active_tool(ToolMode.NONE)
        elif self._zone_preview_shown:
            self._preview_zone_solution()

    def clear_zone_pins(self) -> None:
        """Drop the pins; restores the committed edit if a preview was on the canvas."""
        restore = self._zone_preview_shown
        self._drop_zone_pins()
        if restore:
            self.request_render()

    def _drop_zone_pins(self) -> None:
        if self.state.zone_arm_target is not None:
            self.state.zone_arm_target = None
            self.zone_arm_changed.emit(None)
        if not self.state.zone_pins and not self._zone_preview_shown:
            return
        self.state.zone_pins.clear()
        self._zone_preview_shown = False
        self._pin_dragging = False
        self._pin_solution = None
        self.zone_pins_changed.emit()

    def toggle_ring_around(self, force: Optional[bool] = None) -> None:
        """Print (or clear) the color ring-around — the M/Y filtration proof."""
        self.toggle_test_strip(force, kind="color")

    def toggle_test_strip(self, force: Optional[bool] = None, kind: str = "tone") -> None:
        """Print (or clear) a proof mosaic: the density × grade strip, or the color ring-around.
        Entering dispatches one job, since these need pixels the canvas doesn't have.

        Both share the one proof slot, so asking for the other kind swaps it.
        """
        showing = (self.state.test_strip or self.state.test_strip_pending) and self.state.test_strip_kind == kind
        target = (not showing) if force is None else bool(force)
        if not target:
            self._clear_test_strip()
            return
        if self.state.preview_raw is None:
            return

        # The mosaic replaces the frame, so the split has nothing left to compare against.
        self.exit_compare()

        # Unrotated: one print yields every orientation, so rotating never re-renders.
        grid = RING_GRID if kind == "color" else STRIP_GRID
        overrides = ring_overrides() if kind == "color" else strip_overrides()
        toast = "Printing the color ring-around…" if kind == "color" else "Printing test strip…"

        # Reprinting an unchanged proof re-renders pixels we already have.
        cached = self._strip_memo.get(self.state.current_file_hash or "", self._strip_memo_key(kind))
        if cached is not None:
            self.state.test_strip_kind = kind
            self.on_strip_finished(cached["mosaics"], cached["content_rect"], from_cache=True)
            return

        self.state.test_strip_kind = kind
        self.state.test_strip_pending = True
        self.test_strip_changed.emit(False)
        # A few seconds of renders, so tick the HUD or it reads as wedged.
        self.set_status(toast, 2500)
        self.status_progress_requested.emit(0, len(overrides))
        cam_xyz, camera_wb = self._effective_cam_xyz()
        self.strip_requested.emit(
            TestStripTask(
                buffer=self.state.preview_raw,
                config=self.state.config,
                source_hash=self.state.current_file_hash or "preview",
                # Always preview res, never HQ: full-res renders per patch take minutes, and
                # each patch is shown at a fraction of the frame's width.
                preview_size=float(APP_CONFIG.preview_render_size),
                overrides=tuple(overrides),
                grid=grid,
                gpu_enabled=self.state.gpu_enabled,
                ir_buffer=self.state.preview_ir,
                detect_buffer=self.state.preview_detect,
                cam_xyz=cam_xyz,
                camera_wb=camera_wb,
            )
        )

    def on_strip_progress(self, done: int, total: int) -> None:
        if self.state.test_strip_pending:
            self.status_progress_requested.emit(done, total)

    def _clear_test_strip(self) -> None:
        """Drop the strip and its mosaic; silent when there was nothing up."""
        if not (self.state.test_strip or self.state.test_strip_pending):
            return
        if self.state.test_strip_pending:
            self.status_progress_requested.emit(0, 0)  # total <= 0 hides the bar
        self.state.test_strip = False
        self.state.test_strip_pending = False
        self.state.test_strip_mosaic = None
        self.state.test_strip_mosaics = None
        self.state.test_strip_content_rect = None
        self.test_strip_changed.emit(False)

    def on_strip_finished(self, mosaics: Any, content_rect: Any, from_cache: bool = False) -> None:
        # A render that landed while the strip was building already cancelled it.
        if not (self.state.test_strip_pending or from_cache):
            return
        if not from_cache:
            # Keyed on the config as it stands now, not as it was at dispatch. Measured
            # bounds persist after a render with render=False, so the config drifts mid-print
            # without invalidating anything, and keying at dispatch made every reprint a
            # miss. Safe: a change that mattered would go through request_render, which
            # cancels the strip.
            self._strip_memo.store(
                self.state.current_file_hash or "",
                self._strip_memo_key(self.state.test_strip_kind),
                {"mosaics": mosaics, "content_rect": content_rect},
            )
        self.state.test_strip_pending = False
        self.state.test_strip = True
        self.state.test_strip_mosaics = mosaics
        self.state.test_strip_mosaic = mosaics[self.state.test_strip_rotation]
        self.state.test_strip_content_rect = content_rect
        self.test_strip_changed.emit(True)
        label = "Ring-around" if self.state.test_strip_kind == "color" else "Test strip"
        self.set_status(f"{label} ready — click a patch to keep it", 4000)

    def rotate_test_strip(self, direction: int) -> bool:
        """Turn the ladder rather than the image while a proof is on the canvas; True = consumed.

        Gated on a mosaic being up, never on `pending`: the rotate controls are global, and a
        proof that is only expected must not swallow them. Mid-print the turn falls through to
        the image, dropping the print like any other edit.
        """
        if not (self.state.test_strip and self.state.test_strip_mosaics):
            return False
        self.state.test_strip_rotation = (self.state.test_strip_rotation + direction) % 4
        self.state.test_strip_mosaic = self.state.test_strip_mosaics[self.state.test_strip_rotation]
        self.test_strip_changed.emit(True)
        return True

    def _on_strip_error(self, _message: str) -> None:
        """A print that errors never reaches on_strip_finished, and the stuck `pending` flag holds
        the progress bar open and blocks a reprint."""
        if self.state.test_strip_pending:
            self._clear_test_strip()

    def apply_test_strip_pick(self, row: int, col: int) -> None:
        """Commit the clicked patch's settings, then drop the proof.

        Cast Removal and the Auto toggles are left alone: the patches were rendered under
        them, so flipping one would render something other than the patch that was clicked.
        """
        if not self.state.test_strip:
            return
        exposure = self.state.config.exposure
        color = self.state.test_strip_kind == "color"
        base_grid = RING_GRID if color else STRIP_GRID
        rotation = self.state.test_strip_rotation
        cells = rotate_grid(ring_cells() if color else strip_cells(), base_grid, rotation)
        _, _, first, second = cells[row * proof_grid(base_grid, rotation)[1] + col]
        if color:
            new_exposure = replace(exposure, wb_magenta=first, wb_yellow=second)
        else:
            new_exposure = replace(exposure, density=first, grade=second)
        self._clear_test_strip()
        self.session.update_config(replace(self.state.config, exposure=new_exposure), persist=True)
        self.request_render()

    def handle_crop_rect_changed(self, nx1: float, ny1: float, nx2: float, ny2: float, persist: bool) -> None:
        """Live-updates (persist=False) or commits (persist=True) the manual crop rect
        while the crop tool is open. The tool stays active afterwards — darktable-style
        continuous adjustment, not a one-shot drag-then-close."""
        if self.state.active_tool != ToolMode.CROP_MANUAL:
            return
        # A drag takes ownership of the rect, auto or not, so nothing re-detects over it.
        new_geo = replace(
            self.state.config.geometry,
            crop_rect=(
                min(nx1, nx2),
                min(ny1, ny2),
                max(nx1, nx2),
                max(ny1, ny2),
            ),
            crop_from_auto=False,
        )
        # Defer the bounds recompute to crop-tool close. Clearing here re-normalizes on
        # every drag step.
        self._crop_bounds_dirty = True
        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=persist)
        if persist:
            self.request_render()
        else:
            self._render_debounce.start()

    def handle_crop_rotation_changed(self, angle: float, persist: bool) -> None:
        """Live-updates (persist=False) or commits (persist=True) fine rotation from the
        crop tool's edge rotation handles. Writes the same geometry.fine_rotation the
        sidebar slider drives, so handle drag and slider fine-tuning compose; the crop
        rect is display-space and stays put while the image rotates under it."""
        if self.state.active_tool != ToolMode.CROP_MANUAL:
            return
        new_geo = replace(self.state.config.geometry, fine_rotation=angle)
        # Defer the bounds recompute to crop-tool close, like the rect drag.
        self._crop_bounds_dirty = True
        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=persist)
        self.rotation_guide_requested.emit()
        if persist:
            self.request_render()
        else:
            self._render_debounce.start()

    def handle_straighten_completed(self, delta_deg: float) -> None:
        """Applies the straighten tool's measured correction on top of the current
        fine rotation and closes the tool (one-shot, like a Lightroom straighten
        line). ``delta_deg`` is stored-convention (positive = CCW on screen) and
        display-space, so it composes additively under flips/90° turns."""
        if self.state.active_tool != ToolMode.STRAIGHTEN:
            return
        current = self.state.config.geometry.fine_rotation
        new_angle = float(np.clip(current + delta_deg, -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT))
        new_geo = replace(self.state.config.geometry, fine_rotation=new_angle)
        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=True)
        self.rotation_guide_requested.emit()
        self.set_active_tool(ToolMode.NONE)
        self.request_render()

    def confirm_manual_crop(self) -> None:
        """Close the crop tool (committing the current rect) — invoked by a double-click
        inside the crop box so the user needn't return to the Crop button."""
        if self.state.active_tool == ToolMode.CROP_MANUAL:
            self.set_active_tool(ToolMode.NONE)

    def set_crop_ratio(self, ratio: str) -> None:
        """Sets the Crop card's target ratio, locking the card away from the roll
        the instant it changes and was not already, like any other roll card. If a
        manual crop box is
        already drawn, reshapes it to the new ratio in place — same center, shrunk
        to fit within its current footprint (enforce_roi_aspect_ratio, the same
        centered-reshape auto-crop uses) — instead of leaving the box visually
        stale until the user redrags it.

        Deliberately does NOT invalidate the metering bounds, unlike the other crop
        entry points. Those clear them because the crop decides whether the film
        rebate is inside the metered region (resolve_analysis_region meters within
        context.active_roi), and letting clear base into the meter wrecks the
        bounds. A ratio change can't do that: both this reshape and autocrop's
        _enforce_ratio_by_occupancy only ever shrink the box inside a footprint
        that already excludes the rebate, so the new ROI is a subset of the old
        one. Re-metering there can only drift the per-channel floors/ceils — i.e.
        a visible color shift from what is supposed to be a pure reframe."""
        geom = self.state.config.geometry
        if ratio == geom.autocrop_ratio:
            return
        new_geo = replace(geom, autocrop_ratio=ratio)

        # An auto rect re-detects under the new ratio (it is in the detection key); the
        # frame Auto finds at 5:4 is not the 3:2 frame shrunk to fit.
        rect = None if geom.crop_from_auto else geom.crop_rect
        img = self.state.preview_raw
        if rect is not None and img is not None:
            h, w = img.shape[:2]
            if geom.rotation in (1, 3):
                h, w = w, h
            nx1, ny1, nx2, ny2 = rect
            roi_px = (round(ny1 * h), round(ny2 * h), round(nx1 * w), round(nx2 * w))
            y1, y2, x1, x2 = enforce_roi_aspect_ratio(roi_px, h, w, ratio)
            new_geo = replace(new_geo, crop_rect=(x1 / w, y1 / h, x2 / w, y2 / h))

        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=True)
        self._lock_roll_card("autocrop")
        # Same spinner treatment as reset_crop/apply_auto_crop: the base stage re-runs,
        # since geometry is part of its cache key, and that takes a moment on a large HQ frame.
        self.loading_started.emit()
        self.request_render()

    def handle_analysis_rect_changed(self, nx1: float, ny1: float, nx2: float, ny2: float, persist: bool) -> None:
        """Live-update (persist=False) or commit (persist=True) the freehand analysis
        region while the tool is open. Setting a region re-meters the frame, so a commit
        clears the per-file bounds (unless bounds are locked) and re-renders."""
        if self.state.active_tool != ToolMode.ANALYSIS_DRAW:
            return
        rect = (min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2))
        proc = replace(self.state.config.process, analysis_rect=rect)
        if persist:
            proc = replace(proc, **invalidate_local_bounds(proc))
        self.session.update_config(replace(self.state.config, process=proc), persist=persist)
        if persist:
            self.request_render()
        else:
            self._render_debounce.start()

    def clear_analysis_region(self) -> None:
        """Drop the freehand analysis region; metering falls back to the Analysis Buffer slider."""
        if self.state.config.process.analysis_rect is None:
            return
        proc = replace(self.state.config.process, analysis_rect=None)
        proc = replace(proc, **invalidate_local_bounds(proc))
        self.session.update_config(replace(self.state.config, process=proc), persist=True)
        self.request_render()

    def confirm_analysis_region(self) -> None:
        """Close the analysis-region tool (double-click inside the region)."""
        if self.state.active_tool == ToolMode.ANALYSIS_DRAW:
            self.set_active_tool(ToolMode.NONE)

    def reset_crop(self) -> None:
        self._crop_bounds_dirty = False
        new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(self.state.config.geometry, crop_rect=None, crop_from_auto=False),
                process=new_proc,
            )
        )
        # Same spinner treatment as an initial file load: the bounds recompute above takes
        # a moment on a large HQ frame.
        self.loading_started.emit()
        self.request_render()

    def apply_auto_crop(self) -> None:
        """Arm Auto Crop: clear the rect and let the next render detect one.

        _on_render_finished freezes the rect that render found into the edit, so the
        exported crop is the one on screen."""
        # Autocrop supersedes a manual crop in progress: leave the tool.
        if self.state.active_tool == ToolMode.CROP_MANUAL:
            self.state.active_tool = ToolMode.NONE
            self.tool_sync_requested.emit()
        self._crop_bounds_dirty = False
        new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(
                    self.state.config.geometry,
                    crop_rect=None,
                    crop_from_auto=True,
                ),
                process=new_proc,
            )
        )
        self.loading_started.emit()
        self.request_render()

    def _config_for_batch_asset(self, asset: dict) -> WorkspaceConfig:
        """Resolve per-asset settings, including unsaved edits on the active frame."""
        if asset.get("hash") == self.state.current_file_hash:
            return resolve_asset_hdr(resolve_asset_stitch(resolve_asset_rgbscan(self.state.config, asset), asset), asset)
        return self.session.config_for_asset(asset)

    def request_batch_auto_crop(self) -> None:
        """Analyze visible landscape frames together and persist explicit safe crops."""
        if self._batch_busy("Auto Crop All"):
            return
        if self.state.config.geometry.autocrop_mode != AutocropMode.IMAGE:
            self.set_status("Auto Crop All currently supports Image only mode", 4000)
            return
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return

        frames: list[BatchAutoCropInput] = []
        preflight_skipped = 0
        for asset in visible_files:
            config = self._config_for_batch_asset(asset)
            if has_manual_crop(config.geometry) or config.geometry.autocrop_mode != AutocropMode.IMAGE:
                preflight_skipped += 1
                continue
            frames.append(
                BatchAutoCropInput(
                    file_info=asset,
                    config=config,
                    fingerprint=_autocrop_fingerprint(config, self.state.workspace_color_space),
                )
            )

        if not frames:
            self.set_status(f"Auto Crop All preserved {count_of(preflight_skipped, 'frame')}; nothing to analyze", 4000)
            return

        token = self._begin_batch("autocrop", "Auto cropping roll", abortable=True)
        if token is None:
            return
        self._autocrop_batch_token = token
        self._autocrop_dispatched = len(frames)
        self._autocrop_preflight_skipped = preflight_skipped
        self._autocrop_cancel_requested = False
        self.set_status(f"Auto cropping {count_of(len(frames), 'frame')}...")
        self.batch_autocrop_requested.emit(
            BatchAutoCropTask(
                frames=frames,
                workspace_color_space=self.state.workspace_color_space,
                generation=token,
            )
        )

    def _on_batch_autocrop_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"Auto crop {current}/{total}: {name}")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, name)

    def _on_batch_autocrop_finished(self, results: list[BatchAutoCropResult]) -> None:
        token = self._autocrop_batch_token
        if self._active_batch != "autocrop" or token is None or token != self._active_batch_token:
            return  # stale completion from an older generation
        if self._autocrop_cancel_requested:
            self._on_batch_autocrop_cancelled()
            return

        saved = 0
        conflicted = 0
        failed = 0
        active_changed = False
        changed_hashes: list[str] = []
        try:
            for result in results:
                asset = result.file_info
                try:
                    latest = self._config_for_batch_asset(asset)
                    if has_manual_crop(latest.geometry):
                        conflicted += 1
                        continue
                    if _autocrop_fingerprint(latest, self.state.workspace_color_space) != result.fingerprint:
                        conflicted += 1
                        continue

                    rect = result.crop_rect
                    if len(rect) != 4 or not (0.0 <= rect[0] < rect[2] <= 1.0 and 0.0 <= rect[1] < rect[3] <= 1.0):
                        conflicted += 1
                        continue
                    fine_rotation = latest.geometry.fine_rotation + result.correction_angle
                    if not np.isfinite(fine_rotation) or abs(fine_rotation) > FINE_ROTATION_LIMIT:
                        conflicted += 1
                        continue

                    new_geometry = replace(
                        latest.geometry,
                        crop_rect=tuple(float(value) for value in rect),
                        crop_from_auto=False,
                        fine_rotation=float(fine_rotation),
                    )
                    new_process = replace(latest.process, **invalidate_local_bounds(latest.process))
                    updated = replace(latest, geometry=new_geometry, process=new_process)
                    if asset.get("hash") == self.state.current_file_hash:
                        self.session.persist_active_batch_config(updated)
                        active_changed = True
                    else:
                        self.session.repo.save_file_settings(asset["hash"], updated, file_path=asset["path"])
                        changed_hashes.append(asset["hash"])
                    saved += 1
                except Exception:
                    failed += 1
                    logger.exception("Auto Crop All could not persist %s", asset.get("path", asset.get("hash", "frame")))
        finally:
            self._end_batch("autocrop", token)
            self._autocrop_batch_token = None
            self._autocrop_cancel_requested = False
            self.status_progress_requested.emit(0, 0)

        if changed_hashes:
            self.session.frames_edited_offscreen.emit(changed_hashes)

        unresolved = max(0, self._autocrop_dispatched - len(results))
        preserved = self._autocrop_preflight_skipped + conflicted
        failure_suffix = f", failed {failed}" if failed else ""
        self.set_status(
            f"Auto Crop All: saved {saved}, preserved {preserved}, unchanged {unresolved}{failure_suffix}",
            5000,
        )
        if active_changed:
            self.config_updated.emit()
            self.request_render()

    def _on_batch_autocrop_cancelled(self) -> None:
        token = self._autocrop_batch_token
        if token is None:
            return
        self._end_batch("autocrop", token)
        self._autocrop_batch_token = None
        self._autocrop_cancel_requested = False
        self.status_progress_requested.emit(0, 0)
        self.set_status("Auto Crop All aborted; no crops were saved", 4000)

    def _on_batch_autocrop_error(self, message: str) -> None:
        token = self._autocrop_batch_token
        if token is None:
            return
        self._end_batch("autocrop", token)
        self._autocrop_batch_token = None
        self._autocrop_cancel_requested = False
        self.status_progress_requested.emit(0, 0)
        logger.error("Auto Crop All failed: %s", message)
        self.set_status(f"Auto Crop All failed: {message}", 5000, kind="error")

    @property
    def thumbnail_refresh_running(self) -> bool:
        return self._thumbnail_render_running

    def _preempt_background_thumbnail_refresh(self) -> None:
        """Give real batch work immediate use of `norm_thread` and its CPU: the
        background refresh checks for a cancel between every frame, so it yields
        within one frame's processing time instead of finishing the whole roll first."""
        if self._thumbnail_render_running:
            self.thumbnail_render_worker.cancel(self._thumbnail_render_generation)

    def cancel_thumbnail_refresh(self) -> None:
        """Stop a thumbnail refresh outright — the user's own escape hatch for one
        started on too large a folder by mistake. Unlike a real batch's pre-emption,
        this discards the backlog instead of resuming it once norm_thread is free."""
        self._thumbnail_render_resume.clear()
        if self._thumbnail_render_running:
            self._thumbnail_render_user_cancelled = True
            self.thumbnail_render_worker.cancel(self._thumbnail_render_generation)

    def request_thumbnail_refresh(self, scope: str) -> None:
        """User-triggered escape hatch for stale thumbnails: the same background pass
        a bulk edit dispatches automatically, run on demand over ``scope`` ("selection"
        or "roll") — for staleness an automatic trigger missed, or predates one."""
        if scope == "roll":
            indices = self.session.asset_model.visible_actual_indices_ordered()
        else:
            indices = [
                i for i in (self.state.selected_indices or [self.state.selected_file_idx]) if 0 <= i < len(self.state.uploaded_files)
            ]
        hashes = [self.state.uploaded_files[i]["hash"] for i in indices]
        if not hashes:
            self.set_status("Nothing to update", 2000)
            return
        self.refresh_thumbnails_for(hashes)

    def refresh_thumbnails_for(self, hashes: list[str]) -> None:
        """Re-render the filmstrip thumbnails of frames a bulk settings write touched
        without opening them. Runs off the shared batch lane so it never blocks Export
        or another user-triggered batch, and never pops the batch progress dialog for
        what felt like an instant settings change. A request that arrives while a
        generation is already using `norm_thread` is folded into the resume backlog
        instead of being dropped — otherwise a bulk write landing during, say, Batch
        Analysis's own pre-emption window would be lost outright. A request under system
        memory pressure is retried later instead of growing this refresh's own preview
        cache on top of it."""
        wanted = set(hashes)
        if not wanted:
            return
        if self._thumbnail_render_running:
            self._thumbnail_render_resume |= wanted
            return
        available = available_system_memory_bytes()
        if available is not None and available < MIN_RAM_RESERVE_BYTES:
            QTimer.singleShot(_THUMBNAIL_REFRESH_MEMORY_RETRY_MS, lambda: self.refresh_thumbnails_for(list(wanted)))
            return
        seen_keys: set[str] = set()
        frames: list[ThumbnailRenderInput] = []
        for asset in self.state.uploaded_files:
            asset_hash = asset.get("hash")
            if asset_hash not in wanted or asset_hash == self.state.current_file_hash:
                continue
            # A diptych row's canvas render joins two half configs; a whole-frame
            # render would not reproduce it, and a bulk apply writes the whole-scan
            # hash the diptych render does not read.
            if self.diptych_pair(asset) is not None:
                continue
            key = asset_thumbnail_key(asset)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            config = self._config_for_batch_asset(asset)
            frames.append(
                ThumbnailRenderInput(
                    file_info=asset,
                    config=config,
                    thumbnail_key=key,
                    icc_input_active=bool(self.effective_input_icc(config.process)),
                )
            )

        if not frames:
            return

        self._thumbnail_render_generation += 1
        self._thumbnail_render_running = True
        self._thumbnail_render_pending = {f.file_info.get("hash") for f in frames}
        self.thumbnail_refresh_state_changed.emit(True)
        self.set_status(f"Updating {count_of(len(frames), 'thumbnail')}...")
        self.thumbnail_render_requested.emit(
            ThumbnailRenderTask(
                frames=frames,
                workspace_color_space=self.state.workspace_color_space,
                generation=self._thumbnail_render_generation,
            )
        )

    def _on_thumbnail_render_progress(self, current: int, total: int, name: str) -> None:
        self.set_status(f"Updating thumbnail {current}/{total}: {name}")

    def _on_thumbnail_rendered(self, frame: ThumbnailRenderInput, buffer: np.ndarray) -> None:
        if not self._thumbnail_render_running:
            return  # late frame from an already-finished or pre-empted generation
        asset_hash = frame.file_info.get("hash")
        self._thumbnail_render_pending.discard(asset_hash)
        if asset_hash == self.state.current_file_hash:
            return  # opened since dispatch — the live render already owns its thumbnail
        asset = next((a for a in self.state.uploaded_files if a.get("hash") == asset_hash), None)
        if asset is None or self._config_for_batch_asset(asset) != frame.config:
            return  # removed, or edited again since dispatch — this render is stale
        cs, monitor_bytes, proof = self.display_transform_params(process=frame.config.process)
        self.thumbnail_update_requested.emit(
            ThumbnailUpdateTask(
                file_hash=frame.thumbnail_key,
                buffer=buffer,
                color_space=cs,
                monitor_icc_bytes=monitor_bytes,
                proof=proof,
                persist=True,
            )
        )

    def _on_thumbnail_render_finished(self, count: int) -> None:
        if not self._thumbnail_render_running:
            return  # stale completion from an already-cleared generation
        self.set_status(f"Updated {count_of(count, 'thumbnail')}", 3000)
        self._finish_thumbnail_render_generation()

    def _on_thumbnail_render_cancelled(self) -> None:
        if self._thumbnail_render_user_cancelled:
            # Stopped outright, not pre-empted: discard the backlog instead of the
            # routine path's resume, and say so — this one the user did ask for.
            self._thumbnail_render_user_cancelled = False
            self._thumbnail_render_resume.clear()
            self.set_status("Thumbnail update cancelled", 3000)
            self._finish_thumbnail_render_generation()
            return
        # Fires whenever real batch work pre-empts a running refresh, which is routine
        # rather than something the user asked for, so no status message here — the
        # resume dispatch below, if any, sets its own "Updating N thumbnails..." right
        # after this returns, which is the accurate, visible one.
        self._thumbnail_render_resume |= self._thumbnail_render_pending
        self._finish_thumbnail_render_generation()

    def _on_thumbnail_render_error(self, message: str) -> None:
        if not self._thumbnail_render_running:
            return
        logger.error("Background thumbnail refresh failed: %s", message)
        self.set_status(f"Thumbnail update failed: {message}", 5000, kind="error")
        self._finish_thumbnail_render_generation()

    def _finish_thumbnail_render_generation(self) -> None:
        """Ends the current generation and, if anything is backlogged, immediately
        redispatches it. Safe only because every `norm_thread` batch's own dispatch —
        `_begin_batch` plus its `*_requested.emit(...)` — runs in one straight line with
        no return to the event loop in between; that ordering is what keeps this
        redispatch queued behind, never ahead of, the batch that pre-empted it."""
        self._thumbnail_render_running = False
        self._thumbnail_render_pending = set()
        self.thumbnail_refresh_state_changed.emit(False)
        if self._thumbnail_render_resume:
            leftover = list(self._thumbnail_render_resume)
            self._thumbnail_render_resume.clear()
            self.refresh_thumbnails_for(leftover)

    def detect_aspect_ratio(self) -> None:
        img = self.state.preview_raw
        if img is None:
            return

        geom = self.state.config.geometry
        transformed = img
        if geom.rotation != 0:
            transformed = np.rot90(transformed, k=geom.rotation)
        if geom.flip_horizontal:
            transformed = np.ascontiguousarray(np.fliplr(transformed))
        if geom.flip_vertical:
            transformed = np.ascontiguousarray(np.flipud(transformed))
        if geom.fine_rotation != 0.0:
            transformed = apply_fine_rotation(transformed, geom.fine_rotation)

        # Detection can match a portrait frame to a portrait-only AspectRatio the picker
        # does not display, so canonicalize to an entry it can show (see
        # domain.models.CROP_RATIO_CHOICES). The crop tool auto-orients either way.
        new_ratio = canonical_crop_ratio(detect_closest_aspect_ratio(transformed, fallback=geom.autocrop_ratio))
        if new_ratio == geom.autocrop_ratio:
            return

        new_proc = replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process))
        self.session.update_config(
            replace(
                self.state.config,
                geometry=replace(geom, autocrop_ratio=new_ratio),
                process=new_proc,
            ),
            persist=True,
            render=False,
        )
        self._lock_roll_card("autocrop")
        # Emit manually so UI syncs (combo dropdown updates), but without triggering
        # a render via the state_changed debounce.
        self.config_updated.emit()
        if geom.crop_from_auto:
            self.request_render()

    def clear_retouch(self) -> None:
        from negpy.desktop.view.confirm import confirm_clear_heals

        conf = self.state.config.retouch
        count = len(conf.manual_dust_spots) + len(conf.manual_heal_strokes) + len(conf.scratch_lines)
        if count == 0:
            return
        # Wiping every heal is not step-recoverable like single-heal undo, so confirm.
        if not confirm_clear_heals(None, count):
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=[], manual_heal_strokes=[], scratch_lines=[]),
            ),
            persist=True,
        )
        self.request_render()

    def delete_heal(self, kind: str, index: int) -> None:
        """Removes one placed heal by identity ("stroke"/"spot", index) — lets the
        user pick off a bad patch directly instead of unwinding newer heals first."""
        strokes = list(self.state.config.retouch.manual_heal_strokes)
        spots = list(self.state.config.retouch.manual_dust_spots)
        lines = list(self.state.config.retouch.scratch_lines)
        if kind == "stroke" and 0 <= index < len(strokes):
            strokes.pop(index)
        elif kind == "spot" and 0 <= index < len(spots):
            spots.pop(index)
        elif kind == "line" and 0 <= index < len(lines):
            lines.pop(index)
        else:
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=spots, manual_heal_strokes=strokes, scratch_lines=lines),
            ),
            persist=True,
        )
        self.request_render()

    def undo_last_retouch(self) -> None:
        """
        Removes the most recently added heal (strokes first, then legacy spots).
        """
        strokes = list(self.state.config.retouch.manual_heal_strokes)
        spots = list(self.state.config.retouch.manual_dust_spots)
        lines = list(self.state.config.retouch.scratch_lines)
        if lines:
            lines.pop()
        elif strokes:
            strokes.pop()
        elif spots:
            spots.pop()
        else:
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_dust_spots=spots, manual_heal_strokes=strokes, scratch_lines=lines),
            ),
            persist=True,
        )
        self.request_render()

    def _handle_dust_pick(self, nx: float, ny: float) -> None:
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)
        self._commit_heal_stroke([(rx, ry)])

    def _handle_scratch_line_pick(self, nx: float, ny: float) -> None:
        """One click near a transport scratch: trace the whole line and commit it.

        Traced on the source-frame preview, so the stored line is in raw coordinates and the
        render re-measures the scratch at its own resolution. A click that finds nothing says
        so rather than committing a line that would repair nothing.
        """
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        preview = self.state.preview_raw
        if uv_grid is None or preview is None:
            return
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)
        line = trace_scratch(preview, rx, ry, self.state.config.retouch.scratch_threshold)
        if line is None:
            self.set_status("No scratch found there — click directly on the line", 3000, kind="warning")
            return
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, scratch_lines=list(self.state.config.retouch.scratch_lines) + [line]),
            ),
            persist=True,
        )
        self.request_render()

    def handle_heal_stroke_completed(self, viewport_pts: list) -> None:
        """Commits a scratch-tool polyline (viewport-normalized points)."""
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None or not viewport_pts:
            return
        raw_pts = [CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_pts]
        self._commit_heal_stroke(raw_pts)

    def _commit_heal_stroke(self, raw_pts: list) -> None:
        conf = self.state.config.retouch
        size = float(conf.manual_dust_size)
        # Brush size is a diameter at HEAL_SIZE_REF scale, like the pipeline radius and the
        # overlay cursor. The trailing zeroes are the retired clone-source offset: repairs
        # are content-aware now, but the stroke keeps its four-element shape so stored edits
        # load unchanged.
        stroke = ([[rx, ry] for rx, ry in raw_pts], size, 0.0, 0.0)
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(self.state.config.retouch, manual_heal_strokes=conf.manual_heal_strokes + [stroke]),
            ),
            persist=True,
        )
        self.request_render()

    def handle_dust_exclusion_painted(self, viewport_pts: list) -> None:
        """Commits a right-painted stroke (viewport-normalized points) the optical detector
        must leave alone. One stroke, so the whole path it swept is released as a band."""
        if not viewport_pts:
            return
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return
        conf = self.state.config.retouch
        raw_pts = [CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_pts]
        stroke = ([[rx, ry] for rx, ry in raw_pts], float(conf.manual_dust_size))
        self.session.update_config(
            replace(
                self.state.config,
                retouch=replace(conf, dust_exclusion_strokes=list(conf.dust_exclusion_strokes) + [stroke]),
            ),
            persist=True,
        )
        self.request_render()

    def handle_local_mask_created(self, shape: str, viewport_vertices: list) -> None:
        from negpy.features.local.logic import min_points
        from negpy.features.local.models import LocalMask, MaskShape

        mask_shape = MaskShape(shape)
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None or len(viewport_vertices) < min_points(mask_shape):
            return

        raw_vertices = tuple(CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_vertices)

        mask = LocalMask(vertices=raw_vertices, shape=mask_shape)
        local = self.state.config.local
        new_masks = local.masks + (mask,)
        new_local = replace(local, masks=new_masks)
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.state.local_selected_mask = len(new_masks) - 1
        self.set_active_tool(ToolMode.NONE)  # auto-exit draw mode once the polygon closes
        self.config_updated.emit()
        self.request_render()

    def handle_local_mask_edited(self, index: int, viewport_vertices: list) -> None:
        """Replace a mask's vertices after an on-canvas drag/add edit (persist on release)."""
        from negpy.features.local.logic import min_points

        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        local = self.state.config.local
        if uv_grid is None or not (0 <= index < len(local.masks)):
            return
        if len(viewport_vertices) < min_points(local.masks[index].shape):
            return
        raw_vertices = tuple(CoordinateMapping.map_click_to_raw(nx, ny, uv_grid) for nx, ny in viewport_vertices)
        masks = list(local.masks)
        masks[index] = replace(masks[index], vertices=raw_vertices)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.config_updated.emit()
        self.request_render()

    def delete_local_vertex(self, index: int, vertex_index: int) -> None:
        """Remove one vertex from a polygon mask. Keep a minimum of 3 vertices."""
        from negpy.features.local.models import MaskShape

        local = self.state.config.local
        if not (0 <= index < len(local.masks)):
            return
        mask = local.masks[index]
        if mask.shape != MaskShape.POLYGON:
            return
        if len(mask.vertices) <= 3 or not (0 <= vertex_index < len(mask.vertices)):
            return
        verts = mask.vertices[:vertex_index] + mask.vertices[vertex_index + 1 :]
        masks = list(local.masks)
        masks[index] = replace(mask, vertices=verts)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.config_updated.emit()
        self.request_render()

    def select_local_mask(self, index: int) -> None:
        self.state.local_selected_mask = index
        self.config_updated.emit()

    def set_local_mask_visible(self, index: int, visible: bool) -> None:
        """Show/hide one mask's outline on the canvas (view-only; no re-render)."""
        if not (0 <= index < len(self.state.config.local.masks)):
            return
        hidden = set(self.state.local_hidden_masks)
        if visible:
            hidden.discard(index)
        else:
            hidden.add(index)
        self.state.local_hidden_masks = hidden
        self.session.persist_hidden_masks()
        if self.canvas:
            self.canvas.overlay.update()

    def set_local_mask_enabled(self, index: int, enabled: bool) -> None:
        """Suppress or restore one mask's effect on the render, independent of selection."""
        local = self.state.config.local
        if not (0 <= index < len(local.masks)):
            return
        masks = list(local.masks)
        masks[index] = replace(masks[index], enabled=enabled)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)
        self.config_updated.emit()
        self.request_render()

    def delete_local_mask(self, index: int) -> None:
        local = self.state.config.local
        if not (0 <= index < len(local.masks)):
            return
        from negpy.desktop.view.confirm import confirm_delete_mask

        if not confirm_delete_mask(None):
            return
        new_masks = local.masks[:index] + local.masks[index + 1 :]
        new_local = replace(local, masks=new_masks)
        self.session.update_config(replace(self.state.config, local=new_local), persist=True)

        sel = self.state.local_selected_mask
        self.state.local_selected_mask = -1 if sel == index else (sel - 1 if sel > index else sel)
        self.state.local_hidden_masks = {j - 1 if j > index else j for j in self.state.local_hidden_masks if j != index}
        self.session.persist_hidden_masks()

        self.config_updated.emit()
        self.request_render()

    def update_selected_local_mask(self, persist: bool = True, readback_metrics: bool = True, **changes) -> None:
        local = self.state.config.local
        idx = self.state.local_selected_mask
        if not (0 <= idx < len(local.masks)):
            return
        masks = list(local.masks)
        masks[idx] = replace(masks[idx], **changes)
        new_local = replace(local, masks=tuple(masks))
        self.session.update_config(replace(self.state.config, local=new_local), persist=persist)
        self.request_render(readback_metrics=readback_metrics)

    def _handle_wb_pick(self, nx: float, ny: float) -> None:
        """
        Samples color from viewport coordinates and updates WB shifts to neutralize.
        """
        with self.state.metrics_lock:
            metrics = dict(self.state.last_metrics)

        img = metrics.get("normalized_log")
        is_log = True
        if img is None:
            img = metrics.get("base_positive")
            is_log = False

        if img is None:
            return

        roi = metrics.get("active_roi")
        radius = 4

        if isinstance(img, GPUTexture):
            h, w = img.height, img.width
            if roi and is_log:
                ry1, ry2, rx1, rx2 = roi
                center_y = int(np.clip(ry1 + ny * (ry2 - ry1), 0, h - 1))
                center_x = int(np.clip(rx1 + nx * (rx2 - rx1), 0, w - 1))
            else:
                center_y = int(np.clip(ny * h, 0, h - 1))
                center_x = int(np.clip(nx * w, 0, w - 1))
            x0 = max(center_x - radius, 0)
            y0 = max(center_y - radius, 0)
            rw = min(center_x + radius, w) - x0
            rh = min(center_y + radius, h) - y0
            sampled = img.readback_region(x0, y0, rw, rh).mean(axis=(0, 1))
        elif isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            if roi and is_log:
                ry1, ry2, rx1, rx2 = roi
                center_y = int(np.clip(ry1 + ny * (ry2 - ry1), 0, h - 1))
                center_x = int(np.clip(rx1 + nx * (rx2 - rx1), 0, w - 1))
            else:
                center_y = int(np.clip(ny * h, 0, h - 1))
                center_x = int(np.clip(nx * w, 0, w - 1))
            y0 = max(center_y - radius, 0)
            y1_ = min(center_y + radius, h)
            x0 = max(center_x - radius, 0)
            x1_ = min(center_x + radius, w)
            sampled = img[y0:y1_, x0:x1_].mean(axis=(0, 1))
        else:
            return

        exp = self.state.config.exposure
        bounds = metrics.get("final_bounds") or metrics.get("log_bounds")  # CPU/GPU key names
        if is_log:
            new_m, new_y = calculate_wb_shifts_from_log(sampled[:3], bounds)
        else:
            delta_m, delta_y = calculate_wb_shifts(sampled[:3])
            damping = 0.4
            new_m = exp.wb_magenta + delta_m * damping
            new_y = exp.wb_yellow + delta_y * damping

        region = self.state.wb_pick_region
        if region == 0:
            new_exp = replace(
                exp,
                wb_cyan=0.0,
                wb_magenta=float(np.clip(new_m, -1.0, 1.0)),
                wb_yellow=float(np.clip(new_y, -1.0, 1.0)),
            )
        else:
            # Store the residual over the global pair in the region's fields. Filtration
            # offsets are range-normalized and regional ones are absolute density, so
            # convert by the stretch range. Assumes the picked patch sits in its region.
            c_field, m_field, y_field = (
                ("shadow_cyan", "shadow_magenta", "shadow_yellow"),
                ("highlight_cyan", "highlight_magenta", "highlight_yellow"),
            )[region - 1]
            rng_m = rng_y = 1.0
            if is_log and bounds is not None:
                rng_m = max(abs(bounds.ceils[1] - bounds.floors[1]), 1e-6)
                rng_y = max(abs(bounds.ceils[2] - bounds.floors[2]), 1e-6)
            dm = (new_m - exp.wb_magenta) / rng_m
            dy = (new_y - exp.wb_yellow) / rng_y
            new_exp = replace(
                exp,
                **{
                    c_field: 0.0,
                    m_field: float(np.clip(dm, -1.0, 1.0)),
                    y_field: float(np.clip(dy, -1.0, 1.0)),
                },
            )
        self.session.update_config(replace(self.state.config, exposure=new_exp), persist=True, record_history=True)
        self.request_render()

    def _normalization_targets(self, scene_id: Optional[str]) -> List[dict]:
        """Loaded frames a baseline is written onto: one scene's members, or for the roll
        every frame outside a scene, since a scene's baseline replaces the roll's."""
        by_hash = rolls.scene_by_hash(self.session.repo, self.state.active_roll_id)
        return [f for f in self.state.uploaded_files if by_hash.get(rolls.unforked_hash(f["hash"]), (0, None))[1] == scene_id]

    def _confirm_normalization(self, frames: List[dict], title: str, filtered: bool) -> bool:
        total = len(frames)
        cropped = 0
        for f in frames:
            p = self.session.repo.load_file_settings(f["hash"])
            if p and (p.geometry.crop_rect or p.geometry.crop_from_auto):
                cropped += 1

        if cropped == 0:
            crop_status = f"Crop status: 0 of {total} files are cropped."
            crop_warning = (
                "Strongly recommended: crop all images before running the analysis. "
                "Without a crop, the Analysis Buffer's small centered "
                "margin isn't enough to exclude sprocket holes and empty space outside "
                "the actual frame — that unwanted region gets included in the luma and "
                "color average, producing a less accurate result for every file."
            )
        elif cropped < total:
            crop_status = f"Crop status: {cropped} of {total} files are cropped."
            crop_warning = (
                f"Strongly recommended: crop the remaining {count_of(total - cropped, 'file')} "
                "before running the analysis. Uncropped files rely on the Analysis "
                "Buffer's small centered margin, which isn't enough to exclude sprocket "
                "holes and empty space outside the actual frame — that unwanted region "
                "gets included in the luma and color average, producing a less accurate "
                "result for every file."
            )
        else:
            crop_status = f"Crop status: all {total} files are cropped."
            crop_warning = "Analysis will run on each file's cropped negative area."

        sheet_note = ""
        if filtered and self.session.asset_model.sheet_filter != "all":
            sheet_note = (
                f"Note: the Sheet filter is on — only the {count_of(total, 'visible frame')} {plural(total, 'is', 'are')} analyzed.\n\n"
            )
        if title == "Roll Analysis":
            summary = (
                "Roll Analysis measures the exposure bounds of every file outside a scene and "
                "applies their average to those files, so they share a consistent baseline."
            )
        else:
            summary = "Scene Analysis measures the exposure bounds of each scene's files and applies their average to that scene alone."

        reply = QMessageBox.question(
            None,
            title,
            f"{sheet_note}"
            f"{crop_status}\n"
            f"{crop_warning}\n\n"
            f"{summary} A frame whose color is far from the rest keeps its own exposure and color.\n\n"
            "Two settings from the image you have open right now are applied to every "
            "file before averaging:\n"
            "  • Analysis Buffer — shrinks the analyzed region inward, excluding a "
            "margin around the edges (film borders, light leaks, the scanner mask). "
            "A frame with its own drawn analysis region meters that region instead.\n"
            "  • Luma Range Clip — how aggressively the highlight/shadow tails are "
            "clipped when setting each file's bounds.\n"
            "Set both on the current frame before running.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _dispatch_normalization(self, scene_id: Optional[str], frames: List[dict]) -> None:
        self._normalization_scope = scene_id
        task = NormalizationTask(
            frames=[NormalizationInput(file_info=a, config=self._config_for_batch_asset(a)) for a in frames],
            workspace_color_space=self.state.workspace_color_space,
            override_analysis_buffer=self.state.config.process.analysis_buffer,
            override_luma_range_clip=self.state.config.process.luma_range_clip,
            override_color_range_clip=self.state.config.process.color_range_clip,
        )
        self.normalization_requested.emit(task)

    def _scene_name(self, scene_id: str) -> str:
        entry = dict(rolls.roll_scenes(self.session.repo, self.state.active_roll_id)).get(scene_id)
        return entry["name"] if entry else ""

    def request_batch_normalization(self) -> None:
        """Roll Analysis: the visible frames outside every scene."""
        self._request_normalization(None)

    def request_scene_analysis(self, scene_id: str) -> None:
        """Scene Analysis: one scene's loaded members, whatever the sheet filter shows."""
        self._request_normalization(scene_id)

    def _request_normalization(self, scene_id: Optional[str]) -> None:
        title = "Roll Analysis" if scene_id is None else "Scene Analysis"
        if self._batch_busy(title):
            return
        if scene_id is None:
            in_scene = set(rolls.scene_by_hash(self.session.repo, self.state.active_roll_id))
            visible = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
            frames = [f for f in visible if rolls.unforked_hash(f["hash"]) not in in_scene]
        else:
            frames = self._normalization_targets(scene_id)
        if not frames:
            return
        if not self._confirm_normalization(frames, title, filtered=scene_id is None):
            return
        batch_title = "Analyzing roll" if scene_id is None else f"Analyzing scene “{self._scene_name(scene_id)}”"
        if self._begin_batch("normalization", batch_title, abortable=True) is None:
            return
        self._scene_queue = []
        self.set_status(f"Starting {title}…")
        self._dispatch_normalization(scene_id, frames)

    def request_analyze_all_scenes(self) -> None:
        """Scene Analysis for every scene of the active roll with a loaded frame, one after another."""
        if self._batch_busy("Scene Analysis"):
            return
        queue = [
            (sid, frames)
            for sid, _entry in rolls.roll_scenes(self.session.repo, self.state.active_roll_id)
            if (frames := self._normalization_targets(sid))
        ]
        if not queue:
            self.set_status("No scene has a loaded frame", 3000)
            return
        if not self._confirm_normalization([f for _sid, frames in queue for f in frames], "Scene Analysis", filtered=False):
            return
        if self._begin_batch("normalization", "Analyzing scenes", abortable=True) is None:
            return
        self._scene_queue = [sid for sid, _frames in queue[1:]]
        self._dispatch_normalization(*queue[0])

    def _on_normalization_progress(self, current: int, total: int, name: str, has_crop: bool) -> None:
        """
        Updates UI status during Roll or Scene Analysis.
        """
        marker = "cropped" if has_crop else "full frame"
        self.set_status(f"Analyzing {current}/{total}: {name} [{marker}]...")
        self.status_progress_requested.emit(current, total)
        self.batch_progress.emit(current, total, f"{name} [{marker}]")

    def _push_bounds(
        self,
        targets: List[dict],
        floors: tuple,
        ceils: tuple,
        roll_name: Optional[str],
        source: str,
        outliers: Collection[str] = (),
    ) -> int:
        """Writes a baseline onto *targets*, riding both average axes, except that a frame in
        *outliers* keeps its own bounds. A frame with Lock Bounds on keeps its own exposure.
        Returns how many locked frames were skipped."""
        locked_skipped = 0
        changed_hashes: list[str] = []
        current = self.state.current_file_hash
        for f_info in targets:
            p = self.session.repo.load_file_settings(f_info["hash"]) or self.session.config_for_asset(f_info)
            if p.process.lock_bounds:
                locked_skipped += 1
                continue
            rides = f_info["hash"] not in outliers
            new_process = replace(
                p.process,
                use_luma_average=rides,
                use_color_average=rides,
                locked_floors=floors,
                locked_ceils=ceils,
                roll_name=roll_name,
                baseline_source=source,
            )
            new_p = replace(p, process=new_process)
            # The active file records its step via update_config(persist=True) below.
            if f_info["hash"] != current:
                self.session.push_external_history(f_info["hash"], p, new_p)
                changed_hashes.append(f_info["hash"])
            self.session.repo.save_file_settings(f_info["hash"], new_p, file_path=f_info["path"])

        if changed_hashes:
            self.session.frames_edited_offscreen.emit(changed_hashes)

        if any(f["hash"] == current for f in targets) and not self.state.config.process.lock_bounds:
            rides = current not in outliers
            new_process = replace(
                self.state.config.process,
                use_luma_average=rides,
                use_color_average=rides,
                locked_floors=floors,
                locked_ceils=ceils,
                roll_name=roll_name,
                baseline_source=source,
            )
            self.session.update_config(replace(self.state.config, process=new_process), persist=True)
        return locked_skipped

    def _on_normalization_finished(self, locked_floors: tuple, locked_ceils: tuple, outlier_hashes: list) -> None:
        """
        Applies the pooled baseline to the analyzed scope and records it on the roll
        (rolls.set_roll_normalization) or the scene (rolls.set_scene_normalization).
        *outlier_hashes* are frames whose color is far from the pool: they keep their own
        bounds, now and on every later apply of this baseline.
        """
        scene_id = self._normalization_scope
        roll_id = self.state.active_roll_id
        source = f"scene:{scene_id}" if scene_id else (f"roll:{roll_id}" if roll_id else "")
        outliers = tuple(outlier_hashes)
        targets = self._normalization_targets(scene_id)
        locked_skipped = self._push_bounds(targets, locked_floors, locked_ceils, None, source, outliers)

        if scene_id is None:
            if roll_id is not None:
                rolls.set_roll_normalization(self.session.repo, roll_id, locked_floors, locked_ceils, outliers=outliers)
            message = "Roll analysis complete"
            scope_word = "roll"
        else:
            rolls.set_scene_normalization(self.session.repo, roll_id, scene_id, locked_floors, locked_ceils, outliers=outliers)
            self.session.refresh_scene_marks()
            message = f"Scene “{self._scene_name(scene_id)}” analyzed"
            scope_word = "scene"

        names_by_hash = {f["hash"]: f["name"] for f in self.state.uploaded_files}
        outlier_names = [names_by_hash.get(h, h) for h in outliers]
        timeout = 3000
        if locked_skipped:
            message += f" — {count_of(locked_skipped, 'locked frame')} kept its own exposure"
        if outlier_names:
            shown = ", ".join(outlier_names[:3])
            if len(outlier_names) > 3:
                shown += f" +{len(outlier_names) - 3} more"
            message += f". {count_of(len(outlier_names), 'frame')} far from the {scope_word} color {plural(len(outlier_names), 'keeps its', 'keep their')} own exposure and color: {shown}"
            timeout = 8000
        self.set_status(message, timeout=timeout)

        while self._scene_queue:
            next_id = self._scene_queue.pop(0)
            frames = self._normalization_targets(next_id)
            if frames:
                self._dispatch_normalization(next_id, frames)
                self.request_render()
                return
        self._end_batch("normalization")
        self.status_progress_requested.emit(0, 0)
        self.request_render()

    def request_reset_roll(self) -> None:
        """Reset every visible frame to its own bare defaults -- Reset Settings, applied
        to the whole roll at once."""
        visible = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible:
            return
        self.session.reset_roll(visible)
        self.set_status(f"Reset {count_of(len(visible), 'frame')} to defaults", timeout=3000)
        self.request_render()

    def set_roll_baseline_from_frame(self, roll_id: str) -> None:
        """Save the active frame's rendered bounds as the roll's baseline, then push them
        out the way a Roll Analysis result is pushed. A standing rule, not a one-shot
        copy: a frame loaded later reads the same baseline."""
        bounds = _source_effective_bounds(self.state.config.process)
        if bounds is None:
            self.set_status("Render this frame before taking its bounds", 3000)
            return
        floors, ceils = bounds
        rolls.set_roll_normalization(self.session.repo, roll_id, floors, ceils)
        self.apply_normalization_roll(roll_id)

    def apply_normalization_roll(self, roll_id: str) -> None:
        """
        Loads a roll's saved Roll Analysis baseline onto every loaded frame outside a
        scene that has not locked its own bounds. No-op if that roll has never been analyzed.
        """
        data = rolls.roll_normalization(self.session.repo, roll_id)
        if not data:
            return
        entry = rolls.roll_for_id(self.session.repo, roll_id)
        name = entry["name"] if entry else roll_id
        locked_floors, locked_ceils = data["floors"], data["ceils"]

        locked_skipped = self._push_bounds(
            self._normalization_targets(None), locked_floors, locked_ceils, name, f"roll:{roll_id}", data.get("outliers", ())
        )

        message = f'Applied "{name}"\'s baseline'
        if locked_skipped:
            message += f" — {count_of(locked_skipped, 'locked frame')} kept its own exposure"
        self.set_status(message, 2000)
        self.request_render()

    def _selected_scene_hashes(self) -> List[str]:
        files = self.state.uploaded_files
        targets = [i for i in (self.state.selected_indices or [self.state.selected_file_idx]) if 0 <= i < len(files)]
        return [rolls.unforked_hash(files[i]["hash"]) for i in targets]

    def _edit_scenes(self, edit: Callable[[str], Any]) -> Any:
        roll_id = self.state.active_roll_id
        if roll_id is None:
            self.set_status("Save the Film Strip as a roll before grouping scenes", 3000)
            return None
        result = edit(roll_id)
        self.session.refresh_scene_marks()
        return result

    def request_group_as_scene(self, name: str) -> Optional[str]:
        hashes = self._selected_scene_hashes()
        if not hashes:
            return None
        return self._edit_scenes(lambda roll_id: rolls.create_scene(self.session.repo, roll_id, name, hashes))

    def request_add_to_scene(self, scene_id: str) -> None:
        hashes = self._selected_scene_hashes()
        self._edit_scenes(lambda roll_id: rolls.add_to_scene(self.session.repo, roll_id, scene_id, hashes))

    def request_remove_from_scene(self) -> None:
        hashes = self._selected_scene_hashes()
        self._edit_scenes(lambda roll_id: rolls.remove_from_scenes(self.session.repo, roll_id, hashes))

    def request_rename_scene(self, scene_id: str, name: str) -> None:
        self._edit_scenes(lambda roll_id: rolls.rename_scene(self.session.repo, roll_id, scene_id, name))

    def request_delete_scene(self, scene_id: str) -> None:
        """Forgets the grouping only; members keep the baseline they last took."""
        self._edit_scenes(lambda roll_id: rolls.delete_scene(self.session.repo, roll_id, scene_id))

    def select_scene_frames(self, scene_id: str) -> None:
        members = {f["hash"] for f in self._normalization_targets(scene_id)}
        indices = [i for i, f in enumerate(self.state.uploaded_files) if f["hash"] in members]
        if not indices:
            return
        if self.state.selected_file_idx in indices:
            self.session.update_selection(indices)
        else:
            self.session.select_file(indices[0], selection_override=indices)

    _ROLL_CARDS = (
        "film",
        "sensor",
        "autocrop",
        "baseline",
        "process",
        "demosaic",
        "lens",
        "flatfield",
        "metadata_gear",
        "metadata_capture",
        "metadata_process",
        "metadata_scanning",
        "metadata_exposure",
    )
    _ROLL_CARD_LABELS = {
        "film": "Film Mode",
        "sensor": "Calibration",
        "demosaic": "Raw Decode",
        "baseline": "Roll Analysis",
        "process": "Metering",
        "autocrop": "Crop",
        "lens": "Optics",
        "flatfield": "Optics",
        "metadata_gear": "Analog Gear",
        "metadata_capture": "Capture",
        "metadata_process": "Process",
        "metadata_scanning": "Scanning",
        "metadata_exposure": "Exposure",
    }
    METADATA_CARDS = ("metadata_gear", "metadata_capture", "metadata_process", "metadata_scanning", "metadata_exposure")

    @staticmethod
    def _card_values(config, card_key: str) -> dict:
        """What *card_key* currently holds on *config*, ready to freeze or push."""
        section, fields = rolls.ROLL_DEFAULT_FIELDS[card_key]
        return {name: getattr(getattr(config, section), name) for name in fields}

    @staticmethod
    def _with_card_values(config, card_key: str, values: dict, remeter: bool = True):
        """*config* with *card_key*'s section carrying *values*. Fields that move the
        crop also feed the meter, so a real change to one drops the cached per-frame
        bounds with it; re-freezing a card on its own values must not."""
        section, _fields = rolls.ROLL_DEFAULT_FIELDS[card_key]
        current = getattr(config, section)
        remeter = remeter and any(name in BOUNDS_INPUT_FIELDS and value != getattr(current, name) for name, value in values.items())
        config = replace(config, **{section: replace(current, **values)})
        if remeter:
            config = replace(config, process=replace(config.process, **invalidate_local_bounds(config.process)))
        return config

    def locked_roll_cards(self) -> set:
        """The Roll-tab cards the active frame has locked to its own value, within the
        active roll. Empty with no active roll. Keyed on the unforked hash, like the lock
        itself: it is about this physical frame, not its current edit identity, and must
        read the same locked or not whether or not it is forked."""
        roll_id = self.state.active_roll_id
        if roll_id is None or not self.state.current_file_hash:
            return set()
        return rolls.frame_override_cards(self.session.repo, roll_id, rolls.unforked_hash(self.state.current_file_hash))

    def roll_card_locked(self, card_key: str) -> bool:
        return card_key in self.locked_roll_cards()

    def diverged_roll_cards(self) -> List[str]:
        """Every Roll-tab card locked away from the roll on the active frame -- what
        Apply to Whole Roll / Apply to Selected Frames act on."""
        locked = self.locked_roll_cards()
        return [key for key in self._ROLL_CARDS if key in locked]

    def _lock_roll_card(self, card_key: str) -> None:
        """Locks or unlocks *card_key* to match whether the active frame's own
        already-applied value actually differs from the roll's -- editing a value and
        then editing it back to what the roll already says is not a divergence, so the
        card must not stay marked This Frame Only just because it was touched. Shared
        tail of set_roll_default and set_process_mode/set_positive_source (the "film"
        card). No-op with no active roll."""
        roll_id = self.state.active_roll_id
        if roll_id is None:
            return
        defaults = rolls.roll_defaults(self.session.repo, roll_id)
        # A field the roll has never set at all cannot "match" -- there is nothing yet
        # to differ from, and treating that as a match would hide a card's first-ever
        # edit from Apply until every one of its fields happened to get a roll default.
        current = self._card_values(self.state.config, card_key)
        matches_roll = all(name in defaults and value == defaults[name] for name, value in current.items())
        diverged = not matches_roll
        if diverged == self.roll_card_locked(card_key):
            return
        rolls.set_frame_override(self.session.repo, roll_id, rolls.unforked_hash(self.state.current_file_hash), card_key, diverged)

    def set_process_mode(self, mode: str) -> None:
        """Switches Film Mode for the active frame, locking the "film" card away from
        the roll the instant it changes and was not already -- same as any other
        Roll-tab card (set_roll_default). Apply to Whole Roll pushes it out."""
        exp = self.state.config.exposure
        strength = cast_removal_for_mode(mode, exp.cast_removal_strength)
        new_exposure = replace(exp, cast_removal_strength=strength) if strength != exp.cast_removal_strength else exp
        proc = self.state.config.process
        # Leaving Slide drops Positive, and restores the autos as the toggle would.
        drops_positive = proc.positive_source and mode != ProcessMode.E6
        if drops_positive:
            new_exposure = replace(
                new_exposure,
                auto_exposure=auto_meter_for_positive_source(False, new_exposure.auto_exposure),
                auto_normalize_contrast=auto_meter_for_positive_source(False, new_exposure.auto_normalize_contrast),
            )
        new_process = replace(
            proc,
            process_mode=mode,
            positive_source=proc.positive_source and not drops_positive,
            **invalidate_local_bounds(proc),
        )
        self.apply_config(replace(self.state.config, process=new_process, exposure=new_exposure), persist=True)
        self._lock_roll_card("film")

    def set_positive_source(self, checked: bool) -> None:
        """Toggles Positive for the active frame, locking the "film" card away from
        the roll the instant it changes and was not already -- same treatment as
        Film Mode, since both live on that one card.

        Also rewrites Auto Density/Auto Grade to the mode being switched to
        (auto_meter_for_positive_source): untouched, they carry whichever mode's
        default they last matched, so this only moves them when the user never
        touched them."""
        proc = self.state.config.process
        if proc.process_mode != ProcessMode.E6:
            # The shortcut still reaches the hidden button; the autos must not move.
            return
        new_process = replace(
            proc,
            positive_source=checked,
            **invalidate_local_bounds(proc),
        )
        exp = self.state.config.exposure
        new_exposure = replace(
            exp,
            auto_exposure=auto_meter_for_positive_source(checked, exp.auto_exposure),
            auto_normalize_contrast=auto_meter_for_positive_source(checked, exp.auto_normalize_contrast),
        )
        self.apply_config(replace(self.state.config, process=new_process, exposure=new_exposure), persist=True)
        self._lock_roll_card("film")

    def set_roll_default(self, card_key: str, persist: bool = True, readback_metrics: bool = True, **changes) -> None:
        """Edits *card_key* for the active frame alone, same as any other control --
        marking it locked away from the roll the instant it changes and was not
        already, since the frame no longer matches whatever the roll currently says.
        The card's own Roll button (apply_roll_card) is the only thing that pushes a
        value back out; editing alone never does, here or on an already-locked card.

        persist=False (a slider mid-drag) previews on the active frame only, same as
        any other live preview -- the lock only follows the settled value, not every
        intermediate tick.
        """
        new_config = self._with_card_values(self.state.config, card_key, changes, remeter=persist)
        self.apply_config(new_config, persist=persist, readback_metrics=readback_metrics)
        if persist:
            self._lock_roll_card(card_key)

    def apply_roll_card(self, card_key: str) -> int:
        """One card's Roll button: pushes just that card, leaving every other diverged
        card marked. Force Settings is the Roll tab's own modifier over all of them, so
        it does not widen a single-card push."""
        return self._push_cards_to_roll([card_key] if self.roll_card_locked(card_key) else [], sweep=False)

    def _push_cards_to_roll(self, pushed: List[str], sweep: bool) -> int:
        roll_id = self.state.active_roll_id
        if roll_id is None:
            self.set_status(_NOTHING_TO_APPLY, 2500)
            return 0
        active_hash = self.state.current_file_hash
        for card_key in pushed:
            rolls.set_roll_defaults(self.session.repo, roll_id, **self._card_values(self.state.config, card_key))
            rolls.set_frame_override(self.session.repo, roll_id, rolls.unforked_hash(active_hash), card_key, False)

        touched = set(pushed)
        if sweep:
            for card_key in self._ROLL_CARDS:
                if self._reclaim_locked_frames(card_key, roll_id):
                    touched.add(card_key)

        if not touched:
            self.set_status(_NOTHING_TO_APPLY, 2500)
            return 0
        # Metadata cards do not reach the pixels. A frame locked on every other touched
        # card keeps its own values, so its thumbnail holds.
        rendered = {k for k in touched if rolls.ROLL_DEFAULT_FIELDS[k][0] != "metadata"}
        changed_hashes = []
        for f in self.state.uploaded_files:
            file_hash = f.get("hash")
            if file_hash == active_hash:
                continue
            if rendered <= rolls.frame_override_cards(self.session.repo, roll_id, rolls.unforked_hash(file_hash)):
                continue
            self.state.stale_thumbnails.add(asset_thumbnail_key(f))
            changed_hashes.append(file_hash)
        self.session.asset_model.refresh()
        self.config_updated.emit()
        if changed_hashes:
            self.session.frames_edited_offscreen.emit(changed_hashes)
        names = ", ".join(dict.fromkeys(self._ROLL_CARD_LABELS[k] for k in self._ROLL_CARDS if k in touched))
        self.set_status(f"Applied to the roll: {names}", 3000)
        return len(touched)

    def set_card_scope(self, card_key: Union[str, tuple], scope: str) -> None:
        """A Roll-tab card's scope pair: Roll gives the roll this frame's value for that
        card, Frame pins the card here. The one click each button performs. A tuple is
        one section driving several cards (Optics), pushed in one go."""
        keys = (card_key,) if isinstance(card_key, str) else card_key
        if scope == "roll":
            self._push_cards_to_roll([k for k in keys if self.roll_card_locked(k)], sweep=False)
        else:
            for key in keys:
                self.set_roll_card_locked(key, True)

    def sync_metadata_card_locks(self) -> None:
        """Re-reads every Metadata card's lock after a write that touched several at once.
        The Metadata panel commits its fields in one go rather than control by control, so
        no single edit knows which cards it moved; _lock_roll_card settles each either way."""
        for card_key in self.METADATA_CARDS:
            self._lock_roll_card(card_key)

    def frame_section_scope(self, section_key: str) -> str:
        """Where a frame-level card's values live: "roll" once a whole-roll apply put them
        there and this frame still matches every field it pushed, "frame" otherwise. Edit
        one of those fields and it reads Frame again on its own, with nothing to clear."""
        return self.frame_section_scopes((section_key,))[section_key]

    def frame_section_scopes(self, section_keys: tuple) -> Dict[str, str]:
        """frame_section_scope for many cards, off one read of the roll."""
        roll_id = self.state.active_roll_id
        entry = rolls.roll_for_id(self.session.repo, roll_id) if roll_id is not None else None
        pushes = entry.get("section_pushes", {}) if entry else {}
        sections = section_of_field()
        scopes = {}
        for key in section_keys:
            pushed = pushes.get(key)
            matches = bool(pushed) and all(
                getattr(getattr(self.state.config, sections[f], None), f, None) == v for f, v in pushed.items() if f in sections
            )
            scopes[key] = "roll" if matches else "frame"
        return scopes

    def record_section_push(self, section_key: str, values: dict) -> None:
        """Files a whole-roll apply of a frame-level card, so its scope pair can read Roll
        until the frame drifts off it again."""
        if self.state.active_roll_id is not None and values:
            rolls.set_section_push(self.session.repo, self.state.active_roll_id, section_key, values)
            self.config_updated.emit()

    def set_roll_card_locked(self, card_key: str, locked: bool) -> None:
        """Lock or unlock one Roll-tab card for the active frame, within the active
        roll. Locking seeds the frame's own saved row with whatever is currently in
        effect (the roll's default, most likely), so nothing appears to jump the
        moment it stops following the roll; unlocking drops the flag and the roll's
        current value takes over immediately. No-op with no active roll."""
        roll_id = self.state.active_roll_id
        if roll_id is None or not self.state.current_file_hash:
            return
        if locked:
            frozen = self._card_values(self.state.config, card_key)
            self.session.update_config(self._with_card_values(self.state.config, card_key, frozen), persist=True, render=False)
        rolls.set_frame_override(self.session.repo, roll_id, rolls.unforked_hash(self.state.current_file_hash), card_key, locked)
        if not locked:
            asset = self.state.uploaded_files[self.state.selected_file_idx]
            self.apply_config(self.session.config_for_asset(asset), persist=False)

    def reanalyze_current_file(self) -> None:
        """
        Clears cached local floors and forces a fresh analysis render.
        """
        new_process = replace(
            self.state.config.process,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.session.update_config(replace(self.state.config, process=new_process))
        self.request_render()

    def set_active_flatfield_profile(self, profile_id: str) -> None:
        """
        Selects the rig-global flat-field reference profile (or clears it when
        ``profile_id`` is empty), so a roll that names none of its own picks it up, and
        edits the Flat Field card on the current frame like any other roll card.
        """
        from negpy.services.assets.flatfield import FlatFieldProfiles

        self.session.repo.save_global_setting("flatfield_active_profile", profile_id or "")
        prof = FlatFieldProfiles.get(profile_id) if profile_id else None
        pid = prof.id if prof else ""
        self.set_roll_default("flatfield", profile_id=pid, apply=bool(pid))

    def save_flatfield_profile(self, name: str, path: str) -> None:
        """
        Bakes a reference image into a named flat-field profile and makes it active.
        """
        from negpy.services.assets.flatfield import FlatFieldProfiles

        profile_id = FlatFieldProfiles.create(name, path)
        if profile_id is None:
            self.set_status("Flat Field: could not read that reference image", 3000, kind="error")
            return
        self.set_active_flatfield_profile(profile_id)
        self.set_status(f"Flat Field profile '{name}' saved", 2000)

    def delete_flatfield_profile(self, profile_id: str) -> None:
        """
        Removes a flat-field profile; clears the active correction if it was selected.
        """
        from negpy.features.flatfield.logic import invalidate_gain
        from negpy.services.assets.flatfield import FlatFieldProfiles

        if not profile_id:
            return
        FlatFieldProfiles.delete(profile_id)
        invalidate_gain(profile_id)
        if self.session.repo.get_global_setting("flatfield_active_profile") == profile_id:
            self.set_active_flatfield_profile("")

    def load_gear_library(self):
        from negpy.services.assets.gear import GearProfiles

        return GearProfiles.load_library()

    def save_gear_library(self, library) -> None:
        from negpy.services.assets.gear import GearProfiles

        GearProfiles.save_library(library)

    def set_flatfield_enabled(self, enabled: bool) -> None:
        """
        Toggles flat-field correction on the Flat Field card, locking it away from the
        roll the instant it diverges, like any other roll card.
        """
        self.set_roll_default("flatfield", apply=enabled)

    # ── Scanner integration ───────────────────────────────────────────

    def request_scan_devices(self) -> None:
        """Request device enumeration on the scan worker thread."""
        self.scan_devices_requested.emit()

    def set_scan_backend(self, backend_id: str) -> None:
        """Route the chosen scanner backend to the worker thread."""
        self.scan_backend_requested.emit(backend_id)

    def start_scan(self, req: ScanRequest) -> None:
        """Start a scan. The UI connects to scan signals for state updates."""
        self.scan_worker.prepare_scan()
        self.scan_started.emit()
        self.scan_requested.emit(req)

    def start_batch(self, req: BatchRequest) -> None:
        """Start a frame-range batch scan over a roll/strip feeder."""
        self.scan_worker.prepare_scan()
        self.scan_started.emit()
        self.scan_batch_requested.emit(req)

    def start_roll_preview(self, req: RollPreviewRequest) -> None:
        """Preview strip slots (results via scan_roll_preview_ready, then
        scan_roll_preview_finished). No scan_started — preview is dialog-local."""
        self.scan_worker.prepare_scan()
        self.scan_roll_preview_requested.emit(req)

    def start_prescan(self, req: PrescanRequest) -> None:
        """Low-DPI full-window colour preview for crop setup (dialog-local)."""
        self.scan_worker.prepare_scan()
        self.scan_prescan_requested.emit(req)

    def eject_scanner(self, device_id: str) -> None:
        """Trigger the scanner's eject action on the worker thread."""
        self.scan_eject_requested.emit(device_id)

    def cancel_scan(self) -> None:
        self.scan_worker.cancel()

    def _on_scan_finished(self, path: str) -> None:
        """Auto-add scanned file to NegPy file list and select it."""
        self.scan_finished.emit(path)
        self._pending_scanned_file = path
        self.request_asset_discovery([path])

    def _on_scan_batch_finished(self, paths: list) -> None:
        """Import every frame a batch completed, including a stopped or failed run."""
        self.scan_batch_finished.emit(paths)
        if paths:
            self._pending_scanned_file = paths[-1]
            self.request_asset_discovery(list(paths))

    # ── Stitch (multi-part scan composite) ─────────────────────────────

    def request_stitch_selected(self) -> None:
        """Register the selected frames into one stitched composite asset."""
        if self._batch_busy("Stitch"):
            return
        files = [self.state.uploaded_files[i] for i in sorted(set(self.state.selected_indices)) if 0 <= i < len(self.state.uploaded_files)]
        by_path = {f["path"]: f for f in files}  # half-frame assets share a path
        ordered = sorted(by_path.values(), key=lambda f: os.path.basename(f["path"]).lower())
        if len(ordered) < 2:
            self.set_status("Select two or more frames to stitch", 4000)
            return
        if any(f.get("stitch_paths") for f in ordered):
            self.set_status("Stitching an already-stitched frame is not supported", 4000)
            return
        if self._begin_batch("stitch", "Stitching frames", abortable=True) is None:
            return
        self.stitch_requested.emit(
            StitchTask(
                files=tuple(dict(f) for f in ordered),
                params_by_path={f["path"]: self._batch_params_for(f) for f in ordered},
            )
        )

    def _on_stitch_registered(self, payload: dict) -> None:
        self._end_batch("stitch")
        files = payload["files"]
        part_paths = [f["path"] for f in files]
        triplets = tuple((f.get("green_path") or "", f.get("blue_path") or "") for f in files)
        composite = {
            "name": stitch_name(part_paths),
            "path": part_paths[0],
            "hash": stitch_hash([f["hash"] for f in files]),
            "stitch_paths": tuple(part_paths[1:]),
            "stitch_transforms": payload["transforms"],
            "stitch_canvas": payload["canvas"],
            "stitch_sizes": payload["sizes"],
            "stitch_triplets": triplets,
            "stitch_align": bool(files[0].get("align", True)),
            # Same inheritance as a merge: a composite's fresh hash would otherwise take
            # the stale sticky mode rather than the parts' own.
            "process_mode": self._composite_process_mode(files),
        }
        if all(triplets[0]):
            # Thumbnail decode and the sensor-unmix skip read the primary's pair from here.
            composite.update(green_path=triplets[0][0], blue_path=triplets[0][1], align=composite["stitch_align"])
        wanted = set(part_paths)
        indices = [i for i, f in enumerate(self.state.uploaded_files) if f["path"] in wanted]
        self.session.apply_composite(indices, composite)
        self.set_status(f"Stitched {count_of(len(files), 'frame')}", 4000)
        # The composite bypasses asset discovery, so nothing else queues its thumbnail.
        self.generate_missing_thumbnails()

    def _on_stitch_cancelled(self) -> None:
        self._on_batch_cancelled("stitch")

    def _on_stitch_error(self, message: str) -> None:
        self._end_batch("stitch")
        self.set_status(message, 6000, kind="error")

    def request_unstitch(self) -> None:
        """Dissolve the active stitched composite back into its part frames.

        Part edits restore from the DB by content hash; the composite's edits stay
        keyed under its stitch hash for a future re-stitch of the same parts."""
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        parts = asset.get("stitch_paths")
        if not parts:
            return
        paths = [asset["path"], *parts]
        # Triplet parts must come back as triplet assets, not as loose exposures.
        align = bool(asset.get("stitch_align", True))
        triplets = {path: [green, blue, align] for path, (green, blue) in zip(paths, asset.get("stitch_triplets") or ()) if green and blue}
        for green, blue, _ in triplets.values():
            paths.extend((green, blue))
        self.state.uploaded_files.pop(idx)
        key = asset_thumbnail_key(asset)
        self.session.state.thumbnails.pop(key, None)
        self.session.state.rendered_thumbnails.discard(key)
        self.session.asset_model.refresh()
        forget_composite(self.session.repo, asset["path"])
        self._pending_scanned_file = paths[0]
        self.request_asset_discovery(paths, restore_triplets=triplets or None)

    def _composite_process_mode(self, files: list) -> str:
        """The film process a composite should inherit from its source frames.

        The most common mode among them, ties going to the first (the reference frame /
        primary part). Majority rather than just the primary's: a bracket's extreme
        exposures can autodetect differently — the frame that blows 46% of its area is
        not a reliable vote — while the frames of one physical slide always agree in fact.
        """
        modes = [str(self.session.config_for_asset(f).process.process_mode) for f in files]
        if not modes:
            return str(self.state.config.process.process_mode)
        counts = Counter(modes)
        top = max(counts.values())
        return next(m for m in modes if counts[m] == top)

    # ── HDR (bracketed-exposure merge) ─────────────────────────────────

    def request_hdr_merge_selected(self) -> None:
        """Solve the selected frames into one merged bracket asset."""
        if self._batch_busy("HDR merge"):
            return
        files = [self.state.uploaded_files[i] for i in sorted(set(self.state.selected_indices)) if 0 <= i < len(self.state.uploaded_files)]
        by_path = {f["path"]: f for f in files}  # half-frame assets share a path
        ordered = sorted(by_path.values(), key=lambda f: os.path.basename(f["path"]).lower())
        if len(ordered) < 2:
            self.set_status("Select two or more exposures of the same frame to merge", 4000)
            return
        if any(f.get("hdr_paths") for f in ordered):
            self.set_status("Merging an already-merged frame is not supported", 4000)
            return
        # Both are multi-file source assembly and an asset carries one primary path. The
        # composition order is definable but not wired, so refuse instead of guessing.
        if any(f.get("stitch_paths") for f in ordered):
            self.set_status("HDR merge of a stitched frame is not supported", 4000)
            return
        if any(f.get("green_path") for f in ordered):
            self.set_status("HDR merge of a Trichrome triplet is not supported", 4000)
            return
        # Halves share a path, so by_path already dropped one of each pair and merging them
        # would produce a whole-frame composite. Every other assembly leaves half-frame
        # assets whole for the same reason (see _expand_half_frames).
        if any(f.get("half") for f in ordered):
            self.set_status("HDR merge of a half-frame asset is not supported", 4000)
            return
        if self._begin_batch("hdr", "Merging exposures", abortable=True) is None:
            return
        self.hdr_requested.emit(
            HdrTask(
                files=tuple(dict(f) for f in ordered),
                params_by_path={f["path"]: self._batch_params_for(f) for f in ordered},
            )
        )

    def _on_hdr_solved(self, payload: dict) -> None:
        self._end_batch("hdr")
        files = payload["files"]
        reference = payload["reference"]
        # The reference frame becomes the composite's primary. It is the asset's own path
        # everywhere downstream, and the merge expresses radiance in its units.
        ordered = [files[reference], *[f for i, f in enumerate(files) if i != reference]]
        ratios = payload["ratios"]
        ordered_ratios = (ratios[reference], *[r for i, r in enumerate(ratios) if i != reference])
        frame_paths = [f["path"] for f in ordered]
        composite = {
            "name": hdr_name(frame_paths),
            "path": frame_paths[0],
            "hash": hdr_hash([f["hash"] for f in ordered]),
            "hdr_paths": tuple(frame_paths[1:]),
            "hdr_ratios": tuple(float(r) for r in ordered_ratios),
            "hdr_align": True,
            "hdr_anchor": "",  # bracket middle until the user nominates an exposure
            "hdr_anchor_ev": ANCHOR_EV_UNSET,
            "process_mode": self._composite_process_mode(ordered),
        }
        wanted = set(frame_paths)
        indices = [i for i, f in enumerate(self.state.uploaded_files) if f["path"] in wanted]
        self.session.apply_composite(indices, composite)
        stops = math.log2(max(ratios) / min(ratios)) if min(ratios) > 0 else 0.0
        self.set_status(f"Merged {len(files)} exposures spanning {stops:.1f} stops", 4000)
        # The composite bypasses asset discovery, so nothing else queues its thumbnail.
        self.generate_missing_thumbnails()

    def _on_hdr_cancelled(self) -> None:
        self._on_batch_cancelled("hdr")

    def _on_hdr_error(self, message: str) -> None:
        self._end_batch("hdr")
        self.set_status(message, 6000, kind="error")

    def apply_config(self, config: WorkspaceConfig, persist: bool = False, readback_metrics: bool = True) -> None:
        """Adopt `config` and repaint by whichever route the change actually needs.

        A change to a *source* input — a bracket, a triplet, a stitch, Linear RAW — cannot
        be honoured by re-running the pipeline: assembly happens while the source is
        decoded, so the buffer the pipeline starts from is already the wrong one. Compare
        `source_token` and re-decode when it moves.

        The alternative is every such control remembering to reload for itself, which is
        how the HDR render exposure shipped writing a value that never reached the canvas.
        """
        needs_decode = source_token(config) != source_token(self.state.config)
        # render=False on the decode branch: state_changed would analyse bounds against
        # the stale pre-reload buffer.
        self.session.update_config(config, persist=persist, render=not needs_decode)
        if needs_decode and self.state.current_file_path:
            self.load_file(self.state.current_file_path, preserve_zoom=True)
        else:
            self.request_render(readback_metrics=readback_metrics)

    def set_hdr_anchor(self, path: str) -> None:
        """Render the active merge at `path`'s exposure ("" = the bracket's middle).

        Which frame looks right is intent, not a measurement: the exposure *reference* is
        the longest frame that does not clip, which on a slide is brighter than the capture
        the photographer metered — a slide's own brightest point is denser than clear film.
        Stored on the asset, like the rest of the bracket, so it survives re-hydration.
        """
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        if not asset.get("hdr_paths") or str(asset.get("hdr_anchor", "") or "") == path:
            return
        asset["hdr_anchor"] = path
        asset["hdr_anchor_ev"] = ANCHOR_EV_UNSET  # a named frame supersedes a value
        # The asset dict is authoritative for the bracket, and only the manifest carries it
        # across a restart.
        self.session.persist_session()
        cfg = self.state.config
        self.set_status(f"Rendering the merge as {os.path.basename(path)}" if path else "Rendering the merge at the bracket middle", 4000)
        # apply_config re-decodes. The bracket is merged while the source is decoded, so the
        # scale lives in the buffer the pipeline starts from and a render alone would change
        # nothing.
        self.apply_config(replace(cfg, hdr=replace(cfg.hdr, hdr_anchor=path, hdr_anchor_ev=ANCHOR_EV_UNSET)))

    def set_hdr_anchor_ev(self, ev: float, persist: bool = True) -> None:
        """Render the active merge at `ev` stops below the reference, continuously.

        The menu can only offer exposures the bracket contains, so the render is otherwise
        quantised to the frames that happen to have been shot — and the one that looks right
        is rarely one of them exactly. Setting a value takes precedence over a named frame;
        `ANCHOR_EV_UNSET` hands it back to the menu.
        """
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        if not asset.get("hdr_paths"):
            return
        asset["hdr_anchor_ev"] = float(ev)
        if float(ev) < ANCHOR_EV_UNSET:
            # A value and a frame are two answers to one question, so keep only the live one
            # and the menu's tick cannot disagree with the slider.
            asset["hdr_anchor"] = ""
        self.session.persist_session()
        cfg = self.state.config
        hdr = replace(cfg.hdr, hdr_anchor_ev=float(ev), hdr_anchor="" if float(ev) < ANCHOR_EV_UNSET else cfg.hdr.hdr_anchor)
        # apply_config re-decodes: the scale is applied while the bracket is merged, so a
        # re-render alone would run the pipeline over an already-scaled buffer.
        self.apply_config(replace(cfg, hdr=hdr), persist=persist)

    def request_unmerge_hdr(self) -> None:
        """Dissolve the active merged frame back into its exposures.

        Frame edits restore from the DB by content hash; the composite's edits stay keyed
        under its HDR hash for a future re-merge of the same bracket."""
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        frames = asset.get("hdr_paths")
        if not frames:
            return
        paths = [asset["path"], *frames]
        self.state.uploaded_files.pop(idx)
        key = asset_thumbnail_key(asset)
        self.session.state.thumbnails.pop(key, None)
        self.session.state.rendered_thumbnails.discard(key)
        self.session.asset_model.refresh()
        forget_composite(self.session.repo, asset["path"])
        self._pending_scanned_file = paths[0]
        self.request_asset_discovery(paths)

    def request_undiptych(self) -> None:
        """Turn the active diptych back into one plain scan, deleting both halves' edits.

        The scan leaves the split-scan set, so it stays a plain frame until it is split
        again. Exported ``.negpy`` half sidecars are left alone.
        """
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        file_hash = asset.get("hash") or ""
        if not asset.get("diptych") or not file_hash:
            return
        forget_split_scan(self.session.repo, file_hash)
        for n in (1, 2):
            half = half_hash(file_hash, n)
            self.session.repo.delete_file_settings(half)
            self._measured_half_rows.discard(half)
        asset["diptych"] = False
        self._active_diptych_memo = ("", None)
        self.session.asset_model.refresh()
        if file_hash == self.state.current_file_hash and asset.get("path"):
            self.load_file(asset["path"])
        self.set_status("Diptych unsplit — the halves' edits are deleted", 4000)

    def request_fork_edit_for_roll(self) -> None:
        """Give the active frame its own edit under the active roll, seeded from the
        shared edit as it stands right now, including any unsaved change on this frame.
        """
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        roll_id, path, from_hash = self.state.active_roll_id, asset.get("path"), asset.get("hash")
        if not roll_id or not path or not from_hash:
            return
        is_active = from_hash == self.state.current_file_hash
        seed = self.state.config if is_active else self.session.config_for_asset(asset)
        asset["hash"] = rolls.fork_edit(self.session.repo, roll_id, from_hash, path, seed)
        self.session.asset_model.refresh()
        if is_active:
            self.load_file(path)
        self.set_status("This roll now has its own edit for this frame", 3000)

    def request_unfork_edit_for_roll(self) -> None:
        """Undo `request_fork_edit_for_roll`: delete the active frame's roll-specific
        edit and go back to the shared one."""
        idx = self.state.selected_file_idx
        if not (0 <= idx < len(self.state.uploaded_files)):
            return
        asset = self.state.uploaded_files[idx]
        roll_id, path = self.state.active_roll_id, asset.get("path")
        forked_hash = asset.get("hash") or ""
        from_hash = rolls.unforked_hash(forked_hash)
        if not roll_id or not path or from_hash == forked_hash:
            return
        rolls.unfork_edit(self.session.repo, roll_id, from_hash)
        asset["hash"] = from_hash
        self.session.asset_model.refresh()
        if forked_hash == self.state.current_file_hash:
            self.load_file(path)
        self.set_status("Reverted to this roll's shared edit", 3000)

    def _select_file_by_path(self, path: str) -> bool:
        """Find a file by path in uploaded_files and select it."""
        for i, f_info in enumerate(self.session.state.uploaded_files):
            if f_info.get("path") == path:
                self.session.select_file(i)
                return True
        return False

    # ── Scanlight capture integration ─────────────────────────────────

    def _ensure_capture_thread(self) -> None:
        """Start the capture worker's thread on first use (lazy). Every capture entry point that
        emits to the worker calls this first, so the thread is running when the queued cross-thread
        signal is delivered. The live-view sub-controls and cancel skip it: they only run once a
        session is already up (started here) or touch the worker's thread-safe cancel Event."""
        if not self._capture_thread_started:
            self.capture_thread.start()
            self._capture_thread_started = True

    def set_scanlight_color(self, r: int, g: int, b: int, w: int = 0, port: str = "") -> None:
        """Live light control (no capture): RGB for preview, or white (w) for focus."""
        self._ensure_capture_thread()
        self.capture_light_requested.emit(r, g, b, w, port)

    def start_capture(self, req: CaptureRequest) -> None:
        """Start a capture; the Scanlight sidebar tracks state via signals."""
        self._ensure_capture_thread()
        self._last_capture_req = req
        self.capture_requested.emit(req)

    def cancel_capture(self) -> None:
        self.capture_worker.cancel()

    def start_live_view(self, req: LiveViewRequest) -> None:
        self._ensure_capture_thread()
        self.live_view_requested.emit(req)

    def stop_live_view(self) -> None:
        self.live_view_stop_requested.emit()

    def close_camera_session(self) -> None:
        """Release the held PTP session. Call once neither the scan window nor the
        preset-calibration pop-up is open — some bodies (Fuji) get stuck in a
        tethered-capture state until the session is cleanly exited, and leaving it
        open past the last consuming window makes the next connection attempt hang."""
        if self._capture_thread_started:
            self.camera_session_close_requested.emit()

    def set_focus_magnifier(self, on: bool) -> None:
        self.live_view_focus_magnifier_requested.emit(on)

    def set_focus_magnifier_pos(self, x: int, y: int) -> None:
        self.live_view_focus_magnifier_pos_requested.emit(x, y)

    def set_camera_setting(self, which: str, raw: int) -> None:
        # Ensure the worker thread runs. The sidebar counts these writes and gates Scan until
        # each reports back, so a write queued to an unstarted thread gates forever.
        self._ensure_capture_thread()
        self.live_view_camera_setting_requested.emit(which, raw)

    def start_calibration(self, req: CalibrationRequest) -> None:
        self._ensure_capture_thread()
        self.calibration_requested.emit(req)

    def poll_connection(self, port: str) -> None:
        self._ensure_capture_thread()
        self.poll_connection_requested.emit(port)

    def poll_light_temp(self, port: str) -> None:
        self._ensure_capture_thread()
        self.poll_light_temp_requested.emit(port)

    def _on_capture_finished(self, paths: list) -> None:
        """Feed the captured frame(s) into NegPy. A 3-file RGB triplet → RGB-Scan negative
        (C-41) pipeline; a single white-light slide → E-6/positive; a normal white-light
        camera scan → an ordinary single RAW (RGB-Scan off, process left to NegPy)."""
        self.capture_finished.emit(paths)
        if not paths:
            return
        req = getattr(self, "_last_capture_req", None)
        white = bool(req is not None and req.white_mode)
        rgb = bool(req is not None and getattr(req, "rgb_mode", True))
        # RGB-Scan (triplet merge) is on only for an actual RGB triplet. Off for a single
        # white-light slide and for a normal camera scan.
        self.session.repo.save_global_setting("rgbscan_mode", rgb and not white)
        capture_roll = getattr(req, "roll_name", "") if req is not None else ""
        capture_frame = getattr(req, "frame_number", None) if req is not None else None
        if white:  # slides / B&W negatives force a positive process
            mode = WhiteCaptureMode(req.white_process_mode)
            target = {WhiteCaptureMode.E6: ProcessMode.E6, WhiteCaptureMode.BW: ProcessMode.BW}.get(mode)
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(
                process_mode=target,
                detect_mode=target is None,
                capture_roll=capture_roll,
                capture_frame=capture_frame,
            )
        elif rgb:
            # Independently exposed RGB channels carry no broadband orange-mask signal for
            # the normal classifier. They are negative scans unless capture metadata says
            # otherwise, so carry C-41 through discovery instead of guessing from the merge.
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(
                process_mode=ProcessMode.C41,
                capture_roll=capture_roll,
                capture_frame=capture_frame,
            )
        elif req is not None:
            self._pending_capture_imports[_capture_import_key(paths[0])] = _PendingCaptureImport(
                capture_roll=capture_roll,
                capture_frame=capture_frame,
            )
        self._pending_scanned_file = paths[0]
        # The capture shot these three exposures for one frame, in red/green/blue order
        # (CaptureResult.paths), so hand discovery the triplet instead of asking it to
        # re-derive one it already knows. Deriving it can only refuse a frame it should
        # have kept: a blank or untextured frame gives the content test nothing to match.
        triplet = {paths[0]: [paths[1], paths[2]]} if rgb and not white and len(paths) == 3 else None
        self.request_asset_discovery(list(paths), restore_triplets=triplet)

    def effective_output_icc(self) -> Optional[str]:
        """Profile the *export* converts to and tags with: a custom override, else the
        profile for the selected export color space."""
        return self.state.icc_output_path or ColorSpaceRegistry.get_icc_path(self.state.config.export.export_color_space)

    def effective_proof_icc(self) -> Optional[str]:
        """Profile the preview proofs through. Falls back to the export target, so the
        proof answers "what will the file look like" until a print is named explicitly."""
        return self.state.proof_icc_path or self.effective_output_icc()

    def effective_input_icc(self, process: Optional[ProcessConfig] = None) -> Optional[str]:
        """Source profile for color management: an explicit Input ICC wins; else the
        bundled RGBScan profile when Narrowband Scan applies; else None.

        A transparency never takes the implicit profile — see narrowband_profile_active,
        which owns that rule. An explicit Input ICC still wins there, being a deliberate
        user choice about their own source.
        """
        p = process if process is not None else self.state.config.process
        if self.state.icc_input_path:
            return self.state.icc_input_path
        if narrowband_profile_active(p):
            return get_resource_path("icc/RGBScan.icc")
        return None

    def _effective_cam_xyz(self) -> tuple[Optional[list], Optional[list]]:
        """(cam_xyz, camera_wb) for the transparency transfer. With an Input ICC active,
        `cam_xyz` is stood in for: the decode still needs the white-balance fold, but the
        camera's own primaries rotation would double up on the ICC's, see wb_only_cam_xyz."""
        cam_xyz = self.state.preview_cam_xyz
        if self.effective_input_icc():
            cam_xyz = wb_only_cam_xyz(cam_xyz)
        return cam_xyz, self.state.preview_camera_wb

    def display_transform_params(
        self, splash: bool = False, proofed: bool = True, process: Optional[ProcessConfig] = None
    ) -> tuple[str, Optional[bytes], Optional[tuple]]:
        """Everything the display transform needs for the current render, as
        ``(color_space, monitor_icc_bytes, proof)``.

        Single source of truth for every consumer of a rendered buffer — the canvas
        shader, the CPU overlay and the filmstrip thumbnail must agree, or the same
        frame shows two different colors. Renders arrive in the working space; a
        proof is *not* baked into them, it is folded into the display LUT here (see
        ``get_display_lut``), which is what lets a GPU texture go to the shader
        untouched. ``splash`` marks the embedded camera thumbnail, already sRGB.
        ``proofed`` is False for a working-space buffer that is not a print: the
        negative peek shows the scan, which a paper simulation would misdescribe.
        ``process`` overrides the active frame's for a background render of another
        frame, whose narrowband-profile state can differ.
        """
        if splash:
            return ColorSpace.SRGB.value, self.state.monitor_icc_bytes, None
        proof = self.proof_profiles(process) if proofed else None
        return self.state.workspace_color_space, self.state.monitor_icc_bytes, proof

    def proof_profiles(self, process: Optional[ProcessConfig] = None) -> Optional[ProofCondition]:
        """The `ProofCondition` the preview simulates, or None when off.

        Narrowband Scan supplies an implicit *input* profile whether or not the
        soft-proof toggle is on; the output profile only applies with the toggle.
        """
        p = process if process is not None else self.state.config.process
        proofing = self.state.soft_proof_enabled
        if not (proofing or narrowband_profile_active(p)):
            return None
        icc_input = self.effective_input_icc(p)
        icc_output = self.effective_proof_icc() if proofing else None
        if not (icc_input or icc_output):
            return None
        st = self.state
        return ProofCondition(
            icc_input,
            icc_output,
            st.proof_intent,
            st.proof_black_point,
            st.proof_paper_white,
            st.proof_ink_black,
            # A warning drawn into the display LUT would describe a gamut nothing is being
            # proofed to when the output leg is absent.
            st.proof_gamut_warning and bool(icc_output),
        )

    def set_proof_field(self, name: str, value: Any) -> None:
        """Set one proof-condition field, persist it and re-render.

        One setter for all of them: each is an independent app-level preference that
        reaches the render only by changing the display LUT's cache key.
        """
        if getattr(self.state, name) == value:
            return
        setattr(self.state, name, value)
        self.session.save_icc_prefs()
        self.request_render()

    def save_proof_condition(self, name: str) -> None:
        """Store the current proof settings under `name`, replacing a condition of the
        same name."""
        entry = {
            "name": name,
            "icc": self.state.proof_icc_path,
            "intent": self.state.proof_intent,
            "black_point": self.state.proof_black_point,
            "paper_white": self.state.proof_paper_white,
            "ink_black": self.state.proof_ink_black,
        }
        others = [c for c in self.state.proof_conditions if c.get("name") != name]
        self.state.proof_conditions = sorted([*others, entry], key=lambda c: c["name"].lower())
        self.session.save_icc_prefs()
        self.config_updated.emit()

    def delete_proof_condition(self, name: str) -> None:
        self.state.proof_conditions = [c for c in self.state.proof_conditions if c.get("name") != name]
        self.session.save_icc_prefs()
        self.config_updated.emit()

    def reset_proof_condition(self) -> None:
        """The None preset: proof the export target, simulate nothing.

        Not "proofing off": the proof still runs, it just shows the export's own gamut
        rather than a sheet of paper's limits. Every named preset is a departure from this,
        and it is the state a frame should be judged in before a paper is chosen.
        """
        st = self.state
        st.soft_proof_enabled = True
        st.proof_icc_path = None
        # The intent is part of the baseline too, or None would not match itself and the
        # preset box would go blank the moment you picked it.
        st.proof_intent = ProofIntent.RELATIVE_COLORIMETRIC.value
        st.proof_black_point = False
        st.proof_paper_white = False
        st.proof_ink_black = False
        st.proof_gamut_warning = False
        self.session.save_icc_prefs()
        self.config_updated.emit()
        self.request_render()

    def apply_proof_condition(self, name: str) -> None:
        """Load a saved preset. The gamut warning is deliberately not part of one: it is
        a way of looking at the frame, not a property of the printer and paper."""
        entry = next((c for c in self.state.proof_conditions if c.get("name") == name), None)
        if entry is None:
            return
        st = self.state
        icc = entry.get("icc")
        st.proof_icc_path = icc if icc and os.path.exists(icc) else None
        st.proof_intent = entry.get("intent") if entry.get("intent") in PROOF_INTENT_LABELS else ProofIntent.RELATIVE_COLORIMETRIC.value
        st.proof_black_point = bool(entry.get("black_point", False))
        st.proof_paper_white = bool(entry.get("paper_white", False))
        st.proof_ink_black = bool(entry.get("ink_black", False))
        self.session.save_icc_prefs()
        self.config_updated.emit()
        self.request_render()

    def proof_active(self) -> bool:
        """True when the preview should soft-proof: the toggle is on and an input or
        output profile is available, or Narrowband Scan supplies an implicit input
        profile. Off → preview is the edit on the monitor."""
        if self.effective_input_icc() and narrowband_profile_active(self.state.config.process):
            return True
        return self.state.soft_proof_enabled and bool(self.state.icc_input_path or self.effective_proof_icc())

    def set_soft_proof(self, enabled: bool) -> None:
        """Toggle preview soft-proofing through the Output/Input ICC (preview only)."""
        if self.state.soft_proof_enabled == enabled:
            return
        self.state.soft_proof_enabled = enabled
        self.session.save_icc_prefs()
        self.config_updated.emit()
        self.request_render()

    def _apply_monitor_profile(self) -> None:
        """Resolve the effective display profile (override else detected), push it to
        every preview path, and re-render. Display-only; export is unaffected."""
        from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space

        override = self.state.monitor_profile_override
        effective = icc_bytes_for_space(override) if override else self.state.monitor_icc_detected_bytes
        self.state.monitor_icc_bytes = effective
        if self.canvas is not None:
            self.canvas.set_monitor_profile(effective)
        self.request_render()
        self.monitor_profile_changed.emit()

    def set_monitor_detected(self, detected_bytes: Optional[bytes]) -> None:
        """Record the auto-detected screen profile and re-resolve the effective one."""
        self.state.monitor_icc_detected_bytes = detected_bytes
        self._apply_monitor_profile()

    def set_monitor_override(self, cs_name: Optional[str]) -> None:
        """Set the manual display-profile override (None = use detected) and persist it."""
        self.state.monitor_profile_override = cs_name
        self.session.save_icc_prefs()
        self._apply_monitor_profile()

    def request_render(
        self,
        readback_metrics: bool = True,
        config_override: Optional[WorkspaceConfig] = None,
        ephemeral: bool = False,
        compare_capture: bool = False,
    ) -> None:
        """
        Dispatches a render task to the worker thread.
        Direct callers bypass the debounce; the timer is cancelled to avoid a duplicate.

        config_override renders an alternate config (e.g. the before/after baseline) without
        mutating session state; pass readback_metrics=False so it doesn't disturb
        histogram/bounds persistence.

        compare_capture marks the baseline render whose pixels are stashed for the
        before/after split instead of being displayed.
        """
        self._render_debounce.stop()
        lens_token = lens_decode_token(metadata_lens_corrections(self.state.config), self.state.config.flatfield)
        if not ephemeral and self.state.current_file_path and lens_token != self.state.preview_lens_token:
            self.load_file(self.state.current_file_path, preserve_zoom=True)
            return

        # Any direct render exits the flat preview-peek.
        if config_override is None and self.state.flat_peek:
            self.state.flat_peek = False
            self.flat_peek_changed.emit(False)
        if config_override is None and self.state.negative_peek:
            self.state.negative_peek = False
            self.negative_peek_changed.emit(False)
        if config_override is None and self.state.embedded_peek:
            self.state.embedded_peek = False
            self.embedded_peek_changed.emit(False)

        # The strip's patches were printed from the config as it stood, so once the edit
        # moves they prove something else. Drop them, which also cancels a strip still
        # building. Zone pins die the same way.
        if config_override is None:
            self._clear_test_strip()
            self._drop_zone_pins()

        if self.state.preview_raw is None:
            return

        preview_raw = self.state.preview_raw
        if preview_raw is None:
            return

        # A drag asks for no metrics, the release does. Interactive frames go through the
        # proxy, so full resolution arrives only once the gesture settles. The baseline
        # capture wants no metrics but full resolution: it is painted beside the edit, and a
        # proxy would show softer pixels on one side of the divider.
        interactive = not readback_metrics and not compare_capture
        ir_buffer = self.state.preview_ir
        detect_buffer = self.state.preview_detect
        crop_preview_full = self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW)
        if (interactive or crop_preview_full) and self.state.preview_proxy is not None:
            preview_raw = self.state.preview_proxy
            # The IR and detection planes must follow the image they are read against.
            ir_buffer = self.state.preview_ir_proxy
            detect_buffer = self.state.preview_detect_proxy

        target_size = float(APP_CONFIG.preview_render_size)
        if self.state.hq_preview:
            target_size = float(max(preview_raw.shape[:2]))

        # Only a plain render of the saved edit is reproducible on navigate-back. Overrides,
        # splash and tool previews are not memoized. Interactive frames are excluded: a proxy
        # render filed under the full-resolution key would be painted back as the real one.
        memo_key = ""
        if config_override is None and not ephemeral and not crop_preview_full and not interactive:
            memo_key = self._render_memo_key()
        self._expected_render_key = memo_key

        dip = self.active_diptych()
        cam_xyz, camera_wb = self._effective_cam_xyz()
        task = RenderTask(
            buffer=preview_raw,
            config=config_override if config_override is not None else self.state.config,
            source_hash=self.state.current_file_hash or "preview",
            preview_size=target_size,
            gpu_enabled=self.state.gpu_enabled,
            readback_metrics=readback_metrics,
            ir_buffer=ir_buffer,
            detect_buffer=detect_buffer,
            crop_preview_full=crop_preview_full,
            ephemeral=ephemeral,
            memo_key=memo_key,
            compare=compare_capture,
            interactive=interactive,
            # Mirrors should_update_thumb, minus its pending-task check.
            wants_thumbnail=(not interactive and not ephemeral and config_override is None and self.state.config is not self._thumb_config),
            cam_xyz=cam_xyz,
            camera_wb=camera_wb,
            diptych=dip[1] if dip is not None else None,
            split_x=dip[0]["split_x"] if dip is not None else 0.5,
            gutter_thickness=dip[0]["gutter_thickness"] if dip is not None else 0.0,
        )

        prefetch_was_running = self._cancel_neighbor_prefetch()

        if self._is_rendering:
            self._pending_render_task = task
            return

        if prefetch_was_running:
            self._pending_render_task = task
            return

        self._is_rendering = True
        self.render_requested.emit(task)

    def _baseline_compare_config(self) -> WorkspaceConfig:
        return baseline_compare_config(self.state.config)

    def _compare_before_key(self) -> str:
        """Identity of the stashed baseline frame. Creative edits leave it alone (they are
        reset in the baseline anyway); geometry, process and display changes invalidate it."""
        return self._render_memo_key(self._baseline_compare_config())

    def _request_compare_baseline(self) -> None:
        if self.state.preview_raw is None:
            return
        self.request_render(readback_metrics=False, config_override=self._baseline_compare_config(), compare_capture=True)

    def _capture_compare_before(self, metrics: Dict[str, Any]) -> None:
        """Keep the baseline render's pixels for the split. The GPU pool overwrites its
        textures on the next frame, so read back now rather than holding the texture."""
        buffer = metrics.get("base_positive")
        if isinstance(buffer, GPUTexture):
            try:
                readback = buffer.readback()
            except Exception:
                logger.exception("Failed to read back the before/after baseline frame")
                return
            buffer = np.ascontiguousarray(readback[:, :, :3]) if readback.ndim == 3 and readback.shape[2] >= 3 else readback
        if not isinstance(buffer, np.ndarray):
            return
        self.state.compare_before = buffer
        self.state.compare_before_rect = metrics.get("content_rect")
        self.state.compare_before_key = self._compare_before_key()
        self.compare_frame_ready.emit()

    def exit_compare(self) -> None:
        """Leave the before/after split and drop the stashed baseline frame."""
        self.state.compare_before = None
        self.state.compare_before_rect = None
        self.state.compare_before_key = ""
        if self.state.compare_mode:
            self.state.compare_mode = False
            self.compare_changed.emit(False)

    def toggle_compare(self) -> None:
        """Toggle the before/after split between the edit and the auto baseline."""
        if self.state.preview_raw is None:
            return
        if self.state.compare_mode:
            self.exit_compare()
        else:
            # Mutually exclusive with flat-peek: drop it so its toggle cannot stay lit while
            # the compare baseline is on screen. toggle_flat_peek exits compare the same way.
            if self.state.flat_peek:
                self.state.flat_peek = False
                self.flat_peek_changed.emit(False)
            if self.state.negative_peek:
                self.state.negative_peek = False
                self.negative_peek_changed.emit(False)
            if self.state.embedded_peek:
                self.state.embedded_peek = False
                self.embedded_peek_changed.emit(False)
            # Same reason the strip and the peek are exclusive: both want the canvas.
            self._clear_test_strip()
            self.state.compare_mode = True
            self.compare_changed.emit(True)
            # The edit is already on screen; only the baseline half has to be rendered.
            self._request_compare_baseline()

    def rerender_active_view(self) -> None:
        """Re-render the canvas keeping whatever comparison overlay is active.

        Geometry ops (rotate/flip) change the config but shouldn't kick the user
        out of flat-peek; a plain request_render() would exit it. The compare split
        survives a plain render, and its baseline half re-captures on the key change.
        """
        if self.state.negative_peek:
            self._paint_negative_peek()
        elif self.state.embedded_peek:
            self._paint_embedded_peek()
        elif self.state.flat_peek:
            self.request_render(readback_metrics=False, config_override=flat_master_config(self.state.config))
        else:
            self.request_render()

    # --- Flat ("for editing elsewhere") master output -----------------------

    def set_flat_output(self, enabled: bool) -> None:
        """Toggle the flat digital-intermediate output intent (export + peek)."""
        if self.state.flat_output == enabled:
            return
        self.state.flat_output = enabled
        if enabled:
            self.state.linear_output = False
        self.session.save_flat_output_prefs()
        # Flat masters default to full resolution; only honour Print/Pixels when the
        # user explicitly selects those modes in the export panel.
        if enabled and self.state.config.export.export_resolution_mode == ExportResolutionMode.PRINT.value:
            self.session.update_config(
                replace(
                    self.state.config,
                    export=replace(
                        self.state.config.export,
                        export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
                    ),
                ),
                persist=True,
            )
        self.flat_output_changed.emit(enabled)
        if enabled:
            self.linear_output_changed.emit(False)
        # If a peek is active and flat output was turned off, drop back to the edit.
        if not enabled and self.state.flat_peek:
            self.toggle_flat_peek(force=False)

    def set_linear_output(self, enabled: bool) -> None:
        """Toggle the linear output intent (raw loader dump, no pipeline)."""
        if self.state.linear_output == enabled:
            return
        self.state.linear_output = enabled
        if enabled:
            self.state.flat_output = False
            self.flat_output_changed.emit(False)
            if self.state.flat_peek:
                self.toggle_flat_peek(force=False)
        self.session.save_flat_output_prefs()
        self.linear_output_changed.emit(enabled)

    def toggle_flat_peek(self, force: Optional[bool] = None) -> None:
        """Preview the flat master render in the canvas without changing the saved edit.

        ``force`` sets an explicit state; otherwise toggles. Mutually exclusive with
        the before/after compare view.
        """
        if self.state.preview_raw is None:
            return
        target = (not self.state.flat_peek) if force is None else force
        if target == self.state.flat_peek:
            return

        if target:
            self.exit_compare()
            if self.state.negative_peek:
                self.state.negative_peek = False
                self.negative_peek_changed.emit(False)
            if self.state.embedded_peek:
                self.state.embedded_peek = False
                self.embedded_peek_changed.emit(False)
            self._clear_test_strip()

        self.state.flat_peek = target
        self.flat_peek_changed.emit(target)

        if target:
            self.request_render(readback_metrics=False, config_override=flat_master_config(self.state.config))
        else:
            self.request_render()

    def _paint_negative_peek(self) -> None:
        """Put the decoded source on the canvas: un-inverted, un-normalized, no tone edits.

        Geometry is the one thing the peek does apply, through the same two processors
        the base and crop stages use, so the negative sits at the orientation and
        framing the user set. Everything after it is skipped: no metering, no
        inversion, no look. Only the working OETF follows, or a linear buffer shows as
        near-black. ``content_rect`` is cleared because no border stage ran to inset the
        picture.

        The source is in camera primaries, so the camera matrix runs here: painting those
        numbers as display RGB flattens the film base, which on a C-41 negative reads as a
        mask far weaker than the one in the file. The multipliers fold into the matrix
        whenever the decode skipped them, narrowband included. That is wider than
        `should_fold_camera_wb`, which refuses narrowband because the light has no color
        temperature to reconstruct; this view only has to show the film the way a raw viewer
        does, and unbalanced sensor RGB renders the mask green. Reconstruction can override
        the decode to bake real white balance in instead (`highlight_reconstruction_bakes_wb`);
        folding it again here would double it, so the fold sits out whenever baking did.
        `lightbox_level` supplies the brightness a decode with no auto-brightness never gets.
        The proof stays off: this is the scan, not a print.
        """
        source = self.state.preview_raw
        if source is None:
            return
        geometry = self.state.config.geometry
        original = self.state.original_res if any(self.state.original_res) else source.shape[:2]
        context = PipelineContext(
            original_size=(original[0], original[1]),
            scale_factor=max(original) / float(APP_CONFIG.preview_render_size),
            process_mode=self.state.config.process.process_mode,
            # Mirrors request_render: the crop tool frames against the uncropped frame.
            crop_preview_full=self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW),
            wants_uv_grid=False,
        )
        img = GeometryProcessor(geometry).process(source, context)
        process, render_intent = self.state.config.process, self.state.config.exposure.render_intent
        decoded_without_wb = effective_linear_raw(process, render_intent) and not highlight_reconstruction_bakes_wb(process, render_intent)
        matrix = camera_to_working_matrix(
            self.state.preview_cam_xyz,
            self.state.preview_camera_wb if decoded_without_wb else None,
        )
        level = lightbox_level(img, matrix)
        if not context.crop_preview_full:
            img = CropProcessor(geometry).process(img, context)
        img = apply_camera_matrix(img, matrix)
        if level is not None:
            img = img * level
        with self.state.metrics_lock:
            self.state.last_metrics["base_positive"] = working_oetf_encode(img)
            self.state.last_metrics["content_rect"] = None
            self.state.last_metrics["splash"] = False
            self.state.last_metrics["proof"] = False
            # A prior interactive/peek render (e.g. Flat Peek, which renders with
            # readback_metrics=False) can leave this stale True, which makes
            # right_panel's _update_analysis mistake this settled frame for a
            # mid-gesture one and skip the histogram refresh.
            self.state.last_metrics["interactive"] = False
        self.image_updated.emit()

    def toggle_negative_peek(self, force: Optional[bool] = None) -> None:
        """Show the negative as it was loaded, without changing the saved edit.

        ``force`` sets an explicit state; otherwise toggles. Mutually exclusive with
        the before/after compare view and the flat peek.
        """
        if self.state.preview_raw is None:
            return
        target = (not self.state.negative_peek) if force is None else force
        if target == self.state.negative_peek:
            return

        if target:
            self.exit_compare()
            if self.state.flat_peek:
                self.state.flat_peek = False
                self.flat_peek_changed.emit(False)
            if self.state.embedded_peek:
                self.state.embedded_peek = False
                self.embedded_peek_changed.emit(False)
            self._clear_test_strip()

        self.state.negative_peek = target
        self.negative_peek_changed.emit(target)

        if target:
            self._paint_negative_peek()
        else:
            self.request_render()

    def _load_embedded_preview(self) -> Optional[Any]:
        """The file's own embedded preview, read once per frame and kept in state.

        Read on first peek rather than kept from the load: the splash is skipped on a
        preview-cache hit. The half-frame slice is the active asset's, so a diptych peeks
        the half being edited.
        """
        if self.state.preview_embedded is not None:
            return self.state.preview_embedded
        path = self.state.current_file_path
        if not path:
            return None
        result = PreviewManager.try_splash_preview(path, half_slice=self._active_half())
        if result is None:
            return None
        self.state.preview_embedded = result[0]
        return self.state.preview_embedded

    def _paint_embedded_peek(self) -> None:
        """Put the camera's own preview on the canvas, at the user's geometry.

        The JPEG the camera wrote, so it carries that camera's white balance, tone curve and
        clipping, and none of NegPy's decode. Nothing is measured from it: the analysis chart
        keeps reading the frame's own metrics.

        Geometry runs against the preview's own pixel grid, which is not the raw's, so the
        context is built from its shape rather than `original_res`.
        """
        source = self._load_embedded_preview()
        if source is None:
            return
        geometry = self.state.config.geometry
        height, width = source.shape[:2]
        context = PipelineContext(
            original_size=(height, width),
            scale_factor=max(height, width) / float(APP_CONFIG.preview_render_size),
            process_mode=self.state.config.process.process_mode,
            crop_preview_full=self.state.active_tool in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW),
            wants_uv_grid=False,
        )
        img = GeometryProcessor(geometry).process(source, context)
        if not context.crop_preview_full:
            img = CropProcessor(geometry).process(img, context)
        with self.state.metrics_lock:
            # Already display-encoded sRGB, so no working OETF: the camera's curve is the
            # whole point of the view.
            self.state.last_metrics["base_positive"] = img
            self.state.last_metrics["content_rect"] = None
            self.state.last_metrics["splash"] = True
            self.state.last_metrics["proof"] = False
            self.state.last_metrics["interactive"] = False
        self.image_updated.emit()

    def toggle_embedded_peek(self, force: Optional[bool] = None) -> None:
        """Show the camera's embedded preview, without changing the saved edit.

        ``force`` sets an explicit state; otherwise toggles. Mutually exclusive with the
        other peeks, the before/after compare view and the test strip. A source that carries
        no preview (a scanner TIFF, most DNGs) says so and stays off.
        """
        if self.state.preview_raw is None:
            return
        target = (not self.state.embedded_peek) if force is None else force
        if target == self.state.embedded_peek:
            return

        if target and self._load_embedded_preview() is None:
            self.set_status("This file carries no embedded preview", 3000)
            self.embedded_peek_changed.emit(False)
            return

        if target:
            self.exit_compare()
            if self.state.flat_peek:
                self.state.flat_peek = False
                self.flat_peek_changed.emit(False)
            if self.state.negative_peek:
                self.state.negative_peek = False
                self.negative_peek_changed.emit(False)
            self._clear_test_strip()

        self.state.embedded_peek = target
        self.embedded_peek_changed.emit(target)

        if target:
            self._paint_embedded_peek()
        else:
            self.request_render()

    def _enabled_presets(self) -> List[ExportPreset]:
        return [p for p in self.state.export_presets if p.enabled]

    def _validate_preset_paths(self, presets: List[ExportPreset]) -> bool:
        """Returns True if all absolute-path presets have a valid directory configured."""
        from PyQt6.QtWidgets import QFileDialog

        for p in presets:
            if p.output_mode == ExportPresetOutputMode.ABSOLUTE and not p.output_path.strip():
                new_path = QFileDialog.getExistingDirectory(None, f"Select output folder for preset '{p.name}'", os.path.expanduser("~"))
                if not new_path:
                    return False
                p.output_path = new_path
                self.session.save_export_presets()
        return True

    def _batch_params_for(self, f: dict) -> WorkspaceConfig:
        """Resolve a visible frame's export params: its saved DB config (else the current
        config), with its own RGB-scan green/blue re-injected from the asset dict — the
        same authoritative source individual export gets via select_file.

        For the currently active file, always use the live session config to ensure
        unsaved edits (e.g., crosstalk adjustments not yet persisted) are included
        in the export.

        Half-frame siblings share capture-side spectral-crosstalk calibration — if one
        half has it enabled and the other's DB entry defaults to 0, propagate the active
        session value so both frames get identical dye-unmixing during export.
        """
        # The active file uses the live session config, which may hold unsaved changes the
        # user expects in the export. Other files use their saved DB settings, or the session
        # config when they have none.
        if f.get("hash") == self.state.current_file_hash:
            params = self.state.config
        else:
            params = self.session.repo.load_file_settings(f["hash"]) or self.state.config

        # Propagate capture-side crosstalk between sibling half-frames. The dye-unmix
        # calibration belongs to the scanner-film pair, not to a frame, so if one half was
        # calibrated and the other left at default both need the same correction.
        base = f.get("hash", "")
        half_val = half_of(base)
        if half_val is not None:
            sibling_hash = half_hash(base_hash(base) or base, 3 - half_val)
            sibling_params = self.session.repo.load_file_settings(sibling_hash)
            session_ct = self.state.config.process.crosstalk_strength
            params_ct = params.process.crosstalk_strength
            sibling_ct = sibling_params.process.crosstalk_strength if sibling_params else session_ct
            # If this frame's crosstalk differs from the sibling but is at default (0),
            # inherit the sibling's value so both get identical correction.
            if abs(params_ct) < 1e-9 and abs(sibling_ct) > 1e-9:
                proc = params.process
                params = replace(
                    params,
                    process=replace(
                        proc,
                        crosstalk_strength=sibling_ct,
                        crosstalk_matrix=sibling_params.process.crosstalk_matrix if sibling_params else proc.crosstalk_matrix,
                    ),
                )

        return resolve_asset_hdr(resolve_asset_stitch(resolve_asset_rgbscan(params, f), f), f)

    def _tasks_for_file(
        self,
        file_info: dict,
        params: WorkspaceConfig,
        presets: List[ExportPreset],
        bounds_override=None,
        source_exif=None,
        metadata_config=None,
    ) -> List[ExportTask]:
        file_info, diptych = self._diptych_task(file_info)
        if diptych is not None:
            bounds_override = None  # the active frame's bounds belong to a half, not to the pair
        tasks = []
        for preset in presets:
            task_params, export_settings = resolve_preset_export(preset, params)
            export_settings.icc_input_path = self.effective_input_icc(task_params.process)
            tasks.append(
                ExportTask(
                    file_info=file_info,
                    params=task_params,
                    export_settings=export_settings,
                    gpu_enabled=self.state.gpu_enabled,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=metadata_config,
                    working_color_space=self.state.workspace_color_space,
                    diptych=diptych,
                )
            )
        return tasks

    def _ensure_valid_export_path(self) -> Optional[str]:
        """
        Checks if the current export path is valid. If not, prompts the user.
        Returns the valid path, or None if the user cancelled. The path can come back
        empty in the source-relative modes, which do not use it — callers must test
        `is None`, not truthiness, or an unset path silently cancels the export.
        """
        export_path = self.state.config.export.export_path
        if self.state.config.export.output_mode != ExportPresetOutputMode.ABSOLUTE:
            return export_path  # path irrelevant when the destination follows the source folder
        if export_path.strip().lower() in ["export", "/export", ""]:
            from PyQt6.QtWidgets import QFileDialog

            new_path = QFileDialog.getExistingDirectory(None, "Select Export Directory", os.path.expanduser("~"))
            if new_path:
                new_export = replace(self.state.config.export, export_path=new_path)
                self.session.update_config(replace(self.state.config, export=new_export), persist=True)
                return new_path
            return None
        return export_path

    def _roll_export_root(self, output_mode: ExportPresetOutputMode, subfolder: str) -> Optional[str]:
        """Base folder for Subfolder of Source, redirected from the active roll's own
        (non-existent) folder to the data folder when it is a virtual roll — files
        gathered from a library search or picked by hand share no folder to build a
        subfolder under. None for a folder roll, no active roll, or any other mode,
        which already resolve correctly per file.

        Warns once, since the redirect departs from what the DESTINATION picker shows.
        """
        if output_mode != ExportPresetOutputMode.SUBFOLDER_OF_SOURCE or not self.state.active_roll_id:
            return None
        entry = rolls.roll_for_id(self.session.repo, self.state.active_roll_id)
        if entry is None or entry.get("kind") != "virtual":
            return None
        name = path_safe(entry.get("name", "")) or "Untitled Roll"
        root = os.path.join(get_default_user_dir(), name)
        destination = os.path.join(root, subfolder) if subfolder else root
        self.set_status(f'"{name}" has no single folder — exporting to {destination}', 6000, kind="warning")
        return root

    def history_steps(self) -> List[Dict[str, Any]]:
        """Rows for the History panel: one dict {index, label, is_current} per edit step."""
        file_hash = self.state.current_file_hash
        if not file_hash:
            return []
        configs = dict(self.session.repo.load_all_history(file_hash))
        # The live top step may not be persisted yet: it lives in state.config.
        configs[self.state.undo_index] = self.state.config

        rows: List[Dict[str, Any]] = []
        for i in range(self.state.max_history_index + 1):
            config = configs.get(i)
            if config is None:
                continue
            rows.append(
                {
                    "index": i,
                    "label": history_step_label(configs.get(i - 1), config, i),
                    "is_current": i == self.state.undo_index,
                }
            )
        return rows

    def jump_to_history_step(self, index: int) -> None:
        self.session.jump_to_step(index)

    def export_history_step(self, index: int) -> None:
        """Load a history step, then export it through the normal export path."""
        self.session.jump_to_step(index)
        self.request_export()

    def export_work_print(self, name: str) -> None:
        """Make a named version live, then export it through the normal export path."""
        self.session.load_work_print(name)
        self.request_export()

    def _flush_export_ui(self) -> None:
        """Push pending Export-panel edits into state before any export path reads config."""
        flush = self.flush_export_settings
        if flush is not None:
            flush()

    def request_linear_output_export(self, files: list[dict] | None = None) -> None:
        """Export decoded linear buffers as untagged 16-bit TIFFs to the export folder."""
        from negpy.services.export.linear_output import is_linear_output_supported

        if self._batch_busy("export"):
            return

        export_path = self._ensure_valid_export_path()
        if export_path is None:
            return

        if files is None:
            file_path = self.state.current_file_path
            if not file_path:
                return
            if not is_linear_output_supported(file_path):
                self.set_status("Linear Output is not supported for this file type", 4000)
                return
            # Reuse the asset dict from uploaded_files so the RGB-scan triplet and stitch
            # fields reach _batch_params_for. A bare {path, name, hash} dict makes
            # resolve_asset_rgbscan/resolve_asset_stitch reset those configs and export only
            # the primary narrowband exposure.
            file_info = next(
                (f for f in self.state.uploaded_files if f.get("hash") == self.state.current_file_hash),
                None,
            )
            if file_info is None:
                file_info = {"path": file_path, "name": os.path.basename(file_path), "hash": self.state.current_file_hash}
            files = [file_info]

        supported = [f for f in files if is_linear_output_supported(f["path"])]
        if not supported:
            self.set_status("No files support Linear Output", 4000)
            return

        if len(supported) > 1 and not self._confirm_bulk_export(f"Linear-export {count_of(len(supported), 'frame')}?"):
            return

        tasks = self._linear_output_tasks(supported, export_path)

        self._export_start_time = time.time()
        self._export_failures = 0
        if self._begin_batch("export", "Exporting Linear Output", abortable=True) is None:
            return
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_linear_output",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
        )

    def _linear_output_tasks(self, supported: list[dict], export_path: str) -> list[LinearOutputTask]:
        """Resolve each frame's config and destination on the UI thread; the worker only writes."""
        expansion = self.state.linear_expansion
        linear_fmt = self.state.linear_format
        out_ext = "jxl" if linear_fmt == "jxl" else "tiff"
        # Destination and naming are the Export panel's, shared with print and flat. Only the
        # format belongs to the Linear intent, so the ephemeral preset carries it: `{{ format }}`
        # in a filename template has to name the file that is actually written.
        delivery = replace(
            preset_from_export_config(replace(self.state.config.export, export_path=export_path)),
            export_fmt=ExportFormat.JXL if linear_fmt == "jxl" else ExportFormat.TIFF,
        )
        roll_root = self._roll_export_root(delivery.output_mode, delivery.output_subfolder)
        sync_metadata = self.state.config.metadata.sync_to_batch
        taken: set[str] = set()
        tasks = []
        for f in supported:
            params = self._batch_params_for(f)
            stitch = params.stitch if params.stitch.stitch_enabled else None
            frames = hdr_frame_paths(f)
            out_dir = resolve_output_dir(f["path"], delivery, roll_root)
            # Same naming rule as a normal export: the bracket's first frame, suffixed so
            # the merge does not write over that frame's own linear output. No border and no
            # half: a linear dump is the whole decoded source, whatever the print crop says.
            stem = render_export_filename(
                min(frames, key=lambda p: os.path.basename(p).lower()) if frames else f["path"],
                delivery,
                metadata=self.state.config.metadata if sync_metadata else params.metadata,
                composite="HDR" if frames else "",
            )
            # `_linear` always, on top of whatever the template rendered: without it a dump
            # written next to its source under the default pattern overwrites that source.
            out_path = os.path.join(out_dir, f"{stem}_linear.{out_ext}")
            counter = 2
            # `taken` as well as the disk: the whole batch is named up front, before the
            # worker writes any of it, so same-stem frames would collide.
            while out_path in taken or (os.path.exists(out_path) and not delivery.overwrite):
                out_path = os.path.join(out_dir, f"{stem}_linear_{counter}.{out_ext}")
                counter += 1
            taken.add(out_path)
            tasks.append(
                LinearOutputTask(
                    file_info=f,
                    out_path=out_path,
                    options={
                        "geometry": params.geometry,
                        "expansion": expansion,
                        "rgbscan": params.rgbscan,
                        "stitch": stitch,
                        "hdr": params.hdr,
                        "flatfield": params.flatfield,
                        "process": params.process,
                        "apply_wb": self.state.linear_apply_wb,
                        "apply_flatfield": self.state.linear_apply_flatfield,
                        "apply_sensor": self.state.linear_apply_sensor,
                        "apply_ice": self.state.linear_apply_ice,
                        "retouch": params.retouch,
                        "gamma_key": self.state.linear_gamma_key,
                        "output_format": linear_fmt,
                        "jxl_effort": self.state.linear_jxl_effort,
                        "tiff_compression": self.state.config.export.tiff_compression,
                    },
                )
            )
        return tasks

    def request_export(self) -> None:
        """Exports the current file using the settings currently shown in the Export panel."""
        self._flush_export_ui()
        if self._batch_busy("export"):
            return
        if not self.state.current_file_path:
            return

        export_path = self._ensure_valid_export_path()
        if export_path is None:
            return

        params = self.state.config
        if self.state.flat_output:
            params = flat_master_config(params)
        export_conf = replace(
            self.state.config.export,
            export_path=export_path,
            icc_input_path=self.effective_input_icc(params.process),
            icc_output_path=self.state.icc_output_path,
        )
        if self.state.flat_output:
            export_conf = flat_export_config(export_conf)
        roll_root = self._roll_export_root(export_conf.output_mode, export_conf.output_subfolder)
        source_exif = self.state.source_exif.get(self.state.current_file_hash or "")

        # Reuse the asset dict from uploaded_files so the half-frame fields reach the
        # exporter. Without them process_export gets half=0 and renders the whole scan,
        # dropping the crop and shifting the log bounds.
        file_info = next(
            (f for f in self.state.uploaded_files if f.get("hash") == self.state.current_file_hash),
            None,
        )
        if file_info is None:
            file_info = {
                "name": os.path.basename(self.state.current_file_path),
                "path": self.state.current_file_path,
                "hash": self.state.current_file_hash,
            }

        file_info, diptych = self._diptych_task(file_info)

        bounds_override = None
        if diptych is None and file_info.get("hash") == self.state.current_file_hash:
            with self.state.metrics_lock:
                bounds_override = self.state.last_metrics.get("log_bounds")

        self._run_export_tasks(
            [
                ExportTask(
                    file_info=file_info,
                    params=params,
                    export_settings=preset_from_export_config(export_conf),
                    gpu_enabled=self.state.gpu_enabled,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=self.state.config.metadata,
                    working_color_space=self.state.workspace_color_space,
                    diptych=diptych,
                    roll_export_root=roll_root,
                )
            ]
        )

    def request_export_selected(self) -> None:
        """Batch-exports the currently selected files using the current export settings."""
        selected = [self.state.uploaded_files[i] for i in self.state.selected_indices if 0 <= i < len(self.state.uploaded_files)]
        self.request_batch_export(files=[f for f in selected if not f.get("excluded")])

    def request_batch_export(self, files: list[dict] | None = None) -> None:
        """Batch-exports the given files (all visible by default) using the current export settings."""
        self._flush_export_ui()
        if self._batch_busy("export"):
            return
        export_path = self._ensure_valid_export_path()
        if export_path is None:
            return

        current_export = replace(self.state.config.export, export_path=export_path)
        roll_root = self._roll_export_root(current_export.output_mode, current_export.output_subfolder)
        icc_output = self.state.icc_output_path
        sync_metadata = self.state.config.metadata.sync_to_batch

        if files is None:
            files = [
                self.state.uploaded_files[i]
                for i in self.session.asset_model.visible_actual_indices_ordered()
                if not self.state.uploaded_files[i].get("excluded")
            ]

        if len(files) > 1 and not self._confirm_bulk_export(f"Export {count_of(len(files), 'frame')}?"):
            return

        if self.state.config.export.export_sidecars_enabled:
            self._write_edit_sidecars(files)

        flat = self.state.flat_output

        tasks = []
        for f in files:
            # Delivery settings are session-level. A per-file config from the DB bypasses
            # _apply_sticky_settings and carries the export block current when that frame was
            # last saved, so honouring it exports at a size the panel never shows (#750).
            params = replace(self._batch_params_for(f), export=current_export)

            if flat:
                params = flat_master_config(params)

            final_export = replace(
                params.export,
                icc_input_path=self.effective_input_icc(params.process),
                icc_output_path=icc_output,
            )

            if flat:
                final_export = flat_export_config(final_export)

            file_info, diptych = self._diptych_task(f)

            bounds_override = None
            if diptych is None and f["hash"] == self.state.current_file_hash:
                with self.state.metrics_lock:
                    bounds_override = self.state.last_metrics.get("log_bounds")

            source_exif = self.state.source_exif.get(f["hash"])
            metadata_config = self.state.config.metadata if sync_metadata else params.metadata

            tasks.append(
                ExportTask(
                    file_info=file_info,
                    params=params,
                    export_settings=preset_from_export_config(final_export),
                    gpu_enabled=self.state.gpu_enabled,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=metadata_config,
                    working_color_space=self.state.workspace_color_space,
                    diptych=diptych,
                    roll_export_root=roll_root,
                )
            )

        if tasks:
            self._run_export_tasks(tasks)

    def _preset_export_files_for_selection(self) -> list[dict]:
        """Selected filmstrip frames in display order; single selection exports the preview frame."""
        n = len(self.state.uploaded_files)
        selected = [i for i in self.state.selected_indices if 0 <= i < n]

        if len(selected) <= 1:
            if not self.state.current_file_path or not (0 <= self.state.selected_file_idx < n):
                return []
            file_info = self.state.uploaded_files[self.state.selected_file_idx]
            if file_info.get("excluded"):
                return []
            return [file_info]

        selected_set = set(selected)
        visible_order = self.session.asset_model.visible_actual_indices_ordered()
        ordered = [i for i in visible_order if i in selected_set]
        for i in sorted(selected_set):
            if i not in ordered:
                ordered.append(i)
        files = [self.state.uploaded_files[i] for i in ordered]
        return [f for f in files if not f.get("excluded")]

    def _build_preset_export_tasks(self, files: list[dict], presets: List[ExportPreset]) -> List[ExportTask]:
        sync_metadata = self.state.config.metadata.sync_to_batch
        tasks: List[ExportTask] = []
        for f in files:
            params = self._batch_params_for(f)

            bounds_override = None
            if f["hash"] == self.state.current_file_hash:
                with self.state.metrics_lock:
                    bounds_override = self.state.last_metrics.get("log_bounds")

            source_exif = self.state.source_exif.get(f["hash"])
            metadata_config = self.state.config.metadata if sync_metadata else params.metadata

            tasks.extend(
                self._tasks_for_file(
                    f,
                    params,
                    presets,
                    bounds_override=bounds_override,
                    source_exif=source_exif,
                    metadata_config=metadata_config,
                )
            )
        return tasks

    def _confirm_bulk_export(self, text: str) -> bool:
        reply = QMessageBox.question(
            None,
            "Export",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _dispatch_preset_export(self, files: list[dict]) -> None:
        self._flush_export_ui()
        if self._batch_busy("export"):
            return
        if not files:
            return

        presets = self._enabled_presets()
        if not presets:
            QMessageBox.information(None, "No Presets Enabled", "Enable at least one export preset in the Export panel.")
            return

        if not self._validate_preset_paths(presets):
            return

        if len(files) > 1:
            n_frames = len(files)
            n_presets = len(presets)
            n_files = n_frames * n_presets
            if not self._confirm_bulk_export(
                f"Export {count_of(n_frames, 'frame')} through {count_of(n_presets, 'preset')} ({count_of(n_files, 'file')})?"
            ):
                return

        if self.state.config.export.export_sidecars_enabled:
            self._write_edit_sidecars(files)

        tasks = self._build_preset_export_tasks(files, presets)
        if tasks:
            self._run_export_tasks(tasks)

    def request_preset_export(self) -> None:
        """Initiates high-resolution export for the current file using enabled presets."""
        if not self.state.current_file_path:
            return

        # Reuse the asset dict from uploaded_files so the RGB-scan triplet and stitch fields
        # reach _batch_params_for. A bare {path, name, hash} dict makes
        # resolve_asset_rgbscan/resolve_asset_stitch reset those configs and preset-export
        # only the primary un-merged exposure.
        file_info = next(
            (f for f in self.state.uploaded_files if f.get("hash") == self.state.current_file_hash),
            None,
        )
        if file_info is None:
            file_info = {
                "name": os.path.basename(self.state.current_file_path),
                "path": self.state.current_file_path,
                "hash": self.state.current_file_hash,
            }
        self._dispatch_preset_export([file_info])

    def request_preset_export_selected(self) -> None:
        """Initiates preset export for every selected filmstrip frame."""
        files = self._preset_export_files_for_selection()
        self._dispatch_preset_export(files)

    def request_preset_batch_export(self) -> None:
        """Initiates batch export for all visible files using enabled presets."""
        visible_files = [
            self.state.uploaded_files[i]
            for i in self.session.asset_model.visible_actual_indices_ordered()
            if not self.state.uploaded_files[i].get("excluded")
        ]
        self._dispatch_preset_export(visible_files)

    def _contact_sheet_output_dir(self, visible_files: list) -> Optional[str]:
        """Resolve the contact sheet output folder (custom path or export destination rules)."""
        custom = self.state.config.export.contact_sheet_output_path.strip()
        if custom:
            return custom
        export_path = self._ensure_valid_export_path()
        if export_path is None:
            return None
        export_conf = replace(self.state.config.export, export_path=export_path)
        roll_root = self._roll_export_root(export_conf.output_mode, export_conf.output_subfolder)
        # The sheet covers the whole roll, so the source-relative modes follow the first frame.
        return resolve_output_dir(visible_files[0]["path"], preset_from_export_config(export_conf), roll_root)

    def request_contact_sheet(self) -> None:
        """Renders all visible files small and writes darkroom contact sheet(s)."""
        self._flush_export_ui()
        if self._batch_busy("contact sheet"):
            return
        visible_files = [self.state.uploaded_files[i] for i in self.session.asset_model.visible_actual_indices_ordered()]
        if not visible_files:
            return

        out_dir = self._contact_sheet_output_dir(visible_files)
        if not out_dir:
            return

        if len(visible_files) > 1 and not self._confirm_bulk_export(
            f"Render a contact sheet from {count_of(len(visible_files), 'frame')}?"
        ):
            return

        tasks = []
        for f in visible_files:
            params = self._batch_params_for(f)
            tasks.append(
                ExportTask(
                    file_info=f,
                    params=params,
                    export_settings=params.export,
                    gpu_enabled=self.state.gpu_enabled,
                    working_color_space=self.state.workspace_color_space,
                )
            )

        cs = self.state.config.export
        self._export_start_time = time.time()
        self._export_failures = 0
        if self._begin_batch("contact_sheet", "Contact sheet", abortable=True) is None:
            return
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_contact_sheet",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
            Q_ARG(str, out_dir),
            Q_ARG(int, cs.contact_sheet_cell_px),
            Q_ARG(int, cs.contact_sheet_gap),
            Q_ARG(int, cs.contact_sheet_margin),
            Q_ARG(int, cs.contact_sheet_max_tiles),
            Q_ARG(bool, cs.contact_sheet_show_labels),
            Q_ARG(str, cs.contact_sheet_background_color),
            Q_ARG(str, cs.contact_sheet_label_color),
        )

    def _write_edit_sidecars(self, files: list[dict]) -> tuple[int, int]:
        """Write a .negpy edit sidecar next to each source (each frame's own saved edits).
        Returns (written, failed) — a caller that reports only the written count turns a
        read-only source folder into a silent success."""
        repo = self.session.repo
        written = 0
        failed = 0
        for f in files:
            half = int(f.get("half") or 0)
            params = load_or_promote(
                repo, f["hash"], f["path"], half=half, composite=bool(f.get("hdr_paths") or f.get("stitch_paths"))
            ) or self.session.config_for_asset(f)
            try:
                write_sidecar(f["path"], params, half=half)
                written += 1
            except Exception as exc:
                failed += 1
                logger.warning("Sidecar write failed for %s: %s", f.get("path"), exc)
        return written, failed

    def export_edit_sidecars(self) -> None:
        """Explicit batch sidecar export for all visible files (ignores the on-export toggle)."""
        visible_files = [
            self.state.uploaded_files[i]
            for i in self.session.asset_model.visible_actual_indices_ordered()
            if not self.state.uploaded_files[i].get("excluded")
        ]
        if not visible_files:
            return
        written, failed = self._write_edit_sidecars(visible_files)
        suffix = f" — {failed} failed" if failed else ""
        self.set_status(f"Wrote {count_of(written, 'edit sidecar')}{suffix}", 6000 if failed else 4000)

    def _run_export_tasks(self, tasks: List[ExportTask]) -> None:
        # Reject unencodable format/color-space pairings before anything else.
        blocked = [t for t in tasks if export_blocked(t.export_settings.export_fmt, t.export_settings.export_color_space)]
        if blocked:
            names = ", ".join(sorted({t.file_info.get("name", "?") for t in blocked})[:5])
            QMessageBox.warning(
                None,
                "Export",
                f"JPEG XL can't tag the selected color space ({names}).\n"
                "Choose sRGB, P3 D65, Rec 2020 or Grayscale, or a different format.",
            )
            return

        # Then confirm any overwrites before dispatching to the worker.
        tasks = self._resolve_export_conflicts(tasks)
        if not tasks:
            return

        self._export_start_time = time.time()
        self._export_failures = 0
        if self._begin_batch("export", "Exporting", abortable=True) is None:
            return
        QMetaObject.invokeMethod(
            self.export_worker,
            "run_batch",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(list, tasks),
        )

    def _resolve_export_conflicts(self, tasks: List[ExportTask]) -> Optional[List[ExportTask]]:
        """Decide how to handle existing destination files before dispatching an export.

        If the "Overwrite existing files" preference is on, overwrite silently (no prompt)
        — for single Export and Export All alike. Otherwise, if the batch would clobber
        existing files, prompt (Overwrite / Rename / Cancel); the dialog's "always
        overwrite without asking" toggle persists the preference. Returns the tasks to run
        (overwrite flag set to the chosen action) or None to cancel the whole export."""
        if not tasks:
            return tasks

        if self.state.config.export.overwrite:
            return [replace(t, export_settings=replace(t.export_settings, overwrite=True)) for t in tasks]

        conflicts = find_export_conflicts(tasks)
        if not conflicts:
            return tasks

        choice, remember = self._prompt_overwrite_conflicts(conflicts)
        if choice is None:
            return None
        if remember and choice:
            self._set_overwrite_preference(True)
        return [replace(t, export_settings=replace(t.export_settings, overwrite=choice)) for t in tasks]

    def _set_overwrite_preference(self, value: bool) -> None:
        """Persist the global 'Overwrite existing files' preference (syncs the Export tab
        checkbox and the sticky default) without touching edit history or re-rendering."""
        cfg = self.state.config
        if bool(cfg.export.overwrite) == value:
            return
        new_config = replace(cfg, export=replace(cfg.export, overwrite=value))
        self.session.update_config(new_config, persist=True, render=True, record_history=False)

    @staticmethod
    def _prompt_overwrite_conflicts(conflicts: List[str]) -> tuple[Optional[bool], bool]:
        """Ask how to handle existing destination files. Returns (choice, remember):
        choice is True (overwrite), False (rename with a numbered suffix) or None (cancel);
        remember is whether the user asked to always overwrite without being asked again."""
        n = len(conflicts)
        names = "\n".join("  • " + os.path.basename(p) for p in conflicts[:8])
        if n > 8:
            names += f"\n  … and {n - 8} more"

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Warning)
        if n == 1:
            box.setWindowTitle("File Already Exists")
            box.setText(f"“{os.path.basename(conflicts[0])}” already exists in the export folder.")
        else:
            box.setWindowTitle("Files Already Exist")
            box.setText(f"{count_of(n, 'file')} already {plural(n, 'exists', 'exist')} in the export destination.")
        box.setInformativeText(f"{names}\n\nOverwrite, save with a new name, or cancel?")

        remember_check = QCheckBox("Always overwrite without asking")
        remember_check.setToolTip("Turns on the Export panel's “Overwrite existing files” option; stays on until you turn it off.")
        box.setCheckBox(remember_check)

        overwrite_label = "Overwrite" if n == 1 else "Overwrite All"
        rename_label = "Rename" if n == 1 else "Rename All"
        overwrite_btn = box.addButton(overwrite_label, QMessageBox.ButtonRole.DestructiveRole)
        rename_btn = box.addButton(rename_label, QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(rename_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()

        clicked = box.clickedButton()
        remember = remember_check.isChecked()
        if clicked is overwrite_btn:
            return True, remember
        if clicked is rename_btn:
            return False, remember
        return None, False

    def _on_render_busy(self, label: str) -> None:
        """Slow uncached render step (IR bake, inpaint) — hold a toast until the frame lands."""
        self._busy_toast = True
        self.set_status(label, _BUSY_TOAST_MS)

    def _clear_busy_toast(self) -> None:
        if self._busy_toast:
            self._busy_toast = False
            self.set_status("")

    def _renders_another_frame(self, metrics: Dict[str, Any]) -> bool:
        """Reject pixels and measurements for a different file or superseded edit."""
        src = metrics.get("source_hash")
        key = metrics.get("memo_key")
        return (src is not None and src != (self.state.current_file_hash or "preview")) or bool(
            key and self._expected_render_key and key != self._expected_render_key
        )

    def _on_render_finished(self, _result: Any, metrics: Dict[str, Any]) -> None:
        self._is_rendering = False
        self._clear_busy_toast()

        # The queue still drains — only the frame this render produced is unusable.
        if self._renders_another_frame(metrics):
            self._dispatch_pending_render()
            AppController._continue_background_work(self)
            return

        # The baseline half of the split is stashed, never displayed: it must not reach
        # last_metrics, the memo, the thumbnail or the canvas.
        if metrics.get("compare"):
            if self.state.compare_mode:
                self._capture_compare_before(metrics)
            if self._pending_render_task is not None:
                self._dispatch_pending_render()
            else:
                # The engine pool hands every render the same output texture, so this one
                # has just overwritten the edit the canvas is sampling. Print it again.
                self.request_render()
            AppController._continue_background_work(self)
            return

        if self._first_render_t0 is not None and not metrics.get("ephemeral"):
            logger.info(
                "load-timing first_render %.0fms (buffer -> painted) %s",
                (time.perf_counter() - self._first_render_t0) * 1000,
                self.state.current_file_path,
            )
            self._first_render_t0 = None

        # Config is replaced wholesale on every edit, so identity detects any change.
        # Not mid-gesture: the filmstrip only has to be right once the drag settles.
        should_update_thumb = (
            self._pending_render_task is None
            and not metrics.get("ephemeral")
            and not metrics.get("interactive")
            and self.state.config is not self._thumb_config
        )

        with self.state.metrics_lock:
            self.state.last_metrics.update(metrics)
            self.state.last_metrics["splash"] = False
            # last_metrics carries over between frames, so a peek's suppressed proof must
            # not outlive it onto the next render.
            self.state.last_metrics["proof"] = True

        self._freeze_resolved_auto_crop(metrics)

        result = metrics.get("base_positive")
        memoizable = bool(metrics.get("memo_key")) and metrics.get("source_hash") == self.state.current_file_hash
        # The pool overwrites a GPU texture on the next frame, so only its identity is kept
        # here. load_file files the texture itself on the way out.
        self._last_render_identity = (
            (metrics["source_hash"], metrics["memo_key"], metrics.get("content_rect"))
            if memoizable and isinstance(result, GPUTexture)
            else None
        )

        if metrics.get("gpu_fallback") and not self._gpu_fallback_notified:
            self._gpu_fallback_notified = True
            self.set_status("GPU acceleration failed — using CPU", 5000, kind="warning")

        # A render already in flight when the peek went on would otherwise repaint over it.
        if self.state.negative_peek:
            self._paint_negative_peek()
        elif self.state.embedded_peek:
            self._paint_embedded_peek()
        else:
            self.image_updated.emit()

        # By reference, because display buffers are read-only downstream. After the repaint:
        # overwriting an entry frees the texture the canvas has just stopped sampling.
        if memoizable and isinstance(result, np.ndarray):
            self._render_memo.store(
                metrics["source_hash"],
                metrics["memo_key"],
                {
                    "base_positive": result,
                    "content_rect": metrics.get("content_rect"),
                    "render_long_edge": metrics.get("render_long_edge", 0),
                },
            )

        if should_update_thumb:
            self._thumb_config = self.state.config
            # persist=False: refresh in-memory only; disk JPEG written on switch/save/export.
            self._update_thumbnail_from_state(persist=False)

        # Geometry, process or display changes make the stashed baseline half disagree with
        # the frame beside it; re-capture once the queue is empty.
        if self.state.compare_mode and self._pending_render_task is None and self.state.compare_before_key != self._compare_before_key():
            self._request_compare_baseline()
            AppController._continue_background_work(self)
            return

        self._dispatch_pending_render()
        AppController._continue_background_work(self)

    def _freeze_resolved_auto_crop(self, metrics: Dict[str, Any]) -> None:
        """Store the crop this render detected, so nothing detects it a second time.

        No render is requested: the rect is what was just painted. The key guards the gap
        between the render starting and this landing, so a ratio change mid-flight drops
        the result, which a queued render then re-detects.
        """
        rect = metrics.get("autocrop_resolved_rect")
        if rect is None:
            return
        geom = self.state.config.geometry
        if not geom.crop_from_auto or autocrop_detection_key(geom) != metrics.get("autocrop_resolved_key"):
            return
        if geom.crop_rect == rect and geom.crop_detect_key == metrics["autocrop_resolved_key"]:
            return
        new_geo = replace(geom, crop_rect=tuple(float(v) for v in rect), crop_detect_key=metrics["autocrop_resolved_key"])
        # record_history=False: tail of the Auto press, not a second edit to undo past.
        self.session.update_config(replace(self.state.config, geometry=new_geo), persist=True, render=False, record_history=False)
        self.config_updated.emit()

    def _dispatch_pending_render(self) -> None:
        """Start the render queued while the last one was running, if any."""
        if self._pending_render_task:
            task = self._pending_render_task
            self._pending_render_task = None
            self._is_rendering = True
            self.render_requested.emit(task)

    def _on_metrics_updated(self, metrics: Dict[str, Any]) -> None:
        """
        Handles late-arriving metrics and persists analysis results.
        """
        # A render of a frame the user has left measured that frame, not this one:
        # merging it corrupts the histogram, densitometer and UV grid until the next
        # render replaces every key it touched. The compare baseline measures a config
        # the user never set, so it is dropped for the same reason.
        if self._renders_another_frame(metrics) or metrics.get("compare"):
            return

        with self.state.metrics_lock:
            self.state.last_metrics.update(metrics)
        if "ir_degenerate" in metrics:
            self.state.ir_degenerate = bool(metrics["ir_degenerate"])
        self.metrics_available.emit(metrics)

        # Do not persist bounds from a splash render, or from a frame with no identity of
        # its own: they are not this frame's bounds. Nor from a mid-gesture frame, which
        # measured the proxy rather than the real buffer. A render of another file was
        # already dropped above.
        # A diptych's bounds were measured on one half, under that half's edit; writing them
        # onto the whole-frame config would file a half's measurement as the scan's.
        if metrics.get("ephemeral") or metrics.get("interactive") or metrics.get("diptych"):
            return
        src = metrics.get("source_hash")
        if src is not None and src != self.state.current_file_hash:
            return

        # Persist the per-frame *base*, not the final mix: re-feeding a mix as the next base
        # stacks edits. Skip only when both axes ride the roll baseline.
        proc = self.state.config.process
        bounds = metrics.get("log_bounds_base") or metrics.get("log_bounds")
        if bounds and not (proc.use_luma_average and proc.use_color_average):
            changes = {}
            if not proc.lock_bounds and (bounds.floors != proc.local_floors or bounds.ceils != proc.local_ceils):
                changes["local_floors"] = bounds.floors
                changes["local_ceils"] = bounds.ceils

            if changes:
                new_process = replace(self.state.config.process, **changes)
                self.session.update_config(
                    replace(self.state.config, process=new_process),
                    persist=self._may_persist_measured_bounds(),
                    render=False,
                    record_history=False,
                )
                # render=False: the displayed pixels already reflect these measured bounds.
                # Move the frame's memo entry to the updated config's key so the first
                # navigate-back after an initial render still hits. A GPU render is not filed
                # until navigate-away, so its identity follows too.
                self._render_memo.rekey(
                    src or self.state.current_file_hash or "", self._render_memo_key(), old_key=metrics.get("memo_key", "")
                )
                if self._last_render_identity is not None:
                    self._last_render_identity = (
                        self._last_render_identity[0],
                        self._render_memo_key(),
                        self._last_render_identity[2],
                    )

    def _may_persist_measured_bounds(self) -> bool:
        """Whether an auto-measured bounds write may reach the database.

        A half must not be brought into existence by a measurement. Looking at one half of a
        scan renders it, which meters it, which would file a settings row under `<hash>#1` —
        and the mere existence of that row is what later says the scan is a diptych. Turning
        Half Frame on and straight back off then leaves the frame stuck as one, having never
        been edited. A half the user did edit already has a row, and its bounds keep tracking.
        """
        file_hash = self.state.current_file_hash or ""
        if half_of(file_hash) is None:
            return True
        if file_hash not in self._measured_half_rows:
            if self.session.repo.load_file_settings(file_hash) is None:
                return False
            self._measured_half_rows.add(file_hash)
        return True

    def _on_preview_load_error(self, message: str) -> None:
        self._foreground_preview_generation = None
        self._neighbor_prefetch_generation = None
        self._neighbor_prefetch_queue.clear()
        logger.error(f"Preview load failure: {message}")
        self.set_status(f"Failed to load file: {message}", 5000, kind="error")
        self.load_failed.emit()
        AppController._continue_background_work(self)

    def _on_render_error(self, message: str) -> None:
        self.state.is_processing = self._is_rendering = False
        self._busy_toast = False  # the failure message below replaces the toast
        logger.error(f"Render failure: {message}")
        self.set_status(f"Failed to load file: {message}", 5000, kind="error")
        self.load_failed.emit()
        self._dispatch_pending_render()
        AppController._continue_background_work(self)

    def _on_export_task_error(self, message: str) -> None:
        self._export_failures += 1
        self._report_worker_error("Export", message)

    def _report_worker_error(self, source: str, message: str) -> None:
        """A background job failed. Names the job and leaves the canvas alone: only a failed
        load of the shown frame (_on_render_error) may blank it."""
        logger.error(f"{source} failed: {message}")
        self.set_status(f"{source} failed: {message}", 6000, kind="error")

    def _on_library_search_error(self, message: str) -> None:
        if self._active_batch == "library_index":
            self._end_batch("library_index")
            self._report_worker_error("Library indexing", message)
        else:
            self._report_worker_error("Library search", message)

    def _on_export_finished(self) -> None:
        elapsed = time.time() - self._export_start_time
        owner = self._active_batch if self._active_batch in ("export", "contact_sheet") else "export"
        self._end_batch(owner)
        self.export_finished.emit(elapsed, self._export_failures)
        self._update_thumbnail_from_state()

    def _asset_for_render(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The asset a finished render belongs to — not whichever one is selected now.

        A render carries the hash it was started for, and it can land after the user has
        moved on: select a frame, start its render, click the next one before the decode
        finishes. Keying those pixels by the current selection files one frame's picture
        under another frame's thumbnail, which then shows the wrong image until that frame
        is clicked and re-rendered. The render memo already guards this way.

        Falls back to the selection when the render carries no hash, which is the
        active_file_changing caller — there the outgoing file is still selected.
        """
        source_hash = metrics.get("source_hash")
        if not source_hash:
            return None
        for asset in self.state.uploaded_files:
            if asset.get("hash") == source_hash:
                return asset
        # No fallback to the selected frame. This runs on file switch, on save and after an
        # export as well as from the render itself, so last_metrics can hold a render whose
        # frame has left the list. Guessing files that buffer under whatever is selected now
        # and persists it, so one frame wears another's picture until it renders again. A
        # skipped refresh costs nothing: the next render of that frame writes it.
        return None

    def _update_thumbnail_from_state(self, persist: bool = True) -> None:
        if not self.state.current_file_path or not self.state.current_file_hash:
            return
        with self.state.metrics_lock:
            metrics = dict(self.state.last_metrics)
        asset = self._asset_for_render(metrics)
        if asset is None:
            return
        buffer = metrics.get("base_positive")

        # The render worker supplies host pixels. Reading back here would put a full-frame
        # copy on the UI thread.
        if isinstance(buffer, GPUTexture):
            buffer = metrics.get("thumbnail_source")

        if buffer is not None and not isinstance(buffer, np.ndarray):
            buffer = metrics.get("analysis_buffer")
        if buffer is None or not isinstance(buffer, np.ndarray):
            return

        # The same transform the canvas used for this buffer, so the filmstrip and the canvas
        # cannot disagree about the frame's color.
        display_cs, monitor_bytes, proof = self.display_transform_params(
            splash=bool(metrics.get("splash")), proofed=bool(metrics.get("proof", True))
        )
        # The asset's own key, so the batch (source) path re-serves this rendered positive
        # instead of the uninverted source merge it would decode itself.
        self.thumbnail_update_requested.emit(
            ThumbnailUpdateTask(
                file_hash=asset_thumbnail_key(asset),
                buffer=buffer,
                color_space=display_cs,
                monitor_icc_bytes=monitor_bytes,
                proof=proof,
                persist=persist,
            )
        )

    def cleanup(self) -> None:
        """
        Total system evacuation on exit.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._render_debounce.stop()
        self._cursor_readout_timer.stop()
        if self.render_thread.isRunning():
            self.render_thread.quit()
            self.render_thread.wait()
        if self.export_thread.isRunning():
            self.export_thread.quit()
            self.export_thread.wait()
        if self.thumb_thread.isRunning():
            self.thumb_worker.cancel_pending()
            self.thumb_thread.quit()
            self.thumb_thread.wait()
        self._autocrop_cancel_requested = True
        self.batch_autocrop_worker.cancel(self._autocrop_batch_token)
        self.thumbnail_render_worker.cancel(self._thumbnail_render_generation)
        if self.norm_thread.isRunning():
            self.norm_thread.quit()
            self.norm_thread.wait()
        if self.discovery_thread.isRunning():
            self.discovery_thread.quit()
            self.discovery_thread.wait()
        if self.preview_load_thread.isRunning():
            self.preview_load_thread.quit()
            self.preview_load_thread.wait()
        self.scan_worker.cancel()
        if self.scan_thread.isRunning():
            self.scan_thread.quit()
            self.scan_thread.wait()
        self.capture_worker.shutdown()
        if self.capture_thread.isRunning():
            self.capture_thread.quit()
            self.capture_thread.wait()
        # Memo-owned textures outlive the pool, so they must die before the device.
        self._render_memo.clear()
        self.render_worker.destroy_all()

        # All GPU-touching threads are now joined; release the wgpu device.
        GPUDevice.destroy_singleton()
