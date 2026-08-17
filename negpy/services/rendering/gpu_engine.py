import gc
import math
import os
import struct
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import wgpu  # type: ignore

from negpy.domain.models import AspectRatio, ExportResolutionMode, WorkspaceConfig
from negpy.features.exposure.analysis import DENSITY_HIST_BINS
from negpy.features.finish.logic import carrier_profiles
from negpy.features.finish.processor import carrier_width_px
from negpy.features.exposure import models as exposure_models
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds_from_log,
    contrast_mask_plane,
    luma_source_bounds,
    normalized_roi,
    luminance_density_range,
    measure_anchor_from_log,
    measure_clip_fractions,
    measure_neutral_axis_from_log,
    measure_shadow_refs_from_log,
    effective_crosstalk_matrix,
    unmix_log_image,
    measure_textural_range_from_log,
    prefilter_log_grid,
    resolve_analysis_region,
    resolve_bounds_detailed,
)
from negpy.features.geometry.logic import (
    apply_fine_rotation,
    apply_keystone,
    keystone_inverse_normalized,
    apply_margin_to_roi,
    apply_radial_distortion,
    compute_distortion_scale,
    get_manual_rect_coords,
)
from negpy.features.lab.logic import gaussian_kernel_1d, rl_iterations
from negpy.features.lab.models import SharpenMethod
from negpy.features.altprocess.models import AltProcess
from negpy.features.cyanotype.logic import CYANOTYPE_CONSTANTS, sensitizer_constants
from negpy.features.lith.logic import LITH_CONSTANTS
from negpy.features.local.logic import compute_local_maps
from negpy.features.exposure.transfer import (
    TRANSFER_CONSTANTS,
    ZONE_BLACK_TAPER,
    TRANSFER_DENSITY_RANGE,
    is_transparency_transfer,
    transfer_bounds,
    transfer_curve_params,
    transfer_widths,
    zone_geometry,
)
from negpy.features.process.capture_color import camera_to_working_matrix
from negpy.features.process.logic import effective_linear_raw
from negpy.features.process.models import ProcessMode, per_channel_point_offsets
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUBuffer, GPUTexture
from negpy.infrastructure.gpu.shader_loader import ShaderLoader
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.paths import get_resource_path
from negpy.services.export.print import PrintService
from negpy.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)

# Mirrors ToningUniforms.alt_mode in toning.wgsl.
_ALT_MODE = {AltProcess.NONE: 0, AltProcess.LITH: 1, AltProcess.CYANOTYPE: 2}

# Hardware constants
UNIFORM_ALIGNMENT_DEFAULT = 256
TILE_SIZE = 2048
TILE_HALO = 32
TILING_THRESHOLD_PX = 12_000_000
HISTOGRAM_BINS = 256
# Metrics buffer layout in u32 words: RGBL histogram (metrics.wgsl), then the density
# histogram (density_hist.wgsl). 256 B-aligned offsets, mirrored as WGSL array lengths.
# Append-only.
_METRICS_HIST_WORDS = HISTOGRAM_BINS * 4
_METRICS_DENSITY_BASE = 1024
METRICS_BUFFER_SIZE = 1152 * 4

# Per-frame metrics clear; write_buffer copies at call time, so sharing is safe.
_METRICS_ZEROS = np.zeros(METRICS_BUFFER_SIZE // 4, dtype=np.uint32)


def _downsample_for_analysis(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _binding_identity(idx: int, res: Any) -> tuple:
    """Hashable identity for the bind-group cache. Pooled views/persistent buffers keep the
    same object across frames, so id() is stable."""
    if isinstance(res, dict) and "buffer" in res:
        return (idx, id(res["buffer"]), res.get("offset", 0), res.get("size"))
    if isinstance(res, GPUBuffer):
        return (idx, id(res.buffer))
    return (idx, id(res))


def _keystone_inverse_bytes(converge_v: float, converge_h: float) -> bytes:
    """The keystone inverse packed as two vec4s: (h00, h01, h02, h10), (h11, h12, h20, h21).
    h22 is 1 by construction, so the shader supplies it."""
    m = keystone_inverse_normalized(converge_v, converge_h)
    return struct.pack("ffff", m[0, 0], m[0, 1], m[0, 2], m[1, 0]) + struct.pack("ffff", m[1, 1], m[1, 2], m[2, 0], m[2, 1])


def _analysis_cache_key(settings: WorkspaceConfig, analysis_source_hash: str) -> tuple:
    """Identity of the auto-exposure analysis for a frame. Mirrors the CPU base-stage
    cache key (engine.py) plus the fields that gate refs/anchor/textural, so it survives
    creative-slider drags but invalidates on anything the meter actually reads."""
    e = settings.exposure
    return (
        analysis_source_hash,
        settings.process,
        settings.geometry,
        e.cast_removal_strength > 0.0,
        e.auto_exposure,
        e.auto_normalize_contrast,
        exposure_models.TARGETS_REVISION,
    )


def _fill_analysis_overrides(cache, key, bounds, refs, anchor, textural, neutral):
    """Fill the None overrides from the cache when its key matches; caller overrides win."""
    if cache is None or cache[0] != key:
        return bounds, refs, anchor, textural, neutral
    _, cb, cr, ca, ct, cn = cache
    return (
        bounds if bounds is not None else cb,
        refs if refs is not None else cr,
        anchor if anchor is not None else ca,
        textural if textural is not None else ct,
        neutral if neutral is not None else cn,
    )


def _update_analysis_cache(cache, key, bounds, refs, anchor, textural, neutral):
    """Store the resolved analysis under key, merging (a frame may compute only a subset)."""
    if cache is None or cache[0] != key:
        cb = cr = ca = ct = cn = None
    else:
        _, cb, cr, ca, ct, cn = cache
    return (
        key,
        bounds if bounds is not None else cb,
        refs if refs is not None else cr,
        anchor if anchor is not None else ca,
        textural if textural is not None else ct,
        neutral if neutral is not None else cn,
    )


class GPUEngine:
    """
    Core GPU orchestration engine using WebGPU.
    Manages a 10-stage compute pipeline with unified memory and texture pooling.
    """

    def __init__(self) -> None:
        self.gpu = GPUDevice.get()
        self._shaders = {
            "geometry": get_resource_path(os.path.join("negpy", "features", "geometry", "shaders", "transform.wgsl")),
            "normalization": get_resource_path(os.path.join("negpy", "features", "exposure", "shaders", "normalization.wgsl")),
            "exposure": get_resource_path(os.path.join("negpy", "features", "exposure", "shaders", "exposure.wgsl")),
            "transfer": get_resource_path(os.path.join("negpy", "features", "exposure", "shaders", "transfer.wgsl")),
            "output_encode": get_resource_path(os.path.join("negpy", "features", "exposure", "shaders", "output_encode.wgsl")),
            "autocrop": get_resource_path(os.path.join("negpy", "features", "geometry", "shaders", "autocrop.wgsl")),
            "clahe_hist": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_hist.wgsl")),
            "clahe_cdf": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_cdf.wgsl")),
            "clahe_apply": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_apply.wgsl")),
            "lab_sharpen_h": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "lab_sharpen_h.wgsl")),
            "lab_sharpen_v": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "lab_sharpen_v.wgsl")),
            "rl_init": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "rl_init.wgsl")),
            "rl_blur_h": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "rl_blur_h.wgsl")),
            "rl_div_v": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "rl_div_v.wgsl")),
            "rl_mult_v": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "rl_mult_v.wgsl")),
            "lab": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "lab.wgsl")),
            "lith": get_resource_path(os.path.join("negpy", "features", "lith", "shaders", "lith.wgsl")),
            "cyanotype": get_resource_path(os.path.join("negpy", "features", "cyanotype", "shaders", "cyanotype.wgsl")),
            "toning": get_resource_path(os.path.join("negpy", "features", "toning", "shaders", "toning.wgsl")),
            "finish": get_resource_path(os.path.join("negpy", "features", "finish", "shaders", "finish.wgsl")),
            "metrics": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "metrics.wgsl")),
            "density_hist": get_resource_path(os.path.join("negpy", "features", "exposure", "shaders", "density_hist.wgsl")),
            "layout": get_resource_path(os.path.join("negpy", "features", "toning", "shaders", "layout.wgsl")),
        }
        self._pipelines: Dict[str, Any] = {}
        self._buffers: Dict[str, GPUBuffer] = {}
        self._sampler: Optional[Any] = None
        self._tex_cache: Dict[Tuple[int, int, int, str], GPUTexture] = {}

        self._uniform_names = [
            "geometry",
            "normalization",
            "exposure",
            "transfer",
            "clahe_u",
            "lab",
            "lith",
            "cyanotype",
            "toning",
            "finish",
            "layout",
            "density_hist",
        ]
        # Packed byte size per stage. A stage that exceeds the 256B dynamic-offset
        # alignment (exposure, 336B) occupies multiple aligned slots.
        self._uniform_sizes = {
            "geometry": 64,
            "normalization": 160,
            "exposure": 336,
            "transfer": 144,
            "clahe_u": 32,
            "lab": 96,
            "lith": 64,
            "cyanotype": 64,
            "toning": 64,
            "finish": 60,
            "layout": 48,
            "density_hist": 16,
        }
        self._alignment = UNIFORM_ALIGNMENT_DEFAULT
        self._current_source_hash: Optional[str] = None
        # Once-per-source guard so the analysis timing log fires on load, not every slider.
        self._analysis_timing_hash: Optional[str] = None
        # (key, bounds, shadow_refs, metered_anchor, textural_range, neutral_axis): the
        # per-source meter cache, so creative-slider previews skip the analysis.
        self._analysis_cache: Optional[tuple] = None
        # (analysis_key, per-channel clipped fractions) for the scan-exposure warning.
        self._clip_cache: Optional[tuple] = None
        self._last_settings: Optional[WorkspaceConfig] = None
        self._last_targets_rev: int = -1
        self._last_scale_factor: float = 1.0
        # No config field carries render_size_ref, so a size-only change would
        # otherwise resume past the layout pass.
        self._last_render_size_ref: Optional[float] = None
        # (radius, scale_factor) of the sharpen taps currently in sharpen_k.
        self._sharpen_kernel_key: Optional[tuple] = None

        # Bind groups reference resources, not contents, so they survive across frames.
        # Cache and reuse them (cleared in cleanup()): about 28 fewer wgpu calls per frame.
        self._bind_group_cache: Dict[Tuple, Any] = {}
        self._bind_layout_cache: Dict[str, Any] = {}

        # Persistent staging buffers, so no create_buffer() per readback
        self._metrics_staging: Optional[Any] = None
        # (prb, height, buffer): reused when image size and rotation are unchanged
        self._downsample_staging: Optional[Tuple[int, int, Any]] = None

        # (key, grid): a pure function of geometry, reused across settled frames
        self._uv_grid_cache: Optional[Tuple[Tuple, np.ndarray]] = None
        # Identity of the dodge/burn EV map currently sitting in the local_ev texture.
        self._local_ev_key: Optional[Tuple] = None
        self._mask_plane: Optional[Tuple[Tuple, np.ndarray, float]] = None
        # Identity of the plane currently sitting in the contrast_mask texture.
        self._mask_tex_key: Optional[Tuple] = None

    def _detect_invalidated_stage(self, settings: WorkspaceConfig, scale_factor: float, render_size_ref: Optional[float] = None) -> int:
        """
        Determines the earliest pipeline stage that needs re-running.
        Returns stage index (5 unused — dodge/burn lives in the exposure pass):
        0: Geometry (Source/Transform)
        1: Exposure (Normalization/Grading/Dodge & Burn)
        2: CLAHE (Adaptive Hist)
        3: Retouch (Healing)
        4: Lab (Color/Sharpen)
        6: Toning (Paper/Split)
        7: Finish (Vignette)
        8: Layout (Final compositing)
        """
        if (
            self._last_settings is None
            or self._last_scale_factor != scale_factor
            or self._last_render_size_ref != render_size_ref
            or self._last_settings.process.process_mode != settings.process.process_mode
        ):
            return 0

        last = self._last_settings
        if last.geometry != settings.geometry:
            return 0
        # k1 lives in flatfield config but is applied in the geometry pass (stage 0).
        if last.flatfield.apply != settings.flatfield.apply or last.flatfield.k1 != settings.flatfield.k1:
            return 0
        if last.process != settings.process or last.exposure != settings.exposure:
            return 1
        # Retuned Auto Density/Grade targets live in EXPOSURE_CONSTANTS, invisible to the
        # config diff, but they reshape the print curve in the exposure pass.
        if self._last_targets_rev != exposure_models.TARGETS_REVISION:
            return 1
        if last.local != settings.local:
            return 1
        if last.lab.clahe_strength != settings.lab.clahe_strength:
            return 2
        if last.lab != settings.lab:
            return 4
        if last.altproc != settings.altproc:
            return 5
        if last.toning != settings.toning:
            return 6
        if last.finish != settings.finish:
            return 7
        if last.export != settings.export:
            # Carrier width is mm-of-print, so a print-size change moves the finish pass too.
            if settings.finish.carrier_width > 0.0 and last.export.export_print_size != settings.export.export_print_size:
                return 7
            return 8

        return 9  # Nothing changed

    def _get_intermediate_texture(self, w: int, h: int, usage: int, label: str) -> GPUTexture:
        """Retrieves or creates a texture from the pool.

        Key is (w, h, usage, label). A 90°/270° rotation already swaps w and h
        upstream (see w_rot/h_rot computation), so the key naturally changes
        with geometry — no extra geometry field needed.
        Contents are fully overwritten each render, so no stale-data risk.

        Invariant: callers must pass post-rotation dimensions. If rotation
        handling ever moves downstream of texture allocation, revisit this key.
        """
        key = (w, h, usage, label)
        if key not in self._tex_cache:
            self._tex_cache[key] = GPUTexture(w, h, usage=usage)
        return self._tex_cache[key]

    def _init_resources(self) -> None:
        """Initializes hardware pipelines and persistent buffers."""
        if self._pipelines or not self.gpu.device:
            return
        # Buffers are recreated below, so force the next kernel upload.
        self._sharpen_kernel_key = None
        t0 = time.perf_counter()
        device = self.gpu.device
        self._sampler = device.create_sampler(min_filter="linear", mag_filter="linear")

        hw_min = self.gpu.limits.get("min_uniform_buffer_offset_alignment", 256)
        self._alignment = max(256, hw_min)

        for name, path in self._shaders.items():
            self._pipelines[name] = self._create_pipeline(path)

        # Unified Uniform Buffer (UBO)
        self._buffers["unified_u"] = GPUBuffer(
            sum(self._slot_bytes(n) for n in self._uniform_names),
            wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Storage buffers for intermediate metrics and CLAHE
        self._buffers["clahe_h"] = GPUBuffer(65536, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self._buffers["clahe_c"] = GPUBuffer(
            65536,
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        # Sharpen blur taps (gaussian_kernel_1d): 1024 f32 covers radius <= 511.
        self._buffers["sharpen_k"] = GPUBuffer(4096, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        # Filed-carrier jitter profiles are a fixed table, so upload once.
        self._buffers["carrier_s"] = GPUBuffer(carrier_profiles().nbytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self._buffers["carrier_s"].upload(np.ascontiguousarray(carrier_profiles().ravel(), dtype=np.float32))
        self._buffers["metrics"] = GPUBuffer(
            METRICS_BUFFER_SIZE,
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )

        logger.info(
            "load-timing gpu_init %.0fms (compiled %d shaders/pipelines)",
            (time.perf_counter() - t0) * 1000,
            len(self._pipelines),
        )

    def _create_pipeline(self, shader_path: str) -> Any:
        shader_module = ShaderLoader.load(shader_path)
        assert self.gpu.device is not None
        try:
            return self.gpu.device.create_compute_pipeline(layout="auto", compute={"module": shader_module, "entry_point": "main"})
        except Exception:
            logger.exception(f"Failed to compile pipeline: {shader_path}")
            raise

    def _slot_bytes(self, name: str) -> int:
        """Aligned bytes a stage occupies in the unified UBO (>= 1 slot)."""
        return -(-self._uniform_sizes[name] // self._alignment) * self._alignment

    def _get_uniform_binding(self, name: str) -> Dict[str, Any]:
        """Calculates UBO offset and size for a specific pipeline stage.
        Offsets are cumulative so an oversized stage spans multiple slots."""
        offset = 0
        for n in self._uniform_names:
            if n == name:
                break
            offset += self._slot_bytes(n)
        return {
            "buffer": self._buffers["unified_u"].buffer,
            "offset": offset,
            "size": self._uniform_sizes[name],
        }

    def process_to_texture(
        self,
        img: np.ndarray,
        settings: WorkspaceConfig,
        scale_factor: float = 1.0,
        tiling_mode: bool = False,
        bounds_override: Optional[Any] = None,
        global_offset: Tuple[int, int] = (0, 0),
        full_dims: Optional[Tuple[int, int]] = None,
        clahe_cdf_override: Optional[np.ndarray] = None,
        shadow_refs_override: Optional[Tuple[float, float, float]] = None,
        metered_anchor_override: Optional[float] = None,
        textural_range_override: Optional[float] = None,
        neutral_axis_override: Optional[tuple] = None,
        apply_layout: bool = True,
        render_size_ref: Optional[float] = None,
        source_hash: Optional[str] = None,
        readback_metrics: bool = True,
        vignette_full_crop: Optional[Tuple[int, int, int, int]] = None,
        local_maps: Optional[np.ndarray] = None,
        analysis_source_hash: Optional[str] = None,
        cam_xyz: Optional[list] = None,
        camera_wb: Optional[list] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the full pipeline, returning a GPU texture and associated metrics.

        ``local_maps`` is the pre-rasterised (h, w, 2) dodge/burn EV + local grade
        map already in the post-geometry frame; tiled export passes a per-tile slice.
        When None and masks are present, it is computed here from ``settings.local``.
        """
        if not self.gpu.is_available:
            raise RuntimeError("GPU not available")
        self._init_resources()
        device = self.gpu.device
        assert device is not None

        h, w = img.shape[:2]
        k1_eff = settings.flatfield.k1 if settings.flatfield.apply else 0.0
        source_tex = self._get_intermediate_texture(
            w,
            h,
            wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            "source",
        )

        # Only upload if the source content has changed
        if source_hash is None or source_hash != self._current_source_hash:
            source_tex.upload(img)
            self._current_source_hash = source_hash
            start_stage = 0
        elif tiling_mode:
            start_stage = 0
        else:
            start_stage = self._detect_invalidated_stage(settings, scale_factor, render_size_ref)

        # ROI calculation
        if tiling_mode and full_dims:
            w_rot, h_rot = w, h
            x1, y1 = 0, 0
            crop_w, crop_h = w, h
            actual_full_dims = full_dims
            orig_shape = (h, w)
            roi = (0, h, 0, w)
        else:
            rot = settings.geometry.rotation % 4
            w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
            # Invariant: intermediate textures are allocated with post-rotation
            # dimensions, so the cache key naturally avoids 90°/270° collisions.
            # If rotation handling ever moves downstream of _get_intermediate_texture
            # calls, this invariant must be re-checked.
            assert w_rot > 0 and h_rot > 0
            actual_full_dims, orig_shape = (w_rot, h_rot), (h, w)
            if settings.geometry.crop_rect:
                roi = get_manual_rect_coords(
                    (h_rot, w_rot),
                    settings.geometry.crop_rect,
                    offset_px=settings.geometry.autocrop_offset,
                    scale_factor=scale_factor,
                )
            elif settings.geometry.autocrop_offset > 0:
                margin = settings.geometry.autocrop_offset * scale_factor
                roi = apply_margin_to_roi((0, h_rot, 0, w_rot), h_rot, w_rot, margin)
            else:
                roi = (0, h_rot, 0, w_rot)
            y1, y2, x1, x2 = roi
            crop_w, crop_h = max(1, x2 - x1), max(1, y2 - y1)

        # Reuse the per-source meter across creative-slider previews: fill any missing
        # override from the cache so the needs_* gates below skip the analysis entirely.
        analysis_key = None
        if analysis_source_hash is not None and not tiling_mode:
            analysis_key = _analysis_cache_key(settings, analysis_source_hash)
            (
                bounds_override,
                shadow_refs_override,
                metered_anchor_override,
                textural_range_override,
                neutral_axis_override,
            ) = _fill_analysis_overrides(
                self._analysis_cache,
                analysis_key,
                bounds_override,
                shadow_refs_override,
                metered_anchor_override,
                textural_range_override,
                neutral_axis_override,
            )

        analysis_t0 = time.perf_counter()
        needs_refs = (
            shadow_refs_override is None
            and not tiling_mode
            and settings.exposure.cast_removal_strength > 0.0
            and settings.process.process_mode == ProcessMode.C41
        )
        _roll_luma = settings.process.use_luma_average and settings.process.is_locked_initialized
        _roll_color = settings.process.use_color_average and settings.process.is_locked_initialized
        needs_bounds_analysis = not (bounds_override or (_roll_luma and _roll_color) or settings.process.is_local_initialized)
        # Measure the anchor for the render when Auto Density is on, and for the
        # Analysis-panel stats on every preview whatever the toggle says. The render only
        # *uses* it when auto_exposure is on (see uniforms).
        needs_anchor = metered_anchor_override is None and not tiling_mode and (settings.exposure.auto_exposure or readback_metrics)
        needs_textural = textural_range_override is None and not tiling_mode and settings.exposure.auto_normalize_contrast

        analysis_source = None
        prefiltered = None
        unmix_m = effective_crosstalk_matrix(settings.process, settings.process.process_mode)
        if needs_bounds_analysis or needs_refs or needs_anchor or needs_textural:
            # Use views to avoid copying the full-res image; crop to ROI first.
            analysis_source = img
            if settings.geometry.rotation != 0:
                analysis_source = np.rot90(analysis_source, k=settings.geometry.rotation)
            if settings.geometry.flip_horizontal:
                analysis_source = np.fliplr(analysis_source)
            if settings.geometry.flip_vertical:
                analysis_source = np.flipud(analysis_source)
            # A freehand analysis_rect overrides the crop ROI and centered buffer, like the
            # CPU path. Tiled export uses explicit overrides, so it stays on the ROI.
            base_roi = roi if not tiling_mode else None
            analysis_roi, an_buffer = resolve_analysis_region(
                analysis_source.shape,
                base_roi,
                settings.process.analysis_buffer,
                settings.process.analysis_rect if not tiling_mode else None,
            )
            if analysis_roi is not None:
                ay1, ay2, ax1, ax2 = analysis_roi
                analysis_source = np.ascontiguousarray(analysis_source[ay1:ay2, ax1:ax2])
            if settings.geometry.fine_rotation != 0.0:
                analysis_source = apply_fine_rotation(analysis_source, settings.geometry.fine_rotation)
            # The meters must read the frame the print stage gets. The CPU engine
            # normalizes the keystoned buffer, so this replay has to carry it too or the
            # two engines measure different bounds.
            analysis_source = apply_keystone(analysis_source, settings.geometry.converge_v, settings.geometry.converge_h)

            analysis_source = _downsample_for_analysis(analysis_source, APP_CONFIG.preview_render_size)
            # Shared prefilter, once for all five meters (ROI already applied).
            # Unmixed like the CPU path so every meter reads the unmixed film.
            prefiltered = unmix_log_image(prefilter_log_grid(analysis_source, None, an_buffer), unmix_m)

        scan_clip_fractions = None
        if analysis_source is not None:
            scan_clip_fractions = measure_clip_fractions(analysis_source, None, an_buffer)
            if analysis_key is not None:
                self._clip_cache = (analysis_key, scan_clip_fractions)
        elif analysis_key is not None and self._clip_cache is not None and self._clip_cache[0] == analysis_key:
            scan_clip_fractions = self._clip_cache[1]

        def _analyze_bounds() -> LogNegativeBounds:
            assert prefiltered is not None
            return analyze_log_exposure_bounds_from_log(
                prefiltered,
                None,
                0.0,
                process_mode=settings.process.process_mode,
                e6_normalize=settings.process.e6_normalize,
                percentile_clip=settings.process.luma_range_clip,
                color_clip=settings.process.color_range_clip,
            )

        if bounds_override:
            bounds = base_bounds = anchor_bounds = bounds_override
        else:
            bounds, base_bounds = resolve_bounds_detailed(settings.process, _analyze_bounds)
            anchor_bounds = luma_source_bounds(settings.process, base_bounds)

        shadow_refs = shadow_refs_override
        if needs_refs and prefiltered is not None:
            shadow_refs = measure_shadow_refs_from_log(prefiltered, None, 0.0)

        # Neutral axis for the two-point Cast Removal; normalized at consumption.
        neutral_axis_refs = neutral_axis_override
        if needs_refs and prefiltered is not None:
            neutral_axis_refs = measure_neutral_axis_from_log(prefiltered, bounds, None, 0.0)

        metered_anchor = metered_anchor_override
        if needs_anchor and prefiltered is not None:
            metered_anchor = measure_anchor_from_log(prefiltered, anchor_bounds, None, 0.0)

        textural_range = textural_range_override
        if needs_textural and prefiltered is not None:
            textural_range = measure_textural_range_from_log(prefiltered, None, 0.0)

        if analysis_key is not None:
            self._analysis_cache = _update_analysis_cache(
                self._analysis_cache, analysis_key, bounds, shadow_refs, metered_anchor, textural_range, neutral_axis_refs
            )

        # Same helper, same pre-geometry array as the CPU engine, so the two mask alike.
        # Keyed off the meter, so only the Contrast Mask slider's own value stays live.
        mask_plane = None
        mask_centre = 0.5
        mask_key = None
        if settings.exposure.contrast_mask != 0.0 and not tiling_mode:
            mask_key = (analysis_key, bounds, roi, (h_rot, w_rot))
            if self._mask_plane is None or self._mask_plane[0] != mask_key:
                self._mask_plane = (
                    mask_key,
                    *contrast_mask_plane(
                        img,
                        bounds,
                        unmix_m,
                        rotation=settings.geometry.rotation,
                        fine_rotation=settings.geometry.fine_rotation,
                        flip_horizontal=settings.geometry.flip_horizontal,
                        flip_vertical=settings.geometry.flip_vertical,
                        converge_v=settings.geometry.converge_v,
                        converge_h=settings.geometry.converge_h,
                        distortion_k1=k1_eff,
                        roi_norm=normalized_roi(roi, (h_rot, w_rot)),
                    ),
                )
            mask_plane = self._mask_plane[1]
            mask_centre = self._mask_plane[2]

        # The printed frame in rotated pixels; the shader maps the plane onto it and
        # clamps outside, which is expand_mask_plane's edge padding.
        mask_uniform = None
        if mask_plane is not None:
            y1_m, y2_m, x1_m, x2_m = roi if roi is not None else (0, h_rot, 0, w_rot)
            y1_m, x1_m = max(0, y1_m), max(0, x1_m)
            y2_m, x2_m = min(h_rot, y2_m), min(w_rot, x2_m)
            if y2_m - y1_m < 1 or x2_m - x1_m < 1:
                mask_plane = None
            else:
                from negpy.features.exposure.logic import contrast_mask_scale

                mask_uniform = (
                    contrast_mask_scale(settings.exposure.contrast_mask, luminance_density_range(bounds)),
                    float(x1_m),
                    float(y1_m),
                    float(x2_m - x1_m),
                    float(y2_m - y1_m),
                )

        # CPU meter cost, logged once per source (skips creative-slider re-renders).
        if analysis_source is not None and analysis_source_hash is not None and analysis_source_hash != self._analysis_timing_hash:
            self._analysis_timing_hash = analysis_source_hash
            logger.info(
                "load-timing analysis %.0fms (bounds=%s refs=%s anchor=%s textural=%s)",
                (time.perf_counter() - analysis_t0) * 1000,
                needs_bounds_analysis,
                needs_refs,
                needs_anchor,
                needs_textural,
            )

        pw, ph, cw, ch, ox, oy, _ = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)

        self._upload_unified_uniforms(
            settings,
            bounds,
            global_offset,
            actual_full_dims,
            (0, 0) if tiling_mode else (x1, y1),
            crop_w,
            crop_h,
            tiling_mode,
            render_size_ref,
            scale_factor,
            vignette_full_crop=vignette_full_crop,
            shadow_refs=shadow_refs,
            metered_anchor=metered_anchor,
            textural_range=textural_range,
            neutral_axis_refs=neutral_axis_refs,
            unmix=unmix_m,
            cam_xyz=cam_xyz,
            camera_wb=camera_wb,
            contrast_mask=mask_uniform,
        )
        if clahe_cdf_override is not None:
            self._buffers["clahe_c"].upload(clahe_cdf_override)

        # Texture chain
        tex_geom = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "geom",
        )
        tex_norm = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
            "norm",
        )
        tex_expo = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "expo",
        )
        tex_clahe = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "clahe",
        )
        tex_lab = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "lab",
        )
        # The Contrast Mask plane rides its own analysis-grid texture, uploaded only when the
        # plane itself moves. A 1x1 dummy keeps the bind group valid when it is off (mask.x
        # gates it).
        if mask_plane is not None:
            tex_mask = self._get_intermediate_texture(
                mask_plane.shape[1],
                mask_plane.shape[0],
                wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                "contrast_mask",
            )
            if self._mask_tex_key != mask_key:
                tex_mask.upload(np.dstack([mask_plane] * 3))
                self._mask_tex_key = mask_key
        else:
            tex_mask = self._get_intermediate_texture(
                1,
                1,
                wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                "contrast_mask",
            )
        # The dodge/burn EV map feeds the exposure pass. A zero-initialized 1x1 dummy keeps
        # the bind group valid when no masks are active (ev_scale.w gates it).
        wants_ev_map = bool(settings.local.masks)
        if wants_ev_map:
            tex_local_ev = self._get_intermediate_texture(
                w_rot,
                h_rot,
                wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                "local_ev",
            )
        else:
            tex_local_ev = self._get_intermediate_texture(
                1,
                1,
                wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                "local_ev",
            )
        tex_toning = self._get_intermediate_texture(
            crop_w,
            crop_h,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
            "toning",
        )

        enc = device.create_command_encoder()

        if start_stage <= 0:
            self._dispatch_pass(
                enc,
                "geometry",
                [
                    (0, source_tex.view),
                    (1, tex_geom.view),
                    (2, self._get_uniform_binding("geometry")),
                ],
                w_rot,
                h_rot,
            )

        if start_stage <= 1:
            self._dispatch_pass(
                enc,
                "normalization",
                [
                    (0, tex_geom.view),
                    (1, tex_norm.view),
                    (2, self._get_uniform_binding("normalization")),
                ],
                w_rot,
                h_rot,
            )
            if wants_ev_map:
                # This stage re-runs for any exposure change, but the map only moves
                # with the masks, the geometry and the grade.
                tiled_maps = local_maps is not None
                ev_key = (
                    settings.local,
                    settings.geometry.rotation,
                    settings.geometry.fine_rotation,
                    settings.geometry.flip_horizontal,
                    settings.geometry.flip_vertical,
                    k1_eff,
                    settings.exposure.grade,
                    orig_shape,
                    w_rot,
                    h_rot,
                )
                if tiled_maps or self._local_ev_key != ev_key:
                    if local_maps is None:
                        local_maps = compute_local_maps(
                            settings.local,
                            h_rot,
                            w_rot,
                            orig_shape,
                            rotation=settings.geometry.rotation,
                            fine_rotation=settings.geometry.fine_rotation,
                            flip_horizontal=settings.geometry.flip_horizontal,
                            flip_vertical=settings.geometry.flip_vertical,
                            distortion_k1=k1_eff,
                        )
                    from negpy.features.exposure.logic import local_grade_factor_map

                    if local_maps is None:
                        local_maps = np.zeros((h_rot, w_rot, 2), dtype=np.float32)
                    ev_plane = local_maps[:, :, 0]
                    # r = dodge/burn EV, g = local grade slope factor, b unused. One texture,
                    # so the local-grade map costs no bind slot.
                    tex_local_ev.upload(
                        np.dstack(
                            [
                                ev_plane,
                                local_grade_factor_map(local_maps[:, :, 1], settings.exposure.grade),
                                np.zeros_like(ev_plane),
                            ]
                        )
                    )
                    # A tiled export passes a per-tile slice, which is not reusable.
                    self._local_ev_key = None if tiled_maps else ev_key
            if is_transparency_transfer(settings.process.process_mode, settings.process.e6_normalize):
                # The transfer curve takes no dodge/burn map: local EV is a print-exposure
                # input, and this path replaces the print.
                self._dispatch_pass(
                    enc,
                    "transfer",
                    [
                        (0, tex_norm.view),
                        (1, tex_expo.view),
                        (2, self._get_uniform_binding("transfer")),
                    ],
                    w_rot,
                    h_rot,
                )
            else:
                self._dispatch_pass(
                    enc,
                    "exposure",
                    [
                        (0, tex_norm.view),
                        (1, tex_expo.view),
                        (2, self._get_uniform_binding("exposure")),
                        (3, tex_local_ev.view),
                        (4, tex_mask.view),
                    ],
                    w_rot,
                    h_rot,
                )

        if settings.lab.clahe_strength > 0:
            if clahe_cdf_override is None and start_stage <= 2:
                self._dispatch_pass(
                    enc,
                    "clahe_hist",
                    [(0, tex_expo.view), (1, self._buffers["clahe_h"])],
                    8,
                    8,
                )
                self._dispatch_pass(
                    enc,
                    "clahe_cdf",
                    [
                        (0, self._buffers["clahe_h"]),
                        (1, self._buffers["clahe_c"]),
                        (2, self._get_uniform_binding("clahe_u")),
                    ],
                    8,
                    8,
                )
            if start_stage <= 2:
                self._dispatch_pass(
                    enc,
                    "clahe_apply",
                    [
                        (0, tex_expo.view),
                        (1, tex_clahe.view),
                        (2, self._buffers["clahe_c"]),
                        (3, self._get_uniform_binding("clahe_u")),
                    ],
                    w_rot,
                    h_rot,
                )
            prev_tex = tex_clahe
        else:
            prev_tex = tex_expo

        if start_stage <= 4:
            # Sharpen state (USM blur, or RL deconvolution) feeds the lab pass. A 1x1 dummy
            # keeps binding 3 valid when sharpening is off.
            usage = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
            lab_u = self._get_uniform_binding("lab")
            if settings.lab.sharpen > 0 and settings.lab.sharpen_method == SharpenMethod.RL:
                # Iterative RL: ping-pong two textures through init + N x (blur_h, div_v,
                # blur_h, mult_v). The final estimate lands back in rl_a.
                tex_rl_a = self._get_intermediate_texture(w_rot, h_rot, usage, "rl_a")
                tex_rl_b = self._get_intermediate_texture(w_rot, h_rot, usage, "rl_b")
                self._dispatch_pass(enc, "rl_init", [(0, prev_tex.view), (1, tex_rl_a.view)], w_rot, h_rot)
                sk = self._buffers["sharpen_k"]
                for _ in range(rl_iterations(settings.lab.sharpen_radius)):
                    self._dispatch_pass(enc, "rl_blur_h", [(0, tex_rl_a.view), (1, tex_rl_b.view), (2, lab_u), (3, sk)], w_rot, h_rot)
                    self._dispatch_pass(enc, "rl_div_v", [(0, tex_rl_b.view), (1, tex_rl_a.view), (2, lab_u), (3, sk)], w_rot, h_rot)
                    self._dispatch_pass(enc, "rl_blur_h", [(0, tex_rl_a.view), (1, tex_rl_b.view), (2, lab_u), (3, sk)], w_rot, h_rot)
                    self._dispatch_pass(enc, "rl_mult_v", [(0, tex_rl_b.view), (1, tex_rl_a.view), (2, lab_u), (3, sk)], w_rot, h_rot)
                tex_sharpen_v = tex_rl_a
            elif settings.lab.sharpen > 0:
                tex_sharpen_h = self._get_intermediate_texture(w_rot, h_rot, usage, "sharpen_h")
                tex_sharpen_v = self._get_intermediate_texture(w_rot, h_rot, usage, "sharpen_v")
                self._dispatch_pass(
                    enc,
                    "lab_sharpen_h",
                    [
                        (0, prev_tex.view),
                        (1, tex_sharpen_h.view),
                        (2, lab_u),
                        (3, self._buffers["sharpen_k"]),
                    ],
                    w_rot,
                    h_rot,
                )
                self._dispatch_pass(
                    enc,
                    "lab_sharpen_v",
                    [
                        (0, tex_sharpen_h.view),
                        (1, tex_sharpen_v.view),
                        (2, lab_u),
                        (3, self._buffers["sharpen_k"]),
                    ],
                    w_rot,
                    h_rot,
                )
            else:
                tex_sharpen_v = self._get_intermediate_texture(1, 1, usage, "sharpen_v")
            self._dispatch_pass(
                enc,
                "lab",
                [
                    (0, prev_tex.view),
                    (1, tex_lab.view),
                    (2, lab_u),
                    (3, tex_sharpen_v.view),
                ],
                w_rot,
                h_rot,
            )

        tex_pre_toning = tex_lab

        # --- Alternative processes (lith / cyanotype) ---
        # Mutually exclusive, and no pass at all when neither is picked. Toning then reads
        # tex_lab directly.
        alt = settings.altproc.alt_process
        if alt != AltProcess.NONE and settings.process.process_mode == ProcessMode.BW:
            shader = "lith" if alt == AltProcess.LITH else "cyanotype"
            tex_alt = self._get_intermediate_texture(
                w_rot,
                h_rot,
                wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
                shader,
            )
            if start_stage <= 5:
                self._dispatch_pass(
                    enc,
                    shader,
                    [
                        (0, tex_lab.view),
                        (1, tex_alt.view),
                        (2, self._get_uniform_binding(shader)),
                    ],
                    w_rot,
                    h_rot,
                )
            tex_pre_toning = tex_alt

        if start_stage <= 6:
            self._dispatch_pass(
                enc,
                "toning",
                [
                    (0, tex_pre_toning.view),
                    (1, tex_toning.view),
                    (2, self._get_uniform_binding("toning")),
                ],
                crop_w,
                crop_h,
            )

        # --- Finish (Vignette) ---
        tex_finish = self._get_intermediate_texture(
            crop_w,
            crop_h,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
            "finish_tex",
        )
        if start_stage <= 7:
            self._dispatch_pass(
                enc,
                "finish",
                [
                    (0, tex_toning.view),
                    (1, tex_finish.view),
                    (2, self._get_uniform_binding("finish")),
                    (3, self._buffers["carrier_s"]),
                ],
                crop_w,
                crop_h,
            )
            tex_for_layout = tex_finish
        else:
            tex_for_layout = tex_toning

        if not tiling_mode and apply_layout:
            paper_w, paper_h, content_w, content_h, off_x, off_y, _ = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)
            tex_final = self._get_intermediate_texture(
                paper_w,
                paper_h,
                wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
                "final",
            )
            if start_stage <= 8:
                self._dispatch_pass(
                    enc,
                    "layout",
                    [
                        (0, tex_for_layout.view),
                        (1, tex_final.view),
                        (2, self._get_uniform_binding("layout")),
                    ],
                    paper_w,
                    paper_h,
                )
            content_rect = (off_x, off_y, content_w, content_h)
        else:
            tex_final, content_rect = tex_for_layout, (0, 0, crop_w, crop_h)

        if not tiling_mode and readback_metrics:
            device.queue.write_buffer(self._buffers["metrics"].buffer, 0, _METRICS_ZEROS)
            # Always compute metrics on the content image (tex_toning) before any
            # border/layout pass so that border pixels don't skew the histogram.
            self._dispatch_pass(
                enc,
                "metrics",
                [(0, tex_toning.view), (1, self._buffers["metrics"])],
                crop_w,
                crop_h,
            )
            # The density histogram slice sits past the RGBL bins, so one shared readback.
            self._dispatch_pass(
                enc,
                "density_hist",
                [
                    (0, tex_norm.view),
                    (
                        1,
                        {
                            "buffer": self._buffers["metrics"].buffer,
                            "offset": _METRICS_DENSITY_BASE * 4,
                            "size": DENSITY_HIST_BINS * 4,
                        },
                    ),
                    (2, self._get_uniform_binding("density_hist")),
                ],
                crop_w,
                crop_h,
            )

        # Output transform: scene-linear to display-encoded, so every consumer reads
        # encoded data.
        tex_output = self._get_intermediate_texture(
            tex_final.width,
            tex_final.height,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
            "output_encoded",
        )
        self._dispatch_pass(enc, "output_encode", [(0, tex_final.view), (1, tex_output.view)], tex_final.width, tex_final.height)
        tex_final = tex_output

        device.queue.submit([enc.finish()])
        # The exact stretch the shader normalized with (mirrors the CPU "final_bounds").
        _wp3, _bp3 = per_channel_point_offsets(settings.process, settings.process.process_mode == ProcessMode.E6)
        final_bounds = LogNegativeBounds(
            floors=(bounds.floors[0] + _wp3[0], bounds.floors[1] + _wp3[1], bounds.floors[2] + _wp3[2]),
            ceils=(bounds.ceils[0] + _bp3[0], bounds.ceils[1] + _bp3[1], bounds.ceils[2] + _bp3[2]),
        )
        metrics: Dict[str, Any] = {
            "active_roi": roi,
            "base_positive": tex_final,
            "normalized_log": tex_norm,
            "content_rect": content_rect,
            "log_bounds": bounds,
            "final_bounds": final_bounds,
            "log_bounds_base": base_bounds,
            "norm_density_range": luminance_density_range(bounds),
            "metered_anchor": metered_anchor,
            "contrast_mask_centre": mask_centre,
            "textural_range": textural_range,
            "scan_clip_fractions": scan_clip_fractions,
            # Raw cast refs so the chart can re-solve the exact render curves.
            "shadow_log_refs": shadow_refs,
            "neutral_axis_refs": neutral_axis_refs,
        }

        if not tiling_mode and readback_metrics:
            raw_metrics = self._readback_metrics()
            metrics["histogram_raw"] = raw_metrics[:_METRICS_HIST_WORDS].reshape((4, HISTOGRAM_BINS))
            metrics["histogram_density"] = raw_metrics[_METRICS_DENSITY_BASE : _METRICS_DENSITY_BASE + DENSITY_HIST_BINS].astype(np.float64)
            try:
                uv_key = (
                    h,
                    w,
                    settings.geometry.rotation,
                    settings.geometry.fine_rotation,
                    settings.geometry.flip_horizontal,
                    settings.geometry.flip_vertical,
                    roi,
                    k1_eff,
                )
                if self._uv_grid_cache is not None and self._uv_grid_cache[0] == uv_key:
                    metrics["uv_grid"] = self._uv_grid_cache[1]
                else:
                    uv_grid = CoordinateMapping.create_uv_grid(
                        rh_orig=h,
                        rw_orig=w,
                        rotation=settings.geometry.rotation,
                        fine_rot=settings.geometry.fine_rotation,
                        flip_h=settings.geometry.flip_horizontal,
                        flip_v=settings.geometry.flip_vertical,
                        autocrop=True,
                        autocrop_params={"roi": roi} if roi else None,
                        distortion_k1=k1_eff,
                        converge_v=settings.geometry.converge_v,
                        converge_h=settings.geometry.converge_h,
                    )
                    self._uv_grid_cache = (uv_key, uv_grid)
                    metrics["uv_grid"] = uv_grid
            except Exception as e:
                logger.error(f"GPU Engine metrics error: {e}")

        self._last_settings = settings
        self._last_targets_rev = exposure_models.TARGETS_REVISION
        self._last_scale_factor = scale_factor
        self._last_render_size_ref = render_size_ref
        return tex_final, metrics

    def _upload_unified_uniforms(
        self,
        settings: WorkspaceConfig,
        bounds: Any,
        offset: Tuple[int, int],
        full_dims: Tuple[int, int],
        crop_offset: Tuple[int, int],
        crop_w: int,
        crop_h: int,
        tiling_mode: bool,
        render_size_ref: Optional[float],
        scale_factor: float,
        vignette_full_crop: Optional[Tuple[int, int, int, int]] = None,
        shadow_refs: Optional[Tuple[float, float, float]] = None,
        metered_anchor: Optional[float] = None,
        textural_range: Optional[float] = None,
        neutral_axis_refs: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]
        ] = None,
        unmix: Optional[np.ndarray] = None,
        cam_xyz: Optional[list] = None,
        camera_wb: Optional[list] = None,
        contrast_mask: Optional[Tuple[float, float, float, float, float]] = None,
    ) -> None:
        """Packs and uploads all pipeline parameters to the unified UBO."""
        # scale_s uses the post-rotation dims the geometry pass emits. Zeroed for tiled
        # export below, where geometry runs on the CPU instead.
        w_rot, h_rot = full_dims
        k1_eff = settings.flatfield.k1 if settings.flatfield.apply else 0.0
        scale_s = compute_distortion_scale(k1_eff, w_rot, h_rot) if k1_eff != 0.0 else 1.0
        g_data = struct.pack(
            "ifii",
            int(settings.geometry.rotation),
            float(settings.geometry.fine_rotation),
            (1 if settings.geometry.flip_horizontal else 0),
            (1 if settings.geometry.flip_vertical else 0),
        ) + struct.pack("ffff", float(k1_eff), float(scale_s), 0.0, 0.0)
        # The shader undoes the keystone first, so it gets the CPU's own matrix inverted
        # and normalized to [0,1] coords. Deriving the quad twice would let the two drift.
        g_data += _keystone_inverse_bytes(settings.geometry.converge_v, settings.geometry.converge_h)
        if tiling_mode:
            g_data = b"\x00" * 64

        f, c = bounds.floors, bounds.ceils
        mode_val = 0
        if settings.process.process_mode == ProcessMode.BW:
            mode_val = 1
        elif settings.process.process_mode == ProcessMode.E6:
            mode_val = 2

        # Per-channel WP/BP (global + layer trims, E6-signed) mirror the CPU path. Baked
        # into the packed floors/ceils, so the shader's scalar wp/bp offsets, kept at 0.0
        # for layout, need no per-channel lanes.
        wp3, bp3 = per_channel_point_offsets(settings.process, mode_val == 2)
        adj_floors = (f[0] + wp3[0], f[1] + wp3[1], f[2] + wp3[2])
        adj_ceils = (c[0] + bp3[0], c[1] + bp3[1], c[2] + bp3[2])

        # Transparency transfer: the fixed window, with no WP/BP trims. Mirrors
        # NormalizationProcessor._process_transparency, whose identity they would break.
        if is_transparency_transfer(settings.process.process_mode, settings.process.e6_normalize):
            t_floors, t_ceils = transfer_bounds()
            adj_floors, adj_ceils = t_floors, t_ceils

        # Capture-side dye-unmix rows, resolved once per frame by the caller and shared
        # with NormalizationProcessor. Identity when off.
        if unmix is None:
            unmix = np.eye(3)

        # Working-space-from-camera rows, identity when the source carries no matrix. The
        # shader reads them only on the transfer path.
        # Linear RAW decodes without white balance, so fold the as-shot multipliers back in
        # and the render stops depending on which decode produced the buffer.
        cam = camera_to_working_matrix(
            cam_xyz, camera_wb if effective_linear_raw(settings.process, settings.exposure.render_intent) else None
        )
        if cam is None:
            cam = np.eye(3, dtype=np.float32)

        n_data = (
            struct.pack("ffff", adj_floors[0], adj_floors[1], adj_floors[2], 0.0)
            + struct.pack("ffff", adj_ceils[0], adj_ceils[1], adj_ceils[2], 0.0)
            + struct.pack(
                "IIff",
                mode_val,
                (1 if settings.process.e6_normalize else 0),
                0.0,
                0.0,
            )
            + struct.pack("ffff", unmix[0, 0], unmix[0, 1], unmix[0, 2], 0.0)
            + struct.pack("ffff", unmix[1, 0], unmix[1, 1], unmix[1, 2], 0.0)
            + struct.pack("ffff", unmix[2, 0], unmix[2, 1], unmix[2, 2], 0.0)
            + struct.pack("ffff", float(cam[0, 0]), float(cam[0, 1]), float(cam[0, 2]), 0.0)
            + struct.pack("ffff", float(cam[1, 0]), float(cam[1, 1]), float(cam[1, 2]), 0.0)
            + struct.pack("ffff", float(cam[2, 0]), float(cam[2, 1]), float(cam[2, 2]), 0.0)
        )

        from negpy.features.exposure.logic import (
            _reference_linear_value,
            cast_solve_inputs,
            filtration_offsets,
            per_channel_dye_separation,
            per_channel_toe_shoulder,
            grade_coupled_shape,
            local_ev_scale,
            paper_dmin_rgb,
            per_channel_curve_params,
            per_channel_midtone_gamma,
            per_channel_widths,
            split_grade_deltas,
        )
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS
        from negpy.features.exposure.normalization import LogNegativeBounds, luminance_density_range

        # Transparency transfer params (mirrors transfer.py; inert on the print path).
        tc = TRANSFER_CONSTANTS
        t_exp, t_contrast, t_toe3, t_sh3 = transfer_curve_params(settings.exposure)
        t_tw3, t_sw3 = transfer_widths(settings.exposure)
        t_cmy = filtration_offsets(
            (settings.exposure.wb_cyan, settings.exposure.wb_magenta, settings.exposure.wb_yellow),
            LogNegativeBounds(floors=adj_floors, ceils=adj_ceils),
        )
        t_sh_c, t_hi_c, t_zone_k = zone_geometry()
        tr_data = (
            struct.pack(
                "ffffffff",
                float(t_exp),
                float(t_contrast),
                float(TRANSFER_DENSITY_RANGE),
                float(t_zone_k),
                float(tc["transfer_contrast_pivot"]),
                float(tc["transfer_toe_knee"]),
                float(tc["transfer_shoulder_knee"]),
                float(2.0 ** float(tc["transfer_baseline_ev"])),
            )
            + struct.pack("ffff", t_toe3[0], t_toe3[1], t_toe3[2], 0.0)
            + struct.pack("ffff", t_sh3[0], t_sh3[1], t_sh3[2], 0.0)
            + struct.pack("ffff", t_tw3[0], t_tw3[1], t_tw3[2], 0.0)
            + struct.pack("ffff", t_sw3[0], t_sw3[1], t_sw3[2], 0.0)
            + struct.pack("ffff", t_cmy[0], t_cmy[1], t_cmy[2], 0.0)
            + struct.pack(
                "ffff",
                float(settings.exposure.shadow_density),
                float(settings.exposure.highlight_density),
                float(t_sh_c),
                float(t_hi_c),
            )
            + struct.pack("ffff", float(ZONE_BLACK_TAPER), 0.0, 0.0, 0.0)
        )
        from negpy.features.exposure.papers import (
            compose_density_matrices,
            effective_constants,
            effective_paper_profile,
            resolve_dye_matrix,
            resolve_saturation_matrix,
        )

        exp = settings.exposure
        paper = effective_paper_profile(exp.paper_profile, settings.process.process_mode)
        pc = effective_constants(paper)  # tonal overrides; non-paper keys == EXPOSURE_CONSTANTS
        d_min = paper.d_min if exp.paper_dmin else 0.0
        # metered_anchor may be measured for stats even when auto_exposure is off, so let
        # it move the render only when the toggle is on.
        render_anchor = metered_anchor if exp.auto_exposure else None
        lum_range = luminance_density_range(bounds)
        # adj_floors/adj_ceils (packed above) are the final bounds the shader normalizes
        # with, shared by the Cast Removal shadow refs.
        strength, shadow_refs_norm, neutral_axis_norm = cast_solve_inputs(
            LogNegativeBounds(adj_floors, adj_ceils),
            shadow_refs,
            neutral_axis_refs,
            exp.cast_removal_strength,
        )
        slopes, pivots, curvatures = per_channel_curve_params(
            exp.grade,
            exp.density,
            exp.auto_normalize_contrast,
            strength,
            lum_range,
            shadow_refs_norm,
            textural_range,
            d_min=d_min,
            anchor=render_anchor,
            paper=paper,
            neutral_axis_norm=neutral_axis_norm,
            grade_trims=(exp.grade_trim_red, exp.grade_trim_green, exp.grade_trim_blue),
        )
        cmy_m = EXPOSURE_CONSTANTS["cmy_max_density"]
        _toe_eff, _shoulder_eff = grade_coupled_shape(slopes[1], exp.toe, exp.shoulder)
        _sg3, _hg3 = split_grade_deltas(
            exp.grade,
            exp.shadow_grade,
            exp.highlight_grade,
            shadow_trims=(exp.shadow_grade_trim_red, exp.shadow_grade_trim_green, exp.shadow_grade_trim_blue),
            highlight_trims=(exp.highlight_grade_trim_red, exp.highlight_grade_trim_green, exp.highlight_grade_trim_blue),
        )
        # Per-channel effective toe/shoulder, pre-scaled. The uniform block is full at
        # 256B, so these ride the vec4 w-lanes.
        _ts_k = float(EXPOSURE_CONSTANTS["toe_shoulder_strength"])
        _toe3, _sh3 = per_channel_toe_shoulder(
            _toe_eff,
            _shoulder_eff,
            (exp.toe_trim_red, exp.toe_trim_green, exp.toe_trim_blue),
            (exp.shoulder_trim_red, exp.shoulder_trim_green, exp.shoulder_trim_blue),
        )
        _toe3 = tuple(t * _ts_k for t in _toe3)
        _sh3 = tuple(s * _ts_k for s in _sh3)
        _mg3 = per_channel_midtone_gamma(
            paper,
            exp.midtone_gamma,
            (exp.midtone_gamma_trim_red, exp.midtone_gamma_trim_green, exp.midtone_gamma_trim_blue),
        )
        _tw3, _sw3 = per_channel_widths(
            exp.toe_width,
            exp.shoulder_width,
            (exp.toe_width_trim_red, exp.toe_width_trim_green, exp.toe_width_trim_blue),
            (exp.shoulder_width_trim_red, exp.shoulder_width_trim_green, exp.shoulder_width_trim_blue),
        )
        # Mirrors apply_characteristic_curve (absolute CC, paper base, dye mix).
        wb_offsets = filtration_offsets(
            (exp.wb_cyan, exp.wb_magenta, exp.wb_yellow),
            LogNegativeBounds(adj_floors, adj_ceils),
        )
        dmin_rgb = paper_dmin_rgb(d_min, paper)
        dye = resolve_dye_matrix(paper)
        # Density-space Dye Separation, composed into the same dye_mix slot as the paper's
        # real crosstalk. Mirrors apply_characteristic_curve exactly.
        if settings.process.process_mode == ProcessMode.BW:
            sat_k3 = (1.0, 1.0, 1.0)
            sep_damping = 0.0
            sat = None
        else:
            sat_k3 = per_channel_dye_separation(
                exp.dye_separation,
                (exp.dye_separation_trim_red, exp.dye_separation_trim_green, exp.dye_separation_trim_blue),
            )
            # Separation Damping makes k per-pixel, so it leaves the matrix slot to the
            # paper's coupling and runs in the shader instead.
            sep_damping = float(exp.separation_damping) if sat_k3 != (1.0, 1.0, 1.0) else 0.0
            sat = None if sep_damping > 0.0 else resolve_saturation_matrix(sat_k3)
        composed = compose_density_matrices(dye, sat)
        dye = composed  # use_dye_mix (below) and dye_rows both key off this
        dye_rows = np.eye(3) if dye is None else dye

        # The w-lanes carry per-channel toe (first three vec4s) and shoulder (next three).
        # See the toe3/sh3 reads in exposure.wgsl.
        e_data = (
            struct.pack("ffff", pivots[0], pivots[1], pivots[2], _toe3[0])
            + struct.pack("ffff", slopes[0], slopes[1], slopes[2], _toe3[1])
            + struct.pack("ffff", curvatures[0], curvatures[1], curvatures[2], _toe3[2])
            + struct.pack("ffff", wb_offsets[0], wb_offsets[1], wb_offsets[2], _sh3[0])
            + struct.pack(
                "ffff",
                exp.shadow_cyan * cmy_m,
                exp.shadow_magenta * cmy_m,
                exp.shadow_yellow * cmy_m,
                _sh3[1],
            )
            + struct.pack(
                "ffff",
                exp.highlight_cyan * cmy_m,
                exp.highlight_magenta * cmy_m,
                exp.highlight_yellow * cmy_m,
                _sh3[2],
            )
            # Asymmetric H&D print-curve scalars; mirrors _apply_print_curve_kernel.
            # Per-channel knee widths occupy the dead scalar toe/shoulder/
            # midtone_gamma slots and the former flare pad (see exposure.wgsl).
            + struct.pack(
                "14fI3fIf",
                _tw3[0],
                _tw3[1],
                _tw3[2],
                _sw3[0],
                # Zone Density ΔD shadow offset in the ex-d_min slot; the highlight offset
                # rides d_min_rgb.w.
                exp.shadow_density,
                pc["d_max"],
                pc["toe_sharpness_base"],
                pc["shoulder_sharpness_base"],
                # Separation Damping (toeshoulder_width_ref is a WGSL literal).
                sep_damping,
                pc["toe_height"],
                pc["shoulder_height"],
                pc["anchor_target_density"],
                _sw3[1],
                # Separation Damping's per-layer k, red (ex-surround_gamma slot). Green and
                # blue ride the split_sh/split_hi w-lanes.
                sat_k3[0],
                mode_val,
                _reference_linear_value(d_min, paper),
                _sw3[2],
                float(pc["paper_gamma_width"]),
                1 if dye is not None else 0,
                # BPC flag (former pad). Per-channel toe/shoulder/Snap ride the vec4
                # w-lanes, widths the ex-scalar slots, Zone Density ΔD the ex-d_min slot
                # and d_min_rgb.w, Split Grade the split_sh/split_hi rows past 256B.
                1.0 if not exp.paper_black else 0.0,
            )
            + struct.pack("ffff", dmin_rgb[0], dmin_rgb[1], dmin_rgb[2], exp.highlight_density)
            # Dye-row w-lanes carry the per-channel midtone gamma (Snap).
            + struct.pack("ffff", dye_rows[0, 0], dye_rows[0, 1], dye_rows[0, 2], _mg3[0])
            + struct.pack("ffff", dye_rows[1, 0], dye_rows[1, 1], dye_rows[1, 2], _mg3[1])
            + struct.pack("ffff", dye_rows[2, 0], dye_rows[2, 1], dye_rows[2, 2], _mg3[2])
            # Dodge/burn EV-stop size per channel (local_ev_scale); w = enable flag.
            + struct.pack(
                "ffff",
                *local_ev_scale(LogNegativeBounds(adj_floors, adj_ceils)),
                1.0 if settings.local.masks else 0.0,
            )
            # Split Grade per-channel zone contrast gains (split_grade_deltas). The w-lanes
            # carry Separation Damping's green and blue k.
            + struct.pack("ffff", _sg3[0], _sg3[1], _sg3[2], sat_k3[1])
            + struct.pack("ffff", _hg3[0], _hg3[1], _hg3[2], sat_k3[2])
            # Hue Trim in radians (x; yzw pad). The shader rotates before its encode.
            + struct.pack("ffff", math.radians(float(settings.process.hue_trim)), 0.0, 0.0, 0.0)
            # Contrast Mask: stops per unit of plane (0 = off), then the printed frame's
            # origin and span in rotated pixels. The shader does the upscale.
            + struct.pack("ffff", *(contrast_mask[:3] if contrast_mask else (0.0, 0.0, 0.0)), 0.0)
            + struct.pack("ffff", *(contrast_mask[3:] if contrast_mask else (1.0, 1.0)), 0.0, 0.0)
        )

        cls = float(settings.lab.clahe_strength)
        c_data = (
            struct.pack(
                "ffiiii",
                cls,
                cls * 2.5,
                offset[0],
                offset[1],
                full_dims[0],
                full_dims[1],
            )
            + b"\x00" * 8
        )

        lab = settings.lab
        # Sharpen taps are computed once in Python by gaussian_kernel_1d, the same array
        # the CPU convolves with, and uploaded to sharpen_k. The shaders need only the
        # half-width, derived from the array itself so the support matches.
        sharpen_radius_px = 0
        if lab.sharpen > 0:
            kernel = gaussian_kernel_1d(lab.sharpen_radius)
            sharpen_radius_px = len(kernel) // 2
            kernel_key = (float(lab.sharpen_radius),)
            if self._sharpen_kernel_key != kernel_key:
                self._buffers["sharpen_k"].upload(kernel)
                self._sharpen_kernel_key = kernel_key
        l_data = (
            struct.pack(
                "ffffffffff",
                float(lab.sharpen),
                float(lab.chroma_denoise),
                float(lab.saturation),
                float(lab.glow_amount),
                float(lab.halation_strength),
                # Chroma-denoise scales its blur radius by the preview downsample ratio,
                # like the CPU path (radius * scale_factor).
                float(scale_factor),
                float(sharpen_radius_px),
                float(lab.sharpen_masking),
                1.0 if lab.sharpen_method == SharpenMethod.RL else 0.0,
                float(lab.skin_protection),
            )
            + b"\x00" * 8
        )

        is_bw = 1 if settings.process.process_mode == ProcessMode.BW else 0

        altproc = settings.altproc
        lc = LITH_CONSTANTS
        lith_dmax = float(pc["d_max"])
        lith_over = 0.301 * float(altproc.lith_exposure)
        li_data = (
            struct.pack("ffff", *[float(p[0]) for p in paper.lith_path])
            + struct.pack("ffff", *[float(p[1]) for p in paper.lith_path])
            + struct.pack(
                "ffff",
                lith_over,
                lith_over * float(lc["foot_veil"]),
                lith_dmax * (lc["knee_lo"] - lc["knee_span"] * float(altproc.lith_snatch)),
                max(lc["abrupt_lo"] - lc["abrupt_span"] * float(altproc.lith_abruptness), 0.01),
            )
            # 12 bytes of tail padding: two vec4s force a 16-byte struct align, so
            # WGSL rounds LithUniforms up to 64 and the binding size must match.
            + struct.pack("f", lith_dmax)
            + b"\x00" * 12
        )

        cs = sensitizer_constants(altproc.cyano_sensitizer)
        cc = CYANOTYPE_CONSTANTS
        cy_data = (
            struct.pack("ffff", *[float(p[0]) for p in cs["path"]])
            + struct.pack("ffff", *[float(p[1]) for p in cs["path"]])
            + struct.pack(
                "ffff",
                0.301 * float(altproc.cyano_exposure),
                max(float(altproc.cyano_scale), 0.1),
                float(altproc.cyano_bleach),
                float(altproc.cyano_tannin),
            )
            # 4 bytes of tail padding, as for lith: the leading vec4s round
            # CyanoUniforms up to 64 and the binding size must match.
            + struct.pack(
                "fff",
                float(cs["d_max"]),
                float(cc["brown_dir"][0]),
                float(cc["brown_dir"][1]),
            )
            + b"\x00" * 4
        )

        t_data = (
            struct.pack(
                "ffff",
                float(lab.saturation),
                float(settings.toning.selenium_strength),
                float(settings.toning.sepia_strength),
                2.2,
            )
            + struct.pack(
                "iiIf",
                crop_offset[0],
                crop_offset[1],
                is_bw,
                float(settings.toning.gold_strength),
            )
            + struct.pack(
                "ffff",
                float(settings.toning.shadow_tint_hue),
                float(settings.toning.shadow_tint_strength),
                float(settings.toning.highlight_tint_hue),
                float(settings.toning.highlight_tint_strength),
            )
            + struct.pack(
                "fffI",
                float(settings.toning.blue_strength),
                float(settings.toning.copper_strength),
                float(settings.toning.vanadium_strength),
                _ALT_MODE.get(altproc.alt_process, 0) if is_bw else 0,
            )
        )

        if vignette_full_crop is None:
            v_full_w, v_full_h, v_off_x, v_off_y = crop_w, crop_h, 0, 0
        else:
            v_full_w, v_full_h, v_off_x, v_off_y = vignette_full_crop
        carrier_px = 0.0
        if settings.finish.carrier_width > 0.0:
            carrier_px = carrier_width_px(
                settings.finish.carrier_width,
                settings.export.export_print_size,
                float(max(v_full_w, v_full_h)),
            )
        paper = PrintService.effective_paper_linear(settings.finish, settings.toning)
        f_data = struct.pack(
            "fffffffffffffff",
            float(settings.finish.vignette_stops),
            float(settings.finish.vignette_size),
            float(settings.finish.vignette_roundness),
            float(v_full_w),
            float(v_full_h),
            float(v_off_x),
            float(v_off_y),
            float(carrier_px),
            float(settings.finish.carrier_rough),
            float(settings.finish.carrier_flare),
            float(is_bw),
            float(settings.finish.carrier_corner),
            float(paper[0]),
            float(paper[1]),
            float(paper[2]),
        )

        pw, ph, cw, ch, ox, oy, _ = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)
        color_hex = PrintService.effective_border_color(settings.finish, settings.toning).lstrip("#")
        bg = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        scale = float(cw) / max(1.0, float(crop_w))
        y_data = (
            struct.pack("ffffii", bg[0], bg[1], bg[2], 1.0, ox, oy) + struct.pack("iiii", cw, ch, crop_w, crop_h) + struct.pack("f", scale)
        )

        # ROI offset + crop dims for the density-histogram pass (tex_norm is uncropped).
        dh_data = struct.pack("IIII", crop_offset[0], crop_offset[1], crop_w, crop_h)

        full_buffer = bytearray()
        for name, d in zip(
            self._uniform_names, [g_data, n_data, e_data, tr_data, c_data, l_data, li_data, cy_data, t_data, f_data, y_data, dh_data]
        ):
            full_buffer += d + b"\x00" * (self._slot_bytes(name) - len(d))

        if not self.gpu.device:
            raise RuntimeError("GPU device lost")
        self.gpu.device.queue.write_buffer(self._buffers["unified_u"].buffer, 0, full_buffer)

    def _calculate_layout_dims(
        self, settings: WorkspaceConfig, cw: int, ch: int, size_ref: Optional[float]
    ) -> Tuple[int, int, int, int, int, int, int]:
        """Calculates final paper and image dimensions based on print settings.
        Returns (paper_w, paper_h, content_w, content_h, off_x, off_y, dpi)."""
        mode = settings.export.export_resolution_mode

        # Preview path: size_ref is the desired paper long edge. Derive virtual DPI from it
        # and print_size_cm so the border scales sensibly. Forces non-ORIGINAL math.
        if size_ref:
            dpi = int((size_ref * 2.54) / max(0.1, settings.export.export_print_size))
            paper_long_px = int(size_ref)
            mode = ExportResolutionMode.PRINT
        elif mode == ExportResolutionMode.TARGET_PX:
            dpi = PrintService.effective_dpi(settings.export)
            paper_long_px = max(1, int(settings.export.export_target_long_edge_px))
        else:
            dpi = settings.export.export_dpi
            paper_long_px = int((settings.export.export_print_size / 2.54) * dpi)

        border_px = int((settings.finish.border_size / 2.54) * dpi)
        weight = max(1.0, float(settings.finish.border_bottom_weight))
        border_bottom_px = int(border_px * weight)
        border_y_px = border_px + border_bottom_px

        if mode == ExportResolutionMode.ORIGINAL:
            content_w, content_h = cw, ch

            if settings.export.paper_aspect_ratio == AspectRatio.ORIGINAL:
                paper_w, paper_h = content_w + 2 * border_px, content_h + border_y_px
            else:
                try:
                    w_r, h_r = map(float, settings.export.paper_aspect_ratio.split(":"))
                    paper_ratio = w_r / h_r
                except Exception:
                    paper_ratio = cw / ch

                min_paper_w = content_w + 2 * border_px
                min_paper_h = content_h + border_y_px

                if (min_paper_w / min_paper_h) > paper_ratio:
                    paper_w = min_paper_w
                    paper_h = int(paper_w / paper_ratio)
                else:
                    paper_h = min_paper_h
                    paper_w = int(paper_h * paper_ratio)

            off_x = (paper_w - content_w) // 2
            off_y = PrintService.weighted_offset_y(paper_h, content_h, border_px, border_bottom_px)
        else:
            if settings.export.paper_aspect_ratio == AspectRatio.ORIGINAL:
                if cw >= ch:
                    content_w = max(1, paper_long_px - 2 * border_px)
                    content_h = max(1, int(ch * (content_w / cw)))
                else:
                    content_h = max(1, paper_long_px - border_y_px)
                    content_w = max(1, int(cw * (content_h / ch)))
                paper_w, paper_h = content_w + 2 * border_px, content_h + border_y_px
                off_x, off_y = border_px, border_px
            else:
                paper_w, paper_h = PrintService.paper_dims_from_long_edge(
                    paper_long_px,
                    settings.export.paper_aspect_ratio,
                    cw,
                    ch,
                )
                inner_w, inner_h = paper_w - 2 * border_px, paper_h - border_y_px
                scale = min(inner_w / cw, inner_h / ch)
                content_w, content_h = int(cw * scale), int(ch * scale)

                off_x = (paper_w - content_w) // 2
                off_y = PrintService.weighted_offset_y(paper_h, content_h, border_px, border_bottom_px)

        max_tex = APP_CONFIG.max_texture_size
        if max_tex is not None:
            long_edge = max(paper_w, paper_h)
            if long_edge > max_tex:
                s = max_tex / long_edge
                paper_w = max(1, int(paper_w * s))
                paper_h = max(1, int(paper_h * s))
                content_w = max(1, int(content_w * s))
                content_h = max(1, int(content_h * s))
                off_x = int(off_x * s)
                off_y = int(off_y * s)
                dpi = max(1, int(dpi * s))

        return paper_w, paper_h, content_w, content_h, off_x, off_y, dpi

    def _readback_clahe_cdf(self) -> np.ndarray:
        """Reads back the CLAHE CDF buffer from GPU."""
        device = self.gpu.device
        assert device is not None
        nbytes = 64 * HISTOGRAM_BINS * 4
        read_buf = device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self._buffers["clahe_c"].buffer, 0, read_buf, 0, nbytes)
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(read_buf.read_mapped(), dtype=np.float32).copy()
        read_buf.unmap()
        read_buf.destroy()
        return data

    def _readback_metrics(self) -> np.ndarray:
        """Synchronously reads back the flat metrics buffer (u32 words) from GPU."""
        device = self.gpu.device
        if not device:
            return np.zeros(METRICS_BUFFER_SIZE // 4, dtype=np.uint32)
        if self._metrics_staging is None:
            read_buf = device.create_buffer(
                size=METRICS_BUFFER_SIZE,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            self._metrics_staging = read_buf
        else:
            read_buf = self._metrics_staging
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self._buffers["metrics"].buffer, 0, read_buf, 0, METRICS_BUFFER_SIZE)
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(read_buf.read_mapped(), dtype=np.uint32).copy()
        read_buf.unmap()
        return data

    def _readback_downsampled(self, tex: GPUTexture) -> np.ndarray:
        """Reads back texture as float32 RGB array, handling hardware alignment."""
        device = self.gpu.device
        if not device:
            return np.zeros((1, 1, 3), dtype=np.float32)
        prb = (tex.width * 16 + 255) & ~255
        if self._downsample_staging is None or self._downsample_staging[:2] != (prb, tex.height):
            if self._downsample_staging is not None:
                self._downsample_staging[2].destroy()
            read_buf = device.create_buffer(
                size=prb * tex.height,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            self._downsample_staging = (prb, tex.height, read_buf)
        else:
            read_buf = self._downsample_staging[2]
        encoder = device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": tex.texture},
            {"buffer": read_buf, "bytes_per_row": prb, "rows_per_image": tex.height},
            (tex.width, tex.height, 1),
        )
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        try:
            raw = np.frombuffer(read_buf.read_mapped(), dtype=np.uint8).reshape((tex.height, prb))
            valid = raw[:, : tex.width * 16]
            result = valid.view(np.float32).reshape((tex.height, tex.width, 4))
            # The texture is already display-encoded (output_encode pass).
            return result[:, :, :3]
        finally:
            read_buf.unmap()

    def _dispatch_pass(self, encoder: Any, pipeline_name: str, bindings: list, w: int, h: int) -> None:
        """Configures and dispatches a compute pass."""
        pipeline = self._pipelines.get(pipeline_name)
        if pipeline is None:
            raise RuntimeError(f"Pipeline not initialized: {pipeline_name}")

        if not self.gpu.device:
            raise RuntimeError("GPU device lost")

        wg_x, wg_y = (16, 16) if pipeline_name in ["autocrop", "metrics", "clahe_hist", "density_hist"] else (8, 8)

        cache_key = (pipeline_name, tuple(_binding_identity(idx, res) for idx, res in bindings))
        bind_group = self._bind_group_cache.get(cache_key)
        if bind_group is None:
            entries = []
            for idx, res in bindings:
                if res is None:
                    raise ValueError(
                        f"Binding {idx} in pipeline '{pipeline_name}' is None. "
                        "This usually means a hardware resource was not properly initialized or has been destroyed."
                    )

                if isinstance(res, dict) and "buffer" in res:
                    if res["buffer"] is None:
                        raise ValueError(f"Buffer in binding {idx} ({pipeline_name}) is None")
                    entries.append({"binding": idx, "resource": res})
                elif isinstance(res, GPUBuffer):
                    if res.buffer is None:
                        raise ValueError(f"GPUBuffer in binding {idx} ({pipeline_name}) is None")
                    entries.append(
                        {
                            "binding": idx,
                            "resource": {
                                "buffer": res.buffer,
                                "offset": 0,
                                "size": res.buffer.size,
                            },
                        }
                    )
                else:
                    entries.append({"binding": idx, "resource": res})

            layout = self._bind_layout_cache.get(pipeline_name)
            if layout is None:
                layout = pipeline.get_bind_group_layout(0)
                self._bind_layout_cache[pipeline_name] = layout
            try:
                bind_group = self.gpu.device.create_bind_group(layout=layout, entries=entries)
            except Exception as e:
                logger.error(f"Failed to create bind group for {pipeline_name}: {e}")
                raise
            self._bind_group_cache[cache_key] = bind_group

        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)
        if pipeline_name in ["clahe_hist", "clahe_cdf"]:
            pass_enc.dispatch_workgroups(8, 8)
        else:
            pass_enc.dispatch_workgroups((w + wg_x - 1) // wg_x, (h + wg_y - 1) // wg_y)
        pass_enc.end()

    def process(
        self,
        img: np.ndarray,
        settings: WorkspaceConfig,
        scale_factor: float = 1.0,
        bounds_override: Optional[Any] = None,
        readback_metrics: bool = True,
        cam_xyz: Optional[list] = None,
        camera_wb: Optional[list] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """High-level processing entry point with automatic tiling."""
        self._init_resources()
        h, w = img.shape[:2]
        max_tex = self.gpu.limits.get("max_texture_dimension_2d", 8192)
        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        if w_rot > max_tex or h_rot > max_tex or (w * h > TILING_THRESHOLD_PX):
            return self._process_tiled(img, settings, scale_factor, bounds_override=bounds_override, cam_xyz=cam_xyz, camera_wb=camera_wb)
        tex_final, metrics = self.process_to_texture(
            img,
            settings,
            scale_factor=scale_factor,
            bounds_override=bounds_override,
            readback_metrics=readback_metrics,
            cam_xyz=cam_xyz,
            camera_wb=camera_wb,
        )
        return self._readback_downsampled(tex_final), metrics

    def _process_tiled(
        self,
        img: np.ndarray,
        settings: WorkspaceConfig,
        scale_factor: float,
        bounds_override: Optional[Any] = None,
        cam_xyz: Optional[list] = None,
        camera_wb: Optional[list] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes ultra-high resolution images using memory-efficient tiling."""
        h, w = img.shape[:2]

        # Tiles apply geometry on the CPU (shader uniform zeroed), so distortion too.
        k1_eff = settings.flatfield.k1 if settings.flatfield.apply else 0.0

        img_rot = img
        if settings.geometry.rotation != 0:
            img_rot = np.rot90(img_rot, k=settings.geometry.rotation)
        if settings.geometry.flip_horizontal:
            img_rot = np.fliplr(img_rot)
        if settings.geometry.flip_vertical:
            img_rot = np.flipud(img_rot)
        if settings.geometry.fine_rotation != 0.0:
            img_rot = apply_fine_rotation(img_rot, settings.geometry.fine_rotation)
        if k1_eff != 0.0:
            img_rot = apply_radial_distortion(img_rot, k1_eff)

        # Rasterise the dodge/burn EV map once at full post-geometry resolution; tiles
        # slice it directly, like IR above.
        # ponytail: mask vertices are distortion-mapped so centres land right, but the
        # feathered falloff is not re-warped. Negligible unless a mask sits at the frame
        # edge under strong k1. Rasterise on a warped grid if that combination matters.
        local_maps_rot: Optional[np.ndarray] = None
        if settings.local.masks:
            h_rot_full, w_rot_full = img_rot.shape[:2]
            local_maps_rot = compute_local_maps(
                settings.local,
                h_rot_full,
                w_rot_full,
                (h, w),
                rotation=settings.geometry.rotation,
                fine_rotation=settings.geometry.fine_rotation,
                flip_horizontal=settings.geometry.flip_horizontal,
                flip_vertical=settings.geometry.flip_vertical,
                distortion_k1=k1_eff,
            )

        preview_scale = APP_CONFIG.preview_render_size / max(h, w)
        img_small = cv2.resize(img, (int(w * preview_scale), int(h * preview_scale)))

        # Reuse the CDF from the last preview render when CLAHE settings are unchanged.
        reused_cdf: Optional[np.ndarray] = None
        if (
            settings.lab.clahe_strength > 0
            and self._last_settings is not None
            and self._last_settings.lab.clahe_strength == settings.lab.clahe_strength
        ):
            reused_cdf = self._readback_clahe_cdf()

        _, metrics_ref = self.process_to_texture(
            img_small, settings, scale_factor=scale_factor, clahe_cdf_override=reused_cdf, cam_xyz=cam_xyz, camera_wb=camera_wb
        )

        global_cdfs = reused_cdf if reused_cdf is not None else self._readback_clahe_cdf()

        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        if settings.geometry.crop_rect:
            roi = get_manual_rect_coords(
                (h_rot, w_rot),
                settings.geometry.crop_rect,
                offset_px=settings.geometry.autocrop_offset,
                scale_factor=scale_factor,
            )
        elif settings.geometry.autocrop_offset > 0:
            margin = settings.geometry.autocrop_offset * scale_factor
            roi = apply_margin_to_roi((0, h_rot, 0, w_rot), h_rot, w_rot, margin)
        else:
            roi = (0, h_rot, 0, w_rot)
        y1, y2, x1, x2 = roi
        crop_w, crop_h = x2 - x1, y2 - y1

        # All global meters read the same downsample + scaled ROI; compute lazily once.
        ah, aw = img_rot.shape[:2]
        a_scale = min(1.0, APP_CONFIG.preview_render_size / max(ah, aw))
        analysis_roi = (int(y1 * a_scale), int(y2 * a_scale), int(x1 * a_scale), int(x2 * a_scale))
        analysis_shape = (int(ah * a_scale), int(aw * a_scale))
        # Freehand analysis_rect wins over the crop ROI + centered buffer here too.
        meter_roi, meter_buffer = resolve_analysis_region(
            analysis_shape, analysis_roi, settings.process.analysis_buffer, settings.process.analysis_rect
        )
        analysis_small: Optional[np.ndarray] = None

        def _analysis_img() -> np.ndarray:
            nonlocal analysis_small
            if analysis_small is None:
                analysis_small = _downsample_for_analysis(img_rot, APP_CONFIG.preview_render_size)
            return analysis_small

        # Unmixed like the non-tiled path, lazily: skipped when bounds are locked and no
        # auto refs, anchor or textural range need it.
        unmix_m = effective_crosstalk_matrix(settings.process, settings.process.process_mode)
        prefiltered_cache: Optional[np.ndarray] = None

        def _prefiltered() -> np.ndarray:
            nonlocal prefiltered_cache
            if prefiltered_cache is None:
                prefiltered_cache = unmix_log_image(prefilter_log_grid(_analysis_img(), meter_roi, meter_buffer), unmix_m)
            return prefiltered_cache

        def _analyze_global_bounds() -> LogNegativeBounds:
            return analyze_log_exposure_bounds_from_log(
                _prefiltered(),
                None,
                0.0,
                process_mode=settings.process.process_mode,
                e6_normalize=settings.process.e6_normalize,
                percentile_clip=settings.process.luma_range_clip,
                color_clip=settings.process.color_range_clip,
            )

        if bounds_override:
            global_bounds = global_anchor_bounds = bounds_override
        else:
            global_bounds, global_base_bounds = resolve_bounds_detailed(settings.process, _analyze_global_bounds)
            global_anchor_bounds = luma_source_bounds(settings.process, global_base_bounds)

        global_shadow_refs = None
        global_neutral_axis = None
        if settings.exposure.cast_removal_strength > 0.0 and settings.process.process_mode == ProcessMode.C41:
            global_shadow_refs = measure_shadow_refs_from_log(_prefiltered(), None, 0.0)
            global_neutral_axis = measure_neutral_axis_from_log(_prefiltered(), global_bounds, None, 0.0)

        global_metered_anchor = None
        if settings.exposure.auto_exposure:
            global_metered_anchor = measure_anchor_from_log(_prefiltered(), global_anchor_bounds, None, 0.0)

        global_textural_range = None
        if settings.exposure.auto_normalize_contrast:
            global_textural_range = measure_textural_range_from_log(_prefiltered(), None, 0.0)

        paper_w, paper_h, content_w, content_h, off_x, off_y, _ = self._calculate_layout_dims(settings, crop_w, crop_h, None)
        full_source_res = np.zeros((crop_h, crop_w, 3), dtype=np.float32)

        # Defect repairs are baked into the source before the engine, so no stage here
        # samples beyond its own pixel for them and the halo owes them nothing.
        halo = TILE_HALO
        # The sharpen blur reads ±kernel-radius px, which outgrows TILE_HALO at large
        # radii and shows tile seams in the USM band. RL's influence spreads with the
        # iterations but decays geometrically, so 6x the kernel radius covers it. Capped
        # by the 512 ceiling below.
        if settings.lab.sharpen > 0:
            k_radius = len(gaussian_kernel_1d(settings.lab.sharpen_radius)) // 2
            mult = 6 if settings.lab.sharpen_method == SharpenMethod.RL else 1
            halo = max(halo, k_radius * mult)
        # Glow/halation taps reach up to their radius (max(., . * scale_factor) in
        # lab.wgsl). Without this the bloom seams at tile edges on big exports.
        if settings.lab.glow_amount > 0.0:
            halo = max(halo, int(np.ceil(max(3.0, 15.0 * scale_factor))))
        if settings.lab.halation_strength > 0.0:
            halo = max(halo, int(np.ceil(max(5.0, 25.0 * scale_factor))))
        halo = min(halo, 512)

        for ty in range(0, crop_h, TILE_SIZE):
            for tx in range(0, crop_w, TILE_SIZE):
                tw, th = min(TILE_SIZE, crop_w - tx), min(TILE_SIZE, crop_h - ty)
                ix1, iy1 = max(0, x1 + tx - halo), max(0, y1 + ty - halo)
                ix2, iy2 = (
                    min(w_rot, x1 + tx + tw + halo),
                    min(h_rot, y1 + ty + th + halo),
                )
                maps_tile = np.ascontiguousarray(local_maps_rot[iy1:iy2, ix1:ix2]) if local_maps_rot is not None else None
                ox, oy = x1 + tx - ix1, y1 + ty - iy1
                tile_res, _ = self.process_to_texture(
                    img_rot[iy1:iy2, ix1:ix2],
                    settings,
                    scale_factor=scale_factor,
                    tiling_mode=True,
                    bounds_override=global_bounds,
                    shadow_refs_override=global_shadow_refs,
                    metered_anchor_override=global_metered_anchor,
                    textural_range_override=global_textural_range,
                    neutral_axis_override=global_neutral_axis,
                    global_offset=(ix1, iy1),
                    full_dims=(w_rot, h_rot),
                    clahe_cdf_override=global_cdfs,
                    apply_layout=False,
                    vignette_full_crop=(crop_w, crop_h, tx - ox, ty - oy),
                    local_maps=maps_tile,
                    cam_xyz=cam_xyz,
                    camera_wb=camera_wb,
                )
                full_source_res[ty : ty + th, tx : tx + tw] = self._readback_downsampled(tile_res)[oy : oy + th, ox : ox + tw]

        # Mirrors PrintService.apply_layout: only INTER_AREA is area-correct on a shrink.
        shrinking = content_w < crop_w or content_h < crop_h
        scaled_content = (
            cv2.resize(
                full_source_res,
                (content_w, content_h),
                interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4,
            )
            if (content_w != crop_w or content_h != crop_h)
            else full_source_res
        )
        result = np.zeros((paper_h, paper_w, 3), dtype=np.float32)
        color_hex = settings.finish.border_color.lstrip("#")
        result[:] = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        result[off_y : off_y + content_h, off_x : off_x + content_w] = scaled_content
        return result, metrics_ref

    def cleanup(self, collect: bool = True, retain: Optional[GPUTexture] = None) -> None:
        """Evacuates the texture pool; optionally forces garbage collection.

        ``retain`` is handed to the caller instead: its pool key goes with it, so the
        next render allocates a fresh one rather than painting over borrowed pixels.
        """
        for tex in self._tex_cache.values():
            if tex is not retain:
                tex.destroy()
        self._tex_cache.clear()
        # Bind groups reference the destroyed views, so drop them.
        self._bind_group_cache.clear()
        self._bind_layout_cache.clear()
        self._uv_grid_cache = None
        # The stage textures a resume would paint onto are gone, local_ev included, so the
        # next frame must re-upload and start from stage 0.
        self._current_source_hash = None
        self._last_settings = None
        self._local_ev_key = None
        self._mask_tex_key = None
        if collect:
            gc.collect()
        logger.info("GPUEngine: VRAM resources released")

    def destroy_all(self) -> None:
        """Full resource teardown."""
        self.cleanup()
        if self._metrics_staging is not None:
            self._metrics_staging.destroy()
            self._metrics_staging = None
        if self._downsample_staging is not None:
            self._downsample_staging[2].destroy()
            self._downsample_staging = None
        for buf in self._buffers.values():
            buf.destroy()
        self._buffers.clear()
        self._pipelines.clear()
        self._sampler = None
        logger.info("GPUEngine: Engine decommissioned")
