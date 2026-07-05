import gc
import os
import struct
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import wgpu  # type: ignore

from negpy.domain.models import AspectRatio, ExportResolutionMode, WorkspaceConfig
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds,
    analyze_log_exposure_bounds_from_log,
    luma_source_bounds,
    luminance_density_range,
    measure_anchor,
    measure_anchor_from_log,
    measure_clip_fractions,
    measure_neutral_axis,
    measure_neutral_axis_from_log,
    measure_shadow_log_refs,
    measure_shadow_refs_from_log,
    measure_textural_range,
    resolve_crosstalk_matrix,
    unmix_log_image,
    measure_textural_range_from_log,
    prefilter_log_grid,
    resolve_bounds_detailed,
)
from negpy.features.geometry.logic import (
    AUTOCROP_DETECT_RES,
    apply_fine_rotation,
    apply_margin_to_roi,
    apply_radial_distortion,
    compute_distortion_scale,
    get_autocrop_coords,
    get_manual_rect_coords,
    map_coords_to_geometry,
)
from negpy.features.local.logic import compute_local_ev_map
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUBuffer, GPUTexture
from negpy.infrastructure.gpu.shader_loader import ShaderLoader
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.paths import get_resource_path
from negpy.services.export.print import PrintService
from negpy.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)

# Hardware constants
UNIFORM_ALIGNMENT_DEFAULT = 256
TILE_SIZE = 2048
TILE_HALO = 32
TILING_THRESHOLD_PX = 12_000_000
HISTOGRAM_BINS = 256
METRICS_BUFFER_SIZE = 4096

# Per-frame metrics clear; write_buffer copies at call time, so sharing is safe.
_METRICS_ZEROS = np.zeros(METRICS_BUFFER_SIZE // 4, dtype=np.uint32)


def _detect_autocrop_roi(img: np.ndarray, settings: WorkspaceConfig, h_rot: int, w_rot: int) -> Tuple[int, int, int, int]:
    """
    Computes the autocrop ROI on a detection-resolution copy, mirroring the CPU
    GeometryProcessor transform order (rot90 -> flips -> fine rotation), and
    returns it scaled to full post-rotation resolution (h_rot, w_rot).
    """
    h, w = img.shape[:2]
    det_s = min(1.0, AUTOCROP_DETECT_RES / max(h, w))
    if det_s < 1.0:
        tmp = cv2.resize(img, (max(1, round(w * det_s)), max(1, round(h * det_s))), interpolation=cv2.INTER_AREA)
    else:
        tmp = img
    if settings.geometry.rotation != 0:
        tmp = np.rot90(tmp, k=settings.geometry.rotation)
    if settings.geometry.flip_horizontal:
        tmp = np.fliplr(tmp)
    if settings.geometry.flip_vertical:
        tmp = np.flipud(tmp)
    tmp = np.ascontiguousarray(tmp.astype(np.float32, copy=False))
    if settings.geometry.fine_rotation != 0.0:
        tmp = apply_fine_rotation(tmp, settings.geometry.fine_rotation)
    roi_tmp = get_autocrop_coords(
        tmp,
        offset_px=settings.geometry.autocrop_offset,
        # Margin parity with CPU: (2+offset)*L/preview_size in det coords, upscaled
        # by full/L below, equals the CPU path's (2+offset)*context.scale_factor.
        scale_factor=max(tmp.shape[:2]) / APP_CONFIG.preview_render_size,
        target_ratio_str=settings.geometry.autocrop_ratio,
        mode=settings.geometry.autocrop_mode,
    )
    rh, rw = tmp.shape[:2]
    sy, sx = h_rot / rh, w_rot / rw
    return (
        int(roi_tmp[0] * sy),
        int(roi_tmp[1] * sy),
        int(roi_tmp[2] * sx),
        int(roi_tmp[3] * sx),
    )


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


def _analysis_cache_key(settings: WorkspaceConfig, analysis_source_hash: str) -> tuple:
    """Identity of the auto-exposure analysis for a frame. Mirrors the CPU base-stage
    cache key (engine.py) plus the fields that gate refs/anchor/textural, so it survives
    creative-slider drags but invalidates on anything the meter actually reads."""
    e = settings.exposure
    return (
        analysis_source_hash,
        settings.process,
        settings.geometry,
        e.cast_removal_strength > 0.0 or e.auto_cast_removal,
        e.auto_exposure,
        e.auto_normalize_contrast,
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
            "output_encode": get_resource_path(os.path.join("negpy", "features", "exposure", "shaders", "output_encode.wgsl")),
            "autocrop": get_resource_path(os.path.join("negpy", "features", "geometry", "shaders", "autocrop.wgsl")),
            "clahe_hist": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_hist.wgsl")),
            "clahe_cdf": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_cdf.wgsl")),
            "clahe_apply": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_apply.wgsl")),
            "retouch": get_resource_path(os.path.join("negpy", "features", "retouch", "shaders", "retouch.wgsl")),
            "lab": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "lab.wgsl")),
            "toning": get_resource_path(os.path.join("negpy", "features", "toning", "shaders", "toning.wgsl")),
            "finish": get_resource_path(os.path.join("negpy", "features", "finish", "shaders", "finish.wgsl")),
            "metrics": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "metrics.wgsl")),
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
            "clahe_u",
            "retouch_u",
            "lab",
            "toning",
            "finish",
            "layout",
        ]
        self._alignment = UNIFORM_ALIGNMENT_DEFAULT
        self._current_source_hash: Optional[str] = None
        # Once-per-source guard so the analysis timing log fires on load, not every slider.
        self._analysis_timing_hash: Optional[str] = None
        # (key, bounds, shadow_refs, metered_anchor, textural_range, neutral_axis) — per-source
        # meter cache so creative-slider previews don't re-run the analysis (see _analysis_*).
        self._analysis_cache: Optional[tuple] = None
        # (analysis_key, per-channel clipped fractions) for the scan-exposure warning.
        self._clip_cache: Optional[tuple] = None
        self._last_settings: Optional[WorkspaceConfig] = None
        self._last_scale_factor: float = 1.0
        self._pending_ir_buffer: Optional[np.ndarray] = None
        self._ir_upload_key: Optional[Tuple[int, Any, int, int]] = None

        # Bind groups reference resources, not contents, so they survive across frames;
        # cache and reuse (cleared in cleanup()). Saves ~28 wgpu calls per frame.
        self._bind_group_cache: Dict[Tuple, Any] = {}
        self._bind_layout_cache: Dict[str, Any] = {}

        # Persistent staging buffers — avoid create_buffer() on every readback
        self._metrics_staging: Optional[Any] = None
        # (prb, height, buffer) — reused when image size/rotation is unchanged
        self._downsample_staging: Optional[Tuple[int, int, Any]] = None

        # (key, grid) — pure function of geometry, reused across settled frames
        self._uv_grid_cache: Optional[Tuple[Tuple, np.ndarray]] = None

    def _detect_invalidated_stage(self, settings: WorkspaceConfig, scale_factor: float) -> int:
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
        if last.local != settings.local:
            return 1
        if last.lab.clahe_strength != settings.lab.clahe_strength:
            return 2
        if last.retouch != settings.retouch:
            return 3
        if last.lab != settings.lab:
            return 4
        if last.toning != settings.toning:
            return 6
        if last.finish != settings.finish:
            return 7
        if last.export != settings.export:
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
        t0 = time.perf_counter()
        device = self.gpu.device
        self._sampler = device.create_sampler(min_filter="linear", mag_filter="linear")

        hw_min = self.gpu.limits.get("min_uniform_buffer_offset_alignment", 256)
        self._alignment = max(256, hw_min)

        for name, path in self._shaders.items():
            self._pipelines[name] = self._create_pipeline(path)

        # Unified Uniform Buffer (UBO)
        self._buffers["unified_u"] = GPUBuffer(
            self._alignment * len(self._uniform_names),
            wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Storage buffers for intermediate metrics and CLAHE
        self._buffers["clahe_h"] = GPUBuffer(65536, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self._buffers["clahe_c"] = GPUBuffer(
            65536,
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self._buffers["retouch_s"] = GPUBuffer(8192, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
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

    def _get_uniform_binding(self, name: str) -> Dict[str, Any]:
        """Calculates UBO offset and size for a specific pipeline stage."""
        idx = self._uniform_names.index(name)
        sizes = {
            "geometry": 32,
            "normalization": 112,
            "exposure": 256,
            "clahe_u": 32,
            "retouch_u": 64,
            "lab": 96,
            "toning": 64,
            "finish": 32,
            "layout": 48,
        }
        return {
            "buffer": self._buffers["unified_u"].buffer,
            "offset": idx * self._alignment,
            "size": sizes[name],
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
        ir_buffer: Optional[np.ndarray] = None,
        vignette_full_crop: Optional[Tuple[int, int, int, int]] = None,
        local_ev: Optional[np.ndarray] = None,
        analysis_source_hash: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the full pipeline, returning a GPU texture and associated metrics.

        ``local_ev`` is a pre-rasterised dodge/burn EV map already in the
        post-geometry frame; tiled export passes a per-tile slice. When None and
        masks are present, it is computed here from ``settings.local``.
        """
        if not self.gpu.is_available:
            raise RuntimeError("GPU not available")
        self._pending_ir_buffer = ir_buffer
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
            start_stage = self._detect_invalidated_stage(settings, scale_factor)

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
            if settings.geometry.manual_crop_rect:
                roi = get_manual_rect_coords(
                    (h_rot, w_rot),
                    settings.geometry.manual_crop_rect,
                    orig_shape=orig_shape,
                    rotation_k=settings.geometry.rotation,
                    fine_rotation=settings.geometry.fine_rotation,
                    flip_horizontal=settings.geometry.flip_horizontal,
                    flip_vertical=settings.geometry.flip_vertical,
                    offset_px=settings.geometry.autocrop_offset,
                    scale_factor=scale_factor,
                    distortion_k1=k1_eff,
                )
            elif settings.geometry.auto_crop_enabled:
                roi = _detect_autocrop_roi(img, settings, h_rot, w_rot)
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
            and (settings.exposure.cast_removal_strength > 0.0 or settings.exposure.auto_cast_removal)
            and settings.process.process_mode == ProcessMode.C41
        )
        _roll_luma = settings.process.use_luma_average and settings.process.is_locked_initialized
        _roll_colour = settings.process.use_colour_average and settings.process.is_locked_initialized
        needs_bounds_analysis = not (bounds_override or (_roll_luma and _roll_colour) or settings.process.is_local_initialized)
        # Measure the anchor for the render when Auto Density is on, and for the
        # Analysis-panel stats on every preview (readback) regardless of toggle —
        # it's only *used* in the render when auto_exposure (see uniforms).
        needs_anchor = metered_anchor_override is None and not tiling_mode and (settings.exposure.auto_exposure or readback_metrics)
        needs_textural = textural_range_override is None and not tiling_mode and settings.exposure.auto_normalize_contrast

        analysis_source = None
        prefiltered = None
        unmix_m = resolve_crosstalk_matrix(settings.process.crosstalk_strength, settings.process.crosstalk_matrix)
        if needs_bounds_analysis or needs_refs or needs_anchor or needs_textural:
            # Use views to avoid copying the full-res image; crop to ROI first.
            analysis_source = img
            if settings.geometry.rotation != 0:
                analysis_source = np.rot90(analysis_source, k=settings.geometry.rotation)
            if settings.geometry.flip_horizontal:
                analysis_source = np.fliplr(analysis_source)
            if settings.geometry.flip_vertical:
                analysis_source = np.flipud(analysis_source)
            analysis_roi = roi if not tiling_mode else None
            if analysis_roi is not None:
                ay1, ay2, ax1, ax2 = analysis_roi
                analysis_source = np.ascontiguousarray(analysis_source[ay1:ay2, ax1:ax2])
            if settings.geometry.fine_rotation != 0.0:
                analysis_source = apply_fine_rotation(analysis_source, settings.geometry.fine_rotation)

            analysis_source = _downsample_for_analysis(analysis_source, APP_CONFIG.preview_render_size)
            # Shared prefilter, once for all five meters (ROI already applied).
            # Unmixed like the CPU path so every meter reads the unmixed film.
            prefiltered = unmix_log_image(prefilter_log_grid(analysis_source, None, settings.process.analysis_buffer), unmix_m)

        scan_clip_fractions = None
        if analysis_source is not None:
            scan_clip_fractions = measure_clip_fractions(analysis_source, None, settings.process.analysis_buffer)
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

        pw, ph, cw, ch, ox, oy = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)

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
        )
        self._update_retouch_storage(
            settings.retouch,
            (h, w),
            settings.geometry,
            global_offset,
            actual_full_dims,
            scale_factor,
            distortion_k1=k1_eff,
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
        tex_ret = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "ret",
        )
        tex_ir = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            "ir",
        )
        tex_lab = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "lab",
        )
        # Dodge/burn EV map feeds the exposure pass; a zero-initialized 1x1 dummy
        # keeps the bind group valid when no masks are active (ev_scale.w gates it).
        if settings.local.masks:
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
            if settings.local.masks:
                if local_ev is None:
                    local_ev = compute_local_ev_map(
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
                tex_local_ev.upload(np.stack([local_ev] * 3, axis=-1))
            self._dispatch_pass(
                enc,
                "exposure",
                [
                    (0, tex_norm.view),
                    (1, tex_expo.view),
                    (2, self._get_uniform_binding("exposure")),
                    (3, tex_local_ev.view),
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

        if start_stage <= 3:
            if settings.retouch.ir_dust_remove and self._pending_ir_buffer is not None:
                if tiling_mode:
                    # Tiled export pre-transforms + slices IR per tile in _process_tiled,
                    # so upload it as-is. Cache key doesn't help (each tile is a fresh slice).
                    ir_for_gpu = np.ascontiguousarray(self._pending_ir_buffer.astype(np.float32))
                    tex_ir.upload(np.stack([ir_for_gpu] * 3, axis=-1))
                    self._ir_upload_key = None
                else:
                    upload_key = (id(self._pending_ir_buffer), settings.geometry, w_rot, h_rot, k1_eff)
                    if upload_key != self._ir_upload_key:
                        ir_for_gpu = self._transform_ir_for_gpu(self._pending_ir_buffer, settings.geometry, w_rot, h_rot, k1_eff)
                        tex_ir.upload(np.stack([ir_for_gpu] * 3, axis=-1))
                        self._ir_upload_key = upload_key
            self._dispatch_pass(
                enc,
                "retouch",
                [
                    (0, prev_tex.view),
                    (1, tex_ret.view),
                    (2, self._get_uniform_binding("retouch_u")),
                    (3, self._buffers["retouch_s"]),
                    (4, tex_ir.view),
                ],
                w_rot,
                h_rot,
            )

        if start_stage <= 4:
            self._dispatch_pass(
                enc,
                "lab",
                [
                    (0, tex_ret.view),
                    (1, tex_lab.view),
                    (2, self._get_uniform_binding("lab")),
                ],
                w_rot,
                h_rot,
            )

        tex_pre_toning = tex_lab

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
                ],
                crop_w,
                crop_h,
            )
            tex_for_layout = tex_finish
        else:
            tex_for_layout = tex_toning

        if not tiling_mode and apply_layout:
            paper_w, paper_h, content_w, content_h, off_x, off_y = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)
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

        # Output transform: scene-linear -> display-encoded, so every consumer
        # (readback, display LUT) reads encoded data.
        tex_output = self._get_intermediate_texture(
            tex_final.width,
            tex_final.height,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
            "output_encoded",
        )
        self._dispatch_pass(enc, "output_encode", [(0, tex_final.view), (1, tex_output.view)], tex_final.width, tex_final.height)
        tex_final = tex_output

        device.queue.submit([enc.finish()])
        metrics: Dict[str, Any] = {
            "active_roi": roi,
            "base_positive": tex_final,
            "normalized_log": tex_norm,
            "content_rect": content_rect,
            "log_bounds": bounds,
            "log_bounds_base": base_bounds,
            "norm_density_range": luminance_density_range(bounds),
            "metered_anchor": metered_anchor,
            "textural_range": textural_range,
            "scan_clip_fractions": scan_clip_fractions,
        }

        if not tiling_mode and readback_metrics:
            metrics["histogram_raw"] = self._readback_metrics()
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
                    )
                    self._uv_grid_cache = (uv_key, uv_grid)
                    metrics["uv_grid"] = uv_grid
            except Exception as e:
                logger.error(f"GPU Engine metrics error: {e}")

        self._last_settings = settings
        self._last_scale_factor = scale_factor
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
        if tiling_mode:
            g_data = b"\x00" * 32

        f, c = bounds.floors, bounds.ceils
        mode_val = 0
        if settings.process.process_mode == ProcessMode.BW:
            mode_val = 1
        elif settings.process.process_mode == ProcessMode.E6:
            mode_val = 2

        # E6 mirrors the CPU path (NormalizationProcessor), which negates the offsets.
        offset_sign = -1.0 if mode_val == 2 else 1.0

        # Capture-side dye-unmix rows, resolved once per frame by the caller
        # (shared with NormalizationProcessor); identity when off.
        if unmix is None:
            unmix = np.eye(3)

        n_data = (
            struct.pack("ffff", f[0], f[1], f[2], 0.0)
            + struct.pack("ffff", c[0], c[1], c[2], 0.0)
            + struct.pack(
                "IIff",
                mode_val,
                (1 if settings.process.e6_normalize else 0),
                offset_sign * settings.process.white_point_offset,
                offset_sign * settings.process.black_point_offset,
            )
            + struct.pack("ffff", unmix[0, 0], unmix[0, 1], unmix[0, 2], 0.0)
            + struct.pack("ffff", unmix[1, 0], unmix[1, 1], unmix[1, 2], 0.0)
            + struct.pack("ffff", unmix[2, 0], unmix[2, 1], unmix[2, 2], 0.0)
        )

        from negpy.features.exposure.logic import (
            _reference_linear_value,
            effective_cast_strength,
            filtration_offsets,
            grade_coupled_shape,
            local_ev_scale,
            normalize_refs,
            paper_dmin_rgb,
            per_channel_curve_params,
        )
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS
        from negpy.features.exposure.normalization import LogNegativeBounds, luminance_density_range
        from negpy.features.exposure.papers import effective_constants, effective_paper_profile, resolve_dye_matrix

        exp = settings.exposure
        paper = effective_paper_profile(exp.paper_profile, settings.process.process_mode)
        pc = effective_constants(paper)  # tonal overrides; non-paper keys == EXPOSURE_CONSTANTS
        d_min = paper.d_min if exp.paper_dmin else 0.0
        # metered_anchor may be measured for stats even when auto_exposure is off;
        # only let it move the render when the toggle is on.
        render_anchor = metered_anchor if exp.auto_exposure else None
        lum_range = luminance_density_range(bounds)
        # Final bounds the shader normalizes with (after WP/BP offsets); shared by
        # the Cast Removal shadow refs, mirroring the CPU path.
        wp = offset_sign * settings.process.white_point_offset
        bp = offset_sign * settings.process.black_point_offset
        adj_floors = (f[0] + wp, f[1] + wp, f[2] + wp)
        adj_ceils = (c[0] + bp, c[1] + bp, c[2] + bp)
        shadow_refs_norm = None
        if shadow_refs is not None:
            shadow_refs_norm = normalize_refs(shadow_refs, adj_floors, adj_ceils)
        neutral_axis_norm = None
        cast_confidence = None
        if neutral_axis_refs is not None:
            mid_refs, sh_refs, hl_refs = neutral_axis_refs[0], neutral_axis_refs[1], neutral_axis_refs[2]
            cast_confidence = neutral_axis_refs[3]
            nf = lambda r: normalize_refs(r, adj_floors, adj_ceils) if r is not None else None  # noqa: E731
            neutral_axis_norm = (nf(mid_refs), nf(sh_refs), nf(hl_refs))
        strength = effective_cast_strength(exp.cast_removal_strength, exp.auto_cast_removal, cast_confidence)
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
        )
        cmy_m = EXPOSURE_CONSTANTS["cmy_max_density"]
        _toe_eff, _shoulder_eff = grade_coupled_shape(slopes[1], exp.toe, exp.shoulder)
        # Mirrors apply_characteristic_curve (absolute CC, paper base, dye mix).
        wb_offsets = filtration_offsets(
            (exp.wb_cyan, exp.wb_magenta, exp.wb_yellow),
            LogNegativeBounds(adj_floors, adj_ceils),
        )
        dmin_rgb = paper_dmin_rgb(d_min, paper)
        dye = resolve_dye_matrix(paper)
        dye_rows = np.eye(3) if dye is None else dye

        e_data = (
            struct.pack("ffff", pivots[0], pivots[1], pivots[2], 0.0)
            + struct.pack("ffff", slopes[0], slopes[1], slopes[2], 0.0)
            + struct.pack("ffff", curvatures[0], curvatures[1], curvatures[2], 0.0)
            + struct.pack("ffff", wb_offsets[0], wb_offsets[1], wb_offsets[2], 0.0)
            + struct.pack(
                "ffff",
                exp.shadow_cyan * cmy_m,
                exp.shadow_magenta * cmy_m,
                exp.shadow_yellow * cmy_m,
                0.0,
            )
            + struct.pack(
                "ffff",
                exp.highlight_cyan * cmy_m,
                exp.highlight_magenta * cmy_m,
                exp.highlight_yellow * cmy_m,
                0.0,
            )
            # Asymmetric H&D print-curve scalars; mirrors _apply_print_curve_kernel.
            + struct.pack(
                "14fI3fIf",
                _toe_eff * EXPOSURE_CONSTANTS["toe_shoulder_strength"],
                _shoulder_eff * EXPOSURE_CONSTANTS["toe_shoulder_strength"],
                exp.toe_width,
                exp.shoulder_width,
                d_min,
                pc["d_max"],
                pc["toe_sharpness_base"],
                pc["shoulder_sharpness_base"],
                pc["toeshoulder_width_ref"],
                pc["toe_height"],
                pc["shoulder_height"],
                pc["anchor_target_density"],
                float(EXPOSURE_CONSTANTS["flare_fraction"]) if exp.flare else 0.0,
                float(EXPOSURE_CONSTANTS["target_system_gamma"]) if exp.surround else 1.0,
                mode_val,
                _reference_linear_value(d_min, paper),
                float(pc["paper_midtone_gamma"]),
                float(pc["paper_gamma_width"]),
                1 if dye is not None else 0,
                0.0,  # pad to 16B before the vec4s
            )
            + struct.pack("ffff", dmin_rgb[0], dmin_rgb[1], dmin_rgb[2], 0.0)
            + struct.pack("ffff", dye_rows[0, 0], dye_rows[0, 1], dye_rows[0, 2], 0.0)
            + struct.pack("ffff", dye_rows[1, 0], dye_rows[1, 1], dye_rows[1, 2], 0.0)
            + struct.pack("ffff", dye_rows[2, 0], dye_rows[2, 1], dye_rows[2, 2], 0.0)
            # Dodge/burn EV-stop size per channel (local_ev_scale); w = enable flag.
            + struct.pack("ffff", *local_ev_scale(LogNegativeBounds(adj_floors, adj_ceils)), 1.0 if settings.local.masks else 0.0)
        )

        cls = float(settings.lab.clahe_strength)
        c_data = (
            struct.pack(
                "ffiiii",
                cls,
                max(1.0, cls * 2.5),
                offset[0],
                offset[1],
                full_dims[0],
                full_dims[1],
            )
            + b"\x00" * 8
        )

        ret = settings.retouch
        ir_active = 1 if (ret.ir_dust_remove and self._pending_ir_buffer is not None) else 0
        r_u_data = struct.pack(
            "ffIIiiIIfIff",
            float(ret.dust_threshold),
            float(ret.dust_size),
            len(ret.manual_dust_spots),
            (1 if ret.dust_remove else 0),
            offset[0],
            offset[1],
            full_dims[0],
            full_dims[1],
            float(scale_factor),
            ir_active,
            float(1.0 - ret.ir_threshold),
            float(ret.ir_inpaint_radius),
        )

        lab = settings.lab
        l_data = (
            struct.pack(
                "ffffff",
                float(lab.sharpen),
                float(lab.chroma_denoise),
                float(lab.saturation),
                float(lab.vibrance),
                float(lab.glow_amount),
                float(lab.halation_strength),
            )
            + b"\x00" * 8
        )

        is_bw = 1 if settings.process.process_mode == ProcessMode.BW else 0
        t_data = (
            struct.pack(
                "ffff",
                float(lab.saturation),
                float(settings.toning.selenium_strength),
                float(settings.toning.sepia_strength),
                2.2,
            )
            + struct.pack("iiIf", crop_offset[0], crop_offset[1], is_bw, 0.0)
            + struct.pack(
                "ffff",
                float(settings.toning.shadow_tint_hue),
                float(settings.toning.shadow_tint_strength),
                float(settings.toning.highlight_tint_hue),
                float(settings.toning.highlight_tint_strength),
            )
        )

        if vignette_full_crop is None:
            v_full_w, v_full_h, v_off_x, v_off_y = crop_w, crop_h, 0, 0
        else:
            v_full_w, v_full_h, v_off_x, v_off_y = vignette_full_crop
        f_data = (
            struct.pack(
                "ffffff",
                float(settings.finish.vignette_strength),
                float(settings.finish.vignette_size),
                float(v_full_w),
                float(v_full_h),
                float(v_off_x),
                float(v_off_y),
            )
            + b"\x00" * 8
        )

        pw, ph, cw, ch, ox, oy = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)
        color_hex = settings.finish.border_color.lstrip("#")
        bg = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        scale = float(cw) / max(1.0, float(crop_w))
        y_data = (
            struct.pack("ffffii", bg[0], bg[1], bg[2], 1.0, ox, oy)
            + struct.pack("iiii", cw, ch, crop_w, crop_h)
            + struct.pack("f", scale)
            + b"\x00" * 4
        )

        full_buffer = bytearray()
        for d in [g_data, n_data, e_data, c_data, r_u_data, l_data, t_data, f_data, y_data]:
            full_buffer += d + b"\x00" * (self._alignment - len(d))

        if not self.gpu.device:
            raise RuntimeError("GPU device lost")
        self.gpu.device.queue.write_buffer(self._buffers["unified_u"].buffer, 0, full_buffer)

    def _transform_ir_for_gpu(
        self,
        ir_raw: np.ndarray,
        geom: Any,
        w_rot: int,
        h_rot: int,
        distortion_k1: float = 0.0,
    ) -> np.ndarray:
        """CPU-transforms the IR sidecar (rotation, flip, fine rotation, lens distortion)
        so it aligns with the geometry-transformed RGB texture the retouch shader samples."""
        import cv2

        ir = ir_raw
        if geom.rotation % 4 != 0:
            ir = np.rot90(ir, k=geom.rotation % 4)
        if geom.flip_horizontal:
            ir = np.fliplr(ir)
        if geom.flip_vertical:
            ir = np.flipud(ir)
        ir = np.ascontiguousarray(ir.astype(np.float32))
        if geom.fine_rotation != 0.0:
            ir = apply_fine_rotation(ir, geom.fine_rotation)
        if distortion_k1 != 0.0:
            ir = apply_radial_distortion(ir, distortion_k1)
        if ir.shape[:2] != (h_rot, w_rot):
            ir = cv2.resize(ir, (w_rot, h_rot), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(ir.astype(np.float32))

    def _update_retouch_storage(
        self,
        conf: Any,
        orig_shape: Tuple[int, int],
        geom: Any,
        offset: Tuple[int, int],
        full_dims: Tuple[int, int],
        scale_factor: float,
        distortion_k1: float = 0.0,
    ) -> None:
        """Uploads manual retouch spots to GPU storage buffer."""
        spot_data = bytearray()
        for x, y, size in conf.manual_dust_spots[:512]:
            mx, my = map_coords_to_geometry(
                x,
                y,
                orig_shape,
                geom.rotation,
                geom.fine_rotation,
                geom.flip_horizontal,
                geom.flip_vertical,
                distortion_k1=distortion_k1,
            )
            # Correctly scale radius using scale_factor
            scaled_radius = (size * scale_factor) / max(orig_shape)
            spot_data += struct.pack("ffff", mx, my, scaled_radius, 0.0)
        if spot_data:
            self._buffers["retouch_s"].upload(np.frombuffer(spot_data, dtype=np.uint8))

    def _calculate_layout_dims(
        self, settings: WorkspaceConfig, cw: int, ch: int, size_ref: Optional[float]
    ) -> Tuple[int, int, int, int, int, int]:
        """Calculates final paper and image dimensions based on print settings."""
        mode = settings.export.export_resolution_mode

        # Preview path: size_ref is the desired paper long-edge; derive virtual DPI
        # from it + print_size_cm so border scales sensibly. Forces non-ORIGINAL math.
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

        if mode == ExportResolutionMode.ORIGINAL:
            content_w, content_h = cw, ch

            if settings.export.paper_aspect_ratio == AspectRatio.ORIGINAL:
                paper_w, paper_h = content_w + 2 * border_px, content_h + 2 * border_px
            else:
                try:
                    w_r, h_r = map(float, settings.export.paper_aspect_ratio.split(":"))
                    paper_ratio = w_r / h_r
                except Exception:
                    paper_ratio = cw / ch

                min_paper_w = content_w + 2 * border_px
                min_paper_h = content_h + 2 * border_px

                if (min_paper_w / min_paper_h) > paper_ratio:
                    paper_w = min_paper_w
                    paper_h = int(paper_w / paper_ratio)
                else:
                    paper_h = min_paper_h
                    paper_w = int(paper_h * paper_ratio)

            off_x, off_y = (paper_w - content_w) // 2, (paper_h - content_h) // 2
        else:
            if settings.export.paper_aspect_ratio == AspectRatio.ORIGINAL:
                content_long_px = max(1, paper_long_px - 2 * border_px)
                if cw >= ch:
                    content_w = content_long_px
                    content_h = max(1, int(ch * (content_long_px / cw)))
                else:
                    content_h = content_long_px
                    content_w = max(1, int(cw * (content_long_px / ch)))
                paper_w, paper_h = content_w + 2 * border_px, content_h + 2 * border_px
                off_x, off_y = border_px, border_px
            else:
                paper_w, paper_h = PrintService.paper_dims_from_long_edge(
                    paper_long_px,
                    settings.export.paper_aspect_ratio,
                    cw,
                    ch,
                )
                inner_w, inner_h = paper_w - 2 * border_px, paper_h - 2 * border_px
                scale = min(inner_w / cw, inner_h / ch)
                content_w, content_h = int(cw * scale), int(ch * scale)

                off_x, off_y = (paper_w - content_w) // 2, (paper_h - content_h) // 2

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

        return paper_w, paper_h, content_w, content_h, off_x, off_y

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
        """Synchronously reads back histogram data from GPU."""
        device = self.gpu.device
        if not device:
            return np.zeros((4, HISTOGRAM_BINS), dtype=np.uint32)
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
        return data.reshape((4, HISTOGRAM_BINS))

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

        wg_x, wg_y = (16, 16) if pipeline_name in ["autocrop", "metrics", "clahe_hist"] else (8, 8)

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
        ir_buffer: Optional[np.ndarray] = None,
        readback_metrics: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """High-level processing entry point with automatic tiling."""
        self._init_resources()
        h, w = img.shape[:2]
        max_tex = self.gpu.limits.get("max_texture_dimension_2d", 8192)
        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        if w_rot > max_tex or h_rot > max_tex or (w * h > TILING_THRESHOLD_PX):
            return self._process_tiled(img, settings, scale_factor, bounds_override=bounds_override, ir_buffer=ir_buffer)
        tex_final, metrics = self.process_to_texture(
            img,
            settings,
            scale_factor=scale_factor,
            bounds_override=bounds_override,
            ir_buffer=ir_buffer,
            readback_metrics=readback_metrics,
        )
        return self._readback_downsampled(tex_final), metrics

    def _process_tiled(
        self,
        img: np.ndarray,
        settings: WorkspaceConfig,
        scale_factor: float,
        bounds_override: Optional[Any] = None,
        ir_buffer: Optional[np.ndarray] = None,
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

        # Pre-transform IR once into the post-geometry frame; tiles slice it directly.
        ir_rot: Optional[np.ndarray] = None
        if ir_buffer is not None and settings.retouch.ir_dust_remove:
            h_rot_full, w_rot_full = img_rot.shape[:2]
            try:
                ir_rot = self._transform_ir_for_gpu(ir_buffer, settings.geometry, w_rot_full, h_rot_full, k1_eff)
            except Exception as e:
                logger.warning(f"IR pre-transform failed for tiled export; skipping IR dust removal: {e}")
                ir_rot = None

        # Rasterise the dodge/burn EV map once at full post-geometry resolution;
        # tiles slice it directly (same pattern as IR above).
        # ponytail: mask vertices are distortion-mapped (centres land right), but the
        # feathered falloff isn't re-warped — negligible unless a mask sits at the frame
        # edge under strong k1. Rasterise on a warped grid if that combo ever matters.
        local_ev_rot: Optional[np.ndarray] = None
        if settings.local.masks:
            h_rot_full, w_rot_full = img_rot.shape[:2]
            local_ev_rot = compute_local_ev_map(
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

        _, metrics_ref = self.process_to_texture(img_small, settings, scale_factor=scale_factor, clahe_cdf_override=reused_cdf)

        global_cdfs = reused_cdf if reused_cdf is not None else self._readback_clahe_cdf()

        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        if settings.geometry.manual_crop_rect:
            roi = get_manual_rect_coords(
                (h_rot, w_rot),
                settings.geometry.manual_crop_rect,
                orig_shape=(h, w),
                rotation_k=settings.geometry.rotation,
                fine_rotation=settings.geometry.fine_rotation,
                flip_horizontal=settings.geometry.flip_horizontal,
                flip_vertical=settings.geometry.flip_vertical,
                offset_px=settings.geometry.autocrop_offset,
                scale_factor=scale_factor,
                distortion_k1=k1_eff,
            )
        elif settings.geometry.auto_crop_enabled:
            roi = _detect_autocrop_roi(img, settings, h_rot, w_rot)
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
        analysis_small: Optional[np.ndarray] = None

        def _analysis_img() -> np.ndarray:
            nonlocal analysis_small
            if analysis_small is None:
                analysis_small = _downsample_for_analysis(img_rot, APP_CONFIG.preview_render_size)
            return analysis_small

        def _analyze_global_bounds() -> LogNegativeBounds:
            return analyze_log_exposure_bounds(
                _analysis_img(),
                roi=analysis_roi,
                analysis_buffer=settings.process.analysis_buffer,
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
        if (
            settings.exposure.cast_removal_strength > 0.0 or settings.exposure.auto_cast_removal
        ) and settings.process.process_mode == ProcessMode.C41:
            # Tiles must share one global measurement, like global_bounds.
            global_shadow_refs = measure_shadow_log_refs(
                _analysis_img(),
                roi=analysis_roi,
                analysis_buffer=settings.process.analysis_buffer,
            )
            global_neutral_axis = measure_neutral_axis(
                _analysis_img(),
                global_bounds,
                roi=analysis_roi,
                analysis_buffer=settings.process.analysis_buffer,
            )

        global_metered_anchor = None
        if settings.exposure.auto_exposure:
            # Tiles must share one global anchor, like global_bounds/shadow_refs.
            global_metered_anchor = measure_anchor(
                _analysis_img(),
                global_anchor_bounds,
                roi=analysis_roi,
                analysis_buffer=settings.process.analysis_buffer,
            )

        global_textural_range = None
        if settings.exposure.auto_normalize_contrast:
            # Tiles must share one global textural range, like global_bounds.
            global_textural_range = measure_textural_range(
                _analysis_img(),
                roi=analysis_roi,
                analysis_buffer=settings.process.analysis_buffer,
            )

        paper_w, paper_h, content_w, content_h, off_x, off_y = self._calculate_layout_dims(settings, crop_w, crop_h, None)
        full_source_res = np.zeros((crop_h, crop_w, 3), dtype=np.float32)

        for ty in range(0, crop_h, TILE_SIZE):
            for tx in range(0, crop_w, TILE_SIZE):
                tw, th = min(TILE_SIZE, crop_w - tx), min(TILE_SIZE, crop_h - ty)
                ix1, iy1 = max(0, x1 + tx - TILE_HALO), max(0, y1 + ty - TILE_HALO)
                ix2, iy2 = (
                    min(w_rot, x1 + tx + tw + TILE_HALO),
                    min(h_rot, y1 + ty + th + TILE_HALO),
                )
                ir_tile = np.ascontiguousarray(ir_rot[iy1:iy2, ix1:ix2]) if ir_rot is not None else None
                ev_tile = np.ascontiguousarray(local_ev_rot[iy1:iy2, ix1:ix2]) if local_ev_rot is not None else None
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
                    ir_buffer=ir_tile,
                    vignette_full_crop=(crop_w, crop_h, tx - ox, ty - oy),
                    local_ev=ev_tile,
                )
                full_source_res[ty : ty + th, tx : tx + tw] = self._readback_downsampled(tile_res)[oy : oy + th, ox : ox + tw]

        scaled_content = (
            cv2.resize(full_source_res, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
            if (content_w != crop_w or content_h != crop_h)
            else full_source_res
        )
        result = np.zeros((paper_h, paper_w, 3), dtype=np.float32)
        color_hex = settings.finish.border_color.lstrip("#")
        result[:] = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        result[off_y : off_y + content_h, off_x : off_x + content_w] = scaled_content
        return result, metrics_ref

    def cleanup(self, collect: bool = True) -> None:
        """Evacuates the texture pool; optionally forces garbage collection."""
        for tex in self._tex_cache.values():
            tex.destroy()
        self._tex_cache.clear()
        # Bind groups reference the destroyed views — drop them.
        self._bind_group_cache.clear()
        self._bind_layout_cache.clear()
        self._ir_upload_key = None
        self._uv_grid_cache = None
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
