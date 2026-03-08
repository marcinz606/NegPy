import os
import struct
import numpy as np
import wgpu  # type: ignore
import cv2
import gc
from typing import Any, Optional, Dict, Tuple

from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture, GPUBuffer
from negpy.infrastructure.gpu.shader_loader import ShaderLoader
from negpy.features.process.models import ProcessMode
from negpy.domain.models import WorkspaceConfig, AspectRatio
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.paths import get_resource_path
from negpy.features.geometry.logic import (
    get_manual_rect_coords,
    get_autocrop_coords,
    map_coords_to_geometry,
    apply_fine_rotation,
)
from negpy.features.exposure.normalization import (
    analyze_log_exposure_bounds,
    LogNegativeBounds,
)
from negpy.services.view.coordinate_mapping import CoordinateMapping
from negpy.services.export.print import PrintService

logger = get_logger(__name__)

# Hardware constants
UNIFORM_ALIGNMENT_DEFAULT = 256
TILE_SIZE = 2048
TILE_HALO = 32
TILING_THRESHOLD_PX = 12_000_000
HISTOGRAM_BINS = 256
METRICS_BUFFER_SIZE = 4096


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
            "autocrop": get_resource_path(os.path.join("negpy", "features", "geometry", "shaders", "autocrop.wgsl")),
            "clahe_hist": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_hist.wgsl")),
            "clahe_cdf": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_cdf.wgsl")),
            "clahe_apply": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "clahe_apply.wgsl")),
            "retouch": get_resource_path(os.path.join("negpy", "features", "retouch", "shaders", "retouch.wgsl")),
            "lab": get_resource_path(os.path.join("negpy", "features", "lab", "shaders", "lab.wgsl")),
            "toning": get_resource_path(os.path.join("negpy", "features", "toning", "shaders", "toning.wgsl")),
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
            "layout",
        ]
        self._alignment = UNIFORM_ALIGNMENT_DEFAULT
        self._current_source_hash: Optional[str] = None
        self._last_settings: Optional[WorkspaceConfig] = None
        self._last_scale_factor: float = 1.0

    def _detect_invalidated_stage(self, settings: WorkspaceConfig, scale_factor: float) -> int:
        """
        Determines the earliest pipeline stage that needs re-running.
        Returns stage index:
        0: Geometry (Source/Transform)
        1: Exposure (Normalization/Grading)
        2: CLAHE (Adaptive Hist)
        3: Retouch (Healing)
        4: Lab (Color/Sharpen)
        5: Toning (Paper/Split)
        6: Layout (Final compositing)
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
        if last.process != settings.process or last.exposure != settings.exposure:
            return 1
        if last.lab.clahe_strength != settings.lab.clahe_strength:
            return 2
        if last.retouch != settings.retouch:
            return 3
        if last.lab != settings.lab:
            return 4
        if last.toning != settings.toning:
            return 5
        if last.export != settings.export:
            return 6

        return 7  # Nothing changed

    def _get_intermediate_texture(self, w: int, h: int, usage: int, label: str) -> GPUTexture:
        """Retrieves or creates a texture from the pool."""
        key = (w, h, usage, label)
        if key not in self._tex_cache:
            self._tex_cache[key] = GPUTexture(w, h, usage=usage)
        return self._tex_cache[key]

    def _init_resources(self) -> None:
        """Initializes hardware pipelines and persistent buffers."""
        if self._pipelines or not self.gpu.device:
            return
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

        logger.info("GPU Engine: Hardware resources initialized")

    def _create_pipeline(self, shader_path: str) -> Any:
        shader_module = ShaderLoader.load(shader_path)
        assert self.gpu.device is not None
        return self.gpu.device.create_compute_pipeline(layout="auto", compute={"module": shader_module, "entry_point": "main"})

    def _get_uniform_binding(self, name: str) -> Dict[str, Any]:
        """Calculates UBO offset and size for a specific pipeline stage."""
        idx = self._uniform_names.index(name)
        sizes = {
            "geometry": 32,
            "normalization": 112,
            "exposure": 160,
            "clahe_u": 32,
            "retouch_u": 40,
            "lab": 96,
            "toning": 48,
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
        cast_override: Optional[Any] = None,
        global_offset: Tuple[int, int] = (0, 0),
        full_dims: Optional[Tuple[int, int]] = None,
        clahe_cdf_override: Optional[np.ndarray] = None,
        apply_layout: bool = True,
        render_size_ref: Optional[float] = None,
        source_hash: Optional[str] = None,
        readback_metrics: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the full pipeline, returning a GPU texture and associated metrics.
        """
        if not self.gpu.is_available:
            raise RuntimeError("GPU not available")
        self._init_resources()
        device = self.gpu.device
        assert device is not None

        h, w = img.shape[:2]
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
            roi = (0, h, 0, w)
        else:
            rot = settings.geometry.rotation % 4
            w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
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
                )
            else:
                det_s = APP_CONFIG.preview_render_size / max(h, w)
                tmp = cv2.resize(img, (int(w * det_s), int(h * det_s)))
                if settings.geometry.rotation != 0:
                    tmp = np.rot90(tmp, k=settings.geometry.rotation)
                if settings.geometry.flip_horizontal:
                    tmp = np.fliplr(tmp)
                if settings.geometry.flip_vertical:
                    tmp = np.flipud(tmp)
                roi_tmp = get_autocrop_coords(
                    tmp.astype(np.float32),
                    offset_px=settings.geometry.autocrop_offset,
                    scale_factor=scale_factor,
                    target_ratio_str=settings.geometry.autocrop_ratio,
                )
                rh, rw = tmp.shape[:2]
                sy, sx = h_rot / rh, w_rot / rw
                roi = (
                    int(roi_tmp[0] * sy),
                    int(roi_tmp[1] * sy),
                    int(roi_tmp[2] * sx),
                    int(roi_tmp[3] * sx),
                )
            y1, y2, x1, x2 = roi
            crop_w, crop_h = max(1, x2 - x1), max(1, y2 - y1)

        if bounds_override:
            bounds = bounds_override
        elif settings.process.use_roll_average and settings.process.is_locked_initialized:
            bounds = LogNegativeBounds(
                floors=settings.process.locked_floors,
                ceils=settings.process.locked_ceils,
            )
        elif settings.process.is_local_initialized:
            bounds = LogNegativeBounds(
                floors=settings.process.local_floors,
                ceils=settings.process.local_ceils,
            )
        else:
            analysis_source = img.copy()
            if settings.geometry.rotation != 0:
                analysis_source = np.rot90(analysis_source, k=settings.geometry.rotation)
            if settings.geometry.flip_horizontal:
                analysis_source = np.fliplr(analysis_source)
            if settings.geometry.flip_vertical:
                analysis_source = np.flipud(analysis_source)
            if settings.geometry.fine_rotation != 0.0:
                analysis_source = apply_fine_rotation(analysis_source, settings.geometry.fine_rotation)

            bounds = analyze_log_exposure_bounds(
                analysis_source,
                roi if not tiling_mode else None,
                settings.process.analysis_buffer,
                process_mode=settings.process.process_mode,
                e6_normalize=settings.process.e6_normalize,
            )

        cast = (0.0, 0.0, 0.0)
        if cast_override:
            cast = cast_override
        elif settings.process.shadow_cast_strength > 0:
            if settings.process.use_roll_average:
                cast = settings.process.locked_shadow_cast
            elif any(v != 0.0 for v in settings.process.local_shadow_cast):
                cast = settings.process.local_shadow_cast
            else:
                from negpy.features.exposure.normalization import normalize_log_image
                from negpy.features.exposure.shadows import analyze_shadow_cast

                epsilon = 1e-6

                analysis_source = img.copy()
                if settings.geometry.rotation != 0:
                    analysis_source = np.rot90(analysis_source, k=settings.geometry.rotation)
                if settings.geometry.flip_horizontal:
                    analysis_source = np.fliplr(analysis_source)
                if settings.geometry.flip_vertical:
                    analysis_source = np.flipud(analysis_source)
                if settings.geometry.fine_rotation != 0.0:
                    analysis_source = apply_fine_rotation(analysis_source, settings.geometry.fine_rotation)

                if not tiling_mode and roi:
                    ry1, ry2, rx1, rx2 = roi
                    analysis_source = analysis_source[ry1:ry2, rx1:rx2]

                img_log = np.log10(np.clip(analysis_source, epsilon, 1.0))
                res_norm = normalize_log_image(img_log, bounds)
                cast = analyze_shadow_cast(res_norm, settings.process.shadow_cast_threshold)

        pw, ph, cw, ch, ox, oy = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)

        self._upload_unified_uniforms(
            settings,
            bounds,
            cast,
            global_offset,
            actual_full_dims,
            (0, 0) if tiling_mode else (x1, y1),
            crop_w,
            crop_h,
            tiling_mode,
            render_size_ref,
            scale_factor,
        )
        self._update_retouch_storage(
            settings.retouch,
            (h, w),
            settings.geometry,
            global_offset,
            actual_full_dims,
            scale_factor,
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
        tex_lab = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "lab",
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
            self._dispatch_pass(
                enc,
                "exposure",
                [
                    (0, tex_norm.view),
                    (1, tex_expo.view),
                    (2, self._get_uniform_binding("exposure")),
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
            self._dispatch_pass(
                enc,
                "retouch",
                [
                    (0, prev_tex.view),
                    (1, tex_ret.view),
                    (2, self._get_uniform_binding("retouch_u")),
                    (3, self._buffers["retouch_s"]),
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

        if start_stage <= 5:
            self._dispatch_pass(
                enc,
                "toning",
                [
                    (0, tex_lab.view),
                    (1, tex_toning.view),
                    (2, self._get_uniform_binding("toning")),
                ],
                crop_w,
                crop_h,
            )

        if not tiling_mode and apply_layout:
            paper_w, paper_h, content_w, content_h, off_x, off_y = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)
            tex_final = self._get_intermediate_texture(
                paper_w,
                paper_h,
                wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
                "final",
            )
            if start_stage <= 6:
                self._dispatch_pass(
                    enc,
                    "layout",
                    [
                        (0, tex_toning.view),
                        (1, tex_final.view),
                        (2, self._get_uniform_binding("layout")),
                    ],
                    paper_w,
                    paper_h,
                )
            content_rect = (off_x, off_y, content_w, content_h)
        else:
            tex_final, content_rect = tex_toning, (0, 0, crop_w, crop_h)

        if not tiling_mode and readback_metrics:
            device.queue.write_buffer(self._buffers["metrics"].buffer, 0, np.zeros(1024, dtype=np.uint32))
            self._dispatch_pass(
                enc,
                "metrics",
                [(0, tex_final.view), (1, self._buffers["metrics"])],
                tex_final.width,
                tex_final.height,
            )

        device.queue.submit([enc.finish()])
        metrics: Dict[str, Any] = {
            "active_roi": roi,
            "base_positive": tex_final,
            "normalized_log": tex_norm,
            "content_rect": content_rect,
            "log_bounds": bounds,
            "shadow_cast": cast,
        }

        if not tiling_mode and readback_metrics:
            metrics["histogram_raw"] = self._readback_metrics()
            try:
                metrics["uv_grid"] = CoordinateMapping.create_uv_grid(
                    rh_orig=h,
                    rw_orig=w,
                    rotation=settings.geometry.rotation,
                    fine_rot=settings.geometry.fine_rotation,
                    flip_h=settings.geometry.flip_horizontal,
                    flip_v=settings.geometry.flip_vertical,
                    autocrop=True,
                    autocrop_params={"roi": roi} if roi else None,
                )
            except Exception as e:
                logger.error(f"GPU Engine metrics error: {e}")

        self._last_settings = settings
        self._last_scale_factor = scale_factor
        return tex_final, metrics

    def _upload_unified_uniforms(
        self,
        settings: WorkspaceConfig,
        bounds: Any,
        cast: Any,
        offset: Tuple[int, int],
        full_dims: Tuple[int, int],
        crop_offset: Tuple[int, int],
        crop_w: int,
        crop_h: int,
        tiling_mode: bool,
        render_size_ref: Optional[float],
        scale_factor: float,
    ) -> None:
        """Packs and uploads all pipeline parameters to the unified UBO."""
        g_data = (
            struct.pack(
                "ifii",
                int(settings.geometry.rotation),
                float(settings.geometry.fine_rotation),
                (1 if settings.geometry.flip_horizontal else 0),
                (1 if settings.geometry.flip_vertical else 0),
            )
            + b"\x00" * 16
        )
        if tiling_mode:
            g_data = b"\x00" * 32

        f, c = bounds.floors, bounds.ceils
        mode_val = 0
        if settings.process.process_mode == ProcessMode.BW:
            mode_val = 1
        elif settings.process.process_mode == ProcessMode.E6:
            mode_val = 2

        n_data = (
            struct.pack("ffff", f[0], f[1], f[2], 0.0)
            + struct.pack("ffff", c[0], c[1], c[2], 0.0)
            + struct.pack("ffff", cast[0], cast[1], cast[2], settings.process.shadow_cast_threshold)
            + struct.pack("IIf", mode_val, (1 if settings.process.e6_normalize else 0), settings.process.shadow_cast_strength)
            + struct.pack("ff", settings.process.white_point_offset, settings.process.black_point_offset)
            + b"\x00" * 44
        )

        from negpy.features.exposure.models import EXPOSURE_CONSTANTS

        exp = settings.exposure
        shift = 0.1 + (exp.density * EXPOSURE_CONSTANTS["density_multiplier"])
        slope, pivot = (
            1.0 + (exp.grade * EXPOSURE_CONSTANTS["grade_multiplier"]),
            1.0 - shift,
        )
        cmy_m = EXPOSURE_CONSTANTS["cmy_max_density"]

        e_data = (
            struct.pack("ffff", pivot, pivot, pivot, 0.0)
            + struct.pack("ffff", slope, slope, slope, 0.0)
            + struct.pack(
                "ffff",
                exp.wb_cyan * cmy_m,
                exp.wb_magenta * cmy_m,
                exp.wb_yellow * cmy_m,
                0.0,
            )
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
            + struct.pack(
                "ffffffff",
                exp.toe,
                exp.toe_width,
                exp.toe_hardness,
                exp.shoulder,
                exp.shoulder_width,
                exp.shoulder_hardness,
                exp.shadows,
                exp.highlights,
            )
            + struct.pack("ffI", 4.0, 2.2, mode_val)
            + b"\x00" * 28
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
        r_u_data = struct.pack(
            "ffIIiiIIf",
            float(ret.dust_threshold),
            float(ret.dust_size),
            len(ret.manual_dust_spots),
            (1 if ret.dust_remove else 0),
            offset[0],
            offset[1],
            full_dims[0],
            full_dims[1],
            float(scale_factor),
        )

        lab = settings.lab
        m_raw = lab.crosstalk_matrix
        if m_raw is None:
            if settings.process.process_mode == ProcessMode.E6:
                m_raw = lab.E6_MATRIX
            else:
                m_raw = lab.C41_MATRIX

        sep_strength = max(0.0, lab.color_separation - 1.0)

        cal = np.array(m_raw).reshape(3, 3)
        applied = np.eye(3) * (1.0 - sep_strength) + cal * sep_strength

        # Row-normalization
        applied /= np.maximum(np.sum(applied, axis=1, keepdims=True), 1e-6)
        m = applied.flatten()
        l_data = (
            struct.pack("ffff", m[0], m[1], m[2], 0.0)
            + struct.pack("ffff", m[3], m[4], m[5], 0.0)
            + struct.pack("ffff", m[6], m[7], m[8], 0.0)
            + struct.pack(
                "fffff",
                sep_strength,
                float(lab.sharpen),
                float(lab.chroma_denoise),
                float(lab.saturation),
                float(lab.vibrance),
            )
            + b"\x00" * 28
        )

        from negpy.features.toning.logic import PAPER_PROFILES, PaperProfileName

        prof = settings.toning.paper_profile
        p_obj = PAPER_PROFILES.get(prof, PAPER_PROFILES[PaperProfileName.NONE])
        tint, dmax, is_bw = (
            p_obj.tint,
            p_obj.dmax_boost,
            (1 if settings.process.process_mode == ProcessMode.BW else 0),
        )
        t_data = (
            struct.pack(
                "ffff",
                float(lab.saturation),
                float(settings.toning.selenium_strength),
                float(settings.toning.sepia_strength),
                2.2,
            )
            + struct.pack("ffff", tint[0], tint[1], tint[2], dmax)
            + struct.pack("iiIf", crop_offset[0], crop_offset[1], is_bw, 0.0)
        )

        pw, ph, cw, ch, ox, oy = self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)
        color_hex = settings.export.export_border_color.lstrip("#")
        bg = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        scale = float(cw) / max(1.0, float(crop_w))
        y_data = (
            struct.pack("ffffii", bg[0], bg[1], bg[2], 1.0, ox, oy)
            + struct.pack("iiii", cw, ch, crop_w, crop_h)
            + struct.pack("f", scale)
            + b"\x00" * 4
        )

        full_buffer = bytearray()
        for d in [g_data, n_data, e_data, c_data, r_u_data, l_data, t_data, y_data]:
            full_buffer += d + b"\x00" * (self._alignment - len(d))

        if not self.gpu.device:
            raise RuntimeError("GPU device lost")
        self.gpu.device.queue.write_buffer(self._buffers["unified_u"].buffer, 0, full_buffer)

    def _update_retouch_storage(
        self,
        conf: Any,
        orig_shape: Tuple[int, int],
        geom: Any,
        offset: Tuple[int, int],
        full_dims: Tuple[int, int],
        scale_factor: float,
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
        dpi = settings.export.export_dpi
        if size_ref:
            dpi = int((size_ref * 2.54) / max(0.1, settings.export.export_print_size))
        border_px = int((settings.export.export_border_size / 2.54) * dpi)

        use_orig = settings.export.use_original_res

        if settings.export.paper_aspect_ratio == AspectRatio.ORIGINAL:
            if use_orig:
                content_w, content_h = cw, ch
            else:
                target_long_edge = int((settings.export.export_print_size / 2.54) * dpi)
                if cw >= ch:
                    content_w, content_h = (
                        target_long_edge,
                        int(ch * (target_long_edge / cw)),
                    )
                else:
                    content_h, content_w = (
                        target_long_edge,
                        int(cw * (target_long_edge / ch)),
                    )
            paper_w, paper_h = content_w + 2 * border_px, content_h + 2 * border_px
            off_x, off_y = border_px, border_px
        else:
            if use_orig:
                content_w, content_h = cw, ch
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
            else:
                paper_w, paper_h = PrintService.calculate_paper_px(
                    settings.export.export_print_size,
                    dpi,
                    settings.export.paper_aspect_ratio,
                    cw,
                    ch,
                )
                inner_w, inner_h = paper_w - 2 * border_px, paper_h - 2 * border_px
                scale = min(inner_w / cw, inner_h / ch)
                content_w, content_h = int(cw * scale), int(ch * scale)

            off_x, off_y = (paper_w - content_w) // 2, (paper_h - content_h) // 2
        return paper_w, paper_h, content_w, content_h, off_x, off_y

    def _readback_metrics(self) -> np.ndarray:
        """Synchronously reads back histogram data from GPU."""
        device = self.gpu.device
        if not device:
            return np.zeros((4, HISTOGRAM_BINS), dtype=np.uint32)
        read_buf = device.create_buffer(
            size=METRICS_BUFFER_SIZE,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self._buffers["metrics"].buffer, 0, read_buf, 0, METRICS_BUFFER_SIZE)
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(read_buf.read_mapped(), dtype=np.uint32).copy()
        read_buf.unmap()
        read_buf.destroy()
        return data.reshape((4, HISTOGRAM_BINS))

    def _readback_downsampled(self, tex: GPUTexture) -> np.ndarray:
        """Reads back texture as float32 RGB array, handling hardware alignment."""
        device = self.gpu.device
        if not device:
            return np.zeros((1, 1, 3), dtype=np.float32)
        prb = (tex.width * 16 + 255) & ~255
        read_buf = device.create_buffer(
            size=prb * tex.height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": tex.texture},
            {"buffer": read_buf, "bytes_per_row": prb, "rows_per_image": tex.height},
            (tex.width, tex.height, 1),
        )
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(read_buf.read_mapped(), dtype=np.uint8).reshape((tex.height, prb))
        valid = raw[:, : tex.width * 16]
        result = valid.view(np.float32).reshape((tex.height, tex.width, 4))
        read_buf.unmap()
        read_buf.destroy()
        return result[:, :, :3]

    def _dispatch_pass(self, encoder: Any, pipeline_name: str, bindings: list, w: int, h: int) -> None:
        """Configures and dispatches a compute pass."""
        pipeline = self._pipelines.get(pipeline_name)
        if pipeline is None:
            raise RuntimeError(f"Pipeline not initialized: {pipeline_name}")

        wg_x, wg_y = (16, 16) if pipeline_name in ["autocrop", "metrics", "clahe_hist"] else (8, 8)
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

        if not self.gpu.device:
            raise RuntimeError("GPU device lost")

        try:
            bind_group = self.gpu.device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
        except Exception as e:
            logger.error(f"Failed to create bind group for {pipeline_name}: {e}")
            raise

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
        cast_override: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """High-level processing entry point with automatic tiling."""
        self._init_resources()
        h, w = img.shape[:2]
        max_tex = self.gpu.limits.get("max_texture_dimension_2d", 8192)
        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        if w_rot > max_tex or h_rot > max_tex or (w * h > TILING_THRESHOLD_PX):
            return self._process_tiled(img, settings, scale_factor, bounds_override=bounds_override, cast_override=cast_override)
        tex_final, metrics = self.process_to_texture(
            img, settings, scale_factor=scale_factor, bounds_override=bounds_override, cast_override=cast_override
        )
        return self._readback_downsampled(tex_final), metrics

    def _process_tiled(
        self,
        img: np.ndarray,
        settings: WorkspaceConfig,
        scale_factor: float,
        bounds_override: Optional[Any] = None,
        cast_override: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes ultra-high resolution images using memory-efficient tiling."""
        h, w = img.shape[:2]

        img_rot = img
        if settings.geometry.rotation != 0:
            img_rot = np.rot90(img_rot, k=settings.geometry.rotation)
        if settings.geometry.flip_horizontal:
            img_rot = np.fliplr(img_rot)
        if settings.geometry.flip_vertical:
            img_rot = np.flipud(img_rot)
        if settings.geometry.fine_rotation != 0.0:
            img_rot = apply_fine_rotation(img_rot, settings.geometry.fine_rotation)

        preview_scale = APP_CONFIG.preview_render_size / max(h, w)
        img_small = cv2.resize(img, (int(w * preview_scale), int(h * preview_scale)))
        _, metrics_ref = self.process_to_texture(img_small, settings, scale_factor=scale_factor)

        device = self.gpu.device
        assert device is not None
        nbytes = 64 * HISTOGRAM_BINS * 4
        read_buf = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.MapMode.READ)
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self._buffers["clahe_c"].buffer, 0, read_buf, 0, nbytes)
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        global_cdfs = np.frombuffer(read_buf.read_mapped(), dtype=np.float32).copy()
        read_buf.unmap()
        read_buf.destroy()

        roi, rot = metrics_ref["active_roi"], settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        h_small, w_small = img_small.shape[:2]
        w_small_rot, h_small_rot = (h_small, w_small) if rot in (1, 3) else (w_small, h_small)
        sy, sx = h_rot / h_small_rot, w_rot / w_small_rot
        y1, y2, x1, x2 = int(roi[0] * sy), int(roi[1] * sy), int(roi[2] * sx), int(roi[3] * sx)
        crop_w, crop_h = x2 - x1, y2 - y1

        if bounds_override:
            global_bounds = bounds_override
        elif settings.process.use_roll_average and settings.process.is_locked_initialized:
            global_bounds = LogNegativeBounds(
                floors=settings.process.locked_floors,
                ceils=settings.process.locked_ceils,
            )
        elif settings.process.is_local_initialized:
            global_bounds = LogNegativeBounds(
                floors=settings.process.local_floors,
                ceils=settings.process.local_ceils,
            )
        else:
            global_bounds = analyze_log_exposure_bounds(
                img_rot,
                roi=(y1, y2, x1, x2),
                analysis_buffer=settings.process.analysis_buffer,
                process_mode=settings.process.process_mode,
                e6_normalize=settings.process.e6_normalize,
            )

        if cast_override:
            global_cast = cast_override
        elif settings.process.shadow_cast_strength > 0:
            if settings.process.use_roll_average:
                global_cast = settings.process.locked_shadow_cast
            elif any(v != 0.0 for v in settings.process.local_shadow_cast):
                global_cast = settings.process.local_shadow_cast
            else:
                from negpy.features.exposure.normalization import normalize_log_image
                from negpy.features.exposure.shadows import analyze_shadow_cast

                epsilon = 1e-6

                analysis_src = img_rot
                if roi:
                    analysis_src = img_rot[y1:y2, x1:x2]

                img_log = np.log10(np.clip(analysis_src, epsilon, 1.0))
                res_norm = normalize_log_image(img_log, global_bounds)
                global_cast = analyze_shadow_cast(res_norm, settings.process.shadow_cast_threshold)
        else:
            global_cast = (0.0, 0.0, 0.0)

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
                tile_res, _ = self.process_to_texture(
                    img_rot[iy1:iy2, ix1:ix2],
                    settings,
                    scale_factor=scale_factor,
                    tiling_mode=True,
                    bounds_override=global_bounds,
                    cast_override=global_cast,
                    global_offset=(ix1, iy1),
                    full_dims=(w_rot, h_rot),
                    clahe_cdf_override=global_cdfs,
                    apply_layout=False,
                )
                ox, oy = x1 + tx - ix1, y1 + ty - iy1
                full_source_res[ty : ty + th, tx : tx + tw] = self._readback_downsampled(tile_res)[oy : oy + th, ox : ox + tw]

        scaled_content = (
            cv2.resize(full_source_res, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
            if (content_w != crop_w or content_h != crop_h)
            else full_source_res
        )
        result = np.zeros((paper_h, paper_w, 3), dtype=np.float32)
        color_hex = settings.export.export_border_color.lstrip("#")
        result[:] = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        result[off_y : off_y + content_h, off_x : off_x + content_w] = scaled_content
        return result, metrics_ref

    def cleanup(self) -> None:
        """Evacuates the texture pool and forces garbage collection."""
        for tex in self._tex_cache.values():
            tex.destroy()
        self._tex_cache.clear()
        gc.collect()
        logger.info("GPUEngine: VRAM resources released")

    def destroy_all(self) -> None:
        """Full resource teardown."""
        self.cleanup()
        for buf in self._buffers.values():
            buf.destroy()
        self._buffers.clear()
        self._pipelines.clear()
        self._sampler = None
        logger.info("GPUEngine: Engine decommissioned")
