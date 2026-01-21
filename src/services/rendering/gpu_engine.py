import os
import struct
import numpy as np
import wgpu  # type: ignore
import cv2
from typing import Any, Optional, Dict, Tuple, cast

from src.infrastructure.gpu.device import GPUDevice
from src.infrastructure.gpu.resources import GPUTexture, GPUBuffer
from src.infrastructure.gpu.shader_loader import ShaderLoader
from src.domain.models import WorkspaceConfig, ProcessMode, AspectRatio
from src.kernel.system.logging import get_logger
from src.features.geometry.logic import (
    get_manual_rect_coords,
    get_autocrop_coords,
    map_coords_to_geometry,
)
from src.features.exposure.normalization import (
    measure_log_negative_bounds,
    get_analysis_crop,
)
from src.services.view.coordinate_mapping import CoordinateMapping
from src.services.export.print import PrintService

logger = get_logger(__name__)


class GPUEngine:
    """
    GPU-accelerated image processing engine using WebGPU.
    Orchestrates a 10-stage pipeline with consolidated uniforms and optimized readback.
    """

    def __init__(self) -> None:
        self.gpu = GPUDevice.get()
        self._shaders = {
            "geometry": os.path.join(
                "src", "features", "geometry", "shaders", "transform.wgsl"
            ),
            "normalization": os.path.join(
                "src", "features", "exposure", "shaders", "normalization.wgsl"
            ),
            "exposure": os.path.join(
                "src", "features", "exposure", "shaders", "exposure.wgsl"
            ),
            "autocrop": os.path.join(
                "src", "features", "geometry", "shaders", "autocrop.wgsl"
            ),
            "clahe_hist": os.path.join(
                "src", "features", "lab", "shaders", "clahe_hist.wgsl"
            ),
            "clahe_cdf": os.path.join(
                "src", "features", "lab", "shaders", "clahe_cdf.wgsl"
            ),
            "clahe_apply": os.path.join(
                "src", "features", "lab", "shaders", "clahe_apply.wgsl"
            ),
            "retouch": os.path.join(
                "src", "features", "retouch", "shaders", "retouch.wgsl"
            ),
            "lab": os.path.join("src", "features", "lab", "shaders", "lab.wgsl"),
            "toning": os.path.join(
                "src", "features", "toning", "shaders", "toning.wgsl"
            ),
            "metrics": os.path.join(
                "src", "features", "lab", "shaders", "metrics.wgsl"
            ),
            "layout": os.path.join(
                "src", "features", "toning", "shaders", "layout.wgsl"
            ),
        }
        self._pipelines: Dict[str, Any] = {}
        self._buffers: Dict[str, GPUBuffer] = {}
        self._sampler: Optional[Any] = None
        # Cache key: (width, height, usage, label)
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
        self._alignment = 256

    def _get_intermediate_texture(
        self, w: int, h: int, usage: int, label: str
    ) -> GPUTexture:
        key = (w, h, usage, label)
        if key not in self._tex_cache:
            self._tex_cache[key] = GPUTexture(w, h, usage=usage)
        return self._tex_cache[key]

    def _init_resources(self) -> None:
        if self._pipelines or not self.gpu.device:
            return
        device = self.gpu.device
        self._sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        self._alignment = self.gpu.limits.get(
            "min_uniform_buffer_offset_alignment", 256
        )

        for name in self._shaders.keys():
            self._pipelines[name] = self._create_pipeline(self._shaders[name])

        # Unified Uniform Buffer (UBO)
        self._buffers["unified_u"] = GPUBuffer(
            self._alignment * len(self._uniform_names),
            wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Persistent Storage Buffers
        self._buffers["crop_rows"] = GPUBuffer(
            32768,
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
        self._buffers["crop_cols"] = GPUBuffer(
            32768,
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
        self._buffers["clahe_h"] = GPUBuffer(
            65536, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["clahe_c"] = GPUBuffer(
            65536,
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
        self._buffers["retouch_s"] = GPUBuffer(
            8192, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["metrics"] = GPUBuffer(
            4096,
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

        logger.info("GPU Engine: High-performance Resources Initialized")

    def _create_pipeline(self, shader_path: str) -> Any:
        shader_module = ShaderLoader.load(shader_path)
        assert self.gpu.device is not None
        return self.gpu.device.create_compute_pipeline(
            layout="auto", compute={"module": shader_module, "entry_point": "main"}
        )

    def _get_uniform_binding(self, name: str) -> Dict[str, Any]:
        idx = self._uniform_names.index(name)
        # Layout sizes must match WGSL struct layout precisely
        sizes = {
            "geometry": 32,
            "normalization": 32,
            "exposure": 80,
            "clahe_u": 32,
            "retouch_u": 32,
            "lab": 64,
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
        global_offset: Tuple[int, int] = (0, 0),
        full_dims: Optional[Tuple[int, int]] = None,
        clahe_cdf_override: Optional[np.ndarray] = None,
        apply_layout: bool = True,
        render_size_ref: Optional[float] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
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
        source_tex.upload(img)

        # 1. Determine ROI and Output dimensions
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

                class ShapeMock:
                    def __init__(self, s: Any) -> None:
                        self.shape = s

                roi = get_manual_rect_coords(
                    cast(np.ndarray, ShapeMock((h_rot, w_rot, 3))),
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
                # Optimized GPU-based autocrop detection could go here,
                # but currently we still use the fast CPU logic on preview scale.
                det_s = 1200 / max(h, w)
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

        bounds = (
            bounds_override
            if bounds_override
            else measure_log_negative_bounds(
                get_analysis_crop(
                    np.log10(np.clip(img, 1e-6, 1.0)), settings.exposure.analysis_buffer
                )
                if settings.exposure.analysis_buffer > 0
                else np.log10(np.clip(img, 1e-6, 1.0))
            )
        )

        # Update Pipeline Resources
        self._upload_unified_uniforms(
            settings,
            bounds,
            global_offset,
            actual_full_dims,
            (x1, y1),
            crop_w,
            crop_h,
            tiling_mode,
            render_size_ref,
        )
        self._update_retouch_storage(
            settings.retouch, (h, w), settings.geometry, global_offset, actual_full_dims
        )
        if clahe_cdf_override is not None:
            self._buffers["clahe_c"].upload(clahe_cdf_override)

        # Textures
        tex_geom = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            "geom",
        )
        tex_norm = self._get_intermediate_texture(
            w_rot,
            h_rot,
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
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
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC,
            "toning",
        )

        enc = device.create_command_encoder()
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
            if clahe_cdf_override is None:
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
            paper_w, paper_h, content_w, content_h, off_x, off_y = (
                self._calculate_layout_dims(settings, crop_w, crop_h, render_size_ref)
            )
            tex_final = self._get_intermediate_texture(
                paper_w,
                paper_h,
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC,
                "final",
            )
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

        if not tiling_mode:
            device.queue.write_buffer(
                self._buffers["metrics"].buffer, 0, np.zeros(1024, dtype=np.uint32)
            )
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
            "content_rect": content_rect,
        }
        if not tiling_mode:
            metrics["analysis_buffer"] = self._readback_downsampled(tex_final)
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
    ) -> None:
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
        n_data = struct.pack("ffffffff", f[0], f[1], f[2], 0.0, c[0], c[1], c[2], 0.0)
        from src.features.exposure.models import EXPOSURE_CONSTANTS

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
                "ffffffff",
                exp.toe,
                exp.toe_width,
                exp.toe_hardness,
                exp.shoulder,
                exp.shoulder_width,
                exp.shoulder_hardness,
                4.0,
                2.2,
            )
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
            "ffIIiiII",
            float(ret.dust_threshold),
            float(ret.dust_size),
            len(ret.manual_dust_spots),
            (1 if ret.dust_remove else 0),
            offset[0],
            offset[1],
            full_dims[0],
            full_dims[1],
        )
        lab = settings.lab
        m_raw = (
            lab.crosstalk_matrix
            if lab.crosstalk_matrix
            else [1, 0, 0, 0, 1, 0, 0, 0, 1]
        )
        cal = np.array(m_raw).reshape(3, 3)
        applied = np.eye(3) * (1.0 - lab.color_separation) + cal * lab.color_separation
        applied /= np.maximum(np.sum(applied, axis=1, keepdims=True), 1e-6)
        m = applied.flatten()
        l_data = (
            struct.pack("ffff", m[0], m[1], m[2], 0.0)
            + struct.pack("ffff", m[3], m[4], m[5], 0.0)
            + struct.pack("ffff", m[6], m[7], m[8], 0.0)
            + struct.pack("ffff", 1.0, float(lab.sharpen), 0.0, 0.0)
        )
        from src.features.toning.logic import PAPER_PROFILES, PaperProfileName

        prof = settings.toning.paper_profile
        p_obj = PAPER_PROFILES.get(prof, PAPER_PROFILES[PaperProfileName.NONE])
        tint, dmax, is_bw = (
            p_obj.tint,
            p_obj.dmax_boost,
            (1 if settings.process_mode == ProcessMode.BW else 0),
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

        paper_w, paper_h, content_w, content_h, ox, oy = self._calculate_layout_dims(
            settings, crop_w, crop_h, render_size_ref
        )
        color_hex = settings.export.export_border_color.lstrip("#")
        bg = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        scale = float(content_w) / max(1.0, float(crop_w))
        y_data = (
            struct.pack("ffffii", bg[0], bg[1], bg[2], 1.0, ox, oy)
            + struct.pack("iiii", content_w, content_h, crop_w, crop_h)
            + struct.pack("f", scale)
            + b"\x00" * 4
        )

        full_buffer = bytearray()
        for d in [g_data, n_data, e_data, c_data, r_u_data, l_data, t_data, y_data]:
            full_buffer += d + b"\x00" * (self._alignment - len(d))
        if not self.gpu.device:
            raise RuntimeError("GPU device lost")
        self.gpu.device.queue.write_buffer(
            self._buffers["unified_u"].buffer, 0, full_buffer
        )

    def _update_retouch_storage(
        self,
        conf: Any,
        orig_shape: Tuple[int, int],
        geom: Any,
        offset: Tuple[int, int],
        full_dims: Tuple[int, int],
    ) -> None:
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
            spot_data += struct.pack("ffff", mx, my, size / max(orig_shape), 0.0)
        if spot_data:
            self._buffers["retouch_s"].upload(np.frombuffer(spot_data, dtype=np.uint8))

    def _calculate_layout_dims(
        self, settings: WorkspaceConfig, cw: int, ch: int, size_ref: Optional[float]
    ) -> Tuple[int, int, int, int, int, int]:
        dpi = settings.export.export_dpi
        if size_ref:
            dpi = int((size_ref * 2.54) / max(0.1, settings.export.export_print_size))

        border_px = int((settings.export.export_border_size / 2.54) * dpi)

        if settings.export.paper_aspect_ratio == AspectRatio.ORIGINAL:
            # Scale image to requested size @ DPI
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
            paper_w, paper_h = PrintService.calculate_paper_px(
                settings.export.export_print_size,
                dpi,
                settings.export.paper_aspect_ratio,
                cw,
                ch,
            )
            # Center the scaled image within the paper while respecting borders
            # (Simplified: image fits within paper minus 2x border)
            inner_w, inner_h = paper_w - 2 * border_px, paper_h - 2 * border_px
            scale = min(inner_w / cw, inner_h / ch)
            content_w, content_h = int(cw * scale), int(ch * scale)
            off_x, off_y = (paper_w - content_w) // 2, (paper_h - content_h) // 2

        return paper_w, paper_h, content_w, content_h, off_x, off_y

    def _readback_metrics(self) -> np.ndarray:
        device = self.gpu.device
        if not device:
            return np.zeros((4, 256), dtype=np.uint32)
        read_buf = device.create_buffer(
            size=4096, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(
            self._buffers["metrics"].buffer, 0, read_buf, 0, 4096
        )
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(read_buf.read_mapped(), dtype=np.uint32).copy()
        read_buf.unmap()
        read_buf.destroy()
        return cast(np.ndarray, data.reshape((4, 256)))

    def _readback_downsampled(self, tex: GPUTexture) -> np.ndarray:
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
        raw = np.frombuffer(read_buf.read_mapped(), dtype=np.uint8).reshape(
            (tex.height, prb)
        )
        valid = raw[:, : tex.width * 16]
        result = valid.view(np.float32).reshape((tex.height, tex.width, 4))
        read_buf.unmap()
        read_buf.destroy()
        return result[:, :, :3]

    def _dispatch_pass(
        self, encoder: Any, pipeline_name: str, bindings: list, w: int, h: int
    ) -> None:
        pipeline = self._pipelines[pipeline_name]

        # Hardware-aware workgroup calculation
        # Most shaders are 8x8, some are 16x16
        wg_x, wg_y = 8, 8
        if pipeline_name in ["autocrop", "metrics", "clahe_hist"]:
            wg_x, wg_y = 16, 16
        elif pipeline_name == "clahe_cdf":
            wg_x, wg_y = 8, 8

        entries = []
        for idx, res in bindings:
            if isinstance(res, dict) and "buffer" in res:
                entries.append({"binding": idx, "resource": res})
            elif isinstance(res, GPUBuffer):
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
        bind_group = self.gpu.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0), entries=entries
        )
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)

        # Adaptive Dispatch
        if pipeline_name in ["clahe_hist", "clahe_cdf"]:
            # These use wid fixed at 8x8 tiles
            pass_enc.dispatch_workgroups(8, 8)
        else:
            # Full texture coverage
            pass_enc.dispatch_workgroups((w + wg_x - 1) // wg_x, (h + wg_y - 1) // wg_y)

        pass_enc.end()

    def process(
        self, img: np.ndarray, settings: WorkspaceConfig, scale_factor: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._init_resources()
        h, w = img.shape[:2]
        max_tex = self.gpu.limits.get("max_texture_dimension_2d", 8192)
        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        if w_rot > max_tex or h_rot > max_tex or (w * h > 12000000):
            return self._process_tiled(img, settings, scale_factor)
        tex_final, metrics = self.process_to_texture(
            img, settings, scale_factor=scale_factor
        )
        return self._readback_downsampled(tex_final), metrics

    def _process_tiled(
        self, img: np.ndarray, settings: WorkspaceConfig, scale_factor: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = img.shape[:2]
        epsilon = 1e-6
        global_bounds = measure_log_negative_bounds(
            get_analysis_crop(
                np.log10(np.clip(img, epsilon, 1.0)), settings.exposure.analysis_buffer
            )
            if settings.exposure.analysis_buffer > 0
            else np.log10(np.clip(img, 1e-6, 1.0))
        )
        preview_scale = 1200 / max(h, w)
        img_small = cv2.resize(img, (int(w * preview_scale), int(h * preview_scale)))
        _, metrics_ref = self.process_to_texture(
            img_small, settings, scale_factor=scale_factor
        )

        device = self.gpu.device
        assert device is not None
        nbytes = 64 * 256 * 4
        read_buf = device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(
            self._buffers["clahe_c"].buffer, 0, read_buf, 0, nbytes
        )
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        global_cdfs = np.frombuffer(read_buf.read_mapped(), dtype=np.float32).copy()
        read_buf.unmap()
        read_buf.destroy()

        roi = metrics_ref["active_roi"]
        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)

        h_small, w_small = img_small.shape[:2]
        # Calculate the rotated dimensions of the small image to match the ROI's coordinate system
        w_small_rot, h_small_rot = (
            (h_small, w_small) if rot in (1, 3) else (w_small, h_small)
        )

        sy, sx = h_rot / h_small_rot, w_rot / w_small_rot
        full_roi = (
            int(roi[0] * sy),
            int(roi[1] * sy),
            int(roi[2] * sx),
            int(roi[3] * sx),
        )
        y1, y2, x1, x2 = full_roi
        crop_w, crop_h = x2 - x1, y2 - y1
        paper_w, paper_h, content_w, content_h, off_x, off_y = (
            self._calculate_layout_dims(settings, crop_w, crop_h, None)
        )

        img_rot = img
        if settings.geometry.rotation != 0:
            img_rot = np.rot90(img_rot, k=settings.geometry.rotation)
        if settings.geometry.flip_horizontal:
            img_rot = np.fliplr(img_rot)
        if settings.geometry.flip_vertical:
            img_rot = np.flipud(img_rot)

        # Optimization: Upload global CDFs once outside the tile loop
        self._buffers["clahe_c"].upload(global_cdfs)

        full_source_res = np.zeros((crop_h, crop_w, 3), dtype=np.float32)
        tile_size, halo = 2048, 32
        for ty in range(0, crop_h, tile_size):
            for tx in range(0, crop_w, tile_size):
                tw, th = min(tile_size, crop_w - tx), min(tile_size, crop_h - ty)
                ix1, iy1 = max(0, x1 + tx - halo), max(0, y1 + ty - halo)
                ix2, iy2 = (
                    min(w_rot, x1 + tx + tw + halo),
                    min(h_rot, y1 + ty + th + halo),
                )

                # Pass tiling_mode=True to ensure process_to_texture uses correct ROI logic
                # Pass None for global_cdfs because we already uploaded them to the buffer.
                tile_res, _ = self.process_to_texture(
                    img_rot[iy1:iy2, ix1:ix2],
                    settings,
                    scale_factor=scale_factor,
                    tiling_mode=True,
                    bounds_override=global_bounds,
                    global_offset=(ix1, iy1),
                    full_dims=(w_rot, h_rot),
                    clahe_cdf_override=None,
                    apply_layout=False,
                )
                ox, oy = x1 + tx - ix1, y1 + ty - iy1
                tile_data = self._readback_downsampled(tile_res)

                # Sliced assignment to handle edge tiles correctly
                full_source_res[ty : ty + th, tx : tx + tw] = tile_data[
                    oy : oy + th, ox : ox + tw
                ]

        if content_w != crop_w or content_h != crop_h:
            scaled_content = cv2.resize(
                full_source_res, (content_w, content_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            scaled_content = full_source_res

        result = np.zeros((paper_h, paper_w, 3), dtype=np.float32)
        color_hex = settings.export.export_border_color.lstrip("#")
        result[:] = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        result[off_y : off_y + content_h, off_x : off_x + content_w] = scaled_content

        return result, metrics_ref

    def cleanup(self) -> None:
        """
        Clears the texture cache. Underlying wgpu textures will be destroyed by GC
        once they are no longer in use by the cache or the display widgets.
        """
        self._tex_cache.clear()
        logger.info("GPUEngine: VRAM resources released to GC")
