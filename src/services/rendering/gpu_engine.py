import os
import struct
import numpy as np
import wgpu  # type: ignore
import cv2
from typing import Any, Optional, Dict, Tuple, cast

from src.infrastructure.gpu.device import GPUDevice
from src.infrastructure.gpu.resources import GPUTexture, GPUBuffer
from src.infrastructure.gpu.shader_loader import ShaderLoader
from src.domain.models import WorkspaceConfig, ProcessMode
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

logger = get_logger(__name__)


class GPUEngine:
    """
    GPU-accelerated image processing engine using WebGPU.
    Pipeline: Geometry -> Normalization -> Exposure -> CLAHE -> Retouch -> Lab -> Toning.
    Supports Adaptive Tiling for ultra-high resolution exports.
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
        }
        self._pipelines: Dict[str, Any] = {}
        self._buffers: Dict[str, GPUBuffer] = {}
        self._sampler: Optional[Any] = None
        # Cache key: (width, height, usage, label)
        self._tex_cache: Dict[Tuple[int, int, int, str], GPUTexture] = {}

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

        for name in self._shaders.keys():
            self._pipelines[name] = self._create_pipeline(self._shaders[name])

        self._buffers["geometry"] = GPUBuffer(
            32, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["normalization"] = GPUBuffer(
            32, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["exposure"] = GPUBuffer(
            80, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
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
            65536, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["clahe_u"] = GPUBuffer(
            16, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["retouch_u"] = GPUBuffer(
            16, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["retouch_s"] = GPUBuffer(
            8192, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["lab"] = GPUBuffer(
            64, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["toning"] = GPUBuffer(
            48, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._buffers["metrics"] = GPUBuffer(
            4096,
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

        logger.info("GPU Pipelines Initialized")

    def _create_pipeline(self, shader_path: str) -> Any:
        shader_module = ShaderLoader.load(shader_path)
        assert self.gpu.device is not None
        return self.gpu.device.create_compute_pipeline(
            layout="auto", compute={"module": shader_module, "entry_point": "main"}
        )

    def process_to_texture(
        self,
        img: np.ndarray,
        settings: WorkspaceConfig,
        scale_factor: float = 1.0,
        tiling_mode: bool = False,
        bounds_override: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Runs the pipeline. If tiling_mode is True, bypasses geometry/cropping steps.
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
        source_tex.upload(img)

        if tiling_mode:
            # In tiling mode, input is already rotated/flipped and we don't crop internally.
            w_rot, h_rot = w, h
            roi = (0, h, 0, w)
            y1, y2, x1, x2 = roi
            crop_w, crop_h = w, h
        else:
            rot = settings.geometry.rotation % 4
            w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
            orig_shape = (h, w)

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

        # Normalization
        if bounds_override:
            bounds = bounds_override
        else:
            epsilon = 1e-6
            img_log = np.log10(np.clip(img, epsilon, 1.0))
            analysis_img = (
                get_analysis_crop(img_log, settings.exposure.analysis_buffer)
                if settings.exposure.analysis_buffer > 0
                else img_log
            )
            bounds = measure_log_negative_bounds(analysis_img)

        # Update Pipeline Uniforms
        if tiling_mode:
            # Identity geometry for tiling
            self._buffers["geometry"].upload(np.zeros(32, dtype=np.uint8))
        else:
            self._update_geometry_uniforms(settings.geometry)

        self._update_normalization_uniforms(bounds)
        self._update_exposure_uniforms(settings.exposure)
        self._update_clahe_uniforms(settings.lab)
        self._update_retouch_resources(settings.retouch, (h, w), settings.geometry)
        self._update_lab_uniforms(settings.lab)
        self._update_toning_uniforms(settings, crop_offset=(x1, y1))

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
        tex_final = self._get_intermediate_texture(
            crop_w,
            crop_h,
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC,
            "final",
        )

        enc = device.create_command_encoder()
        self._dispatch_pass(
            enc,
            "geometry",
            [(0, source_tex.view), (1, tex_geom.view), (2, self._buffers["geometry"])],
            w_rot,
            h_rot,
        )
        self._dispatch_pass(
            enc,
            "normalization",
            [
                (0, tex_geom.view),
                (1, tex_norm.view),
                (2, self._buffers["normalization"]),
            ],
            w_rot,
            h_rot,
        )
        self._dispatch_pass(
            enc,
            "exposure",
            [(0, tex_norm.view), (1, tex_expo.view), (2, self._buffers["exposure"])],
            w_rot,
            h_rot,
        )

        if settings.lab.clahe_strength > 0:
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
                    (2, self._buffers["clahe_u"]),
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
                    (3, self._buffers["clahe_u"]),
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
                (2, self._buffers["retouch_u"]),
                (3, self._buffers["retouch_s"]),
            ],
            w_rot,
            h_rot,
        )
        self._dispatch_pass(
            enc,
            "lab",
            [(0, tex_ret.view), (1, tex_lab.view), (2, self._buffers["lab"])],
            w_rot,
            h_rot,
        )
        self._dispatch_pass(
            enc,
            "toning",
            [(0, tex_lab.view), (1, tex_final.view), (2, self._buffers["toning"])],
            crop_w,
            crop_h,
        )

        if not tiling_mode:
            device.queue.write_buffer(
                self._buffers["metrics"].buffer, 0, np.zeros(1024, dtype=np.uint32)
            )
            self._dispatch_pass(
                enc,
                "metrics",
                [(0, tex_final.view), (1, self._buffers["metrics"])],
                crop_w,
                crop_h,
            )

        device.queue.submit([enc.finish()])

        metrics: Dict[str, Any] = {"active_roi": roi, "base_positive": tex_final}
        if not tiling_mode:
            metrics["analysis_buffer"] = self._readback_downsampled(tex_final)
            metrics["histogram_raw"] = self._readback_metrics()
            try:
                uv_grid = CoordinateMapping.create_uv_grid(
                    rh_orig=h,
                    rw_orig=w,
                    rotation=settings.geometry.rotation,
                    fine_rot=settings.geometry.fine_rotation,
                    flip_h=settings.geometry.flip_horizontal,
                    flip_v=settings.geometry.flip_vertical,
                    autocrop=True,
                    autocrop_params={"roi": roi} if roi else None,
                )
                metrics["uv_grid"] = uv_grid
            except Exception as e:
                logger.error(f"GPU Engine metrics error: {e}")

        return tex_final, metrics

    def _read_buffer_u32(self, buf: GPUBuffer, count: int) -> np.ndarray:
        device = self.gpu.device
        if not device:
            return np.zeros(count, dtype=np.uint32)
        nbytes = count * 4
        read_buf = device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(buf.buffer, 0, read_buf, 0, nbytes)
        device.queue.submit([encoder.finish()])
        read_buf.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(read_buf.read_mapped(), dtype=np.uint32).copy()
        read_buf.unmap()
        read_buf.destroy()
        return data

    def _readback_metrics(self) -> np.ndarray:
        return self._read_buffer_u32(self._buffers["metrics"], 1024).reshape((4, 256))

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
        entries = []
        for idx, res in bindings:
            if isinstance(res, GPUBuffer):
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
        if pipeline_name in ["clahe_hist", "clahe_cdf", "autocrop"]:
            pass_enc.dispatch_workgroups(w, h)
        else:
            pass_enc.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        pass_enc.end()

    def process(
        self, img: np.ndarray, settings: WorkspaceConfig, scale_factor: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._init_resources()
        h, w = img.shape[:2]
        if w * h > 12000000:
            return self._process_tiled(img, settings, scale_factor)
        tex_final, metrics = self.process_to_texture(
            img, settings, scale_factor=scale_factor
        )
        return self._readback_downsampled(tex_final), metrics

    def _process_tiled(
        self, img: np.ndarray, settings: WorkspaceConfig, scale_factor: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = img.shape[:2]
        # Calculate full-res ROI and Bounds once
        epsilon = 1e-6
        img_log = np.log10(np.clip(img, epsilon, 1.0))
        analysis_img = (
            get_analysis_crop(img_log, settings.exposure.analysis_buffer)
            if settings.exposure.analysis_buffer > 0
            else img_log
        )
        global_bounds = measure_log_negative_bounds(analysis_img)

        preview_scale = 1200 / max(h, w)
        img_small = cv2.resize(img, (int(w * preview_scale), int(h * preview_scale)))
        _, metrics_ref = self.process_to_texture(
            img_small, settings, scale_factor=scale_factor
        )
        roi = metrics_ref["active_roi"]

        rot = settings.geometry.rotation % 4
        w_rot, h_rot = (h, w) if rot in (1, 3) else (w, h)
        sy, sx = h_rot / img_small.shape[0], w_rot / img_small.shape[1]
        full_roi = (
            int(roi[0] * sy),
            int(roi[1] * sy),
            int(roi[2] * sx),
            int(roi[3] * sx),
        )
        y1, y2, x1, x2 = full_roi
        final_w, final_h = x2 - x1, y2 - y1

        img_rot = img
        if settings.geometry.rotation != 0:
            img_rot = np.rot90(img_rot, k=settings.geometry.rotation)
        if settings.geometry.flip_horizontal:
            img_rot = np.fliplr(img_rot)
        if settings.geometry.flip_vertical:
            img_rot = np.flipud(img_rot)

        result = np.zeros((final_h, final_w, 3), dtype=np.float32)
        tile_size, halo = 2048, 32

        for ty in range(0, final_h, tile_size):
            for tx in range(0, final_w, tile_size):
                tw, th = min(tile_size, final_w - tx), min(tile_size, final_h - ty)
                ix1, iy1 = max(0, x1 + tx - halo), max(0, y1 + ty - halo)
                ix2, iy2 = (
                    min(w_rot, x1 + tx + tw + halo),
                    min(h_rot, y1 + ty + th + halo),
                )
                tile_src = img_rot[iy1:iy2, ix1:ix2]

                # Pass tiling_mode=True to bypass geometry and use identity crop
                tile_res, _ = self.process_to_texture(
                    tile_src,
                    settings,
                    scale_factor=scale_factor,
                    tiling_mode=True,
                    bounds_override=global_bounds,
                )
                tile_np = self._readback_downsampled(tile_res)

                ox, oy = x1 + tx - ix1, y1 + ty - iy1
                result[ty : ty + th, tx : tx + tw] = tile_np[oy : oy + th, ox : ox + tw]
        return result, metrics_ref

    def _update_geometry_uniforms(self, conf: Any) -> None:
        rot, fine, fh, fv = (
            int(conf.rotation),
            float(conf.fine_rotation),
            (1 if conf.flip_horizontal else 0),
            (1 if conf.flip_vertical else 0),
        )
        self._buffers["geometry"].upload(
            np.frombuffer(
                struct.pack("ifii", rot, fine, fh, fv) + b"\x00" * 16, dtype=np.uint8
            )
        )

    def _update_normalization_uniforms(self, bounds: Any) -> None:
        f, c = bounds.floors, bounds.ceils
        self._buffers["normalization"].upload(
            np.array([f[0], f[1], f[2], 0.0, c[0], c[1], c[2], 0.0], dtype=np.float32)
        )

    def _update_exposure_uniforms(self, exp: Any) -> None:
        from src.features.exposure.models import EXPOSURE_CONSTANTS

        m_ref, shift = (
            1.0,
            0.1 + (exp.density * EXPOSURE_CONSTANTS["density_multiplier"]),
        )
        slope, pivot = (
            1.0 + (exp.grade * EXPOSURE_CONSTANTS["grade_multiplier"]),
            m_ref - shift,
        )
        p, s = (
            np.array([pivot] * 3 + [0], dtype=np.float32),
            np.array([slope] * 3 + [0], dtype=np.float32),
        )
        cmy_m = EXPOSURE_CONSTANTS["cmy_max_density"]
        cmy = np.array(
            [exp.wb_cyan * cmy_m, exp.wb_magenta * cmy_m, exp.wb_yellow * cmy_m, 0.0],
            dtype=np.float32,
        )
        sc = np.array(
            [
                exp.toe,
                exp.toe_width,
                exp.toe_hardness,
                exp.shoulder,
                exp.shoulder_width,
                exp.shoulder_hardness,
                4.0,
                2.2,
            ],
            dtype=np.float32,
        )
        self._buffers["exposure"].upload(np.concatenate([p, s, cmy, sc]))

    def _update_clahe_uniforms(self, conf: Any) -> None:
        self._buffers["clahe_u"].upload(
            np.array(
                [
                    float(conf.clahe_strength),
                    max(1.0, float(conf.clahe_strength) * 2.5),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
        )

    def _update_retouch_resources(
        self, conf: Any, orig_shape: Tuple[int, int], geom: Any
    ) -> None:
        u_data = struct.pack(
            "ffII",
            float(conf.dust_threshold),
            float(conf.dust_size),
            len(conf.manual_dust_spots),
            (1 if conf.dust_remove else 0),
        )
        self._buffers["retouch_u"].upload(np.frombuffer(u_data, dtype=np.uint8))
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

    def _update_lab_uniforms(self, conf: Any) -> None:
        strength, sharpen = float(conf.color_separation), float(conf.sharpen)
        matrix = (
            conf.crosstalk_matrix
            if conf.crosstalk_matrix
            else [1, 0, 0, 0, 1, 0, 0, 0, 1]
        )
        cal_matrix = np.array(matrix).reshape(3, 3)
        applied_matrix = np.eye(3) * (1.0 - strength) + cal_matrix * strength
        applied_matrix /= np.maximum(
            np.sum(applied_matrix, axis=1, keepdims=True), 1e-6
        )
        m = applied_matrix.flatten()
        buf = np.zeros(16, dtype=np.float32)
        buf[0:3], buf[4:7], buf[8:11], buf[12], buf[13] = (
            m[0:3],
            m[3:6],
            m[6:9],
            1.0,
            sharpen,
        )
        self._buffers["lab"].upload(buf)

    def _update_toning_uniforms(
        self, settings: WorkspaceConfig, crop_offset: Tuple[int, int] = (0, 0)
    ) -> None:
        lab, toning = settings.lab, settings.toning
        sat, sel, sep = (
            float(lab.saturation),
            float(toning.selenium_strength),
            float(toning.sepia_strength),
        )
        from src.features.toning.logic import PAPER_PROFILES, PaperProfileName

        profile = PAPER_PROFILES.get(
            toning.paper_profile, PAPER_PROFILES[PaperProfileName.NONE]
        )
        is_bw = 1 if settings.process_mode == ProcessMode.BW else 0
        tint, dmax = profile.tint, profile.dmax_boost
        data = (
            struct.pack("ffff", sat, sel, sep, 2.2)
            + struct.pack("ffff", tint[0], tint[1], tint[2], dmax)
            + struct.pack("iiIf", crop_offset[0], crop_offset[1], is_bw, 0.0)
        )
        self._buffers["toning"].upload(np.frombuffer(data, dtype=np.uint8))
