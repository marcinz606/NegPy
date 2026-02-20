import os
import io
import tifffile
import numpy as np
from PIL import Image, ImageCms
from typing import Tuple, Optional, Any, Dict
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.config import APP_CONFIG
from negpy.domain.types import ImageBuffer
from negpy.domain.models import (
    WorkspaceConfig,
    ExportConfig,
    ExportFormat,
)
from negpy.features.process.models import ProcessMode
from negpy.domain.interfaces import PipelineContext
from negpy.services.rendering.engine import DarkroomEngine
from negpy.services.rendering.gpu_engine import GPUEngine
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.image.logic import (
    float_to_uint8,
    float_to_uint16,
    ensure_rgb,
    uint16_to_float32,
    float_to_uint_luma,
)
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm
from negpy.services.export.print import PrintService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry

logger = get_logger(__name__)


class ImageProcessor:
    """
    Coordinates multi-backend image processing.
    Seamlessly switches between CPU (DarkroomEngine) and GPU (GPUEngine).
    """

    def __init__(self) -> None:
        self.engine_cpu = DarkroomEngine()
        self.engine_gpu: Optional[GPUEngine] = None

        if APP_CONFIG.use_gpu:
            gpu = GPUDevice.get()
            if gpu.is_available:
                self.engine_gpu = GPUEngine()
                logger.info("ImageProcessor: Acceleration backend ready")
            else:
                logger.warning("ImageProcessor: GPU unavailable, using CPU fallback")

    @property
    def backend_name(self) -> str:
        if self.engine_gpu:
            return self.engine_gpu.gpu.backend_name or "WEBGPU"
        return "CPU"

    def run_pipeline(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        source_hash: str,
        render_size_ref: float,
        metrics: Optional[Dict[str, Any]] = None,
        prefer_gpu: bool = True,
        readback_metrics: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes rendering pipeline. Returns result (ndarray/GPUTexture) and metrics.
        """
        h_orig, w_cols = img.shape[:2]
        scale_factor = max(h_orig, w_cols) / float(render_size_ref)

        context = PipelineContext(
            scale_factor=scale_factor,
            original_size=(h_orig, w_cols),
            process_mode=settings.process.process_mode,
        )
        if metrics:
            context.metrics.update(metrics)

        if prefer_gpu and self.engine_gpu:
            try:
                processed, gpu_metrics = self.engine_gpu.process_to_texture(
                    img,
                    settings,
                    scale_factor=scale_factor,
                    render_size_ref=render_size_ref,
                    readback_metrics=readback_metrics,
                )
                context.metrics.update(gpu_metrics)
                return processed, context.metrics
            except Exception as e:
                logger.error(f"Hardware acceleration failed: {e}")

        processed = self.engine_cpu.process(img, settings, source_hash, context)
        return processed, context.metrics

    def buffer_to_pil(self, buffer: Any, settings: WorkspaceConfig, bit_depth: int = 8) -> Image.Image:
        """Converts float32 buffer to calibrated PIL Image."""
        if not isinstance(buffer, np.ndarray):
            raise ValueError("Direct GPU textures cannot be converted to PIL without readback.")

        is_toned = (
            settings.toning.selenium_strength != 0.0 or settings.toning.sepia_strength != 0.0 or settings.toning.paper_profile != "None"
        )
        is_bw = settings.process.process_mode == ProcessMode.BW and not is_toned

        if is_bw:
            img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=bit_depth)
            return Image.fromarray(img_int)

        if bit_depth == 8:
            return Image.fromarray(float_to_uint8(buffer))
        elif bit_depth == 16:
            if buffer.ndim == 2 or (buffer.ndim == 3 and buffer.shape[2] == 1):
                return Image.fromarray(float_to_uint16(buffer))
            return Image.fromarray(float_to_uint8(buffer))
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    def process_export(
        self,
        file_path: str,
        params: WorkspaceConfig,
        export_settings: ExportConfig,
        source_hash: str,
        metrics: Optional[Dict[str, Any]] = None,
        prefer_gpu: bool = True,
        bounds_override: Optional[Any] = None,
    ) -> Tuple[Optional[bytes], str]:
        """Performs high-resolution export with color management."""
        try:
            ctx_mgr, metadata = loader_factory.get_loader(file_path)
            source_cs = metadata.get("color_space", "Adobe RGB")
            raw_color_space = ColorSpaceRegistry.get_rawpy_space(source_cs)
            target_cs = export_settings.export_color_space
            if target_cs == "Same as Source":
                target_cs = source_cs
            color_space = str(target_cs)

            with ctx_mgr as raw:
                algo = get_best_demosaic_algorithm(raw)
                use_camera_wb = params.exposure.use_camera_wb
                user_wb = None if use_camera_wb else [1, 1, 1, 1]
                rgb = raw.postprocess(
                    gamma=(1, 1),
                    no_auto_bright=True,
                    use_camera_wb=use_camera_wb,
                    user_wb=user_wb,
                    output_bps=16,
                    output_color=raw_color_space,
                    demosaic_algorithm=algo,
                )
                rgb = ensure_rgb(rgb)

            f32_buffer = uint16_to_float32(np.ascontiguousarray(rgb))
            h_raw, w_raw = f32_buffer.shape[:2]
            export_scale = max(h_raw, w_raw) / float(APP_CONFIG.preview_render_size)

            if prefer_gpu and self.engine_gpu:
                buffer, gpu_metrics = self.engine_gpu.process(
                    f32_buffer, params, scale_factor=export_scale, bounds_override=bounds_override
                )
            else:
                buffer, _ = self.run_pipeline(
                    f32_buffer,
                    params,
                    source_hash,
                    render_size_ref=float(APP_CONFIG.preview_render_size),
                    metrics=metrics or {"log_bounds": bounds_override} if bounds_override else metrics,
                    prefer_gpu=False,
                )
                buffer = self._apply_scaling_and_border_f32(buffer, params, export_settings)

            is_greyscale = export_settings.export_color_space == "Greyscale"
            is_tiff = export_settings.export_fmt != ExportFormat.JPEG

            if is_tiff:
                img_out_f32 = buffer
                img_int = (
                    float_to_uint_luma(np.ascontiguousarray(img_out_f32), bit_depth=16) if is_greyscale else float_to_uint16(img_out_f32)
                )

                if export_settings.apply_icc:
                    pil_img, icc_bytes = self._apply_color_management(
                        Image.fromarray(img_int),
                        color_space,
                        export_settings.icc_profile_path,
                        export_settings.icc_invert,
                    )
                    img_out = np.array(pil_img)
                else:
                    img_out = img_int
                    icc_bytes = self._get_target_icc_bytes(
                        color_space,
                        export_settings.icc_profile_path,
                        export_settings.icc_invert,
                    )

                output_buf = io.BytesIO()
                tifffile.imwrite(
                    output_buf,
                    img_out,
                    photometric="rgb" if img_out.ndim == 3 else "minisblack",
                    iccprofile=icc_bytes,
                    compression="lzw",
                )
                return output_buf.getvalue(), "tiff"
            else:
                img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=8) if is_greyscale else float_to_uint8(buffer)
                icc_path_to_use = export_settings.icc_profile_path if export_settings.apply_icc else None
                icc_invert_to_use = export_settings.icc_invert if export_settings.apply_icc else False

                pil_img, icc_bytes = self._apply_color_management(
                    Image.fromarray(img_int),
                    color_space,
                    icc_path_to_use,
                    icc_invert_to_use,
                )
                output_buf = io.BytesIO()
                self._save_to_pil_buffer(pil_img, output_buf, export_settings, icc_bytes)
                return output_buf.getvalue(), "jpg"

        except Exception as e:
            logger.error(f"Export pipeline failed: {e}")
            return None, str(e)

    def _apply_scaling_and_border_f32(self, img: np.ndarray, params: WorkspaceConfig, export_settings: ExportConfig) -> np.ndarray:
        """CPU fallback for layout application."""
        result, _ = PrintService.apply_layout(img, export_settings)
        return result

    def _get_target_icc_bytes(self, color_space: str, icc_path: Optional[str], inverse: bool = False) -> Optional[bytes]:
        """Loads ICC profile data for embedding."""
        if not inverse and icc_path and os.path.exists(icc_path):
            with open(icc_path, "rb") as f:
                return f.read()
        path = ColorSpaceRegistry.get_icc_path(color_space)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    def _apply_color_management(
        self,
        pil_img: Image.Image,
        color_space: str,
        icc_path: Optional[str],
        inverse: bool = False,
    ) -> Tuple[Image.Image, Optional[bytes]]:
        """Applies ICC profile transformations."""
        path_src = ColorSpaceRegistry.get_icc_path(color_space)
        profile_working = ImageCms.getOpenProfile(path_src) if path_src and os.path.exists(path_src) else ImageCms.createProfile("sRGB")

        try:
            profile_selected = None
            if icc_path and os.path.exists(icc_path):
                profile_selected = ImageCms.getOpenProfile(icc_path)
            else:
                path_dst = ColorSpaceRegistry.get_icc_path(color_space)
                if path_dst and os.path.exists(path_dst):
                    profile_selected = ImageCms.getOpenProfile(path_dst)

            if profile_selected:
                p_src, p_dst = (profile_selected, profile_working) if inverse else (profile_working, profile_selected)
                if pil_img.mode not in ("RGB", "L"):
                    pil_img = pil_img.convert("RGB" if pil_img.mode != "I;16" else "L")

                result_pil = ImageCms.profileToProfile(
                    pil_img,
                    p_src,
                    p_dst,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB" if pil_img.mode != "L" else "L",
                    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
                )
                if result_pil:
                    pil_img = result_pil
                icc_bytes = self._get_target_icc_bytes(color_space, icc_path) if not inverse else None
            else:
                icc_bytes = self._get_target_icc_bytes(color_space, None)
            return pil_img, icc_bytes
        except Exception as e:
            logger.error(f"CMS transformation failed: {e}")
            return pil_img, None

    def _save_to_pil_buffer(
        self,
        pil_img: Image.Image,
        buf: io.BytesIO,
        export_settings: ExportConfig,
        icc_bytes: Optional[bytes],
    ) -> None:
        """Encodes PIL image to byte stream."""
        fmt = "JPEG" if export_settings.export_fmt == ExportFormat.JPEG else "TIFF"
        pil_img.save(
            buf,
            format=fmt,
            quality=95,
            dpi=(export_settings.export_dpi, export_settings.export_dpi),
            icc_profile=icc_bytes,
            compression="tiff_lzw" if fmt == "TIFF" else None,
        )

    def cleanup(self) -> None:
        """Evacuates transient GPU resources."""
        if self.engine_gpu:
            self.engine_gpu.cleanup()

    def destroy_all(self) -> None:
        """Teardown GPU engine."""
        if self.engine_gpu:
            self.engine_gpu.destroy_all()
