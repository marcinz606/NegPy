import os
import io
import tifffile
import numpy as np
from PIL import Image, ImageCms
from typing import Tuple, Optional, Any, Dict
from src.kernel.system.logging import get_logger
from src.kernel.system.config import APP_CONFIG
from src.domain.types import ImageBuffer
from src.domain.models import (
    WorkspaceConfig,
    ExportConfig,
    ProcessMode,
    ExportFormat,
)
from src.domain.interfaces import PipelineContext
from src.services.rendering.engine import DarkroomEngine
from src.services.rendering.gpu_engine import GPUEngine
from src.infrastructure.gpu.device import GPUDevice
from src.kernel.image.logic import (
    float_to_uint8,
    float_to_uint16,
    ensure_rgb,
    uint16_to_float32,
    float_to_uint_luma,
)
from src.infrastructure.loaders.factory import loader_factory
from src.infrastructure.loaders.helpers import get_best_demosaic_algorithm
from src.services.export.print import PrintService
from src.infrastructure.display.color_spaces import ColorSpaceRegistry

logger = get_logger(__name__)


class ImageProcessor:
    """
    Pipeline runner for exports & previews. Supports CPU and GPU backends.
    """

    def __init__(self) -> None:
        self.engine_cpu = DarkroomEngine()
        self.engine_gpu: Optional[GPUEngine] = None

        if APP_CONFIG.use_gpu:
            gpu = GPUDevice.get()
            if gpu.is_available:
                self.engine_gpu = GPUEngine()
                logger.info("ImageProcessor: GPU backend initialized")
            else:
                logger.warning(
                    "ImageProcessor: GPU requested but not available, using CPU"
                )

    def run_pipeline(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        source_hash: str,
        render_size_ref: float,
        metrics: Optional[Dict[str, Any]] = None,
        prefer_gpu: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the engine, returns buffer (ndarray or GPUTexture) + metrics.
        """
        h_orig, w_cols = img.shape[:2]
        scale_factor = max(h_orig, w_cols) / float(render_size_ref)

        context = PipelineContext(
            scale_factor=scale_factor,
            original_size=(h_orig, w_cols),
            process_mode=settings.process_mode,
        )
        if metrics:
            context.metrics.update(metrics)

        # GPU Path
        if prefer_gpu and self.engine_gpu:
            try:
                processed, gpu_metrics = self.engine_gpu.process_to_texture(
                    img,
                    settings,
                    scale_factor=scale_factor,
                    render_size_ref=render_size_ref,
                )
                context.metrics.update(gpu_metrics)
                return processed, context.metrics
            except Exception as e:
                logger.error(f"GPU Processing failed, falling back to CPU: {e}")

        # CPU Fallback
        processed = self.engine_cpu.process(img, settings, source_hash, context)
        return processed, context.metrics

    def buffer_to_pil(
        self, buffer: Any, settings: WorkspaceConfig, bit_depth: int = 8
    ) -> Image.Image:
        """
        Buffer (Any) -> PIL (uint8/16).
        """
        if not isinstance(buffer, np.ndarray):
            raise ValueError(
                "buffer_to_pil received GPU texture. Readback must be handled by processor."
            )

        # CPU ndarray logic
        is_toned = (
            settings.toning.selenium_strength != 0.0
            or settings.toning.sepia_strength != 0.0
            or settings.toning.paper_profile != "None"
        )
        is_bw = settings.process_mode == ProcessMode.BW and not is_toned

        if is_bw:
            img_int = float_to_uint_luma(
                np.ascontiguousarray(buffer), bit_depth=bit_depth
            )
            return Image.fromarray(img_int)

        if bit_depth == 8:
            img_int = float_to_uint8(buffer)
            return Image.fromarray(img_int)
        elif bit_depth == 16:
            if buffer.ndim == 2 or (buffer.ndim == 3 and buffer.shape[2] == 1):
                img_int = float_to_uint16(buffer)
                return Image.fromarray(img_int)
            else:
                return Image.fromarray(float_to_uint8(buffer))
        else:
            raise ValueError("Unsupported bit depth. Use 8 or 16.")

    def process_export(
        self,
        file_path: str,
        params: WorkspaceConfig,
        export_settings: ExportConfig,
        source_hash: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[bytes], str]:
        """
        Full-res render + encoding (TIFF/JPEG).
        """
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

            h, w = rgb.shape[:2]
            f32_buffer = uint16_to_float32(np.ascontiguousarray(rgb))

            if self.engine_gpu:
                buffer, gpu_metrics = self.engine_gpu.process(
                    f32_buffer, params, scale_factor=1.0
                )
                if metrics:
                    metrics.update(gpu_metrics)
            else:
                buffer, _ = self.run_pipeline(
                    f32_buffer,
                    params,
                    source_hash,
                    render_size_ref=float(h),
                    metrics=metrics,
                    prefer_gpu=False,
                )
                # Apply layout for CPU path
                buffer = self._apply_scaling_and_border_f32(
                    buffer, params, export_settings
                )

            is_greyscale = export_settings.export_color_space == "Greyscale"
            is_tiff = export_settings.export_fmt != ExportFormat.JPEG

            if is_tiff:
                if is_greyscale:
                    img_out = float_to_uint_luma(
                        np.ascontiguousarray(buffer), bit_depth=16
                    )
                else:
                    img_out = float_to_uint16(buffer)

                target_icc_bytes = self._get_target_icc_bytes(
                    color_space,
                    export_settings.icc_profile_path,
                    export_settings.icc_invert,
                )

                output_buf = io.BytesIO()
                tifffile.imwrite(
                    output_buf,
                    img_out,
                    photometric="rgb" if img_out.ndim == 3 else "minisblack",
                    iccprofile=target_icc_bytes,
                    compression="lzw",
                )
                return output_buf.getvalue(), "tiff"
            else:
                if is_greyscale:
                    img_int = float_to_uint_luma(
                        np.ascontiguousarray(buffer), bit_depth=8
                    )
                    pil_img = Image.fromarray(img_int)
                else:
                    pil_img = self.buffer_to_pil(buffer, params, bit_depth=8)

                pil_img, target_icc_bytes = self._apply_color_management(
                    pil_img,
                    color_space,
                    export_settings.icc_profile_path,
                    export_settings.icc_invert,
                )

                output_buf = io.BytesIO()
                self._save_to_pil_buffer(
                    pil_img, output_buf, export_settings, target_icc_bytes
                )
                return output_buf.getvalue(), "jpg"

        except Exception as e:
            logger.error(f"Export Processing Error: {e}")
            return None, str(e)

    def _apply_scaling_and_border_f32(
        self,
        img: np.ndarray,
        params: WorkspaceConfig,
        export_settings: ExportConfig,
    ) -> np.ndarray:
        """
        Legacy CPU layout application for testing and fallback.
        """
        result, _ = PrintService.apply_layout(img, export_settings)
        return result

    def _get_target_icc_bytes(
        self, color_space: str, icc_path: Optional[str], inverse: bool = False
    ) -> Optional[bytes]:
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
        target_icc_bytes = None
        profile_working: Any
        path_src = ColorSpaceRegistry.get_icc_path(color_space)
        if path_src and os.path.exists(path_src):
            profile_working = ImageCms.getOpenProfile(path_src)
        else:
            profile_working = ImageCms.createProfile("sRGB")

        try:
            profile_selected: Optional[Any] = None
            if icc_path and os.path.exists(icc_path):
                profile_selected = ImageCms.getOpenProfile(icc_path)
            else:
                path_dst = ColorSpaceRegistry.get_icc_path(color_space)
                if path_dst and os.path.exists(path_dst):
                    profile_selected = ImageCms.getOpenProfile(path_dst)

            if profile_selected:
                profile_src = profile_selected if inverse else profile_working
                profile_dst = profile_working if inverse else profile_selected

                if pil_img.mode not in ("RGB", "L"):
                    pil_img = pil_img.convert("RGB" if pil_img.mode != "I;16" else "L")

                result_pil = ImageCms.profileToProfile(
                    pil_img,
                    profile_src,
                    profile_dst,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB" if pil_img.mode != "L" else "L",
                    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
                )
                if result_pil is not None:
                    pil_img = result_pil

                if not inverse:
                    target_icc_bytes = self._get_target_icc_bytes(color_space, icc_path)
            else:
                target_icc_bytes = self._get_target_icc_bytes(color_space, None)

        except Exception as e:
            logger.error(f"ICC Error: {e}")

        return pil_img, target_icc_bytes

    def _save_to_pil_buffer(
        self,
        pil_img: Image.Image,
        buf: io.BytesIO,
        export_settings: ExportConfig,
        icc_bytes: Optional[bytes],
    ) -> None:
        if export_settings.export_fmt == ExportFormat.JPEG:
            pil_img.save(
                buf,
                format="JPEG",
                quality=95,
                dpi=(export_settings.export_dpi, export_settings.export_dpi),
                icc_profile=icc_bytes,
            )
        else:
            pil_img.save(
                buf,
                format="TIFF",
                compression="tiff_lzw",
                dpi=(export_settings.export_dpi, export_settings.export_dpi),
                icc_profile=icc_bytes,
            )

    def cleanup(self) -> None:
        """
        Releases engine resources.
        """
        if self.engine_gpu:
            self.engine_gpu.cleanup()

    def destroy_all(self) -> None:
        """
        Completely releases all GPU resources.
        """
        if self.engine_gpu:
            self.engine_gpu.destroy_all()
