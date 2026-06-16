import os
import io
import rawpy
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
    ColorSpace,
)
from negpy.features.process.models import ProcessMode
from negpy.features.flatfield.logic import apply_flatfield, flatfield_token
from negpy.domain.interfaces import PipelineContext
from negpy.services.rendering.engine import DarkroomEngine
from negpy.services.rendering.gpu_engine import GPUEngine
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.image.logic import (
    apply_exif_orientation,
    float_to_uint8,
    float_to_uint16,
    ensure_rgb,
    uint16_to_float32,
    float_to_uint_luma,
)
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm
from negpy.services.export.print import PrintService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry, WORKING_COLOR_SPACE
from negpy.infrastructure.display.icc_lut import apply_icc_u16_rgb

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
        ir_buffer: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes rendering pipeline. Returns result (ndarray/GPUTexture) and metrics.
        """
        # Flat-field is a source pre-correction (before geometry/crop); folding its token
        # into source_hash invalidates the engine cache when it changes.
        img = apply_flatfield(img, settings.flatfield)
        source_hash = source_hash + flatfield_token(settings.flatfield)

        h_orig, w_cols = img.shape[:2]
        scale_factor = max(h_orig, w_cols) / float(APP_CONFIG.preview_render_size)

        context = PipelineContext(
            scale_factor=scale_factor,
            original_size=(h_orig, w_cols),
            process_mode=settings.process.process_mode,
            ir_buffer=ir_buffer,
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
                    ir_buffer=ir_buffer,
                )
                context.metrics.update(gpu_metrics)
                return processed, context.metrics
            except Exception:
                logger.exception("Hardware acceleration failed, falling back to CPU")
                context.metrics["gpu_fallback"] = True

        processed = self.engine_cpu.process(img, settings, source_hash, context)
        return processed, context.metrics

    def buffer_to_pil(self, buffer: Any, settings: WorkspaceConfig, bit_depth: int = 8) -> Image.Image:
        """Converts float32 buffer to calibrated PIL Image."""
        if not isinstance(buffer, np.ndarray):
            raise ValueError("Direct GPU textures cannot be converted to PIL without readback.")

        is_toned = settings.toning.selenium_strength != 0.0 or settings.toning.sepia_strength != 0.0
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

    def _load_source_f32(self, file_path: str, params: WorkspaceConfig) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Decode a source file to a flatfield-corrected, EXIF-oriented float32 buffer.

        Returns (f32_buffer, ir_buffer, source_color_space).
        """
        ctx_mgr, metadata = loader_factory.get_loader(file_path)
        source_cs = str(metadata.get("color_space", ColorSpace.ADOBE_RGB.value))
        ir_full = metadata.get("ir")

        with ctx_mgr as raw:
            algo = get_best_demosaic_algorithm(raw)
            linear_raw = params.exposure.linear_raw
            user_wb = [1, 1, 1, 1] if linear_raw else None
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=not linear_raw,
                user_wb=user_wb,
                output_bps=16,
                output_color=rawpy.ColorSpace.raw,
                demosaic_algorithm=algo,
                user_flip=0,
            )
            rgb = ensure_rgb(rgb)

        f32_buffer = uint16_to_float32(rgb)
        orientation = metadata.get("orientation", 1)
        f32_buffer = apply_exif_orientation(f32_buffer, orientation)
        f32_buffer = apply_flatfield(f32_buffer, params.flatfield)
        if ir_full is not None:
            ir_full = apply_exif_orientation(ir_full, orientation)
        return f32_buffer, ir_full, source_cs

    def process_export(
        self,
        file_path: str,
        params: WorkspaceConfig,
        export_settings,  # ExportConfig or ExportPreset
        source_hash: str,
        metrics: Optional[Dict[str, Any]] = None,
        prefer_gpu: bool = True,
        bounds_override: Optional[Any] = None,
        working_color_space: str = WORKING_COLOR_SPACE,
    ) -> Tuple[Optional[bytes], str]:
        """Performs high-resolution export with color management."""
        try:
            from dataclasses import replace as dc_replace

            # Ensure both GPU and CPU paths use the same export settings.
            params = dc_replace(params, export=export_settings)

            f32_buffer, ir_full, source_cs = self._load_source_f32(file_path, params)
            target_cs = export_settings.export_color_space
            if target_cs == ColorSpace.SAME_AS_SOURCE.value:
                target_cs = source_cs
            color_space = str(target_cs)

            h_raw, w_raw = f32_buffer.shape[:2]
            export_scale = max(h_raw, w_raw) / float(APP_CONFIG.preview_render_size)

            if prefer_gpu and self.engine_gpu:
                buffer, gpu_metrics = self.engine_gpu.process(
                    f32_buffer, params, scale_factor=export_scale, bounds_override=bounds_override, ir_buffer=ir_full
                )
            else:
                buffer, _ = self.run_pipeline(
                    f32_buffer,
                    params,
                    source_hash,
                    render_size_ref=float(APP_CONFIG.preview_render_size),
                    metrics=metrics or {"log_bounds": bounds_override} if bounds_override else metrics,
                    prefer_gpu=False,
                    ir_buffer=ir_full,
                )
                buffer = self._apply_scaling_and_border_f32(buffer, params, params.export)
                # Release full-res arrays pinned in the CPU stage cache.
                self.engine_cpu.cache.clear()

            return self._encode_export(buffer, export_settings, color_space, working_color_space)

        except Exception as e:
            logger.error(f"Export pipeline failed: {e}")
            return None, str(e)

    def _encode_export(
        self,
        buffer: np.ndarray,
        export_settings,
        color_space: str,
        working_color_space: str = WORKING_COLOR_SPACE,
    ) -> Tuple[bytes, str]:
        """Encodes a processed float buffer to the target format's file bytes.

        Input ICC overrides the source, output ICC the destination; both are always
        applied so the file matches the preview.
        """
        is_greyscale = color_space == ColorSpace.GREYSCALE.value
        fmt = export_settings.export_fmt
        icc_input = export_settings.icc_input_path
        icc_output = export_settings.icc_output_path

        if fmt == ExportFormat.TIFF:
            img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16) if is_greyscale else float_to_uint16(buffer)
            if is_greyscale:
                img_out, icc_bytes = self._apply_color_management_u16_greyscale(
                    img_int, working_color_space, color_space, icc_output, icc_input
                )
            else:
                img_out, icc_bytes = self._apply_color_management_u16_rgb(img_int, working_color_space, color_space, icc_output, icc_input)

            output_buf = io.BytesIO()
            tifffile.imwrite(
                output_buf,
                img_out,
                photometric="rgb" if img_out.ndim == 3 else "minisblack",
                iccprofile=icc_bytes,
                compression="lzw",
            )
            return output_buf.getvalue(), "tiff"
        elif fmt == ExportFormat.PNG:
            if is_greyscale:
                # PIL "I;16" supports 16-bit greyscale, so keep full bit depth here.
                img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16)
                img_out, icc_bytes = self._apply_color_management_u16_greyscale(
                    img_int, working_color_space, color_space, icc_output, icc_input
                )
                pil_img = Image.fromarray(img_out)
            else:
                # PIL has no 16-bit RGB mode, so RGB PNG is 8-bit (TIFF is the 16-bit
                # lossless path). Mirror the JPEG branch for color management.
                img_int = float_to_uint8(buffer)
                pil_img, icc_bytes = self.apply_color_management(
                    Image.fromarray(img_int), working_color_space, color_space, icc_output, icc_input
                )
            output_buf = io.BytesIO()
            save_kwargs: Dict[str, Any] = {"format": "PNG", "compress_level": 6}
            if icc_bytes:
                save_kwargs["icc_profile"] = icc_bytes
            pil_img.save(output_buf, **save_kwargs)
            return output_buf.getvalue(), "png"
        else:
            img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=8) if is_greyscale else float_to_uint8(buffer)

            pil_img, icc_bytes = self.apply_color_management(
                Image.fromarray(img_int),
                working_color_space,
                color_space,
                icc_output,
                icc_input,
            )
            output_buf = io.BytesIO()
            self._save_to_pil_buffer(pil_img, output_buf, export_settings, icc_bytes)
            return output_buf.getvalue(), "jpg"

    def render_display_array(
        self,
        file_path: str,
        params: WorkspaceConfig,
        source_hash: str,
        target_long_px: int,
        prefer_gpu: bool = True,
        working_color_space: str = WORKING_COLOR_SPACE,
    ) -> Optional[np.ndarray]:
        """Render a file (with its edits) to a small sRGB uint8 RGB array for tiling.

        Mirrors the export render path but at small resolution and in display space,
        so a contact-sheet tile matches the on-canvas look. Returns None on failure.
        """
        try:
            from negpy.infrastructure.display.color_mgmt import apply_display_transform

            f32_buffer, ir_full, _ = self._load_source_f32(file_path, params)
            h_raw, w_raw = f32_buffer.shape[:2]
            scale_factor = max(1.0, max(h_raw, w_raw) / float(target_long_px))

            if prefer_gpu and self.engine_gpu:
                buffer, _ = self.engine_gpu.process(f32_buffer, params, scale_factor=scale_factor, ir_buffer=ir_full)
            else:
                buffer, _ = self.run_pipeline(
                    f32_buffer,
                    params,
                    source_hash,
                    render_size_ref=float(target_long_px),
                    prefer_gpu=False,
                    ir_buffer=ir_full,
                )
                buffer = self._apply_scaling_and_border_f32(buffer, params, params.export)
                self.engine_cpu.cache.clear()

            if isinstance(buffer, np.ndarray) and buffer.ndim == 3 and buffer.shape[2] == 4:
                buffer = buffer[:, :, :3]
            buffer = apply_display_transform(buffer, working_color_space)
            return float_to_uint8(buffer)
        except Exception as e:
            logger.error(f"Contact-sheet tile render failed for {file_path}: {e}")
            return None

    def _apply_scaling_and_border_f32(self, img: np.ndarray, params: WorkspaceConfig, export_settings: ExportConfig) -> np.ndarray:
        """CPU fallback for layout application."""
        result, _ = PrintService.apply_layout(
            img, export_settings, border_size=params.finish.border_size, border_color=params.finish.border_color
        )
        return result

    def _get_target_icc_bytes(self, color_space: str, icc_path: Optional[str]) -> Optional[bytes]:
        """Loads ICC profile data for embedding (custom output profile or target space)."""
        if icc_path and os.path.exists(icc_path):
            with open(icc_path, "rb") as f:
                return f.read()
        path = ColorSpaceRegistry.get_icc_path(color_space)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    @staticmethod
    def _has_custom_icc(input_icc_path: Optional[str], output_icc_path: Optional[str]) -> bool:
        """True when an input or output ICC override file is present."""
        return bool((input_icc_path and os.path.exists(input_icc_path)) or (output_icc_path and os.path.exists(output_icc_path)))

    @staticmethod
    def _resolve_src_profile(working_color_space: str, input_icc_path: Optional[str]) -> Any:
        """Source profile: input ICC override if present, else the working space."""
        if input_icc_path and os.path.exists(input_icc_path):
            return ImageCms.getOpenProfile(input_icc_path)
        path_src = ColorSpaceRegistry.get_icc_path(working_color_space)
        return ImageCms.getOpenProfile(path_src) if path_src and os.path.exists(path_src) else ImageCms.createProfile("sRGB")

    @staticmethod
    def _resolve_dst_profile(color_space: str, output_icc_path: Optional[str]) -> Any:
        """Destination profile: output ICC override if present, else the target space (or None)."""
        if output_icc_path and os.path.exists(output_icc_path):
            return ImageCms.getOpenProfile(output_icc_path)
        path_dst = ColorSpaceRegistry.get_icc_path(color_space)
        return ImageCms.getOpenProfile(path_dst) if path_dst and os.path.exists(path_dst) else None

    @staticmethod
    def _is_print_profile(profile: Any) -> bool:
        """True for a paper/printer output profile (gets a paper-white soft proof)."""
        device_class = (getattr(profile.profile, "device_class", "") or "").strip()
        color_space = (getattr(profile.profile, "xcolor_space", "") or "").strip()
        return device_class == "prtr" or color_space == "CMYK"

    def _apply_color_management_u16_rgb(
        self,
        img_u16: np.ndarray,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[bytes]]:
        """ICC RGB transform for 16-bit arrays (PIL has no 16-bit RGB mode).

        Source is the input override or the working space; destination is the output
        override or the target space. One src→dst transform, so the embedded profile
        matches the pixels.
        """
        has_custom = self._has_custom_icc(input_icc_path, output_icc_path)
        if not has_custom and working_color_space == color_space:
            return img_u16, self._get_target_icc_bytes(color_space, None)
        try:
            p_src = self._resolve_src_profile(working_color_space, input_icc_path)
            p_dst = self._resolve_dst_profile(color_space, output_icc_path)
            if p_dst is None:
                return img_u16, self._get_target_icc_bytes(color_space, None)
            result = apply_icc_u16_rgb(
                img_u16,
                p_src,
                p_dst,
                ImageCms.Intent.RELATIVE_COLORIMETRIC,
                ImageCms.Flags.BLACKPOINTCOMPENSATION,
            )
            return result, self._get_target_icc_bytes(color_space, output_icc_path)
        except Exception as e:
            logger.error(f"CMS transformation failed: {e}")
            return img_u16, None

    def _apply_color_management_u16_greyscale(
        self,
        img_u16: np.ndarray,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[bytes]]:
        """Tag a (H,W) uint16 luma buffer with the target grey profile.

        The buffer is already luma in the working gamma (matching GrayGamma2.2), so no
        pixel transform is run — an RGB working profile can't drive a 1-channel transform.
        """
        return img_u16, self._get_target_icc_bytes(color_space, output_icc_path)

    def apply_color_management(
        self,
        pil_img: Image.Image,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
    ) -> Tuple[Image.Image, Optional[bytes]]:
        """ICC transform for export. Source is the input override or working space;
        destination is the output override or target space."""
        has_custom = self._has_custom_icc(input_icc_path, output_icc_path)
        if not has_custom and working_color_space == color_space:
            return pil_img, self._get_target_icc_bytes(color_space, None)

        try:
            p_src = self._resolve_src_profile(working_color_space, input_icc_path)
            p_dst = self._resolve_dst_profile(color_space, output_icc_path)
            if p_dst is None:
                return pil_img, self._get_target_icc_bytes(color_space, None)

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
            return pil_img, self._get_target_icc_bytes(color_space, output_icc_path)
        except Exception as e:
            logger.error(f"CMS transformation failed: {e}")
            return pil_img, None

    def soft_proof_preview(
        self,
        pil_img: Image.Image,
        working_color_space: str,
        input_icc_path: Optional[str],
        output_icc_path: Optional[str],
        monitor_icc_bytes: Optional[bytes] = None,
    ) -> Image.Image:
        """Soft-proof the preview into display space.

        For a paper/printer output profile, simulate the print on screen (paper white +
        ink) via a proof transform. For an export color space, do a gamut-only proof
        (relative colorimetric + BPC) ending at the display. ``display`` is the monitor
        profile when detected (``monitor_icc_bytes``), else sRGB. The output always
        lands in display space — otherwise it would leak output-space numbers to the
        screen and shift per output space (issue #243). The caller shows the result raw
        (no further display transform).
        """
        try:
            from negpy.infrastructure.display.color_mgmt import open_profile_from_bytes

            # littleCMS needs RGB against the RGB working/output profiles.
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            p_src = self._resolve_src_profile(working_color_space, input_icc_path)
            # Custom output profile, or the working space when only an input is set.
            p_dst = self._resolve_dst_profile(working_color_space, output_icc_path)
            if p_dst is None:
                return pil_img
            # Display the proof lands on: the monitor profile when detected, else sRGB.
            p_display = open_profile_from_bytes(monitor_icc_bytes) if monitor_icc_bytes else ImageCms.createProfile("sRGB")

            if self._is_print_profile(p_dst):
                # Paper/printer profile: simulate the print on screen (paper white + ink)
                # via a proof transform — relative-colorimetric source→paper, then
                # absolute-colorimetric paper→display so the paper white/Dmax show.
                # Handles RGB and CMYK paper profiles (proof space is internal).
                proof = ImageCms.buildProofTransform(
                    p_src,
                    p_display,
                    p_dst,
                    "RGB",
                    "RGB",
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    proofRenderingIntent=ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
                    flags=ImageCms.Flags.SOFTPROOFING,
                )
                result = ImageCms.applyTransform(pil_img, proof)
                return result if result is not None else pil_img

            # Export color space / display-class profile: gamut-only proof. GRAY
            # destinations need an "L" intermediate.
            dst_space = (getattr(p_dst.profile, "xcolor_space", "RGB ") or "RGB ").strip()
            out_mode = "L" if dst_space == "GRAY" else "RGB"
            result = ImageCms.profileToProfile(
                pil_img,
                p_src,
                p_dst,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode=out_mode,
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
            )
            if result is None:
                return pil_img
            result = result if result.mode == "RGB" else result.convert("RGB")
            # Final output → display transform so the proof is shown in display space
            # rather than reinterpreted by the viewer. Always runs (not just when a
            # monitor profile is known): without it the proof would leak output-space
            # numbers to the screen and shift per output space (issue #243). Skipped for
            # GRAY outputs, whose `result` is no longer in `p_dst`'s space after RGB-ising.
            if out_mode == "RGB":
                proofed = ImageCms.profileToProfile(
                    result,
                    p_dst,
                    p_display,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB",
                    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
                )
                if proofed is not None:
                    result = proofed
            return result
        except Exception as e:
            logger.error(f"Soft-proof preview failed: {e}")
            return pil_img

    def _save_to_pil_buffer(
        self,
        pil_img: Image.Image,
        buf: io.BytesIO,
        export_settings,
        icc_bytes: Optional[bytes],
    ) -> None:
        """Encodes PIL image to byte stream."""
        fmt = "JPEG" if export_settings.export_fmt == ExportFormat.JPEG else "TIFF"
        quality = getattr(export_settings, "jpeg_quality", 95)
        pil_img.save(
            buf,
            format=fmt,
            quality=quality,
            subsampling=0,
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
