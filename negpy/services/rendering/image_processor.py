import os
import io
import rawpy
import tifffile
import imagecodecs
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
from negpy.features.process.logic import linear_raw_token
from negpy.features.exposure.models import RenderIntent
from negpy.features.flatfield.logic import apply_flatfield, flatfield_token
from negpy.features.rgbscan.logic import merge_rgb_triplet, rgbscan_token
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
    working_oetf_decode,
)
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm
from negpy.services.export.print import PrintService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry, WORKING_COLOR_SPACE
from negpy.infrastructure.display.icc_lut import apply_icc_u16_rgb

logger = get_logger(__name__)

# (photometric, primaries, transfer) for JXL's enumerated color encoding (D65 white
# only, no ICC). Other spaces must hard-fail. Transfers verified against the bundled
# icc/*.icc: Rec 2020 uses the Rec.709/BT.2020 OETF (BT709, not sRGB) and
# GrayGamma2.2.icc holds the sRGB TRC despite its name (SRGB, not gamma 2.2).
_JXL_COLOR = {
    ColorSpace.SRGB.value: ("RGB", "SRGB", "SRGB"),
    ColorSpace.P3_D65.value: ("RGB", "P3", "SRGB"),
    ColorSpace.REC2020.value: ("RGB", "BT2100", "BT709"),
    ColorSpace.GREYSCALE.value: ("GRAY", None, "SRGB"),
}


class ImageProcessor:
    """
    Coordinates multi-backend image processing.
    Seamlessly switches between CPU (DarkroomEngine) and GPU (GPUEngine).
    """

    def __init__(self) -> None:
        self.engine_cpu = DarkroomEngine()
        self.engine_gpu: Optional[GPUEngine] = None

        # Last decoded full-res source (decode+flatfield) — skips re-decode on repeat export.
        # One entry only (full-res buffers are large); treated read-only downstream.
        self._source_cache_key: Optional[tuple] = None
        self._source_cache_value: Optional[Tuple[np.ndarray, Optional[np.ndarray], str]] = None

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

    @staticmethod
    def _is_flat(settings: WorkspaceConfig) -> bool:
        """Flat (digital-intermediate) renders run on the CPU engine only, so the
        master is numerically exact and never subject to the looser GPU parity."""
        return settings.exposure.render_intent == RenderIntent.FLAT

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
        crop_preview_full: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes rendering pipeline. Returns result (ndarray/GPUTexture) and metrics.
        """
        # Flat-field is a source pre-correction (before geometry/crop); folding its token
        # into source_hash invalidates the engine cache when it changes.
        img = apply_flatfield(img, settings.flatfield)
        source_hash = (
            source_hash + flatfield_token(settings.flatfield) + rgbscan_token(settings.rgbscan) + linear_raw_token(settings.process)
        )

        h_orig, w_cols = img.shape[:2]
        scale_factor = max(h_orig, w_cols) / float(APP_CONFIG.preview_render_size)

        context = PipelineContext(
            scale_factor=scale_factor,
            original_size=(h_orig, w_cols),
            process_mode=settings.process.process_mode,
            ir_buffer=ir_buffer,
            crop_preview_full=crop_preview_full,
        )
        if metrics:
            context.metrics.update(metrics)

        if self._is_flat(settings) or crop_preview_full:
            # The crop tool's "show full uncropped frame" preview only needs a single
            # CPU render per settings change (dragging only moves an overlay rect), so
            # we sidestep the GPU engine's ROI-fused compute dispatch entirely here.
            prefer_gpu = False

        if prefer_gpu and self.engine_gpu:
            try:
                processed, gpu_metrics = self.engine_gpu.process_to_texture(
                    img,
                    settings,
                    scale_factor=scale_factor,
                    render_size_ref=render_size_ref,
                    readback_metrics=readback_metrics,
                    ir_buffer=ir_buffer,
                    analysis_source_hash=source_hash,
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

    def _decode_sensor_rgb(self, file_path: str, linear_raw: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode one RAW to sensor-native (output_color=raw), linear uint16 RGB.

        Returns (rgb_uint16, loader_metadata).
        """
        ctx_mgr, metadata = loader_factory.get_loader(file_path)
        with ctx_mgr as raw:
            algo = get_best_demosaic_algorithm(raw)
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
        return rgb, metadata

    def _load_source_f32(self, file_path: str, params: WorkspaceConfig) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Decode a source file to a flatfield-corrected, EXIF-oriented float32 buffer.

        Returns (f32_buffer, ir_buffer, source_color_space).
        """
        linear_raw = params.process.linear_raw

        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0.0
        cache_key = (file_path, mtime, linear_raw, rgbscan_token(params.rgbscan), flatfield_token(params.flatfield))
        if cache_key == self._source_cache_key and self._source_cache_value is not None:
            return self._source_cache_value

        rgb, metadata = self._decode_sensor_rgb(file_path, linear_raw)
        source_cs = str(metadata.get("color_space", ColorSpace.ADOBE_RGB.value))
        ir_full = metadata.get("ir")

        rgbcfg = params.rgbscan
        if rgbcfg.enabled and rgbcfg.green_path and rgbcfg.blue_path:
            # Assemble one frame from the R/G/B exposures; the primary (red) file is
            # already decoded above, so reuse it and decode only green/blue.
            def _decode(path: str) -> np.ndarray:
                if path == file_path:
                    return rgb
                return self._decode_sensor_rgb(path, linear_raw)[0]

            rgb = merge_rgb_triplet(_decode, file_path, rgbcfg.green_path, rgbcfg.blue_path, align=rgbcfg.align)

        f32_buffer = uint16_to_float32(rgb)

        orientation = metadata.get("orientation", 1)
        f32_buffer = apply_exif_orientation(f32_buffer, orientation)
        f32_buffer = apply_flatfield(f32_buffer, params.flatfield)
        if ir_full is not None:
            ir_full = apply_exif_orientation(ir_full, orientation)
        result = (f32_buffer, ir_full, source_cs)
        self._source_cache_key = cache_key
        self._source_cache_value = result
        return result

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

            if self._is_flat(params):
                prefer_gpu = False

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
        elif fmt == ExportFormat.DNG:
            # Linear digital-negative master. Greyscale is promoted to RGB so the DNG
            # is always a 3-sample LinearRaw the host can open. Colour-managed to the
            # target space so the values match the TIFF master.
            if is_greyscale:
                img_lum = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16)
                img_int = np.stack([img_lum, img_lum, img_lum], axis=-1) if img_lum.ndim == 2 else img_lum
            else:
                img_int = float_to_uint16(buffer)
            img_out, _icc = self._apply_color_management_u16_rgb(img_int, working_color_space, color_space, icc_output, icc_input)
            return self._encode_dng_bytes(img_out), "dng"
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
        elif fmt == ExportFormat.JXL:
            tag = _JXL_COLOR.get(color_space)
            if tag is None:
                raise ValueError(
                    f"JPEG XL export does not support the {color_space} color space. "
                    "Use sRGB, P3 D65, Rec 2020, or Greyscale, or pick another format."
                )
            photometric, primaries, transfer = tag
            # 16-bit, colour-managed to target; ICC discarded (libjxl tags enumeratively).
            if is_greyscale:
                img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16)
                img_out, _icc = self._apply_color_management_u16_greyscale(img_int, working_color_space, color_space, icc_output, icc_input)
            else:
                img_int = float_to_uint16(buffer)
                img_out, _icc = self._apply_color_management_u16_rgb(img_int, working_color_space, color_space, icc_output, icc_input)
            bits = imagecodecs.jpegxl_encode(
                np.ascontiguousarray(img_out),
                bitspersample=16,
                photometric=photometric,
                primaries=primaries,
                transfer=transfer,
                lossless=export_settings.jxl_lossless,
                distance=None if export_settings.jxl_lossless else export_settings.jxl_distance,
                effort=export_settings.jxl_effort,
                numthreads=0,  # all cores; single-threaded otherwise (~7x slower)
            )
            return bytes(bits), "jxl"
        elif fmt == ExportFormat.WEBP:
            # 8-bit only (WebP has no higher bit depth). Lossy or lossless via a
            # flag; PIL embeds the ICC profile for any colour space.
            img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=8) if is_greyscale else float_to_uint8(buffer)
            pil_img, icc_bytes = self.apply_color_management(
                Image.fromarray(img_int), working_color_space, color_space, icc_output, icc_input
            )
            if max(pil_img.size) > 16383:
                raise ValueError("WebP max dimension is 16383 px; use TIFF/PNG for larger exports.")
            output_buf = io.BytesIO()
            save_kwargs: Dict[str, Any] = {
                "format": "WEBP",
                "lossless": export_settings.webp_lossless,
                "quality": export_settings.webp_quality,
                "method": export_settings.webp_method,
            }
            if icc_bytes:
                save_kwargs["icc_profile"] = icc_bytes
            pil_img.save(output_buf, **save_kwargs)
            return output_buf.getvalue(), "webp"
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

    @staticmethod
    def _encode_dng_bytes(rgb_u16: np.ndarray) -> bytes:
        """Write a 16-bit RGB buffer as a LinearRaw DNG and return its bytes."""
        import shutil
        import tempfile

        from negpy.infrastructure.scanners.result import ScanResult
        from negpy.services.scanning.writer import write_dng_linear

        result = ScanResult(rgb=np.ascontiguousarray(rgb_u16), ir=None, dpi=300, device_model="NegPy Flat Master")
        tmpdir = tempfile.mkdtemp()
        try:
            written = write_dng_linear(result, os.path.join(tmpdir, "flat_master"))
            with open(written, "rb") as fh:
                return fh.read()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

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

            if self._is_flat(params):
                prefer_gpu = False

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
        """Re-encode a (H,W) uint16 luma buffer to the target grey profile's gamma.

        The buffer is luma in the working TRC (ProPhoto ROMM 1.8); the grey profile
        (GrayGamma2.2) expects a 2.2 gamma. Decode the working TRC to linear and
        re-encode to 2.2 so the tagged output matches — an RGB working profile can't
        drive a 1-channel ICC transform.
        """
        if working_color_space == color_space:
            return img_u16, self._get_target_icc_bytes(color_space, output_icc_path)
        lin = np.asarray(working_oetf_decode(img_u16.astype(np.float32) / 65535.0))
        gray = np.clip(lin, 0.0, 1.0) ** (1.0 / 2.2)
        out = np.clip(gray * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
        return out, self._get_target_icc_bytes(color_space, output_icc_path)

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
        self._source_cache_key = None
        self._source_cache_value = None
        if self.engine_gpu:
            self.engine_gpu.cleanup()

    def destroy_all(self) -> None:
        """Teardown GPU engine."""
        if self.engine_gpu:
            self.engine_gpu.destroy_all()
