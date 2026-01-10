import numpy as np
import rawpy
import io
import os
from PIL import Image, ImageOps, ImageCms
from typing import Tuple, Optional, Any

from src.logging_config import get_logger
from src.config import APP_CONFIG
from src.core.models import WorkspaceConfig, ExportConfig
from src.helpers import ensure_rgb
from src.application.engine import DarkroomEngine
from src.infrastructure.loaders.factory import loader_factory

logger = get_logger(__name__)

# Single engine instance for heavy lifting
engine = DarkroomEngine()


def get_best_demosaic_algorithm(raw: Any) -> Any:
    """
    Returns the optimal demosaic algorithm based on sensor type.
    """
    try:
        if raw.raw_type == rawpy.RawType.XTrans:
            return rawpy.DemosaicAlgorithm.XT_1PASS
        return rawpy.DemosaicAlgorithm.LMMSE
    except AttributeError:
        return None


def load_raw_and_process(
    file_path: str,
    params: WorkspaceConfig,
    export_settings: ExportConfig,
    source_hash: str,
) -> Tuple[Optional[bytes], str]:
    """
    Worker function for high-resolution processing and export.
    """
    try:
        color_space = str(export_settings.export_color_space)
        raw_color_space = rawpy.ColorSpace.sRGB
        if color_space == "Adobe RGB":
            raw_color_space = rawpy.ColorSpace.Adobe

        with loader_factory.get_loader(file_path) as raw:
            algo = get_best_demosaic_algorithm(raw)

            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                user_wb=[1, 1, 1, 1],
                output_bps=16,
                output_color=raw_color_space,
                demosaic_algorithm=algo,
            )
            rgb = ensure_rgb(rgb)

        img = rgb.astype(np.float32) / 65535.0

        # Execute Photometric Engine
        img = engine.process(img, params, source_hash=source_hash)

        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        # Scale and Add Border
        pil_img = _apply_scaling_and_border(pil_img, params, export_settings)

        # Handle Color Management
        pil_img, target_icc_bytes = _apply_color_management(
            pil_img, color_space, export_settings.icc_profile_path
        )

        # Encode to target format
        output_buf = io.BytesIO()
        ext = "jpg" if export_settings.export_fmt == "JPEG" else "tiff"
        _save_to_buffer(pil_img, output_buf, export_settings, target_icc_bytes)

        return output_buf.getvalue(), ext
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return None, str(e)


def _apply_scaling_and_border(
    pil_img: Image.Image, params: WorkspaceConfig, export_settings: ExportConfig
) -> Image.Image:
    side_inch = export_settings.export_print_size / 2.54
    dpi = export_settings.export_dpi
    total_target_px = int(side_inch * dpi)
    border_px = (
        int((export_settings.export_border_size / 2.54) * dpi)
        if export_settings.export_add_border
        else 0
    )
    content_target_px = max(10, total_target_px - (2 * border_px))

    if pil_img.width >= pil_img.height:
        target_w = content_target_px
        target_h = int(target_w * (pil_img.height / pil_img.width))
    else:
        target_h = content_target_px
        target_w = int(target_h * (pil_img.width / pil_img.height))

    pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    is_toned = (
        params.toning.selenium_strength != 0.0
        or params.toning.sepia_strength != 0.0
        or params.toning.paper_profile != "None"
    )
    if export_settings.export_color_space == "Greyscale" or (
        params.process_mode == "B&W" and not is_toned
    ):
        pil_img = pil_img.convert("L")

    if export_settings.export_add_border and border_px > 0:
        pil_img = ImageOps.expand(
            pil_img, border=border_px, fill=export_settings.export_border_color
        )

    return pil_img


def _apply_color_management(
    pil_img: Image.Image, color_space: str, icc_path: Optional[str]
) -> Tuple[Image.Image, Optional[bytes]]:
    target_icc_bytes = None
    if icc_path and os.path.exists(icc_path):
        try:
            profile_src: Any
            if color_space == "Adobe RGB" and os.path.exists(
                APP_CONFIG.adobe_rgb_profile
            ):
                profile_src = ImageCms.getOpenProfile(APP_CONFIG.adobe_rgb_profile)
            else:
                profile_src = ImageCms.createProfile("sRGB")

            dst_profile: Any = ImageCms.getOpenProfile(icc_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            result_pil = ImageCms.profileToProfile(
                pil_img,
                profile_src,
                dst_profile,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB",
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
            )
            if result_pil is not None:
                pil_img = result_pil

            with open(icc_path, "rb") as f:
                target_icc_bytes = f.read()
        except Exception as e:
            logger.error(f"ICC Error: {e}")
    elif color_space == "Adobe RGB" and os.path.exists(APP_CONFIG.adobe_rgb_profile):
        with open(APP_CONFIG.adobe_rgb_profile, "rb") as f:
            target_icc_bytes = f.read()

    return pil_img, target_icc_bytes


def _save_to_buffer(
    pil_img: Image.Image,
    buf: io.BytesIO,
    export_settings: ExportConfig,
    icc_bytes: Optional[bytes],
) -> None:
    if export_settings.export_fmt == "JPEG":
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
