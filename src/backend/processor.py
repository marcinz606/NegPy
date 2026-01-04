import numpy as np
import rawpy
import io
import os
from PIL import Image, ImageOps, ImageCms
from typing import Tuple, Optional, Any

from src.logging_config import get_logger
from src.config import APP_CONFIG
from src.domain_objects import ImageSettings, ExportSettings
from src.helpers import ensure_rgb, imread_raw
from src.backend.engine import DarkroomEngine

logger = get_logger(__name__)

# Single engine instance
engine = DarkroomEngine()


def process_image_core(img: np.ndarray, params: ImageSettings) -> np.ndarray:
    """
    Orchestrates the image processing pipeline by delegating to the DarkroomEngine.
    """
    return engine.process(img, params)


def load_raw_and_process(
    file_path: str,
    params: ImageSettings,
    export_settings: ExportSettings,
) -> Tuple[Optional[bytes], str]:
    """
    Worker function for processing a RAW file to a final output format (JPEG/TIFF).
    """
    try:
        output_format = export_settings.output_format
        print_width_cm = float(export_settings.print_width_cm)
        dpi = int(export_settings.dpi)
        add_border = bool(export_settings.add_border)
        border_size_cm = float(export_settings.border_size_cm)
        border_color = str(export_settings.border_color)
        icc_profile_path = export_settings.icc_profile_path
        color_space = str(export_settings.color_space)

        raw_color_space = rawpy.ColorSpace.sRGB
        if color_space == "Adobe RGB":
            raw_color_space = rawpy.ColorSpace.Adobe

        with imread_raw(file_path) as raw:
            # PURE RAW: Essential for darkroom-style analytical WB.
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                user_wb=[1, 1, 1, 1],
                output_bps=16,
                output_color=raw_color_space,
            )
            rgb = ensure_rgb(rgb)
        img = rgb.astype(np.float32) / 65535.0

        # Delegate to Engine
        img = engine.process(img, params)

        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        # Resizing
        side_inch = print_width_cm / 2.54
        total_target_px = int(side_inch * dpi)
        border_px = int((border_size_cm / 2.54) * dpi) if add_border else 0
        content_target_px = max(10, total_target_px - (2 * border_px))

        if pil_img.width >= pil_img.height:
            target_w = content_target_px
            target_h = int(target_w * (pil_img.height / pil_img.width))
        else:
            target_h = content_target_px
            target_w = int(target_h * (pil_img.width / pil_img.height))

        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        is_toned = (
            params.selenium_strength != 0.0
            or params.sepia_strength != 0.0
            or params.paper_profile != "None"
        )

        if color_space == "Greyscale" or (params.is_bw and not is_toned):
            pil_img = pil_img.convert("L")

        # ICC Profile Handling
        target_icc_bytes = None
        if icc_profile_path and os.path.exists(icc_profile_path):
            try:
                profile_src: Any
                if color_space == "Adobe RGB" and os.path.exists(
                    APP_CONFIG.adobe_rgb_profile
                ):
                    profile_src = ImageCms.getOpenProfile(APP_CONFIG.adobe_rgb_profile)
                else:
                    profile_src = ImageCms.createProfile("sRGB")

                dst_profile: Any = ImageCms.getOpenProfile(icc_profile_path)
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

                with open(icc_profile_path, "rb") as f_icc:
                    target_icc_bytes = f_icc.read()
            except Exception as e:
                logger.error(f"ICC Error: {e}")
        elif color_space == "Adobe RGB":
            if os.path.exists(APP_CONFIG.adobe_rgb_profile):
                with open(APP_CONFIG.adobe_rgb_profile, "rb") as f_adobe:
                    target_icc_bytes = f_adobe.read()

        if add_border and border_px > 0:
            pil_img = ImageOps.expand(pil_img, border=border_px, fill=border_color)

        output_buf = io.BytesIO()
        ext = "jpg" if output_format == "JPEG" else "tiff"
        if output_format == "JPEG":
            pil_img.save(
                output_buf,
                format="JPEG",
                quality=95,
                dpi=(dpi, dpi),
                icc_profile=target_icc_bytes,
            )
        else:
            pil_img.save(
                output_buf,
                format="TIFF",
                compression="tiff_lzw",
                dpi=(dpi, dpi),
                icc_profile=target_icc_bytes,
            )
        return output_buf.getvalue(), ext
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return None, str(e)
