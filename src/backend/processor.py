import numpy as np
import rawpy
import io
import os
from PIL import Image, ImageOps, ImageCms
from typing import Tuple, Optional, cast

from src.logging_config import get_logger
from src.config import APP_CONFIG
from src.domain_objects import ProcessingParams, ExportSettings
from src.helpers import ensure_rgb, imread_raw
from src.backend.image_logic.post import (
    apply_post_color_grading,
    apply_output_sharpening,
)
from src.backend.image_logic.color import (
    apply_shadow_desaturation,
    calculate_auto_mask_wb,
    apply_manual_color_balance_neg,
    apply_shadow_highlight_grading,
    convert_to_monochrome,
    apply_color_separation,
    measure_film_base,
)
from src.backend.image_logic.exposure import (
    apply_contrast,
    apply_film_characteristic_curve,
    apply_chromaticity_preserving_black_point,
    measure_log_negative_bounds,
)
from src.backend.image_logic.gamma import apply_gamma_to_img
from src.backend.image_logic.retouch import (
    apply_autocrop,
    apply_dust_removal,
    apply_chroma_noise_removal,
    apply_fine_rotation,
)
from src.backend.image_logic.local import apply_local_adjustments

logger = get_logger(__name__)


def process_image_core(img: np.ndarray, params: ProcessingParams) -> np.ndarray:
    """
    High-level orchestration of the image processing pipeline.
    True Darkroom Model: Filtered Light -> Negative -> Scanner Gain -> Inversion.
    """
    img = ensure_rgb(img)

    h_orig, w_cols = img.shape[:2]
    scale_factor = max(h_orig, w_cols) / float(APP_CONFIG["preview_max_res"])

    is_bw = params.get("process_mode") == "B&W"

    # 1. Log-Expansion Normalization (Scanner Gain Simulation)
    # Move to Log Domain first
    epsilon = 1e-6
    img_log = np.log10(np.clip(img, epsilon, 1.0))
    
    # Per-Channel Stretch in Log Domain
    # This preserves stop ratios and maximizes ECN-2 contrast potential.
    floors, ceils = measure_log_negative_bounds(img)
    for ch in range(3):
        f, c = floors[ch], ceils[ch]
        img_log[:, :, ch] = np.clip((img_log[:, :, ch] - f) / (max(c - f, epsilon)), 0, 1)

    # 2. Normalization & Scanner Gain (Photometric Characteristic Curve)
    
    # In normalized log space: 0.0 is D-max (Shadows), 1.0 is D-min (Base).
    # Master Reference is now anchored to the top of the normalized range.
    master_ref = 1.0
    
    # User Parameters
    density = params.get("density", 1.0)
    wb_cyan = params.get("wb_cyan", 0.0)
    wb_magenta = params.get("wb_magenta", 0.0)
    wb_yellow = params.get("wb_yellow", 0.0)
    
    # Exposure Shift Logic:
    # 1.0 (Normal) corresponds to shift of 0.35 units (High-Key focus)
    exposure_shift = 0.1 + (density * 0.25)
    
    pivot_r = master_ref - exposure_shift - (wb_cyan / 100.0)
    pivot_g = master_ref - exposure_shift - (wb_magenta / 100.0)
    pivot_b = master_ref - exposure_shift - (wb_yellow / 100.0)
    
    # Slope (Contrast) - High-Latitude Formula
    # UI 2.0 -> Slope 3.4. UI 2.5 -> Slope 4.0. UI 5.0 -> Slope 7.0.
    # This gives flat ECN-2 the expansion it needs.
    current_slope = 1.0 + (params.get("grade", 2.0) * 1.2)

    # Construct per-channel parameters
    params_r = (pivot_r, current_slope)
    params_g = (pivot_g, current_slope)
    params_b = (pivot_b, current_slope)

    # Extract Recovery Params
    toe = float(params.get("toe", 0.0))
    shoulder = float(params.get("shoulder", 0.0))

    # Apply Photometric Characteristic Curve (Returns Positive)
    # NOTE: Since we are already in Log Domain, we must bypass the Log10 
    # step inside apply_film_characteristic_curve if it exists, or update it.
    img_pos = apply_film_characteristic_curve(
        img_log, # Passing log data directly
        params_r,
        params_g,
        params_b,
        toe=toe,
        shoulder=shoulder,
        pre_logged=True # Pass flag to skip internal log
    )
    img = np.clip(img_pos, 0, 1)    
    # If B&W mode, convert to monochrome base BEFORE toning
    if is_bw:
        img = convert_to_monochrome(img)
    
        # 3. Paper Toning (Legacy Negative Domain Logic)
        # The Toning functions expect a Negative (Bright=Shadows).
        # Since we now have a Positive, we must invert to Negative for Toning, then invert back.
        img_neg = 1.0 - img
        
        img_neg = apply_manual_color_balance_neg(img_neg, params)
        img_neg = apply_shadow_highlight_grading(img_neg, params)
        
        # Inversion (Creating the positive back from toned negative)
        img = 1.0 - img_neg
    
        # 5. Black Point (Luminance-based)
        img = apply_chromaticity_preserving_black_point(img, 0.05)
    # 6. Retouching
    img = apply_dust_removal(img, params, scale_factor)
    img = apply_chroma_noise_removal(img, params, scale_factor)

    # 7. Post-Inversion Tonality
    # Exposure (Linear trim)
    img *= 2.0 ** params.get("exposure", 0.0)

    img = apply_local_adjustments(
        img, params.get("local_adjustments", []), scale_factor
    )

    if not is_bw:
        img = apply_color_separation(img, params.get("color_separation", 1.0))
        img *= params.get("saturation", 1.0)

    img = apply_shadow_desaturation(img, params.get("shadow_desat_strength", 1.0))

    # 9. Geometry
    rotation = params.get("rotation", 0)
    if rotation != 0:
        img = np.rot90(img, k=rotation)

    fine_rot = params.get("fine_rotation", 0.0)
    if fine_rot != 0.0:
        img = apply_fine_rotation(img, fine_rot)

    if params.get("autocrop"):
        img = apply_autocrop(
            img,
            offset_px=params.get("autocrop_offset", 0),
            scale_factor=scale_factor,
            ratio=params.get("autocrop_ratio", "3:2"),
        )

    return img


def load_raw_and_process(
    file_path: str,
    params: ProcessingParams,
    export_settings: ExportSettings,
) -> Tuple[Optional[bytes], str]:
    """
    Worker function for processing a RAW file to a final output format (JPEG/TIFF).
    """
    try:
        output_format = export_settings.get("output_format", "JPEG")
        print_width_cm = float(export_settings.get("print_width_cm", 27.0))
        dpi = int(export_settings.get("dpi", 300))
        sharpen_amount = float(export_settings.get("sharpen_amount", 0.75))
        add_border = bool(export_settings.get("add_border", False))
        border_size_cm = float(export_settings.get("border_size_cm", 1.0))
        border_color = str(export_settings.get("border_color", "#000000"))
        icc_profile_path = export_settings.get("icc_profile_path")
        color_space = str(export_settings.get("color_space", "sRGB"))

        # Determine Rawpy Output Color Space
        # 0=raw, 1=sRGB, 2=Adobe, 3=Wide, 4=ProPhoto, 5=XYZ, 6=ACES
        raw_color_space = rawpy.ColorSpace.sRGB
        if color_space == "Adobe RGB":
            raw_color_space = rawpy.ColorSpace.Adobe

        with imread_raw(file_path) as raw:
            # PURE RAW: Use [1,1,1,1] to see the full physical orange mask.
            # This is essential for our darkroom-style analytical WB to work.
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

        if params.get("auto_wb"):
            m_r, m_g, m_b = calculate_auto_mask_wb(img)
            # These will be converted to CMY in the frontend
            params["wb_manual_r"], params["wb_manual_g"], params["wb_manual_b"] = (
                m_r,
                m_g,
                m_b,
            )

        img = process_image_core(img, params)
        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        pil_img = apply_post_color_grading(pil_img, params)

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

        # Sharpening
        pil_img = apply_output_sharpening(pil_img, sharpen_amount)

        # Force Greyscale if requested or if B&W and NOT toned
        is_bw = params.get("is_bw", False)
        is_toned = (
            params.get("temperature", 0.0) != 0.0
            or params.get("shadow_temp", 0.0) != 0.0
            or params.get("highlight_temp", 0.0) != 0.0
        )

        if color_space == "Greyscale" or (is_bw and not is_toned):
            pil_img = pil_img.convert("L")

        # ICC Profile Handling
        target_icc_bytes = None

        # 1. Custom Soft-Proofing Profile (User uploaded)
        if icc_profile_path and os.path.exists(icc_profile_path):
            try:
                src_profile = cast(
                    ImageCms.ImageCmsProfile, ImageCms.createProfile("sRGB")
                )
                if color_space == "Adobe RGB" and os.path.exists(
                    APP_CONFIG.get("adobe_rgb_profile", "")
                ):
                    src_profile = ImageCms.getOpenProfile(
                        APP_CONFIG["adobe_rgb_profile"]
                    )

                dst_profile = ImageCms.getOpenProfile(icc_profile_path)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                pil_img = cast(
                    Image.Image,
                    ImageCms.profileToProfile(
                        pil_img,
                        src_profile,
                        dst_profile,
                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                        outputMode="RGB",
                        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
                    ),
                )
                with open(icc_profile_path, "rb") as f_icc:
                    target_icc_bytes = f_icc.read()
            except Exception as e:
                logger.error(f"ICC Error: {e}")

        # 2. Standard Color Space Tagging (if no custom profile applied)
        elif color_space == "Adobe RGB":
            adobe_path = APP_CONFIG.get("adobe_rgb_profile")
            if adobe_path and os.path.exists(adobe_path):
                with open(adobe_path, "rb") as f_adobe:
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
