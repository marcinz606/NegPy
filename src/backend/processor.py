import numpy as np
import rawpy
import io
import os
import traceback
from PIL import Image, ImageOps, ImageCms
from typing import Dict, Any, Tuple, Optional

from src.backend.image_logic.post import apply_post_color_grading, apply_output_sharpening
from src.backend.config import APP_CONFIG
from src.backend.image_logic.color import (
    apply_shadow_desaturation,
    calculate_auto_mask_wb, 
    apply_manual_color_balance_neg,
    apply_shadow_highlight_grading
)
from src.backend.image_logic.exposure import (
    apply_contrast,
    apply_scan_gain_with_toe,
    apply_chromaticity_preserving_black_point
)
from src.backend.image_logic.gamma import (
    apply_gamma_to_img
)
from src.backend.image_logic.retouch import (
    apply_autocrop, 
    apply_dust_removal, 
    apply_chroma_noise_removal,
    apply_fine_rotation
)
from src.backend.image_logic.local import apply_local_adjustments

def process_image_core(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    High-level orchestration of the image processing pipeline.
    True Darkroom Model: Filtered Light -> Negative -> Scanner Gain -> Inversion.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    h_orig, w_cols = img.shape[:2]
    scale_factor = max(h_orig, w_cols) / float(APP_CONFIG['preview_max_res'])

    is_bw = params.get('process_mode') == 'B&W'

    # 1. White Balance (Dichroic Filtration - APPLIED FIRST)
    # Only apply filtration in color mode.
    if not is_bw:
        img[:, :, 0] *= params.get('wb_manual_r', 1.0)
        img[:, :, 1] *= params.get('wb_manual_g', 1.0)
        img[:, :, 2] *= params.get('wb_manual_b', 1.0)
        img = np.clip(img, 0, 1)

    # 2. Normalization & Scanner Gain
    # We identify the mask level AFTER filtration.
    shadow_bases = np.percentile(img, 99.5, axis=(0, 1))
    
    scan_gain = params.get('scan_gain', 1.0)
    channel_gains = [scan_gain, scan_gain, scan_gain]

    # Apply tonality recovery based on the filtered/normalized data
    img = apply_scan_gain_with_toe(
        img, 
        channel_gains, 
        shadow_toe=params.get('scan_gain_s_toe', 0.0),
        highlight_shoulder=params.get('scan_gain_h_shoulder', 0.0),
        shadow_bases=shadow_bases
    )
    img = np.clip(img, 0, 1)

    # If B&W mode, convert to monochrome base BEFORE toning
    if is_bw and img.shape[2] == 3:
        gray = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        img = np.stack([gray, gray, gray], axis=2)

    # 3. Paper Toning (Negative Domain)
    img = apply_manual_color_balance_neg(img, params)
    img = apply_shadow_highlight_grading(img, params)

    # 4. Inversion (Creating the positive)
    img = 1.0 - img

    # 5. Black Point (Luminance-based)
    img = apply_chromaticity_preserving_black_point(img, 0.05)

    # 6. Retouching
    img = apply_dust_removal(img, params, scale_factor)
    img = apply_chroma_noise_removal(img, params, scale_factor)

    # 7. Post-Inversion Tonality
    # Exposure (Linear trim)
    img *= (2.0 ** params.get('exposure', 0.0))
    
    img = apply_local_adjustments(img, params.get('local_adjustments', []), scale_factor)

    # Apply global Gamma (Paper Grade - Coupled with Grade slider)
    img = apply_gamma_to_img(img, params.get('gamma', 1.0), mode=params.get('gamma_mode', 'Standard'))
    
    img = apply_contrast(img, params.get('contrast', 1.0))
    img = apply_shadow_desaturation(img, params.get('shadow_desat_strength', 1.0))

    # 9. Geometry
    rotation = params.get('rotation', 0)
    if rotation != 0:
        img = np.rot90(img, k=rotation)

    fine_rot = params.get('fine_rotation', 0.0)
    if fine_rot != 0.0:
        img = apply_fine_rotation(img, fine_rot)

    if params.get('autocrop'):
        img = apply_autocrop(img, offset_px=params.get('autocrop_offset', 0), scale_factor=scale_factor, ratio=params.get('autocrop_ratio', '3:2'))

    return img

def load_raw_and_process(file_bytes: bytes, params: Dict[str, Any], output_format: str, print_width_cm: float, dpi: int, sharpen_amount: float, filename: str = "", add_border: bool = False, border_size_cm: float = 1.0, border_color: str = "#000000", icc_profile_path: Optional[str] = None, color_space: str = "Adobe RGB") -> Tuple[Optional[bytes], str]:
    """
    Worker function for processing a RAW file to a final output format (JPEG/TIFF).
    """
    try:
        # Determine Rawpy Output Color Space
        # 0=raw, 1=sRGB, 2=Adobe, 3=Wide, 4=ProPhoto, 5=XYZ, 6=ACES
        raw_color_space = rawpy.ColorSpace.sRGB
        if color_space == "Adobe RGB":
            raw_color_space = rawpy.ColorSpace.Adobe
            
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            # PURE RAW: Use [1,1,1,1] to see the full physical orange mask.
            # This is essential for our darkroom-style analytical WB to work.
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, user_wb=[1, 1, 1, 1], output_bps=16, output_color=raw_color_space)
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
        img = rgb.astype(np.float32) / 65535.0
        
        if params.get('auto_wb'):
            m_r, m_g, m_b = calculate_auto_mask_wb(img)
            # These will be converted to CMY in the frontend
            params['wb_manual_r'], params['wb_manual_g'], params['wb_manual_b'] = m_r, m_g, m_b

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
        is_bw = params.get('is_bw', False)
        is_toned = (params.get('temperature', 0.0) != 0.0 or 
                    params.get('shadow_temp', 0.0) != 0.0 or 
                    params.get('highlight_temp', 0.0) != 0.0)
        
        if color_space == "Greyscale" or (is_bw and not is_toned):
            pil_img = pil_img.convert("L")

        # ICC Profile Handling
        target_icc_bytes = None
        
        # 1. Custom Soft-Proofing Profile (User uploaded)
        if icc_profile_path and os.path.exists(icc_profile_path):
            try:
                src_profile = ImageCms.createProfile("sRGB")
                if color_space == "Adobe RGB" and os.path.exists(APP_CONFIG.get('adobe_rgb_profile', '')):
                     src_profile = ImageCms.getOpenProfile(APP_CONFIG['adobe_rgb_profile'])
                     
                dst_profile = ImageCms.getOpenProfile(icc_profile_path)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                pil_img = ImageCms.profileToProfile(pil_img, src_profile, dst_profile, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode='RGB', flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
                with open(icc_profile_path, "rb") as f:
                    target_icc_bytes = f.read()
            except Exception as e:
                print(f"ICC Error: {e}")
            
        # 2. Standard Color Space Tagging (if no custom profile applied)
        elif color_space == "Adobe RGB":
            adobe_path = APP_CONFIG.get('adobe_rgb_profile')
            if adobe_path and os.path.exists(adobe_path):
                with open(adobe_path, "rb") as f:
                    target_icc_bytes = f.read()

        if add_border and border_px > 0:
            pil_img = ImageOps.expand(pil_img, border=border_px, fill=border_color)

        output_buf = io.BytesIO()
        ext = "jpg" if output_format == 'JPEG' else "tiff"
        if output_format == 'JPEG':
            pil_img.save(output_buf, format="JPEG", quality=95, dpi=(dpi, dpi), icc_profile=target_icc_bytes)
        else:
            pil_img.save(output_buf, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi), icc_profile=target_icc_bytes)
        return output_buf.getvalue(), ext
    except Exception as e:
        print(traceback.format_exc())
        return None, str(e)
