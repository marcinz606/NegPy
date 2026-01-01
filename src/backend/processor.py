import numpy as np
import cv2
import rawpy
import io
import os
import traceback
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageCms
from typing import Dict, Any, Tuple, Optional

from src.backend.utils import apply_color_separation
from src.backend.config import APP_CONFIG
from src.backend.image_logic.color import (
    get_luminance,
    apply_shadow_desaturation,
    calculate_auto_mask_wb, 
    apply_manual_color_balance_neg,
    apply_shadow_highlight_grading
)
from src.backend.image_logic.exposure import (
    apply_contrast,
    apply_scan_gain_with_toe,
    apply_split_exposure,
    apply_chromaticity_preserving_black_point
)
from src.backend.image_logic.gamma import (
    apply_gamma_to_img,
    apply_split_gamma,
    calculate_balancing_gammas
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

    h_orig, w_cols = img.shape[:2]
    scale_factor = max(h_orig, w_cols) / float(APP_CONFIG['preview_max_res'])

    # 1. White Balance (Dichroic Filtration - APPLIED FIRST)
    # This mimics the filtered light source of an enlarger.
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

    # 3. Paper Toning (Negative Domain)
    img = apply_manual_color_balance_neg(img, params)
    img = apply_shadow_highlight_grading(img, params)

    # 4. Inversion (Creating the positive)
    img = 1.0 - img

    # 5. Black Point (Luminance-based)
    img = apply_chromaticity_preserving_black_point(img, 0.05)

    # 6. Retouching
    img = apply_dust_removal(img, params, scale_factor)
    img = apply_chroma_noise_removal(img, params)

    # 7. Post-Inversion Tonality
    # Exposure (Linear trim)
    img *= (2.0 ** params.get('exposure', 0.0))
    
    lum = get_luminance(img)
    img = apply_split_exposure(
        img,
        params.get('exposure_shadows', 0.0),
        params.get('exposure_highlights', 0.0),
        params.get('exposure_shadows_range', 1.0),
        params.get('exposure_highlights_range', 1.0),
        lum
    )
    
    img = apply_local_adjustments(img, params.get('local_adjustments', []), scale_factor)

    # Apply global Gamma (Paper Grade)
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
        img = apply_autocrop(img, offset_px=params.get('autocrop_offset', 0), scale_factor=scale_factor)

    if params.get('monochrome', False):
        gray = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        img = np.stack([gray, gray, gray], axis=2)

    return img

def load_raw_and_process(file_bytes: bytes, params: Dict[str, Any], output_format: str, print_width_cm: float, dpi: int, sharpen_amount: float, filename: str = "", add_border: bool = False, border_size_cm: float = 1.0, border_color: str = "#000000", icc_profile_path: Optional[str] = None) -> Tuple[Optional[bytes], str]:
    """
    Worker function for processing a RAW file to a final output format (JPEG/TIFF).
    """
    try:
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            # PURE RAW: Use [1,1,1,1] to see the full physical orange mask.
            # This is essential for our darkroom-style analytical WB to work.
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, user_wb=[1, 1, 1, 1], output_bps=16)
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
        
        if not params.get('monochrome', False):
            img_arr = np.array(pil_img)
            img_sep = apply_color_separation(img_arr, params.get('color_separation', 1.0))
            pil_img = Image.fromarray(img_sep)
            
            sat = params.get('saturation', 1.0)
            if sat != 1.0:
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(sat)

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
        if sharpen_amount > 0:
            img_lab = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            l_pil = Image.fromarray(l)
            l_sharpened = l_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen_amount * 250), threshold=5))
            img_lab_sharpened = cv2.merge([np.array(l_sharpened), a, b])
            pil_img = Image.fromarray(cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB))

        if params.get('monochrome', False):
            pil_img = pil_img.convert("L")

        # ICC
        target_icc_bytes = None
        if icc_profile_path and os.path.exists(icc_profile_path):
            try:
                src_profile = ImageCms.createProfile("sRGB")
                dst_profile = ImageCms.getOpenProfile(icc_profile_path)
                if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
                pil_img = ImageCms.profileToProfile(pil_img, src_profile, dst_profile, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode='RGB', flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
                with open(icc_profile_path, "rb") as f: target_icc_bytes = f.read()
            except Exception as e: print(f"ICC Error: {e}")

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
