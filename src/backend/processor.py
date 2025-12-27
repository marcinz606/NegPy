import numpy as np
import cv2
import rawpy
import io
import traceback
from PIL import Image, ImageEnhance, ImageFilter
from src.backend.utils import create_curve_lut, apply_color_separation
from src.backend.config import APP_CONFIG
from src.backend.image_logic.color import (
    apply_grade_to_img, 
    calculate_auto_mask_wb, 
    apply_manual_color_corrections
)
from src.backend.image_logic.retouch import (
    apply_autocrop, 
    apply_dust_removal, 
    apply_chroma_noise_removal
)
from src.backend.image_logic.local import apply_local_adjustments
from typing import Dict, Any, Tuple, Optional

def process_image_core(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    High-level orchestration of the image processing pipeline.
    Coordinates rotation, inversion, cropping, healing, and color grading.
    
    Args:
        img (np.ndarray): Input linear RGB array (float [0, 1]).
        params (Dict[str, Any]): Processing parameters.
        
    Returns:
        np.ndarray: Fully processed image array.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # 0. Rotation
    rotation = params.get('rotation', 0)
    if rotation != 0:
        img = np.rot90(img, k=rotation)

    h_orig, w_cols = img.shape[:2]
    scale_factor = max(h_orig, w_cols) / float(APP_CONFIG['preview_max_res'])

    # 1. Mask Neutralization & Manual WB (BEFORE Inversion)
    img[:, :, 0] *= (params.get('mask_r', 1.0) * params.get('wb_manual_r', 1.0))
    img[:, :, 1] *= (params.get('mask_g', 1.0) * params.get('wb_manual_g', 1.0))
    img[:, :, 2] *= (params.get('mask_b', 1.0) * params.get('wb_manual_b', 1.0))
    img = np.clip(img, 0, 1)

    # 2. Inversion
    img = 1.0 - img

    # 2a. Auto-Crop
    if params.get('autocrop'):
        img = apply_autocrop(img, offset_px=params.get('autocrop_offset', 0), scale_factor=scale_factor)

    # 2b. Dust removal (Auto & Manual)
    img = apply_dust_removal(img, params, scale_factor)

    # 2c. Color Noise Removal
    img = apply_chroma_noise_removal(img, params)

    # 3. Manual Color Balance & Shadow/Highlight Color Correction
    img = apply_manual_color_corrections(img, params)

    # 4. Exposure, Gamma & BP/WP
    exposure_factor = 2.0 ** (params.get('exposure', 0.0) * 0.50)
    img *= exposure_factor
    
    # Soft Highlight Compression
    mask_high = img > 0.8
    if np.any(mask_high):
        vals = img[mask_high]
        img[mask_high] = 0.8 + (vals - 0.8) / (1.0 + (vals - 0.8) * 1.2)

    # 4a. Local Adjustments (Dodge/Burn)
    img = apply_local_adjustments(img, params.get('local_adjustments', []), scale_factor)

    # Base Grade
    img = apply_grade_to_img(img, params.get('gamma', 2.5))
    
    bp = params.get('black_point', 0.0)
    wp = params.get('white_point', 1.0)
    if bp != 0.0 or wp != 1.0:
        img = (img - bp) / (wp - bp + 1e-6)
        img = np.clip(img, 0, 1)
        
    black_p = np.percentile(img, 0.5)
    img = (img - black_p) / (1.0 - black_p + 1e-6)
    img = np.clip(img, 0.0, 1.0)

    # 4b. Global Contrast
    contrast = params.get('contrast', 1.0)
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5
        img = np.clip(img, 0, 1)

    # 5. Split-Grade
    img_s = apply_grade_to_img(img, params.get('grade_shadows', 2.5))
    img_h = apply_grade_to_img(img, params.get('grade_highlights', 2.5))
    luma = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    mask = np.clip(luma, 0, 1)
    img = img_s * (1.0 - mask[:,:,None]) + img_h * mask[:,:,None]

    # 5b. Tone Curve
    if 'curve_lut_x' in params and 'curve_lut_y' in params:
        img = np.interp(img, params['curve_lut_x'], params['curve_lut_y'])

    # 7. Monochrome Mode
    if params.get('monochrome', False):
        gray = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        img = np.stack([gray, gray, gray], axis=2)

    return img

def load_raw_and_process(file_bytes: bytes, params: Dict[str, Any], output_format: str, print_width_cm: float, dpi: int, sharpen_amount: float) -> Tuple[Optional[bytes], str]:
    """
    Worker function for processing a RAW file to a final output format (JPEG/TIFF).
    Used for both single export and batch processing.
    
    Returns:
        Tuple[Optional[bytes], str]: (Image bytes, Extension or error message)
    """
    try:
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16)
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
        img = rgb.astype(np.float32) / 65535.0
        
        # Handle Auto-WB calculation for batch processing
        if params.get('auto_wb'):
            m_r, m_g, m_b = calculate_auto_mask_wb(img)
            params['mask_r'] = m_r
            params['mask_g'] = m_g
            params['mask_b'] = m_b

        img = process_image_core(img, params)
        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        
        if not params.get('monochrome', False):
            img_arr = np.array(pil_img)
            img_sep = apply_color_separation(img_arr, params['saturation'])
            pil_img = Image.fromarray(img_sep)

        side_inch = print_width_cm / 2.54
        if pil_img.width >= pil_img.height:
            target_width_px = int(side_inch * dpi)
            target_height_px = int(target_width_px * (pil_img.height / pil_img.width))
        else:
            target_height_px = int(side_inch * dpi)
            target_width_px = int(target_height_px * (pil_img.width / pil_img.height))
        pil_img = pil_img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
        
        if sharpen_amount > 0:
            img_lab = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            l_pil = Image.fromarray(l)
            l_sharpened = l_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen_amount * 250), threshold=5))
            img_lab_sharpened = cv2.merge([np.array(l_sharpened), a, b])
            pil_img = Image.fromarray(cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB))

        if params.get('monochrome', False):
            pil_img = pil_img.convert("L")

        output_buf = io.BytesIO()
        ext = "jpg" if output_format == 'JPEG' else "tiff"
        if output_format == 'JPEG':
            pil_img.save(output_buf, format="JPEG", quality=95, dpi=(dpi, dpi))
        else:
            pil_img.save(output_buf, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
        return output_buf.getvalue(), ext
    except Exception as e:
        print(traceback.format_exc())
        return None, str(e)