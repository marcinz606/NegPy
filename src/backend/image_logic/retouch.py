import numpy as np
import cv2
from typing import Dict, Any, Tuple
from src.config import APP_CONFIG
from src.backend.utils import ensure_rgb, get_luminance
from src.logging_config import get_logger

logger = get_logger(__name__)

def apply_fine_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates the image by a specific angle (in degrees).
    Keeps the original image dimensions, filling new areas with black.
    
    Args:
        img (np.ndarray): Input image array (float [0, 1]).
        angle (float): Rotation angle in degrees.
        
    Returns:
        np.ndarray: Rotated image.
    """
    if angle == 0.0:
        return img
    
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def get_autocrop_coords(img: np.ndarray, offset_px: int = 0, scale_factor: float = 1.0, target_ratio_str: str = "3:2") -> Tuple[int, int, int, int]:
    """
    Calculates the autocrop coordinates.
    Returns (y1, y2, x1, x2) in pixels relative to input img.
    """
    img = ensure_rgb(img)
    h, w, _ = img.shape
    detect_res = APP_CONFIG['autocrop_detect_res']
    det_scale = detect_res / max(h, w)
    img_small = cv2.resize(img, (int(w * det_scale), int(h * det_scale)), interpolation=cv2.INTER_AREA)
    lum = get_luminance(img_small)
    
    # Threshold for film base detection (detecting darker frame)
    rows_det = np.where(np.mean(lum, axis=1) < 0.96)[0]
    cols_det = np.where(np.mean(lum, axis=0) < 0.96)[0]
    
    if len(rows_det) < 10 or len(cols_det) < 10: 
        return 0, h, 0, w
        
    y1, y2 = rows_det[0] / det_scale, rows_det[-1] / det_scale
    x1, x2 = cols_det[0] / det_scale, cols_det[-1] / det_scale
    
    margin = (2 + offset_px) * scale_factor
    y1, y2, x1, x2 = y1 + margin, y2 - margin, x1 + margin, x2 - margin
    cw, ch = x2 - x1, y2 - y1
    
    if cw <= 0 or ch <= 0: 
        return 0, h, 0, w
    
    # Parse target ratio
    try:
        w_r, h_r = map(float, target_ratio_str.split(':'))
        target_aspect = w_r / h_r
    except Exception as e:
        logger.error(f"Invalid aspect ratio: {target_ratio_str}, defaulting to 3:2. Error: {e}")
        target_aspect = 1.5 # Default 3:2
        
    # Handle Orientation
    is_vertical = ch > cw
    if is_vertical:
        if target_aspect > 1.0:
            target_aspect = 1.0 / target_aspect
    else:
        if target_aspect < 1.0:
            target_aspect = 1.0 / target_aspect
        
    # Enforce Ratio
    current_aspect = cw / ch
    
    if current_aspect > target_aspect:
        # Too wide, crop width
        target_w = ch * target_aspect
        x1 = x1 + (cw - target_w) // 2
        x2 = x1 + target_w
    else:
        # Too tall, crop height
        target_h = cw / target_aspect
        y1 = y1 + (ch - target_h) // 2
        y2 = y1 + target_h
        
    return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))

def apply_autocrop(img: np.ndarray, offset_px: int = 0, scale_factor: float = 1.0, ratio: str = "3:2") -> np.ndarray:
    """
    Detects film edges and automatically crops the image to the specified aspect ratio frame.
    
    Args:
        img (np.ndarray): Input image array.
        offset_px (int): Additional margin to add to the crop.
        scale_factor (float): Scaling factor for the current processing resolution.
        ratio (str): Target aspect ratio string (e.g., "3:2", "6:7").
        
    Returns:
        np.ndarray: Cropped image array.
    """
    y1, y2, x1, x2 = get_autocrop_coords(img, offset_px, scale_factor, ratio)
    return img[y1:y2, x1:x2]

def apply_dust_removal(img: np.ndarray, params: Dict[str, Any], scale_factor: float) -> np.ndarray:
    """
    Applies both automatic and manual dust removal (healing).
    Automatic uses median blur replacement, manual uses inpainting with grain matching.
    
    Args:
        img (np.ndarray): Input image array (float [0, 1]).
        params (Dict[str, Any]): Processing parameters dictionary.
        scale_factor (float): Scaling factor for the current processing resolution.
        
    Returns:
        np.ndarray: Healed image array.
    """
    manual_spots = params.get('manual_dust_spots', [])
    if not (params.get('dust_remove') or manual_spots):
        return img

    # --- Automatic Detection & Healing ---
    if params.get('dust_remove'):
        d_size = int(params.get('dust_size', 3) * 2.0 * scale_factor) | 1
        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        img_median_u8 = cv2.medianBlur(img_uint8, d_size)
        img_median = img_median_u8.astype(np.float32) / 255.0

        gray = get_luminance(img)
        blur_win = int(15 * scale_factor) | 1
        mean = cv2.blur(gray, (blur_win, blur_win))
        sq_mean = cv2.blur(gray**2, (blur_win, blur_win))
        std = np.sqrt(np.clip(sq_mean - mean**2, 0, None))
        flatness = np.clip(1.0 - (std / 0.08), 0, 1)
        flatness_weight = np.sqrt(flatness)
        brightness = np.clip(gray, 0, 1)
        highlight_sens = np.clip((brightness - 0.4) * 1.5, 0, 1) 
        detail_boost = (1.0 - flatness) * 0.05 
        sens_factor = (1.0 - 0.98 * flatness_weight) * (1.0 - 0.5 * highlight_sens)
        adaptive_thresh = params['dust_threshold'] * sens_factor + detail_boost
        
        diff = np.max(np.abs(img - img_median), axis=2)
        raw_mask = (diff > adaptive_thresh).astype(np.float32)
        exclusion_mask = (std > 0.2).astype(np.float32)
        raw_mask = raw_mask * (1.0 - exclusion_mask)
        
        if np.any(raw_mask > 0):
            m_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, m_kernel_close)
            m_kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, m_kernel_dilate, iterations=2)
            feather = d_size | 1
            mask = cv2.GaussianBlur(mask, (feather, feather), 0)
            img = img * (1.0 - mask[:,:,None]) + img_median * mask[:,:,None]

    # --- Manual Healing (using Inpainting) ---
    if manual_spots:
        h_img, w_img = img.shape[:2]
        m_size_param = params.get('manual_dust_size', 10)
        m_radius = int(m_size_param * scale_factor)
        if m_radius < 1:
            m_radius = 1
        
        manual_mask_u8 = np.zeros((h_img, w_img), dtype=np.uint8)
        for (nx, ny) in manual_spots:
            px = int(nx * w_img)
            py = int(ny * h_img)
            cv2.circle(manual_mask_u8, (px, py), m_radius, 255, -1)
        
        img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        inpaint_rad = int(3 * scale_factor) | 1
        img_inpainted_u8 = cv2.inpaint(img_u8, manual_mask_u8, inpaint_rad, cv2.INPAINT_TELEA)
        
        # Grain Matching
        noise = np.random.normal(0, 3.5, img_inpainted_u8.shape).astype(np.float32)
        lum = get_luminance(img_inpainted_u8) / 255.0
        modulation = 5.0 * lum * (1.0 - lum)
        noise = noise * modulation[:,:,None]
        
        mask_f = (manual_mask_u8.astype(np.float32) / 255.0)[:,:,None]
        mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0)
        if mask_f.ndim == 2: mask_f = mask_f[:,:,None]

        img_inpainted_f = img_inpainted_u8.astype(np.float32) + noise * mask_f
        img = np.clip(img_inpainted_f, 0, 255) / 255.0
        
    return img

def apply_chroma_noise_removal(img: np.ndarray, params: Dict[str, Any], scale_factor: float = 1.0) -> np.ndarray:
    """
    Reduces color noise using bilateral filtering and specific deep-shadow blurring in LAB space.
    Targets shadows aggressively where film scan color noise is most prevalent.
    
    Args:
        img (np.ndarray): Input image array (float [0, 1]).
        params (Dict[str, Any]): Processing parameters dictionary.
        scale_factor (float): Scaling factor for the current processing resolution.
        
    Returns:
        np.ndarray: Denoised image array.
    """
    if not params.get('c_noise_remove'):
        return img
        
    strength = params.get('c_noise_strength', 50)
    if strength <= 0:
        return img

    img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 1. Bilateral Filter (Base Denoising)
    # Stronger parameters to smooth out mottling
    # Scale diameter with resolution to maintain relative effect size
    d = int(9 * scale_factor) | 1
    sc = strength * 2.0  # Increased color sigma
    ss = strength * 0.75 * scale_factor # Increased spatial sigma, scaled
    
    a_bilat = cv2.bilateralFilter(a, d, sc, ss)
    b_bilat = cv2.bilateralFilter(b, d, sc, ss)
    
    # 2. Strong Blur for Deep Shadows (The "Nuclear Option" for dark noise)
    # Bilateral can fail on very grainy darks, preserving the noise as "texture".
    # We blur the color channels aggressively in the deep blacks.
    base_k = 11 if strength > 50 else 7
    k_size = int(base_k * scale_factor) | 1
    a_blur = cv2.GaussianBlur(a_bilat, (k_size, k_size), 0)
    b_blur = cv2.GaussianBlur(b_bilat, (k_size, k_size), 0)
    
    l_float = l.astype(np.float32)
    
    # Deep Shadow Mask: 1.0 at L=0, fades to 0.0 at L=60 (~23%)
    deep_mask = np.clip(1.0 - (l_float / 60.0), 0.0, 1.0)
    deep_mask = deep_mask * deep_mask # Quadratic falloff to keep it tight to blacks
    
    # Blend Blur into Bilateral
    a_combined = a_bilat.astype(np.float32) * (1.0 - deep_mask) + a_blur.astype(np.float32) * deep_mask
    b_combined = b_bilat.astype(np.float32) * (1.0 - deep_mask) + b_blur.astype(np.float32) * deep_mask

    # 3. Broad Masking (Application to Image)
    # Apply to Shadows and Midtones, protecting only bright Highlights
    # 1.0 at L=0..150, fades to 0 at L=230 (~90%)
    broad_mask = np.clip(1.0 - ((l_float - 150.0) / 80.0), 0.0, 1.0)
    
    # Smooth the mask to avoid transitions
    broad_mask = cv2.GaussianBlur(broad_mask, (21, 21), 0)
    
    # Final Blend
    a_final = np.clip(a.astype(np.float32) * (1.0 - broad_mask) + a_combined * broad_mask, 0, 255).astype(np.uint8)
    b_final = np.clip(b.astype(np.float32) * (1.0 - broad_mask) + b_combined * broad_mask, 0, 255).astype(np.uint8)
    
    img = cv2.cvtColor(cv2.merge([l, a_final, b_final]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return img
