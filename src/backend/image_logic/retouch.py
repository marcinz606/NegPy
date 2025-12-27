import numpy as np
import cv2
from typing import Dict, Any
from src.backend.config import APP_CONFIG

def apply_autocrop(img: np.ndarray, offset_px: int = 0, scale_factor: float = 1.0) -> np.ndarray:
    """
    Detects film edges and automatically crops the image to the 2:3 or 3:2 frame.
    
    Args:
        img (np.ndarray): Input image array.
        offset_px (int): Additional margin to add to the crop.
        scale_factor (float): Scaling factor for the current processing resolution.
        
    Returns:
        np.ndarray: Cropped image array.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w, _ = img.shape
    detect_res = APP_CONFIG['autocrop_detect_res']
    det_scale = detect_res / max(h, w)
    img_small = cv2.resize(img, (int(w * det_scale), int(h * det_scale)), interpolation=cv2.INTER_AREA)
    lum = 0.2126 * img_small[:,:,0] + 0.7152 * img_small[:,:,1] + 0.0722 * img_small[:,:,2]
    rows_det = np.where(np.mean(lum, axis=1) < 0.96)[0]
    cols_det = np.where(np.mean(lum, axis=0) < 0.96)[0]
    if len(rows_det) < 10 or len(cols_det) < 10: return img
    y1, y2 = rows_det[0] / det_scale, rows_det[-1] / det_scale
    x1, x2 = cols_det[0] / det_scale, cols_det[-1] / det_scale
    margin = (2 + offset_px) * scale_factor
    y1, y2, x1, x2 = y1 + margin, y2 - margin, x1 + margin, x2 - margin
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0: return img
    ratio = 2/3 if ch > cw else 3/2
    if cw / ch > ratio:
        target_w = ch * ratio
        x1 = x1 + (cw - target_w) // 2
        x2 = x1 + target_w
    else:
        target_h = cw / ratio
        y1 = y1 + (ch - target_h) // 2
        y2 = y1 + target_h
    return img[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]

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
        d_size = int(params.get('dust_size', 2) * 2.0 * scale_factor) | 1
        if d_size < 3: d_size = 3
        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        img_median_u8 = cv2.medianBlur(img_uint8, d_size)
        img_median = img_median_u8.astype(np.float32) / 255.0

        gray = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
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
        if m_radius < 1: m_radius = 1
        
        manual_mask_u8 = np.zeros((h_img, w_img), dtype=np.uint8)
        for (nx, ny) in manual_spots:
            px = int(nx * w_img)
            py = int(ny * h_img)
            cv2.circle(manual_mask_u8, (px, py), m_radius, 255, -1)
        
        img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        img_inpainted_u8 = cv2.inpaint(img_u8, manual_mask_u8, 3, cv2.INPAINT_TELEA)
        
        # Grain Matching
        noise = np.random.normal(0, 1.8, img_inpainted_u8.shape).astype(np.float32)
        lum = (0.2126 * img_inpainted_u8[:,:,0] + 0.7152 * img_inpainted_u8[:,:,1] + 0.0722 * img_inpainted_u8[:,:,2]) / 255.0
        modulation = 4.0 * lum * (1.0 - lum)
        noise = noise * modulation[:,:,None]
        
        mask_f = (manual_mask_u8.astype(np.float32) / 255.0)[:,:,None]
        mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0)
        if mask_f.ndim == 2: mask_f = mask_f[:,:,None]

        img_inpainted_f = img_inpainted_u8.astype(np.float32) + noise * mask_f
        img = np.clip(img_inpainted_f, 0, 255) / 255.0
        
    return img

def apply_chroma_noise_removal(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Reduces color noise in the shadows using bilateral filtering in LAB space.
    
    Args:
        img (np.ndarray): Input image array (float [0, 1]).
        params (Dict[str, Any]): Processing parameters dictionary.
        
    Returns:
        np.ndarray: Denoised image array.
    """
    if not params.get('c_noise_remove'):
        return img
    img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    strength = params.get('c_noise_strength', 20)
    a_denoised = cv2.bilateralFilter(a, 5, strength * 2, strength // 2)
    b_denoised = cv2.bilateralFilter(b, 5, strength * 2, strength // 2)
    shadow_mask = np.clip(1.0 - (l.astype(np.float32) / 128.0), 0, 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    a_final = np.clip(np.nan_to_num(a.astype(np.float32) * (1.0 - shadow_mask) + a_denoised.astype(np.float32) * shadow_mask), 0, 255).astype(np.uint8)
    b_final = np.clip(np.nan_to_num(b.astype(np.float32) * (1.0 - shadow_mask) + b_denoised.astype(np.float32) * shadow_mask), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge([l, a_final, b_final]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return img
