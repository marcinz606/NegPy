import numpy as np
from typing import Tuple, Dict, Any

def apply_grade_to_img(img: np.ndarray, grade_val: float) -> np.ndarray:
    """
    Applies a grade (pure gamma) to an image.
    
    Args:
        img (np.ndarray): Input image array (float [0, 1]).
        grade_val (float): Grade value (0..5, neutral is 2.5).
        
    Returns:
        np.ndarray: Graded image array.
    """
    g_val = 1.0 / (1.0 + (grade_val - 2.5) * 0.4)
    res = np.power(img, g_val)
    return np.clip(res, 0.0, 1.0)

def calculate_auto_mask_wb(raw_preview: np.ndarray) -> Tuple[float, float, float]:
    """
    Robustly finds the film base color by analyzing the brightest pixels in the raw negative.
    
    Args:
        raw_preview (np.ndarray): Linear RAW RGB data (float [0, 1]).
        
    Returns:
        Tuple[float, float, float]: RGB gains (R, G, B) to neutralize the mask.
    """
    if raw_preview.ndim == 2:
        return 1.0, 1.0, 1.0
        
    pixels = raw_preview.reshape(-1, 3)
    if pixels.shape[0] > 200000:
        pixels = pixels[::pixels.shape[0]//200000]
        
    lum = 0.2126 * pixels[:,0] + 0.7152 * pixels[:,1] + 0.0722 * pixels[:,2]
    
    # Attempt Mask Detection
    thresh = np.percentile(lum, 98.5)
    mask_pixels = pixels[lum > thresh]
    
    if len(mask_pixels) > 0:
        mask_color = np.mean(mask_pixels, axis=0)
        if mask_color[0] > mask_color[1] * 1.05 and mask_color[1] > mask_color[2] * 1.05:
            max_val = np.max(mask_color)
            gains = max_val / (mask_color + 1e-6)
            return float(gains[0]), float(gains[1]), float(gains[2])

    # Fallback Grey World
    avg_color = np.mean(pixels, axis=0)
    max_avg = np.max(avg_color)
    gains = max_avg / (avg_color + 1e-6)
    gains = 0.5 * gains + 0.5 * np.array([1.0, 1.0, 1.0])
    gains /= (gains[1] + 1e-6)
    
    return float(gains[0]), float(gains[1]), float(gains[2])

def apply_manual_color_corrections(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Handles manual color balance and shadow/highlight tints based on user parameters.
    
    Args:
        img (np.ndarray): Input image array (float [0, 1]).
        params (Dict[str, Any]): Processing parameters dictionary.
        
    Returns:
        np.ndarray: Color corrected image array.
    """
    # Global balance
    img[:, :, 0] *= (params['cr_balance'] * (1.0 + params['temperature']))
    img[:, :, 1] *= params['mg_balance']
    img[:, :, 2] *= (params['yb_balance'] * (1.0 - params['temperature']))

    # Shadow/Highlight tints
    lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    w_shadow = (1.0 - lum)[:,:,None]
    w_highlight = lum[:,:,None]
    
    s_temp = params.get('shadow_temp', 0.0)
    s_r = params.get('shadow_cr', 1.0) * (1.0 + s_temp)
    s_g = params.get('shadow_mg', 1.0)
    s_b = params.get('shadow_yb', 1.0) * (1.0 - s_temp)
    
    if s_r != 1.0 or s_g != 1.0 or s_b != 1.0:
        img *= (w_shadow * np.array([s_r, s_g, s_b]) + (1.0 - w_shadow))

    h_temp = params.get('highlight_temp', 0.0)
    h_r = params.get('highlight_cr', 1.0) * (1.0 + h_temp)
    h_g = params.get('highlight_mg', 1.0)
    h_b = params.get('highlight_yb', 1.0) * (1.0 - h_temp)
    
    if h_r != 1.0 or h_g != 1.0 or h_b != 1.0:
        img *= (w_highlight * np.array([h_r, h_g, h_b]) + (1.0 - w_highlight))

    return np.clip(img, 0, 1)
