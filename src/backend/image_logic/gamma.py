import numpy as np
from typing import Any
from .color import get_luminance

def apply_gamma_to_img(img: np.ndarray, gamma_val: Any, mode: str = "Standard") -> np.ndarray:
    """
    Applies gamma correction to an image.
    Standard: Classic per-channel power law.
    Chromaticity Preserving: Adjusts luminance while keeping color ratios identical.
    """
    if isinstance(gamma_val, (list, tuple, np.ndarray)):
        gammas = np.array(gamma_val)
    else:
        gammas = np.array([gamma_val, gamma_val, gamma_val])

    if np.all(gammas == 1.0):
        return img
        
    if mode == "Standard":
        # Per-channel power law
        g_inv = 1.0 / np.maximum(0.01, gammas)
        res = np.power(img, g_inv)
        return np.clip(res, 0.0, 1.0)
    
    else: # Chromaticity Preserving
        # Professional mode: Adjusts brightness/contrast without shifting hues.
        # Uses the mean gamma for luma stretch.
        avg_gamma = np.mean(gammas)
        g_inv_avg = 1.0 / max(0.01, avg_gamma)
        
        luma_orig = get_luminance(img)
        luma_target = np.power(luma_orig, g_inv_avg)
        gain = luma_target / (luma_orig + 1e-6)
        
        # Apply the SAME gain to all channels to preserve chromaticity
        res = img * gain[:, :, None]
        return np.clip(res, 0.0, 1.0)

def apply_split_gamma(img: np.ndarray, gamma_s: Any, gamma_h: Any, range_s: float, range_h: float, luminance: np.ndarray, mode: str = "Standard") -> np.ndarray:
    """
    Applies split gamma correction based on provided luminance.
    """
    w_s = np.clip(1.0 - (luminance / (range_s + 1e-6)), 0.0, 1.0)
    start_h = 1.0 - range_h
    w_h = np.clip((luminance - start_h) / (range_h + 1e-6), 0.0, 1.0)
    
    img_s = apply_gamma_to_img(img, gamma_s, mode=mode)
    img_h = apply_gamma_to_img(img, gamma_h, mode=mode)
    
    diff_s = img_s - img
    diff_h = img_h - img
    
    res = img + diff_s * w_s[:, :, None] + diff_h * w_h[:, :, None]
    return np.clip(res, 0.0, 1.0)

def calculate_balancing_gammas(img: np.ndarray, target_percentile: float) -> np.ndarray:
    """
    Calculates per-channel gamma required to map the specified percentile of 
    each channel to the Green channel's value.
    Corrected formula for mapping P to T: P^g = T => g = log(T) / log(P)
    """
    points = np.percentile(img, target_percentile, axis=(0, 1))
    target = np.clip(points[1], 0.01, 0.99) # Anchor to Green
    points = np.clip(points, 0.01, 0.99)
    # CORRECT MATH: g = log(T) / log(P)
    return np.log(target) / (np.log(points) + 1e-6)