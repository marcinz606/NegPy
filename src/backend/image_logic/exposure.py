import numpy as np
from typing import Any
from src.backend.utils import get_luminance

def apply_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    """
    Applies simple contrast adjustment around 0.5 midpoint.
    """
    if contrast == 1.0:
        return img
    res = (img - 0.5) * contrast + 0.5
    return np.clip(res, 0.0, 1.0)

def apply_scan_gain_with_toe(img: np.ndarray, gain: Any, shadow_toe: float, highlight_shoulder: float = 0.0, shadow_bases: Any = None) -> np.ndarray:
    """
    Applies Chromaticity-Preserving Scanner Gain with Dual-End Recovery.
    - Shadow Toe: Lifts and recovers the deepest blacks.
    - Highlight Shoulder: Compresses and recovers peak whites.
    """
    if isinstance(gain, (list, tuple, np.ndarray)):
        gains = np.array(gain)
    else:
        gains = np.array([gain, gain, gain])

    if shadow_bases is None:
        v_neg = np.clip(img, 0.0, 1.0)
    else:
        master_base = np.max(shadow_bases)
        v_neg = np.clip(img / (master_base + 1e-6), 0.0, 1.0)
    
    # 1. Master Channel Analysis
    v_max = np.max(v_neg, axis=-1)
    
    # 2. Linear Scanner Gain (Master Exposure)
    avg_g = np.mean(gains)
    v_exp = v_max * avg_g
    
    # 3. Enhanced Shadow Recovery (Toe)
    if shadow_toe > 0:
        mask_s = v_exp > 0.5
        if np.any(mask_s):
            v_val = v_exp[mask_s]
            # k drives rational recovery
            k = shadow_toe * 5.0
            # Subtle linear lift (0.15 max shift) to 'open' the shadows
            lift = shadow_toe * 0.15 * (v_val - 0.5)
            v_exp[mask_s] = 0.5 + (v_val - 0.5 - lift) / (1.0 + k * (v_val - 0.5))
            
    # 4. Enhanced Highlight Recovery (Shoulder)
    if highlight_shoulder > 0:
        mask_h = v_exp < 0.5
        if np.any(mask_h):
            x = v_exp[mask_h] / 0.5
            # k drives log compression
            k_log = highlight_shoulder * 15.0
            # Subtle linear compression (0.15 max shift) to pull peak detail
            v_recovered = 0.5 * (np.log(1.0 + k_log * x) / (np.log(1.0 + k_log) + 1e-6))
            v_exp[mask_h] = v_recovered * (1.0 - highlight_shoulder * 0.15)

    # 5. Calculate Chromaticity-Preserving Scalar
    scalar = v_exp / (v_max * avg_g + 1e-6)
    
    # Apply Scan Gain and the 'Recovery Factor' uniformly to all channels
    res = np.clip(v_neg * avg_g * scalar[:, :, None], 0.0, 1.0)
    
    return res

def calculate_auto_exposure_params(img_raw: np.ndarray, wb_r: float, wb_g: float, wb_b: float) -> tuple[float, float, float]:
    """
    Analytically calculates optimal Grade, Shadow Toe, and Highlight Shoulder.
    Targets negative intensities: 0.05, 0.15, 0.25.
    Returns: (grade, shadow_toe, highlight_shoulder)
    """
    # 1. Get neutralized negative
    img = img_raw.copy()
    img[:, :, 0] *= wb_r
    img[:, :, 1] *= wb_g
    img[:, :, 2] *= wb_b
    img = np.clip(img, 0, 1)
    
    mx = np.max(img, axis=-1)
    
    # 2. Compute exposure points (Highlights -> Midtones)
    pts = np.percentile(mx, [1, 5, 15])
    p1, p5, p15 = pts[0], pts[1], pts[2]
    
    # 3. Targeted negative intensities
    targets = [0.05, 0.15, 0.25]
    
    # Suggested gains at each point
    g1 = targets[0] / (p1 + 1e-6)
    g5 = targets[1] / (p5 + 1e-6)
    g15 = targets[2] / (p15 + 1e-6)
    
    # 4. Solve for required Exposure Gain
    # Mid-upper range (p15) is our primary exposure anchor.
    required_gain = (g15 * 0.50) + (g5 * 0.30) + (g1 * 0.20)
    
    # 5. Map required_gain to the Grade scale (0 to 5)
    # grade = ((gain - 1.0) / 0.6) + 2.5
    new_grade = ((required_gain - 1.0) / 0.6) + 2.5
    new_grade = float(np.clip(new_grade, 0.0, 5.0))
    
    # Re-calculate gain based on clipped grade for accurate toe/shoulder solving
    final_gain = 1.0 + (new_grade - 2.5) * 0.6
    
    # 6. Shadow Toe Solver (Target detail at 0.90 density)
    p99 = np.percentile(mx, 99)
    crush_val = p99 * final_gain
    shadow_detail_target = 0.90
    if crush_val > shadow_detail_target:
        dist = crush_val - 0.5
        target_dist = shadow_detail_target - 0.5
        new_s_toe = ((dist / target_dist) - 1.0) / (8.0 * dist + 1e-6)
    else:
        new_s_toe = 0.0
        
    # 7. Highlight Shoulder Solver (Target detail at 0.12 density)
    p01_after = p1 * final_gain
    highlight_detail_target = 0.12
    if p01_after < highlight_detail_target:
        gamma_needed = np.log(highlight_detail_target / 0.5) / np.log(np.clip(p01_after / 0.5, 0.01, 0.99))
        new_h_shoulder = ( (1.0 / gamma_needed) - 1.0 ) / 3.0
    else:
        new_h_shoulder = 0.0
    
    return new_grade, float(np.clip(new_s_toe, 0.0, 0.5)), float(np.clip(new_h_shoulder, 0.0, 0.5))

def apply_chromaticity_preserving_black_point(img: np.ndarray, percentile: float) -> np.ndarray:
    """
    Neutralizes the overall black level of the print while preserving 
    intentional color balance and warmth in the shadows.
    """
    # 1. Calculate Luminance
    lum = get_luminance(img)
    
    # 2. Find the global black point percentile
    bp = np.percentile(lum, percentile)
    
    # 3. Apply uniform stretch to all channels to preserve color ratios
    res = (img - bp) / (1.0 - bp + 1e-6)
    return np.clip(res, 0.0, 1.0)
