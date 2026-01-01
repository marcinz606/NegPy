import numpy as np
from typing import Any

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
    Applies Pro-Scanner Gain with Chromaticity-Preserving Recovery.
    - shadow_toe = 0: Shadows crush naturally.
    - shadow_toe > 0: Activates 'Shadow Saver' to lift and reveal detail.
    """
    if isinstance(gain, (list, tuple, np.ndarray)):
        gains = np.array(gain)
    else:
        gains = np.array([gain, gain, gain])

    if shadow_bases is None:
        v_neg = np.clip(img, 0.0, 1.0)
    else:
        # Chromaticity-preserving normalization
        master_base = np.max(shadow_bases)
        v_neg = np.clip(img / (master_base + 1e-6), 0.0, 1.0)
    
    # 1. Master Channel Analysis
    v_max = np.max(v_neg, axis=-1)
    
    # 2. Linear Scanner Gain
    avg_g = np.mean(gains)
    v_exp = v_max * avg_g
    
    # 3. Shadow Recovery (Toe)
    # We pull crushed blacks away from the 1.0 limit (and beyond).
    # This prevents the 'inky' look from becoming a flat blob.
    if shadow_toe > 0:
        # Target the top half of the range (Shadows)
        mask_s = v_exp > 0.5
        if np.any(mask_s):
            # We calculate how much to 'pull back' the shadow values.
            # Even if v_exp is 1.5 (heavily crushed), this pulls it back.
            v_val = v_exp[mask_s]
            # Strength coefficient
            k = shadow_toe * 4.0
            # Exponential lift that anchors at 0.5 and compresses everything towards it.
            # This ensures no value can ever 'invert' or turn white.
            v_exp[mask_s] = 0.5 + (v_val - 0.5) / (1.0 + k * (v_val - 0.5))
            
    # 4. Highlight Recovery (Shoulder)
    # We apply a logarithmic roll-off from the 50% midtone point.
    # This mimics the smooth compression of a physical print's shoulder.
    if highlight_shoulder > 0:
        # Target the highlights (lower intensity in negative domain)
        mask_h = v_exp < 0.5
        if np.any(mask_h):
            # Normalize the highlight region to [0, 1] for compression
            x = v_exp[mask_h] / 0.5
            # Logarithmic Detail Recovery formula
            # k drives the strength of the roll-off
            k = highlight_shoulder * 12.0
            v_exp[mask_h] = 0.5 * (np.log(1.0 + k * x) / (np.log(1.0 + k) + 1e-6))

    # 5. Calculate Chromaticity-Preserving Scalar
    scalar = v_exp / (v_max * avg_g + 1e-6)
    
    # Apply Scan Gain and the 'Recovery Factor' uniformly to all channels
    res = np.clip(v_neg * avg_g * scalar[:, :, None], 0.0, 1.0)
    
    return res

def calculate_auto_exposure_params(img_raw: np.ndarray, wb_r: float, wb_g: float, wb_b: float) -> tuple[float, float, float]:
    """
    Analytically calculates optimal Scanner Gain, Shadow Toe, and Highlight Shoulder
    using user-requested targets: 0.05, 0.15, 0.25.
    """
    # 1. Get neutralized negative
    img = img_raw.copy()
    img[:, :, 0] *= wb_r
    img[:, :, 1] *= wb_g
    img[:, :, 2] *= wb_b
    img = np.clip(img, 0, 1)
    
    mx = np.max(img, axis=-1)
    
    # 2. Compute exposure points (Highlights -> Midtones)
    # p1: Peak white, p5: Diffuse white, p15: Upper mids
    pts = np.percentile(mx, [1, 5, 15])
    p1, p5, p15 = pts[0], pts[1], pts[2]
    
    # 3. Targeted negative intensities (User Requested)
    targets = [0.05, 0.15, 0.25]
    
    # Suggested gains at each point
    g1 = targets[0] / (p1 + 1e-6)
    g5 = targets[1] / (p5 + 1e-6)
    g15 = targets[2] / (p15 + 1e-6)
    
    # 4. Final Scan Gain (Weighted average of the three points)
    # Mid-upper range (p15) is our primary exposure anchor.
    new_gain = (g15 * 0.50) + (g5 * 0.30) + (g1 * 0.20)
    new_gain = float(np.clip(new_gain, 1.0, 5.0))
    
    # 5. Shadow Toe Solver (Target detail at 0.90 density)
    # Lifts shadows to maintain separation
    p99 = np.percentile(mx, 99)
    crush_val = p99 * new_gain
    shadow_detail_target = 0.90
    if crush_val > shadow_detail_target:
        dist = crush_val - 0.5
        target_dist = shadow_detail_target - 0.5
        new_s_toe = ((dist / target_dist) - 1.0) / (8.0 * dist + 1e-6)
    else:
        new_s_toe = 0.0
        
    # 6. Highlight Shoulder Solver (Target detail at 0.12 density)
    # Pulls highlights in relative to the requested 0.05 peak.
    p01_after = p1 * new_gain
    highlight_detail_target = 0.12
    if p01_after < highlight_detail_target:
        gamma_needed = np.log(highlight_detail_target / 0.5) / np.log(np.clip(p01_after / 0.5, 0.01, 0.99))
        new_h_shoulder = ( (1.0 / gamma_needed) - 1.0 ) / 3.0
    else:
        new_h_shoulder = 0.0
    
    return new_gain, float(np.clip(new_s_toe, 0.0, 0.5)), float(np.clip(new_h_shoulder, 0.0, 0.5))

def apply_split_exposure(img: np.ndarray, exp_s: float, exp_h: float, range_s: float, range_h: float, luminance: np.ndarray) -> np.ndarray:
    """
    Applies split exposure correction based on provided luminance.
    """
    if exp_s == 0.0 and exp_h == 0.0:
        return img
        
    w_s = np.clip(1.0 - (luminance / (range_s + 1e-6)), 0.0, 1.0)
    start_h = 1.0 - range_h
    w_h = np.clip((luminance - start_h) / (range_h + 1e-6), 0.0, 1.0)
    
    f_s = 2.0 ** exp_s
    f_h = 2.0 ** exp_h
    
    mult_map = 1.0 + (f_s - 1.0) * w_s + (f_h - 1.0) * w_h
    res = img * mult_map[:,:,None]
    return np.clip(res, 0.0, 1.0)

def apply_chromaticity_preserving_black_point(img: np.ndarray, percentile: float) -> np.ndarray:
    """
    Neutralizes the overall black level of the print while preserving 
    intentional color balance and warmth in the shadows.
    """
    # 1. Calculate Luminance using Rec. 709 coefficients
    lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    
    # 2. Find the global black point percentile
    bp = np.percentile(lum, percentile)
    
    # 3. Apply uniform stretch to all channels to preserve color ratios
    res = (img - bp) / (1.0 - bp + 1e-6)
    return np.clip(res, 0.0, 1.0)
