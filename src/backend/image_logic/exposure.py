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
    Applies Linear Scanner Gain with Dual-End Recovery.
    - Scan Gain (gain): Linear multiplier for highlight recovery and darkening.
    - Shadow Toe (shadow_toe): RECOVERS shadows by pulling them away from 1.0 (Black).
    - Highlight Shoulder (highlight_shoulder): RECOVERS whites by reducing local gamma.
    """
    if isinstance(gain, (list, tuple, np.ndarray)):
        gains = np.array(gain)
    else:
        gains = np.array([gain, gain, gain])

    if shadow_bases is None:
        bases = np.array([1.0, 1.0, 1.0])
    else:
        bases = np.array(shadow_bases)

    res = img.copy()
    
    for c in range(3):
        g = gains[c]
        b = bases[c]
        
        # 1. Normalize to Film Base (Mask = 1.0, Highlights = 0.0)
        v_neg = np.clip(img[:, :, c] / (b + 1e-6), 0.0, 1.0)
        
        # 2. Linear Scanner Gain
        # Darkens print, recovers highlights, but pushes shadows towards 1.0+
        v_exp = v_neg * g
        
        # 3. Highlight Shoulder (Local Highlight Gamma Recovery)
        if highlight_shoulder > 0:
            # We target the lower half of the negative range (The Whites)
            mask_h = v_exp < 0.5
            if np.any(mask_h):
                x = v_exp[mask_h] / 0.5
                # Reducing gamma stretches the values away from 0.0
                gamma_h = 1.0 / (1.0 + highlight_shoulder * 3.0)
                v_exp[mask_h] = 0.5 * np.power(x, gamma_h)
            
        # 4. Shadow Toe (Shadow Recovery / Black Protection)
        # Pulls values away from 1.0+. Increasing this makes shadows LIGHTER.
        if shadow_toe > 0:
            # Rational compression targeted at the top half of the range
            mask_s = v_exp > 0.5
            if np.any(mask_s):
                dist = v_exp[mask_s] - 0.5
                # Pulls potential 1.0+ values back towards visible range
                v_exp[mask_s] = 0.5 + dist / (1.0 + shadow_toe * 8.0 * dist)

        res[:, :, c] = np.clip(v_exp, 0.0, 1.0)
        
    return res

def calculate_auto_exposure_params(img_raw: np.ndarray, wb_r: float, wb_g: float, wb_b: float) -> tuple[float, float, float]:
    """
    Analytically calculates optimal Scanner Gain, Shadow Toe, and Highlight Shoulder.
    Targets the 'almost perfect' exposure level.
    """
    # 1. Get neutralized negative
    img = img_raw.copy()
    img[:, :, 0] *= wb_r
    img[:, :, 1] *= wb_g
    img[:, :, 2] *= wb_b
    img = np.clip(img, 0, 1)
    
    mx = np.max(img, axis=-1)
    
    # 2. Compute exposure points (Highlights -> Midtones)
    pts = np.percentile(mx, [1, 5, 15, 30])
    p1, p5, p15, p30 = pts[0], pts[1], pts[2], pts[3]
    
    # 3. Target intensities (Restored 'Almost Perfect' Targets)
    targets = [0.05, 0.10, 0.23, 0.43]
    
    # Suggested gains
    g1 = targets[0] / (p1 + 1e-6)
    g5 = targets[1] / (p5 + 1e-6)
    g15 = targets[2] / (p15 + 1e-6)
    g30 = targets[3] / (p30 + 1e-6)
    
    # 4. Final Scan Gain (Weighted average)
    new_gain = (g30 * 0.35) + (g15 * 0.35) + (g5 * 0.20) + (g1 * 0.10)
    new_gain = float(np.clip(new_gain, 1.0, 5.0))
    
    # 5. Shadow Toe Solver (Rational Target: 0.92)
    p99 = np.percentile(mx, 99)
    crush_val = p99 * new_gain
    if crush_val > 0.92:
        dist = crush_val - 0.5
        new_s_toe = ((dist / 0.42) - 1.0) / (8.0 * dist + 1e-6)
    else:
        new_s_toe = 0.0
        
    # 6. Highlight Shoulder Solver (Power-Law Target: 0.10)
    p01_after = p1 * new_gain
    if p01_after < 0.10:
        gamma_needed = np.log(0.10 / 0.5) / np.log(np.clip(p01_after / 0.5, 0.01, 0.99))
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

def apply_per_channel_black_point(img: np.ndarray, percentile: float) -> np.ndarray:
    """
    Neutralizes shadow crossover by stretching each channel to black 
    based on the specified percentile.
    """
    res = img.copy()
    for c in range(3):
        chan_bp = np.percentile(res[:, :, c], percentile)
        res[:, :, c] = (res[:, :, c] - chan_bp) / (1.0 - chan_bp + 1e-6)
    return np.clip(res, 0.0, 1.0)
