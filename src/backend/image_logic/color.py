import numpy as np
import cv2
from typing import Tuple, Dict, Any

def get_luminance(img: np.ndarray) -> np.ndarray:
    """
    Calculates the relative luminance of an RGB image using Rec. 709 coefficients.
    Supports both 3D (H, W, 3) and 2D (N, 3) arrays.
    """
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

def apply_shadow_desaturation(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Automatically reduces saturation in the darkest parts of the image to 
    prevent "electric" shadows when lifting them.
    
    Args:
        img (np.ndarray): Input RGB image.
        strength (float): Strength of the effect (0.0 to 2.0). 1.0 is standard.
    """
    if strength <= 0:
        return img
        
    luma = get_luminance(img)
    mask = np.clip(1.0 - (luma / 0.3), 0.0, 1.0)
    mask = mask * mask * strength
    luma_3d = luma[:, :, None]
    res = img * (1.0 - mask[:, :, None]) + luma_3d * mask[:, :, None]
    return np.clip(res, 0.0, 1.0)

def calculate_auto_mask_wb(raw_preview: np.ndarray) -> Tuple[float, float, float]:
    """
    Identifies white balance by finding the 'Statistical Mode' of color ratios 
    in the high-density range of the negative. 
    Extremely robust against colorful scene content and missing borders.
    """
    if raw_preview.ndim == 2:
        return 1.0, 1.0, 1.0
        
    # Resize for fast statistical processing
    h, w = raw_preview.shape[:2]
    small = cv2.resize(raw_preview, (w//4, h//4), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    
    # 1. Filter for the 'Dense' part of the negative (Top 25% of brightness)
    # The film mask/base is always in the upper quartile of a linear scan.
    dense_mask = np.max(pixels, axis=1) > 0.4
    top_pixels = pixels[dense_mask]
    
    if len(top_pixels) > 100:
        # 2. Calculate Ratios relative to Green
        r_g_ratios = top_pixels[:, 0] / (top_pixels[:, 1] + 1e-6)
        b_g_ratios = top_pixels[:, 2] / (top_pixels[:, 1] + 1e-6)
        
        # 3. Find the MODE (most frequent value) of the ratios.
        # High-res histograms to find the 'True Mask' peak
        hist_r, edges_r = np.histogram(r_g_ratios, bins=256, range=(1.0, 5.0))
        hist_b, edges_b = np.histogram(b_g_ratios, bins=256, range=(0.05, 1.5))
        
        # The peak of the histogram is our 'Mask Ratio'
        r_mode = edges_r[np.argmax(hist_r)]
        b_mode = edges_b[np.argmax(hist_b)]
        
        # 4. Calculate gains to neutralize these ratios, anchored to Green
        r_gain = 1.0 / (r_mode + 1e-6)
        b_gain = 1.0 / (b_mode + 1e-6)
        
        return float(r_gain), 1.0, float(b_gain)

    return 1.0, 1.0, 1.0

def apply_manual_color_balance_neg(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Applies 'Paper Warmth' (Temperature) in the NEGATIVE domain.
    Inverted Math: warmth (+) decreases neg red, increasing positive red.
    """
    res = img.copy()
    warmth = params.get('temperature', 0.0)
    
    if warmth != 0.0:
        # Increase warmth (+): decrease neg red, increase neg blue
        res[:, :, 0] *= (1.0 - warmth)
        res[:, :, 2] *= (1.0 + warmth)
    
    return np.clip(res, 0, 1)

def apply_shadow_highlight_grading(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Applies Split Toning in the NEGATIVE domain.
    Inverted Math: tone (+) decreases neg red, increasing positive red (Amber).
    """
    res = img.copy()
    lum = get_luminance(res)
    w_shadow = (1.0 - lum)[:,:,None]
    w_highlight = lum[:,:,None]
    
    # Shadow Tone (Amber <-> Blue)
    s_tone = params.get('shadow_temp', 0.0)
    if s_tone != 0.0:
        s_r = 1.0 - s_tone
        s_b = 1.0 + s_tone
        res *= (w_shadow * np.array([s_r, 1.0, s_b]) + (1.0 - w_shadow))

    # Highlight Tone (Amber <-> Blue)
    h_tone = params.get('highlight_temp', 0.0)
    if h_tone != 0.0:
        h_r = 1.0 - h_tone
        h_b = 1.0 + h_tone
        res *= (w_highlight * np.array([h_r, 1.0, h_b]) + (1.0 - w_highlight))

    return np.clip(res, 0, 1)
