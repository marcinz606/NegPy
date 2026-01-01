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
    Identifies white balance by targeting the physical transparency peak of 
    the Red channel. Anchors the result to Red (gain=1.0) so that Cyan 
    filtration remains at 0 while Magenta and Yellow are calculated.
    """
    if raw_preview.ndim == 2:
        return 1.0, 1.0, 1.0
        
    # Resize for fast processing
    h, w = raw_preview.shape[:2]
    small = cv2.resize(raw_preview, (w//4, h//4), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    
    # 1. Filter out clipped pixels
    valid_mask = np.all(small < 0.98, axis=-1)
    valid_pixels = small[valid_mask]
    if len(valid_pixels) < 100: valid_pixels = pixels
        
    # 2. Target the 'Transparency Peak' of the Red channel (Top 0.1%)
    r_vals = valid_pixels[:, 0]
    r_thresh = np.percentile(r_vals, 99.9)
    mask_pixels = valid_pixels[r_vals >= r_thresh]
    
    if len(mask_pixels) > 5:
        # 3. Use the Median color of this physical limit.
        mask_color = np.median(mask_pixels, axis=0)
        
        # 4. Calculate gains anchored to RED = 1.0
        # This forces Cyan to 0 in the darkroom model.
        r_val = mask_color[0]
        g_gain = r_val / (mask_color[1] + 1e-6)
        b_gain = r_val / (mask_color[2] + 1e-6)
        
        return 1.0, float(g_gain), float(b_gain)

    # Standard Fallback
    return 1.0, 1.5, 4.0

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
    # In a negative: High Lum (Clear) = Shadow on Print. Low Lum (Dense) = Highlight on Print.
    w_shadow = lum[:,:,None]
    w_highlight = (1.0 - lum)[:,:,None]
    
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
