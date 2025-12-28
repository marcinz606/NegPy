import numpy as np
import cv2
from typing import Dict, Any, List

def apply_selective_color(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Applies selective color adjustments (Hue, Saturation, Luminance) to specific color ranges.
    
    Args:
        img (np.ndarray): Input image RGB array (float32 [0, 1]).
        params (Dict[str, Any]): Processing parameters.
        
    Returns:
        np.ndarray: Adjusted image RGB array.
    """
    # Check if any selective adjustments are non-default to avoid unnecessary processing
    colors = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']
    has_changes = False
    for c in colors:
        if (params.get(f'selective_{c}_hue', 0.0) != 0.0 or
            params.get(f'selective_{c}_sat', 0.0) != 0.0 or
            params.get(f'selective_{c}_lum', 0.0) != 0.0):
            has_changes = True
            break
    
    if not has_changes:
        return img

    # Convert to HSV (OpenCV float32: H[0-360], S[0-1], V[0-1])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Define color centers and approximate widths (degrees)
    # This is a simple implementation using Gaussian-like falloff or hard ranges with feathering.
    # For performance and smoothness, we'll calculate a weight mask for each color.
    
    color_defs = {
        'red':     {'center': 0.0,   'width': 30.0}, # Special handling for wrap-around
        'orange':  {'center': 35.0,  'width': 20.0},
        'yellow':  {'center': 60.0,  'width': 25.0},
        'green':   {'center': 120.0, 'width': 50.0},
        'aqua':    {'center': 180.0, 'width': 30.0},
        'blue':    {'center': 225.0, 'width': 45.0},
        'purple':  {'center': 270.0, 'width': 30.0},
        'magenta': {'center': 315.0, 'width': 30.0}
    }

    for color_name, props in color_defs.items():
        hue_shift = params.get(f'selective_{color_name}_hue', 0.0)
        sat_shift = params.get(f'selective_{color_name}_sat', 0.0)
        lum_shift = params.get(f'selective_{color_name}_lum', 0.0)

        if hue_shift == 0.0 and sat_shift == 0.0 and lum_shift == 0.0:
            continue

        center = props['center']
        width = props['width'] * params.get(f'selective_{color_name}_range', 1.0)

        # Calculate distance from center hue
        delta = np.abs(h - center)
        # Handle wrap-around (e.g. 350 is close to 0)
        delta = np.minimum(delta, 360.0 - delta)
        
        # Create a soft mask (Gaussian-ish)
        # 1.0 at center, 0.0 at center +/- width
        # Simple linear falloff for speed: max(0, 1 - delta/width)
        # Smoothstep could be nicer: 3x^2 - 2x^3
        
        mask = np.clip(1.0 - (delta / width), 0.0, 1.0)
        
        # Optional: Apply smoothstep for softer transitions
        mask = mask * mask * (3 - 2 * mask)
        
        # Weight by saturation (low saturation shouldn't be affected as much)
        # This prevents gray areas from being tinted
        mask *= np.clip(s * 2.0, 0.0, 1.0) # Full effect at S >= 0.5

        if hue_shift != 0.0:
            # Shift hue
            h += mask * hue_shift
            # Wrap hue
            h = np.mod(h, 360.0)

        if sat_shift != 0.0:
            # Adjust saturation (sat_shift is -1.0 to 1.0 usually, or percentage)
            # Let's assume input is -1.0 to 1.0 (relative) or just an offset.
            # If we treat it as an offset:
            # s = np.clip(s + mask * sat_shift, 0.0, 1.0)
            # If we treat it as a multiplier: (1 + shift)
            # Let's go with multiplier-like behavior for positive, divisor for negative?
            # Or just simple addition:
            s = np.clip(s + mask * sat_shift, 0.0, 1.0)

        if lum_shift != 0.0:
            # Adjust Luminance (Value)
            v = np.clip(v + mask * lum_shift, 0.0, 1.0)

    # Merge and convert back
    hsv_mod = cv2.merge([h, s, v])
    img_mod = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2RGB)
    
    return img_mod
