import os
import json
import io
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from src.config import APP_CONFIG, DEFAULT_SETTINGS

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Ensures the input image is a 3-channel RGB array.
    """
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img] * 3, axis=-1)
    return img

def get_luminance(img: np.ndarray) -> np.ndarray:
    """
    Calculates relative luminance using Rec. 709 coefficients.
    Supports both 3D (H, W, 3) and 2D (N, 3) arrays.
    """
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

def save_preset(name: str, settings: Dict[str, Any]) -> None:
    """
    Saves a filtered subset of settings to a JSON file in the presets folder.
    Excludes image-specific settings and obsolete parameters.
    
    Args:
        name (str): The name of the preset file.
        settings (Dict[str, Any]): The full settings dictionary to filter and save.
    """
    os.makedirs(APP_CONFIG['presets_dir'], exist_ok=True)
    
    # 1. Define image-specific keys that should NEVER be in a global preset
    exclude_keys = {
        'rotation', 'fine_rotation', 
        'autocrop', 'autocrop_offset',
        'manual_dust_spots', 'local_adjustments',
        'active_adjustment_idx',
        'scan_gain', 'scan_gain_toe',
        'wb_manual_r', 'wb_manual_g', 'wb_manual_b'
    }
    
    # 2. Get current valid parameter names from DEFAULT_SETTINGS
    valid_keys = set(DEFAULT_SETTINGS.keys())
    
    # 3. Only save keys that are BOTH in DEFAULT_SETTINGS and NOT in exclude_keys
    save_keys = valid_keys - exclude_keys
    
    filtered = {k: settings[k] for k in save_keys if k in settings}
    
    filepath = os.path.join(APP_CONFIG['presets_dir'], f"{name}.json")
    with open(filepath, 'w') as f:
        json.dump(filtered, f, indent=4)

def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Loads a preset from a JSON file.
    
    Args:
        name (str): The name of the preset (without .json extension).
        
    Returns:
        Optional[Dict[str, Any]]: The preset settings or None if not found.
    """
    filepath = os.path.join(APP_CONFIG['presets_dir'], f"{name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def list_presets() -> List[str]:
    """
    Lists all available preset names from the presets directory.
    
    Returns:
        List[str]: A list of preset names.
    """
    if not os.path.exists(APP_CONFIG['presets_dir']):
        return []
    return [f[:-5] for f in os.listdir(APP_CONFIG['presets_dir']) if f.endswith('.json')]

def get_thumbnail_worker(file_bytes: bytes) -> Optional[Image.Image]:
    """
    Worker function for parallel thumbnail generation from RAW bytes.
    """
    import rawpy
    try:
        ts = APP_CONFIG['thumbnail_size']
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            rgb = raw.postprocess(use_camera_wb=False, user_wb=[1, 1, 1, 1], half_size=True, no_auto_bright=True, bright=1.0)
            rgb = ensure_rgb(rgb)
            img = Image.fromarray(rgb)
            
            img.thumbnail((ts, ts))
            square_img = Image.new('RGB', (ts, ts), (14, 17, 23))
            square_img.paste(img, ((ts - img.width) // 2, (ts - img.height) // 2))
            return square_img
    except Exception:
        return None

def transform_point(x: float, y: float, params: Dict[str, Any], raw_w: int, raw_h: int, inverse: bool = False) -> Tuple[float, float]:
    """
    Transforms a normalized (0..1) point between Raw Space and Display Space.
    inverse=True: Display -> Raw (for saving clicks)
    inverse=False: Raw -> Display (for visualization)
    """
    rotation = params.get('rotation', 0) % 4
    
    if not inverse:
        # Raw -> Display (Forward rotation)
        if rotation == 0: return x, y
        if rotation == 1: return 1.0 - y, x
        if rotation == 2: return 1.0 - x, 1.0 - y
        if rotation == 3: return y, 1.0 - x
    else:
        # Display -> Raw (Inverse rotation)
        if rotation == 0: return x, y
        if rotation == 1: return y, 1.0 - x
        if rotation == 2: return 1.0 - x, 1.0 - y
        if rotation == 3: return 1.0 - y, x
    
    return x, y

