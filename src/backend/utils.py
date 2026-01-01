import os
import json
import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import rawpy
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from src.backend.config import APP_CONFIG

from src.backend.config import APP_CONFIG, DEFAULT_SETTINGS

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

def plot_histogram(img_arr: np.ndarray, figsize: Tuple[float, float] = (6, 1), dpi: int = 150) -> plt.Figure:
    """
    Generates a professional RGB + Luminance histogram plot.
    
    Args:
        img_arr (np.ndarray): Image data as uint8 array.
        figsize (Tuple[float, float]): Figure size in inches.
        dpi (int): Plot resolution.
        
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('#1a1c23') 
    fig.patch.set_facecolor('#0e1117')
    
    lum = 0.2126 * img_arr[..., 0] + 0.7152 * img_arr[..., 1] + 0.0722 * img_arr[..., 2]
    colors = ('#ff4b4b', '#28df99', '#3182ce')
    
    for i, color in enumerate(colors):
        hist, bins = np.histogram(img_arr[..., i], bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color, lw=1.2, alpha=0.8)
        ax.fill_between(bins[:-1], hist, color=color, alpha=0.1)

    l_hist, bins = np.histogram(lum, bins=256, range=(0, 256))
    l_hist = ndimage.gaussian_filter1d(l_hist, sigma=1)
    ax.plot(bins[:-1], l_hist, color='#ffffff', lw=1.5, alpha=0.9, label='Luma')
    ax.fill_between(bins[:-1], l_hist, color='#ffffff', alpha=0.05)
    
    ax.axvline(x=128, color='#ffffff', alpha=0.1, lw=1, ls='--')
    ax.axvline(x=64, color='#ffffff', alpha=0.05, lw=0.8, ls=':')
    ax.axvline(x=192, color='#ffffff', alpha=0.05, lw=0.8, ls=':')

    ax.set_xlim(0, 256)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig

def get_thumbnail_worker(file_bytes: bytes) -> Optional[Image.Image]:
    """
    Worker function for parallel thumbnail generation from RAW bytes.
    
    Args:
        file_bytes (bytes): RAW file content.
        
    Returns:
        Optional[Image.Image]: A PIL Image square thumbnail or None.
    """
    try:
        ts = APP_CONFIG['thumbnail_size']
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            rgb = raw.postprocess(use_camera_wb=False, user_wb=[1, 1, 1, 1], half_size=True, no_auto_bright=True, bright=1.0)
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            img = Image.fromarray(rgb)
            
            img.thumbnail((ts, ts))
            square_img = Image.new('RGB', (ts, ts), (14, 17, 23))
            square_img.paste(img, ((ts - img.width) // 2, (ts - img.height) // 2))
            return square_img
    except Exception:
        return None

def apply_color_separation(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Increases/decreases color separation (saturation) without shifting luminance.
    Refined: Tapers intensity in shadows to prevent "nuclear" darks.
    
    Args:
        img (np.ndarray): Input image array (H, W, 3).
        intensity (float): Separation multiplier (1.0 is neutral).
        
    Returns:
        np.ndarray: Processed image array.
    """
    is_float = img.dtype.kind == 'f'
    if not is_float:
        img = img.astype(np.float32) / 255.0
        
    lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    
    # Create a luma mask to protect shadows from excessive separation
    # 1.0 at L=0.2 and above, fades to 0.0 at L=0.0
    luma_mask = np.clip(lum / 0.2, 0.0, 1.0)
    # Quadratic for smoother transition
    luma_mask = luma_mask * luma_mask
    
    # Calculate effective intensity per pixel
    # We want to interpolate between 1.0 (neutral) and the user target 'intensity'
    # based on the luma_mask.
    effective_intensity = 1.0 + (intensity - 1.0) * luma_mask
    
    # Apply separation
    lum_3d = lum[:,:,None]
    res = lum_3d + (img - lum_3d) * effective_intensity[:,:,None]
    res = np.clip(res, 0.0, 1.0)
    
    if not is_float:
        res = (res * 255.0).astype(np.uint8)
    return res

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

