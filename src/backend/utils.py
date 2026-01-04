import json
import os
from PIL import Image
from typing import List, Dict, Any, Optional
from src.config import APP_CONFIG, DEFAULT_SETTINGS
import numpy as np
from src.helpers import ensure_rgb, imread_raw, get_luminance
from src.domain_objects import ImageSettings


def save_preset(name: str, settings: ImageSettings) -> None:
    """
    Saves a filtered subset of settings to a JSON file in the presets folder.
    Excludes image-specific settings and obsolete parameters.

    Args:
        name (str): The name of the preset file.
        settings (ImageSettings): The full settings object to filter and save.
    """
    os.makedirs(APP_CONFIG.presets_dir, exist_ok=True)

    # 1. Define image-specific keys that should NEVER be in a global preset
    exclude_keys = {
        "rotation",
        "fine_rotation",
        "autocrop",
        "autocrop_offset",
        "manual_dust_spots",
        "local_adjustments",
        "active_adjustment_idx",
    }

    settings_dict = settings.to_dict()
    default_dict = DEFAULT_SETTINGS.to_dict()

    # 2. Only save keys that are BOTH in ImageSettings and NOT in exclude_keys
    filtered = {
        k: v
        for k, v in settings_dict.items()
        if k in default_dict and k not in exclude_keys
    }

    filepath = os.path.join(APP_CONFIG.presets_dir, f"{name}.json")
    with open(filepath, "w") as f_out:
        json.dump(filtered, f_out, indent=4)


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Loads a preset from a JSON file.

    Args:
        name (str): The name of the preset (without .json extension).

    Returns:
        Optional[Dict[str, Any]]: The preset settings dict if found, else None.
    """
    filepath = os.path.join(APP_CONFIG.presets_dir, f"{name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f_in:
        res: Dict[str, Any] = json.load(f_in)
        return res


def list_presets() -> List[str]:
    """
    Lists all available preset names from the presets directory.

    Returns:
        List[str]: A list of preset names.
    """
    if not os.path.exists(APP_CONFIG.presets_dir):
        return []
    return [f[:-5] for f in os.listdir(APP_CONFIG.presets_dir) if f.endswith(".json")]


def get_thumbnail_worker(file_path: str) -> Optional[Image.Image]:
    """
    Worker function for parallel thumbnail generation from RAW file path.
    """
    try:
        ts = APP_CONFIG.thumbnail_size
        with imread_raw(file_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=False,
                user_wb=[1, 1, 1, 1],
                half_size=True,
                no_auto_bright=True,
                bright=1.0,
            )
            rgb = ensure_rgb(rgb)
            img = Image.fromarray(rgb)

            img.thumbnail((ts, ts))
            square_img = Image.new("RGB", (ts, ts), (14, 17, 23))
            square_img.paste(img, ((ts - img.width) // 2, (ts - img.height) // 2))
            return square_img
    except Exception:
        return None


def convert_to_monochrome(img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to a 3-channel monochrome (greyscale) image.
    """
    if img.shape[2] != 3:
        return img
    lum = get_luminance(img)
    res: np.ndarray = np.stack([lum, lum, lum], axis=2)
    return res
