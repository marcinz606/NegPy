from typing import Dict, List, Tuple, Any

# --- TONE CURVE PRESETS ---
# Maps preset names to (x_coords, y_coords) tuples for interpolation
TONE_CURVES_PRESETS: Dict[str, Tuple[List[float], List[float]]] = {
    "Linear": ([0, 1], [0, 1]),
    "Medium Contrast": ([0, 0.25, 0.75, 1], [0, 0.2, 0.8, 1]),
    "Soft Contrast": ([0, 0.25, 0.75, 1], [0, 0.30, 0.70, 1]),
    "Soft Highs": ([0, 0.5, 0.8, 1], [0, 0.5, 0.75, 0.9]),
    "Smooth": ([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], [0, 0.066, 0.225, 0.445, 0.7, 0.9, 0.975]),
    "Dense": ([0, 0.5, 1], [0.0, 0.33, 0.99]),
    "Thin": ([0, 0.5, 1], [0, 0.66, 1]),
    "LOG": ([0, 1], [0.05, 0.95]),
}

# --- DEFAULT SETTINGS ---
# The baseline parameters for every newly imported RAW file
DEFAULT_SETTINGS: Dict[str, Any] = {
    'scan_gain': 1.0,
    'scan_gain_s_toe': 0.0,
    'scan_gain_h_shoulder': 0.0,
    'auto_wb': False,
    'autocrop': True,
    'monochrome': False,
    'autocrop_offset': 0,
    'temperature': 0.0,
    'shadow_temp': 0.0,
    'highlight_temp': 0.0,
    'color_separation': 1.0,
    'saturation': 1.0,
    'gamma': 1.0,
    'gamma_mode': 'Standard',
    'shadow_desat_strength': 1.0,
    'contrast': 1.0,
    'black_point': 0.0,
    'white_point': 1.0,
    'exposure': 0.0,
    'exposure_shadows': 0.0,
    'exposure_shadows_range': 1.0,
    'exposure_highlights': 0.0,
    'exposure_highlights_range': 1.0,
    'curve_mode': "Smooth",
    'curve_strength': 0.5,
    'dust_remove': True,
    'dust_threshold': 0.55,
    'dust_size': 2,
    'c_noise_remove': True,
    'c_noise_strength': 25,
    'sharpen': 0.2,
    'rotation': 0,
    'fine_rotation': 0.0,
    'wb_cyan': 0,
    'wb_magenta': 0,
    'wb_yellow': 0,
    'manual_dust_spots': [],
    'manual_dust_size': 4,
    'local_adjustments': [],
    'active_adjustment_idx': -1,
}

# --- APP CONFIGURATION ---
# Global application constants
APP_CONFIG: Dict[str, Any] = {
    'preview_max_res': 1800,
    'display_width': 1800,
    'thumbnail_size': 120,
    'max_workers': 8,
    'db_path': "settings.db",
    'presets_dir': "presets",
    'default_export_dir': "processed",
    'autocrop_detect_res': 1800
}
