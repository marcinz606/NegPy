from typing import Dict, List, Tuple, Any

# --- TONE CURVE PRESETS ---
# Maps preset names to (x_coords, y_coords) tuples for interpolation
TONE_CURVES_PRESETS: Dict[str, Tuple[List[float], List[float]]] = {
    "Linear": ([0, 1], [0, 1]),
    "Medium Contrast": ([0, 0.25, 0.75, 1], [0, 0.2, 0.8, 1]),
    "Soft Contrast": ([0, 0.25, 0.75, 1], [0, 0.30, 0.70, 1]),
    "Soft Highs": ([0, 0.5, 0.8, 1], [0, 0.5, 0.75, 0.9]),
    "Smooth": ([0, 0.2, 0.33, 0.5, 0.8, 0.9, 1], [0, 0.175, 0.25, 0.375, 0.75, 0.925, 0.98]), # Raised shadows, rolled off highs/mids, anchored black
    "Dense": ([0, 0.5, 1], [0, 0.33, 0.975]), # Darkens midtones for overexposed/dense negs
    "Thin": ([0, 0.5, 1], [0, 0.66, 1]),  # Brightens midtones for underexposed/thin negs
    "LOG": ([0, 1], [0.05, 0.95]),
}

# --- DEFAULT SETTINGS ---
# The baseline parameters for every newly imported RAW file
DEFAULT_SETTINGS: Dict[str, Any] = {
    'auto_wb': True,
    'autocrop': True,
    'monochrome': False,
    'autocrop_offset': 0,
    'temperature': 0.0,
    'shadow_temp': 0.0,
    'highlight_temp': 0.0,
    'cr_balance': 1.0,
    'mg_balance': 1.0,
    'yb_balance': 1.0,
    'shadow_cr': 1.0,
    'shadow_mg': 1.0,
    'shadow_yb': 1.0,
    'highlight_cr': 1.0,
    'highlight_mg': 1.0,
    'highlight_yb': 1.0,
    'saturation': 1.0,
    'gamma': 2.5,
    'black_point': 0.0,
    'white_point': 1.0,
    'exposure': 0.0,
    'contrast': 1.0,
    'grade_shadows': 2.5,
    'grade_highlights': 2.5,
    'curve_mode': "Smooth",
    'curve_strength': 1.0,
    'dust_remove': True,
    'dust_threshold': 0.55,
    'dust_size': 2,
    'c_noise_remove': True,
    'c_noise_strength': 50,
    'sharpen': 1.0,
    'rotation': 0,
    'wb_manual_r': 1.0,
    'wb_manual_g': 1.0,
    'wb_manual_b': 1.0,
    'manual_dust_spots': [],
    'manual_dust_size': 4,
    'local_adjustments': [],
    'active_adjustment_idx': -1
}

# --- APP CONFIGURATION ---
# Global application constants
APP_CONFIG: Dict[str, Any] = {
    'preview_max_res': 1600,
    'display_width': 1600,
    'thumbnail_size': 120,
    'max_workers': 8,
    'db_path': "settings.db",
    'presets_dir': "presets",
    'default_export_dir': "processed",
    'autocrop_detect_res': 1600
}
