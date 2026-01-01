from typing import Dict, List, Tuple, Any

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
    'exposure': 0.0,
    'exposure_shadows': 0.0,
    'exposure_shadows_range': 1.0,
    'exposure_highlights': 0.0,
    'exposure_highlights_range': 1.0,
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
