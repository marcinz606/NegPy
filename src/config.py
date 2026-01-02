from typing import Dict, Any

# --- DEFAULT SETTINGS ---
# The baseline parameters for every newly imported RAW file
DEFAULT_SETTINGS: Dict[str, Any] = {
    'grade': 2.5,
    'scan_gain_s_toe': 0.0,
    'scan_gain_h_shoulder': 0.0,
    'auto_wb': False,
    'shadow_desat_strength': 1.0,
    'contrast': 1.0,
    'exposure': 0.0,
    'dust_remove': True,
    'dust_threshold': 0.55,
    'dust_size': 2,
    'c_noise_remove': True,
    'c_noise_strength': 25,
    'sharpen': 0.20,
    'rotation': 0,
    'fine_rotation': 0.0,
    'wb_cyan': 0,
    'wb_magenta': 0,
    'wb_yellow': 0,
    'temperature': 0.0,
    'shadow_temp': 0.0,
    'highlight_temp': 0.0,
    'color_separation': 1.0,
    'saturation': 1.0,
    'autocrop': True,
    'autocrop_offset': 5,
    'autocrop_ratio': "3:2",
    'export_color_space': "sRGB",
    'process_mode': "C41",
    'manual_dust_spots': [],
    'manual_dust_size': 4,
    'local_adjustments': [],
    'active_adjustment_idx': -1,
}

# --- APP CONFIGURATION ---
# Global application constants
APP_CONFIG: Dict[str, Any] = {
    'preview_max_res': 1500,
    'display_width': 1500,
    'thumbnail_size': 120,
    'max_workers': 8,
    'edits_db_path': "user/edits.db",
    'settings_db_path': "user/settings.db",
    'presets_dir': "user/presets",
    'default_export_dir': "processed",
    'autocrop_detect_res': 1500,
    'adobe_rgb_profile': "icc/AdobeCompat-v4.icc"
}
