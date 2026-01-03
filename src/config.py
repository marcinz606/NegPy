from src.domain_objects import AppConfig, ImageSettings

# --- APP CONFIGURATION ---
# Global application constants
APP_CONFIG = AppConfig(
    preview_max_res=1200,
    display_width=1200,
    thumbnail_size=120,
    max_workers=8,
    edits_db_path="user/edits.db",
    settings_db_path="user/settings.db",
    presets_dir="user/presets",
    cache_dir="user/cache",
    user_icc_dir="user/icc",
    default_export_dir="user/export",
    autocrop_detect_res=1200,
    adobe_rgb_profile="icc/AdobeCompat-v4.icc",
)

# --- PIPELINE CONSTANTS ---
# Centralized multipliers and targets for the photometric engine
PIPELINE_CONSTANTS = {
    "cmy_max_density": 0.2,  # Max absolute density shift for CMY sliders
    "density_multiplier": 0.4,  # Maps Density slider to Log Exposure shift
    "grade_multiplier": 1.5,  # Maps Grade slider to Sigmoid Slope
    "auto_grade_target": 2.7,  # Target output range for Auto-Grade solver
    "auto_density_target": 0.45,  # Target highlight pivot for Auto-Density solver
    "paper_warmth_strength": 0.5,  # Multiplier for paper warmth density shift
    "toning_strength": 0.5,  # Multiplier for shadow/highlight toning density shift
}

# --- DEFAULT SETTINGS ---
# The baseline parameters for every newly imported RAW file
DEFAULT_SETTINGS = ImageSettings(
    density=1.0,
    grade=2.5,
    toe=0.0,
    toe_width=3.0,
    toe_hardness=1.0,
    shoulder=0.0,
    shoulder_width=3.0,
    shoulder_hardness=1.0,
    auto_wb=False,
    shadow_desat_strength=1.0,
    exposure=0.0,
    dust_remove=True,
    dust_threshold=0.55,
    dust_size=3,
    c_noise_remove=True,
    c_noise_strength=25,
    sharpen=0.20,
    rotation=0,
    fine_rotation=0.0,
    wb_cyan=0.0,
    wb_magenta=0.0,
    wb_yellow=0.0,
    temperature=0.0,
    shadow_temp=0.0,
    highlight_temp=0.0,
    color_separation=1.0,
    saturation=1.0,
    autocrop=True,
    autocrop_offset=5,
    autocrop_ratio="3:2",
    export_color_space="sRGB",
    export_fmt="JPEG",
    export_size=27.0,
    export_dpi=300,
    export_add_border=True,
    export_border_size=0.50,
    export_border_color="#ffffff",
    export_path=APP_CONFIG.default_export_dir,
    process_mode="C41",
    manual_dust_spots=[],
    manual_dust_size=4,
    local_adjustments=[],
    active_adjustment_idx=-1,
)
