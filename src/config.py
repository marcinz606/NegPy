import os
from src.core.types import AppConfig
from src.core.session.models import WorkspaceConfig, ExportConfig
from src.features.exposure.models import ExposureConfig
from src.features.geometry.models import GeometryConfig
from src.features.lab.models import LabConfig
from src.features.retouch.models import RetouchConfig
from src.features.toning.models import ToningConfig

# User dir env (mapped to OS Documents folder)
BASE_USER_DIR = os.path.abspath(os.getenv("DARKROOM_USER_DIR", "user"))

# Global application constants
APP_CONFIG = AppConfig(
    thumbnail_size=100,
    max_workers=max(1, (os.cpu_count() or 1) - 1),
    preview_render_size=1200,
    edits_db_path=os.path.join(BASE_USER_DIR, "edits.db"),
    settings_db_path=os.path.join(BASE_USER_DIR, "settings.db"),
    presets_dir=os.path.join(BASE_USER_DIR, "presets"),
    cache_dir=os.path.join(BASE_USER_DIR, "cache"),
    user_icc_dir=os.path.join(BASE_USER_DIR, "icc"),
    default_export_dir=os.path.join(BASE_USER_DIR, "export"),
    adobe_rgb_profile="icc/AdobeCompat-v4.icc",
)

# Centralized multipliers and targets for the photometric engine
PIPELINE_CONSTANTS = {
    "cmy_max_density": 0.2,  # Max absolute density shift for CMY sliders
    "density_multiplier": 0.4,  # Maps Density slider to Log Exposure shift
    "grade_multiplier": 1.5,  # Maps Grade slider to Sigmoid Slope
    "target_paper_range": 2.1,  # To mimic exposure range of darkroom paper.
    "anchor_midpoint": 0.0,  # Sigmoid Center in centered log space (Zone V)
}

# The modular default configuration for every newly imported RAW file
DEFAULT_WORKSPACE_CONFIG = WorkspaceConfig(
    process_mode="C41",
    exposure=ExposureConfig(
        density=1.0,
        grade=2.5,
        toe=0.0,
        toe_width=3.0,
        toe_hardness=1.0,
        shoulder=0.0,
        shoulder_width=3.0,
        shoulder_hardness=1.0,
    ),
    geometry=GeometryConfig(
        rotation=0,
        fine_rotation=0.0,
        autocrop=True,
        autocrop_offset=1,
        autocrop_ratio="3:2",
    ),
    lab=LabConfig(
        color_separation=1.0,
        hypertone_strength=0.0,
        c_noise_strength=0.0,
        sharpen=0.25,
    ),
    toning=ToningConfig(
        paper_profile="None",
        selenium_strength=0.0,
        sepia_strength=0.0,
    ),
    retouch=RetouchConfig(
        dust_remove=True,
        dust_threshold=0.75,
        dust_size=3,
    ),
    export=ExportConfig(
        export_fmt="JPEG",
        export_color_space="sRGB",
        export_print_size=27.0,
        export_dpi=300,
        export_add_border=False,
        export_border_size=0.5,
        export_border_color="#FFFFFF",
        export_path=APP_CONFIG.default_export_dir,
    ),
)
