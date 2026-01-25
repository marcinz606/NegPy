import os
from src.kernel.system.paths import get_resource_path, get_default_user_dir
from src.domain.types import AppConfig
from src.domain.models import (
    WorkspaceConfig,
    ExportConfig,
    ColorSpace,
    ProcessMode,
    ExportFormat,
    AspectRatio,
)
from src.features.exposure.models import ExposureConfig
from src.features.geometry.models import GeometryConfig
from src.features.lab.models import LabConfig
from src.features.retouch.models import RetouchConfig
from src.features.toning.models import ToningConfig, PaperProfileName


BASE_USER_DIR = get_default_user_dir()
APP_CONFIG = AppConfig(
    thumbnail_size=120,
    max_workers=max(1, (os.cpu_count() or 1)),
    preview_render_size=2000,
    edits_db_path=os.path.join(BASE_USER_DIR, "edits.db"),
    settings_db_path=os.path.join(BASE_USER_DIR, "settings.db"),
    presets_dir=os.path.join(BASE_USER_DIR, "presets"),
    cache_dir=os.path.join(BASE_USER_DIR, "cache"),
    user_icc_dir=os.path.join(BASE_USER_DIR, "icc"),
    default_export_dir=os.path.join(BASE_USER_DIR, "export"),
    adobe_rgb_profile=get_resource_path("icc/AdobeCompat-v4.icc"),
    use_gpu=True,
)


DEFAULT_WORKSPACE_CONFIG = WorkspaceConfig(
    process_mode=ProcessMode.C41,
    exposure=ExposureConfig(
        density=1.0,
        grade=2.0,
        toe=0.0,
        toe_width=3.0,
        toe_hardness=1.0,
        shoulder=0.0,
        shoulder_width=3.0,
        shoulder_hardness=1.0,
        analysis_buffer=0.07,
    ),
    geometry=GeometryConfig(
        rotation=0,
        fine_rotation=0.0,
        autocrop_offset=1,
        autocrop_ratio=AspectRatio.R_3_2,
    ),
    lab=LabConfig(
        color_separation=1.0,
        clahe_strength=0.0,
        saturation=1.0,
        sharpen=0.25,
    ),
    toning=ToningConfig(
        paper_profile=PaperProfileName.NONE,
        selenium_strength=0.0,
        sepia_strength=0.0,
    ),
    retouch=RetouchConfig(
        dust_remove=False,
        dust_threshold=0.66,
        dust_size=4,
        manual_dust_size=6,
    ),
    export=ExportConfig(
        export_fmt=ExportFormat.JPEG,
        export_color_space=ColorSpace.ADOBE_RGB.value,
        export_print_size=30.0,
        export_dpi=300,
        export_border_size=0.0,
        export_border_color="#ffffff",
        export_path=APP_CONFIG.default_export_dir,
    ),
)
