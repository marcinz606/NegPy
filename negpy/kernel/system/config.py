import os

from negpy.domain.models import (
    AspectRatio,
    ColorSpace,
    ExportConfig,
    ExportFormat,
    WorkspaceConfig,
)
from negpy.domain.types import AppConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lab.models import LabConfig
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.retouch.models import RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.kernel.system.paths import get_default_user_dir, get_resource_path

BASE_USER_DIR = get_default_user_dir()
APP_CONFIG = AppConfig(
    thumbnail_size=120,
    max_workers=max(1, (os.cpu_count() or 1)),
    preview_render_size=1600,
    max_history_steps=100,
    edits_db_path=os.path.join(BASE_USER_DIR, "edits.db"),
    settings_db_path=os.path.join(BASE_USER_DIR, "settings.db"),
    presets_dir=os.path.join(BASE_USER_DIR, "presets"),
    cache_dir=os.path.join(BASE_USER_DIR, "cache"),
    user_icc_dir=os.path.join(BASE_USER_DIR, "icc"),
    crosstalk_dir=os.path.join(BASE_USER_DIR, "crosstalk"),
    default_export_dir=os.path.join(BASE_USER_DIR, "export"),
    adobe_rgb_profile=get_resource_path("icc/AdobeCompat-v4.icc"),
    use_gpu=True,
    override_toml_path=os.path.join(BASE_USER_DIR, "override.toml"),
    preview_cache_max_entries=8,
    preview_cache_max_bytes=1_200_000_000,
    canvas_zoom_min=0.25,
    canvas_zoom_max=8.0,
)


DEFAULT_WORKSPACE_CONFIG = WorkspaceConfig(
    process=ProcessConfig(
        process_mode=ProcessMode.C41,
        analysis_buffer=0.05,
    ),
    exposure=ExposureConfig(
        density=1.0,
        grade=2.5,
        toe=0.0,
        toe_width=2.5,
        shoulder=0.0,
        shoulder_width=2.5,
    ),
    geometry=GeometryConfig(
        rotation=0,
        fine_rotation=0.0,
        autocrop_offset=1,
        autocrop_ratio=AspectRatio.R_3_2,
    ),
    lab=LabConfig(
        color_separation=1.5,
        clahe_strength=0.25,
        saturation=1.0,
        sharpen=0.25,
    ),
    toning=ToningConfig(
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
        export_color_space=ColorSpace.SRGB.value,
        export_print_size=30.0,
        export_dpi=300,
        export_path=APP_CONFIG.default_export_dir,
    ),
    finish=FinishConfig(
        border_size=0.0,
        border_color="#ffffff",
    ),
)
