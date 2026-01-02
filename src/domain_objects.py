from typing import List, TypedDict, Tuple, Optional


class LocalAdjustment(TypedDict, total=False):
    points: List[Tuple[float, float]]
    strength: float
    radius: int
    feather: float
    luma_range: Tuple[float, float]
    luma_softness: float


class ProcessingParams(TypedDict, total=False):
    density: float
    scan_gain: float
    scan_gain_s_toe: float
    scan_gain_h_shoulder: float
    toe: float
    shoulder: float
    auto_wb: bool
    shadow_desat_strength: float
    grade: float
    exposure: float
    dust_remove: bool
    dust_threshold: float
    dust_size: int
    c_noise_remove: bool
    c_noise_strength: int
    sharpen: float
    rotation: int
    fine_rotation: float
    wb_cyan: float
    wb_magenta: float
    wb_yellow: float
    wb_manual_r: float
    wb_manual_g: float
    wb_manual_b: float
    temperature: float
    shadow_temp: float
    highlight_temp: float
    color_separation: float
    saturation: float
    autocrop: bool
    autocrop_offset: int
    autocrop_ratio: str
    export_color_space: str
    export_fmt: str
    export_size: float
    export_dpi: int
    export_add_border: bool
    export_border_size: float
    export_border_color: str
    export_path: str
    process_mode: str
    is_bw: bool
    manual_dust_spots: List[Tuple[float, float]]
    manual_dust_size: int
    local_adjustments: List[LocalAdjustment]
    active_adjustment_idx: int
    gamma: float


class AppConfig(TypedDict):
    preview_max_res: int
    display_width: int
    thumbnail_size: int
    max_workers: int
    edits_db_path: str
    settings_db_path: str
    presets_dir: str
    cache_dir: str
    default_export_dir: str
    autocrop_detect_res: int
    adobe_rgb_profile: str


class SidebarData(TypedDict, total=False):
    # From Export Section
    out_fmt: str
    color_space: str
    print_width: float
    print_dpi: int
    export_path: str
    add_border: bool
    border_size: float
    border_color: str
    apply_icc: bool
    process_btn: bool
    # From Retouch Section (currently empty dict but reserved)


class ExportSettings(TypedDict, total=False):
    output_format: str
    print_width_cm: float
    dpi: int
    sharpen_amount: float
    filename: str
    add_border: bool
    border_size_cm: float
    border_color: str
    icc_profile_path: Optional[str]
    color_space: str
