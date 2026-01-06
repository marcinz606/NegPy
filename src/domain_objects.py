from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class LocalAdjustment:
    points: List[Tuple[float, float]] = field(default_factory=list)
    strength: float = 0.0
    radius: int = 50
    feather: float = 0.5
    luma_range: Tuple[float, float] = (0.0, 1.0)
    luma_softness: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperSubstrate:
    name: str
    tint: Tuple[float, float, float]
    dmax_boost: float


@dataclass
class ImageSettings:
    density: float = 1.0
    toe: float = 0.0
    toe_width: float = 3.0
    toe_hardness: float = 1.0
    shoulder: float = 0.0
    shoulder_width: float = 3.0
    shoulder_hardness: float = 1.0
    auto_wb: bool = False
    grade: float = 2.5
    exposure: float = 0.0
    dust_remove: bool = True
    dust_threshold: float = 0.75
    dust_size: int = 2
    c_noise_strength: float = 0.25
    sharpen: float = 0.25
    rotation: int = 0
    fine_rotation: float = 0.0
    wb_cyan: float = 0.0
    wb_magenta: float = 0.0
    wb_yellow: float = 0.0
    paper_profile: str = "None"
    selenium_strength: float = 0.0
    sepia_strength: float = 0.0
    color_separation: float = 1.0
    crosstalk_matrix: List[float] = field(
        default_factory=lambda: [1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0]
    )
    hypertone_strength: float = 0.0
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    autocrop: bool = True
    autocrop_offset: int = 5
    autocrop_ratio: str = "3:2"
    export_color_space: str = "sRGB"
    export_fmt: str = "JPEG"
    export_size: float = 27.0
    export_dpi: int = 300
    export_add_border: bool = True
    export_border_size: float = 0.25
    export_border_color: str = "#ffffff"
    export_path: str = "export"
    process_mode: str = "C41"
    is_bw: bool = False
    manual_dust_spots: List[Tuple[float, float, float]] = field(default_factory=list)
    manual_dust_size: int = 5
    local_adjustments: List[LocalAdjustment] = field(default_factory=list)
    active_adjustment_idx: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageSettings":
        # Handle nested dataclasses
        if "local_adjustments" in data:
            data = data.copy()
            data["local_adjustments"] = [
                LocalAdjustment(**adj) if isinstance(adj, dict) else adj
                for adj in data["local_adjustments"]
            ]

        # Filter out keys not in the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class AppConfig:
    thumbnail_size: int
    max_workers: int
    preview_render_size: int
    edits_db_path: str
    settings_db_path: str
    presets_dir: str
    cache_dir: str
    user_icc_dir: str
    default_export_dir: str
    adobe_rgb_profile: str


@dataclass
class SidebarData:
    out_fmt: str = "JPEG"
    color_space: str = "sRGB"
    print_width: float = 27.0
    print_dpi: int = 300
    export_path: str = "export"
    add_border: bool = True
    border_size: float = 0.25
    border_color: str = "#ffffff"
    apply_icc: bool = False
    process_btn: bool = False
    export_btn: bool = False


@dataclass
class ExportSettings:
    output_format: str = "JPEG"
    print_width_cm: float = 27.0
    dpi: int = 300
    sharpen_amount: float = 0.75
    filename: str = ""
    add_border: bool = False
    border_size_cm: float = 1.0
    border_color: str = "#000000"
    icc_profile_path: Optional[str] = None
    color_space: str = "sRGB"
