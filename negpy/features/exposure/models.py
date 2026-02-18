from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ExposureConfig:
    """
    Print parameters (Density, Grade, Color).
    """

    density: float = 1.0
    grade: float = 2.0
    use_camera_wb: bool = False
    wb_cyan: float = 0.0
    wb_magenta: float = 0.0
    wb_yellow: float = 0.0
    shadow_cyan: float = 0.0
    shadow_magenta: float = 0.0
    shadow_yellow: float = 0.0
    highlight_cyan: float = 0.0
    highlight_magenta: float = 0.0
    highlight_yellow: float = 0.0
    toe: float = 0.0
    toe_width: float = 3.0
    toe_hardness: float = 1.0
    shoulder: float = 0.0
    shoulder_width: float = 3.0
    shoulder_hardness: float = 1.0
    shadows: float = 0.0
    highlights: float = 0.0


EXPOSURE_CONSTANTS: Dict[str, Any] = {
    "cmy_max_density": 0.2,
    "density_multiplier": 0.2,
    "grade_multiplier": 2.0,
    "target_paper_range": 2.2,
    "anchor_midpoint": 0.0,
}
