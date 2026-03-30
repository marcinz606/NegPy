from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ExposureConfig:
    """
    Print parameters (Density, Grade, Color).
    """

    density: float = 1.0
    grade: float = 2.5
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
    toe_width: float = 2.5
    shoulder: float = 0.0
    shoulder_width: float = 2.5


EXPOSURE_CONSTANTS: Dict[str, Any] = {
    "cmy_max_density": 0.15,
    "density_multiplier": 0.15,
    "grade_multiplier": 1.75,
}
