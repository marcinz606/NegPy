from dataclasses import dataclass
from typing import Dict, Any

EXPOSURE_CONSTANTS: Dict[str, Any] = {
    "cmy_max_density": 0.1,  # Max absolute density shift for CMY sliders
    "density_multiplier": 0.2,  # Maps Density slider to Log Exposure shift
    "grade_multiplier": 2.0,  # Maps Grade slider to Sigmoid Slope
    "target_paper_range": 2.1,  # To mimic exposure range of darkroom paper.
    "anchor_midpoint": 0.0,  # Sigmoid Center in centered log space (Zone V)
}


@dataclass(frozen=True)
class ExposureConfig:
    """
    Configuration for the Photometric Exposure step.
    """

    # Primary Controls
    density: float = 1.0  # Exposure (Pivot shift)
    grade: float = 2.5  # Contrast (Slope)

    # Color Filtration (CMY)
    wb_cyan: float = 0.0
    wb_magenta: float = 0.0
    wb_yellow: float = 0.0

    # Curve Shape (Toe/Shoulder)
    toe: float = 0.0
    toe_width: float = 3.0
    toe_hardness: float = 1.0

    shoulder: float = 0.0
    shoulder_width: float = 3.0
    shoulder_hardness: float = 1.0
