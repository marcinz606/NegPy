from dataclasses import dataclass


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
