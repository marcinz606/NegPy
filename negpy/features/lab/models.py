from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class LabConfig:
    """
    Scanner emulation (Sharpening, CLAHE).
    """

    color_separation: float = 1.0
    saturation: float = 1.0
    clahe_strength: float = 0.0
    sharpen: float = 0.25
    crosstalk_matrix: Optional[List[float]] = None

    C41_MATRIX: List[float] = field(default_factory=lambda: [1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0])

    E6_MATRIX: List[float] = field(default_factory=lambda: [1.1, -0.06, -0.04, -0.04, 1.1, -0.06, -0.04, -0.06, 1.1])
