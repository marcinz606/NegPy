from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class LabConfig:
    """
    Scanner and high-end lab emulation parameters.
    """

    color_separation: float = 1.0
    hypertone_strength: float = 0.0
    c_noise_strength: float = 0.25
    sharpen: float = 0.25
    crosstalk_matrix: Optional[List[float]] = field(
        default_factory=lambda: [
            1.0,
            -0.05,
            -0.02,
            -0.04,
            1.0,
            -0.08,
            -0.01,
            -0.1,
            1.0,
        ]
    )
