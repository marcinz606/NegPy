from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class LabConfig:
    color_separation: float = 1.0
    hypertone_strength: float = 0.0
    c_noise_strength: float = 0.25
    sharpen: float = 0.25
    crosstalk_matrix: Optional[List[float]] = None
    exposure: float = 0.0  # Linear exposure trim
