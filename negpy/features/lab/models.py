from dataclasses import dataclass, field
from typing import List, Optional

# ProPhoto working-space look compensation: pulled below 1.0 so the wide-gamut default
# render lands near the old Adobe RGB look. Single knob — tune by eye in `make run`.
DEFAULT_SATURATION = 0.85


@dataclass(frozen=True)
class LabConfig:
    """
    Scanner emulation (Sharpening, CLAHE).
    """

    color_separation: float = 1.0
    saturation: float = DEFAULT_SATURATION
    vibrance: float = 1.0
    clahe_strength: float = 0.0
    sharpen: float = 0.25
    chroma_denoise: float = 0.0
    glow_amount: float = 0.0
    halation_strength: float = 0.0
    crosstalk_profile: str = "Default"
    crosstalk_matrix: Optional[List[float]] = None

    DEFAULT_MATRIX: List[float] = field(default_factory=lambda: [1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0])
