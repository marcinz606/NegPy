from dataclasses import dataclass
from typing import Tuple


@dataclass
class PaperSubstrate:
    name: str
    tint: Tuple[float, float, float]
    dmax_boost: float


@dataclass(frozen=True)
class ToningConfig:
    """
    Paper & Toner params.
    """

    paper_profile: str = "None"
    selenium_strength: float = 0.0
    sepia_strength: float = 0.0
