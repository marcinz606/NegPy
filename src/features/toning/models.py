from dataclasses import dataclass
from enum import StrEnum
from typing import Tuple


class PaperProfileName(StrEnum):
    NONE = "None"
    NEUTRAL_RC = "Neutral RC"
    COOL_GLOSSY = "Cool Glossy"
    WARM_FIBER = "Warm Fiber"


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

    paper_profile: str = PaperProfileName.NONE
    selenium_strength: float = 0.0
    sepia_strength: float = 0.0
