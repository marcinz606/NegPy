from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LocalAdjustmentConfig:
    points: List[Tuple[float, float]] = field(default_factory=list)
    strength: float = 0.0
    radius: int = 50
    feather: float = 0.5
    luma_range: Tuple[float, float] = (0.0, 1.0)
    luma_softness: float = 0.2


@dataclass(frozen=True)
class RetouchConfig:
    dust_remove: bool = True
    dust_threshold: float = 0.75
    dust_size: int = 2
    manual_dust_spots: List[Tuple[float, float, float]] = field(default_factory=list)
    manual_dust_size: int = 4
    local_adjustments: List[LocalAdjustmentConfig] = field(default_factory=list)
