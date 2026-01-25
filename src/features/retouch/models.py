from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class RetouchConfig:
    dust_remove: bool = False
    dust_threshold: float = 0.66
    dust_size: int = 4
    manual_dust_spots: List[Tuple[float, float, float]] = field(default_factory=list)
    manual_dust_size: int = 6
