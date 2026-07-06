from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class RetouchConfig:
    dust_remove: bool = False
    dust_threshold: float = 0.66
    dust_size: int = 4
    manual_dust_spots: List[Tuple[float, float, float]] = field(default_factory=list)
    # Each stroke: (points, size, src_dx, src_dy); points = [[nx, ny], ...] source-normalized,
    # size in source px, (src_dx, src_dy) = source-normalized offset to the clone source.
    # A single-point stroke is a spot. manual_dust_spots is the legacy pre-stroke format.
    manual_heal_strokes: List[Tuple] = field(default_factory=list)
    manual_dust_size: int = 6
    ir_dust_remove: bool = False
    ir_threshold: float = 0.5
    ir_inpaint_radius: int = 3
