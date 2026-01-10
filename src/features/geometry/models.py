from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GeometryConfig:
    rotation: int = 0
    fine_rotation: float = 0.0

    autocrop: bool = True
    autocrop_offset: int = 2
    autocrop_ratio: str = "3:2"
    keep_full_frame: bool = False
    autocrop_assist_point: Optional[Tuple[float, float]] = None
    autocrop_assist_luma: Optional[float] = None
