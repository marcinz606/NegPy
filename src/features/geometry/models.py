from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GeometryConfig:
    rotation: int = 0
    fine_rotation: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    autocrop_offset: int = 2
    autocrop_ratio: str = "3:2"
    manual_crop_rect: Optional[Tuple[float, float, float, float]] = None
