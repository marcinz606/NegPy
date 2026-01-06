from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryConfig:
    rotation: int = 0
    fine_rotation: float = 0.0

    autocrop: bool = True
    autocrop_offset: int = 2
    autocrop_ratio: str = "3:2"
