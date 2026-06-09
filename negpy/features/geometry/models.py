from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GeometryConfig:
    rotation: int = 0
    fine_rotation: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    auto_crop_enabled: bool = False

    autocrop_offset: int = 0
    autocrop_ratio: str = "3:2"
    manual_crop_rect: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """Ensure a JSON-loaded list is converted back to a tuple, keeping the
        frozen dataclass hashable for pipeline cache keys."""
        if self.manual_crop_rect is not None:
            object.__setattr__(self, "manual_crop_rect", tuple(self.manual_crop_rect))
