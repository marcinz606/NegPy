from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Tuple


# Valid fine_rotation span (degrees): shared by the sidebar slider and the crop
# tool's rotation handles so both controls cover the same range. Anything beyond
# ±45° is an orientation change and belongs to the 90° rotate buttons.
FINE_ROTATION_LIMIT = 45.0


class AutocropMode(StrEnum):
    IMAGE = "image"  # crop to exposed image area (default)
    FILM = "film"  # crop to film extent, keep rebate/sprockets


@dataclass(frozen=True)
class GeometryConfig:
    rotation: int = 0
    fine_rotation: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    auto_crop_enabled: bool = False

    autocrop_offset: int = 0
    autocrop_ratio: str = "3:2"
    autocrop_mode: str = AutocropMode.IMAGE
    manual_crop_rect: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """Ensure a JSON-loaded list is converted back to a tuple, keeping the
        frozen dataclass hashable for pipeline cache keys."""
        if self.manual_crop_rect is not None:
            object.__setattr__(self, "manual_crop_rect", tuple(self.manual_crop_rect))
        if self.autocrop_mode not in (AutocropMode.IMAGE, AutocropMode.FILM):
            object.__setattr__(self, "autocrop_mode", AutocropMode.IMAGE.value)
