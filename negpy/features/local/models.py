from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PolygonMask:
    # Vertices in raw-image normalised coords [0,1]×[0,1].
    vertices: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    strength: float = 0.3  # EV stops; positive = dodge, negative = burn
    feather: float = 0.04  # Gaussian sigma as fraction of shorter image dimension


@dataclass(frozen=True)
class LocalAdjustmentsConfig:
    masks: Tuple[PolygonMask, ...] = field(default_factory=tuple)
