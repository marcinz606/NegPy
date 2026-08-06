from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PolygonMask:
    # Vertices in raw-image normalised coords [0,1]×[0,1].
    vertices: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    # Print exposure in stops, darkroom-signed like vignette_stops: positive = burn
    # (longer exposure, darker paper), negative = dodge. 0 = the frame's own exposure,
    # so a freshly drawn mask changes nothing until it is given a value.
    stops: float = 0.0
    feather: float = 0.04  # Gaussian sigma as fraction of shorter image dimension
    # Local grade in ISO-R points off the global grade (negative = harder), the
    # darkroom's "burn this in through the hard filter". 0 = print at the frame's grade.
    grade: float = 0.0


@dataclass(frozen=True)
class LocalAdjustmentsConfig:
    masks: Tuple[PolygonMask, ...] = field(default_factory=tuple)
