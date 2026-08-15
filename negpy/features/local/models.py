from dataclasses import dataclass, field
from enum import StrEnum
from typing import Tuple


class MaskShape(StrEnum):
    """How to read a mask's vertices.

    POLYGON: control points of a closed, smooth outline.
    OVAL: 3 points, the centre and one end of each axis. The outline is the unit
        circle under the matrix [u v], u = p1 - c, v = p2 - c. The axes can be oblique.
    GRADIENT: 2 points. The effect is full at the first point and zero at the second.
    """

    POLYGON = "polygon"
    OVAL = "oval"
    GRADIENT = "gradient"


@dataclass(frozen=True)
class LocalMask:
    # Vertices in raw-image normalised coords [0,1]. The `shape` field tells how to read them.
    vertices: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    # Print exposure in stops, darkroom-signed like vignette_stops: positive is a burn, a
    # longer exposure and darker paper, and negative is a dodge. 0 is the frame's own
    # exposure, so a freshly drawn mask changes nothing until it is given a value.
    stops: float = 0.0
    feather: float = 0.04  # Gaussian sigma as fraction of shorter image dimension
    # Local grade in ISO-R points off the global grade, where negative is harder: the
    # darkroom's "burn this in through the hard filter". 0 prints at the frame's grade.
    grade: float = 0.0
    shape: MaskShape = MaskShape.POLYGON
    # Apply the mask outside the shape, not inside it.
    invert: bool = False


@dataclass(frozen=True)
class LocalAdjustmentsConfig:
    masks: Tuple[LocalMask, ...] = field(default_factory=tuple)
