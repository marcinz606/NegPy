from dataclasses import dataclass, field
from enum import StrEnum
from typing import Tuple

Point = Tuple[float, float]


class LeakKind(StrEnum):
    BAND = "band"
    EDGE = "edge"
    SPOT = "spot"
    PAINT = "paint"


class LeakColor(StrEnum):
    """Where a leak's color comes from. AUTO reads it off the film; BACK is light through the base,
    which reaches the red-sensitive layer first; FRONT is light from the lens side, which reaches the
    blue-sensitive layer first and prints near neutral."""

    AUTO = "auto"
    BACK = "back"
    FRONT = "front"


@dataclass(frozen=True)
class LeakShape:
    """One leak the user placed or Auto found. *points* are raw-normalized, before any geometry: the
    ends of a band's line, the ends of the stretch of picture edge an edge leak comes in from, or a
    spot's center and the ends of its two axes. *strokes* are a painted leak's brush strokes, each
    (raw-normalized points, diameter at HEAL_SIZE_REF). *width* is a band's reach on each side of its line,
    (left, right) of the line's direction, and an edge's (reach, 0), into the picture on its left,
    in long sides of the frame. *strength* trims the leak the film reads; *falloff* shapes how a band,
    a spot or a painted region fades toward its edge, 0 holding a shoulder, 1 fading early and softly. *color* says
    where the leak's color comes from, and *trim* takes that share more of the layer the color leads in, as light
    from the front leads in blue and light through the base in red."""

    kind: LeakKind = LeakKind.BAND
    points: Tuple[Point, ...] = ()
    strokes: Tuple[Tuple[Tuple[Point, ...], float], ...] = ()
    width: Tuple[float, float] = (0.0, 0.0)
    strength: float = 1.0
    falloff: float = 0.5
    color: LeakColor = LeakColor.AUTO
    trim: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        # JSON gives back lists; tuples keep the frozen config hashable for change detection.
        object.__setattr__(self, "kind", LeakKind(self.kind))
        object.__setattr__(self, "color", LeakColor(self.color))
        object.__setattr__(self, "trim", float(self.trim))
        object.__setattr__(self, "points", tuple((float(p[0]), float(p[1])) for p in self.points))
        object.__setattr__(self, "width", (float(self.width[0]), float(self.width[1])))
        object.__setattr__(
            self, "strokes", tuple((tuple((float(p[0]), float(p[1])) for p in points), float(size)) for points, size in self.strokes)
        )


@dataclass(frozen=True)
class LeakConfig:
    """Light leak reduction, a bake on the linear negative before every dust repair."""

    leak_remove: bool = False
    leak_shapes: Tuple[LeakShape, ...] = field(default_factory=tuple)
    # The film's maximum density over its base, where its curve flattens; 0 is a straight line.
    leak_dmax: float = 0.0
    # Rebuild a layer the leak saturated from the layers that still hold picture.
    leak_recover: bool = True
    # Clean-up where leak came off: grain reduction and a local contrast lift, 0..1.
    leak_grain: float = 0.5
    leak_contrast: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "leak_shapes", tuple(s if isinstance(s, LeakShape) else LeakShape(**s) for s in self.leak_shapes))
