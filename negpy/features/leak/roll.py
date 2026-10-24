"""Leaks across a roll. A camera throws its leak at the same place on every frame it exposes, so a
leak's shapes carry from one frame to another by the film frame, the picture box, and each frame
then reads the leak's own strength and color: how long the film sat behind the gap sets them."""

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

from negpy.features.leak.film import Rect, work_copy
from negpy.features.leak.models import LeakShape

FrameBox = Tuple[Rect, Tuple[int, int]]


def picture_box(img: np.ndarray, geometry) -> Optional[FrameBox]:
    """((y0, y1, x0, x1), (h, w)) of the picture in the frame's work copy; None without one."""
    from negpy.features.leak.rebate import picture_rect

    small = work_copy(img)
    rect = picture_rect(small, geometry)
    return None if rect is None else (rect, small.shape[:2])


def map_shapes(shapes: Sequence[LeakShape], source: FrameBox, target: FrameBox) -> Tuple[LeakShape, ...]:
    """*shapes* moved from the source frame's picture box to the target's: each point keeps its place
    in the box, and each width and brush size its share of the box's long side."""
    (sy0, sy1, sx0, sx1), (sh, sw) = source
    (ty0, ty1, tx0, tx1), (th, tw) = target

    def point(p) -> Tuple[float, float]:
        u = (p[0] * sw - sx0) / max(sx1 - sx0, 1)
        v = (p[1] * sh - sy0) / max(sy1 - sy0, 1)
        return (float((tx0 + u * (tx1 - tx0)) / tw), float((ty0 + v * (ty1 - ty0)) / th))

    scale = (max(ty1 - ty0, tx1 - tx0) / max(th, tw)) / (max(sy1 - sy0, sx1 - sx0) / max(sh, sw))
    return tuple(
        replace(
            s,
            points=tuple(point(p) for p in s.points),
            strokes=tuple((tuple(point(p) for p in pts), size * scale) for pts, size in s.strokes),
            width=(s.width[0] * scale, s.width[1] * scale),
        )
        for s in shapes
    )
