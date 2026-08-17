import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from negpy.features.local.models import LocalAdjustmentsConfig, MaskShape
from negpy.features.geometry.logic import map_coords_to_geometry, smooth_polyline

_OVAL_SAMPLES = 64

Point = Tuple[float, float]


def min_points(shape: MaskShape) -> int:
    """The minimum number of vertices that this shape needs."""
    return 2 if shape == MaskShape.GRADIENT else 3


def outline_points(shape: MaskShape, pts: Sequence[Point]) -> List[Point]:
    """The closed mask outline, in the same space as `pts`.

    The result is empty for a gradient, which has no boundary. The rasteriser, the
    canvas overlay and the printing-notes map all use this function, so they agree.
    """
    if shape == MaskShape.GRADIENT:
        return []
    if shape == MaskShape.OVAL:
        (cx, cy), (px, py), (qx, qy) = pts[:3]
        ux, uy = px - cx, py - cy
        vx, vy = qx - cx, qy - cy
        return [
            (cx + ux * math.cos(t) + vx * math.sin(t), cy + uy * math.cos(t) + vy * math.sin(t))
            for t in (2.0 * math.pi * i / _OVAL_SAMPLES for i in range(_OVAL_SAMPLES))
        ]
    return smooth_polyline(list(pts), closed=True)


def _rasterise_ramp(a: Point, b: Point, h: int, w: int) -> np.ndarray:
    """The card-edge ramp. Alpha is 1 at `a` and 0 at `b`, and constant beyond each."""
    ax, ay = a[0] * w, a[1] * h
    dx, dy = b[0] * w - ax, b[1] * h - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return np.zeros((h, w), dtype=np.float32)
    xs = (np.arange(w, dtype=np.float32) - ax) * (dx / denom)
    ys = (np.arange(h, dtype=np.float32) - ay) * (dy / denom)
    t = np.clip(xs[None, :] + ys[:, None], 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)


def rasterise(
    shape: MaskShape,
    pts: Sequence[Point],
    h: int,
    w: int,
    feather_sigma: float,
    invert: bool = False,
) -> np.ndarray:
    """Rasterise the control points, normalised to [0,1], to a float32 alpha [h, w].

    `feather_sigma` is a Gaussian sigma in pixels on the hard fill. A gradient ignores
    it, because the distance between its two points sets the softness.
    """
    if shape == MaskShape.GRADIENT:
        alpha = _rasterise_ramp(pts[0], pts[1], h, w)
    else:
        outline = np.array([[x * w, y * h] for x, y in outline_points(shape, pts)], dtype=np.float32)
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(filled, [outline.astype(np.int32)], 255)
        alpha = filled.astype(np.float32) / 255.0
        if feather_sigma > 1e-3:
            k = int(feather_sigma * 3) | 1  # odd kernel covering ~3 sigma
            alpha = cv2.GaussianBlur(alpha, (k, k), feather_sigma)
    return 1.0 - alpha if invert else alpha


def compute_local_maps(
    config: LocalAdjustmentsConfig,
    h: int,
    w: int,
    orig_shape: Tuple[int, int],
    rotation: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    distortion_k1: float = 0.0,
    converge_v: float = 0.0,
    converge_h: float = 0.0,
) -> np.ndarray:
    """
    Build the per-pixel dodge/burn maps [h, w, 2] float32, each plane the sum over
    masks of the mask's value times its feathered alpha: plane 0 is print exposure
    in stops (positive = burn, negative = dodge), plane 1 the local grade delta in
    ISO-R points. One rasterisation feeds both. All-zeros when there are no masks.
    """
    maps = np.zeros((h, w, 2), dtype=np.float32)
    if not config.masks:
        return maps

    short_side = float(min(h, w))
    for mask in config.masks:
        if len(mask.vertices) < min_points(mask.shape):
            continue

        transformed = [
            map_coords_to_geometry(
                rx,
                ry,
                orig_shape,
                rotation,
                fine_rotation,
                flip_horizontal,
                flip_vertical,
                distortion_k1=distortion_k1,
                converge_v=converge_v,
                converge_h=converge_h,
            )
            for rx, ry in mask.vertices
        ]

        alpha = rasterise(mask.shape, transformed, h, w, mask.feather * short_side, mask.invert)
        maps[:, :, 0] += mask.stops * alpha
        if mask.grade:
            maps[:, :, 1] += mask.grade * alpha

    return maps
