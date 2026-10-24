import math
from typing import List, Sequence, Tuple

import negpy_core
import numpy as np

from negpy.features.local.models import MAX_KEYED_MASKS, LocalAdjustmentsConfig, LocalMask, MaskKey, MaskShape
from negpy.features.geometry.logic import map_coords_to_geometry, smooth_polyline

_OVAL_SAMPLES = 64
# Overlap testing grid, and the alpha below which a tint is too faint to hide anything, so a
# feather tail brushing a neighbour is not an overlap.
_OVERLAP_GRID = 96
_OVERLAP_ALPHA = 0.25

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


def _mask_spec(shape: MaskShape, pts: Sequence[Point], h: int, w: int, feather_sigma: float) -> tuple:
    """The alpha half of a negpy_core mask spec: ("fill", outline px, (sigma, ...)) or a gradient's
    card-edge ramp, 1 at the first point and 0 at the second, constant beyond each."""
    if shape == MaskShape.GRADIENT:
        (a, b) = pts[0], pts[1]
        ax, ay = a[0] * w, a[1] * h
        dx, dy = b[0] * w - ax, b[1] * h - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return ("empty", None, (0.0, 0.0, 0.0, 0.0, 0.0))
        return ("ramp", None, (0.0, ax, ay, dx / denom, dy / denom))
    outline = np.array([[x * w, y * h] for x, y in outline_points(shape, pts)], dtype=np.float32)
    return ("fill", outline.astype(np.int32), (float(feather_sigma), 0.0, 0.0, 0.0, 0.0))


def rasterise(
    shape: MaskShape,
    pts: Sequence[Point],
    h: int,
    w: int,
    feather_sigma: float,
    invert: bool = False,
) -> np.ndarray:
    """Rasterise the control points, normalised to [0,1], to a float32 alpha [h, w].

    `feather_sigma` is a Gaussian sigma in pixels on the hard fill, over a kernel of about 3
    sigma. A gradient ignores it, because the distance between its two points sets the softness.
    """
    kind, outline, params = _mask_spec(shape, pts, h, w, feather_sigma)
    return negpy_core.rasterise_mask((h, w), (kind, outline, params, bool(invert), -1, (0.0, 0.0, 0.0)))


def _tinted_area(mask, grid: int) -> np.ndarray | None:
    """The mask's alpha on the overlap grid, or None if it has too few points."""
    if len(mask.vertices) < min_points(mask.shape):
        return None
    return rasterise(mask.shape, mask.vertices, grid, grid, mask.feather * grid, mask.invert)


def overlapping_masks(config: LocalAdjustmentsConfig, index: int, grid: int = _OVERLAP_GRID) -> frozenset:
    """The indices of the masks whose tinted area meets mask `index`'s own.

    Tested on a coarse square grid in raw-normalised space, through the same rasteriser the
    render uses, so a gradient's whole side of the frame and an inverted mask's surround
    count as the area they really cover. Geometry moves every mask together, so an overlap
    here is an overlap on the canvas at any rotation, zoom or pan.
    """
    masks = config.masks
    if not 0 <= index < len(masks):
        return frozenset()
    target = _tinted_area(masks[index], grid)
    if target is None:
        return frozenset()
    hits = set()
    for i, mask in enumerate(masks):
        if i == index:
            continue
        other = _tinted_area(mask, grid)
        if other is not None and negpy_core.alphas_overlap(target, other, _OVERLAP_ALPHA):
            hits.add(i)
    return frozenset(hits)


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
    """Per-pixel dodge/burn maps [h, w, 3 + K] float32. Planes 0-2 sum the unlimited masks' value x feathered alpha:
    stops (positive = burn), grade delta in ISO-R points, flash fraction. Plane 3 + k is the alpha of the k-th of
    `limited_masks`, which cannot be pre-summed because its weight depends on tone. All zeros with no masks."""
    limited = limited_indices(config)
    maps = np.zeros((h, w, 3 + len(limited)), dtype=np.float32)
    if not config.masks:
        return maps

    short_side = float(min(h, w))
    specs = []
    for i, mask in enumerate(config.masks):
        if not mask.enabled or len(mask.vertices) < min_points(mask.shape):
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
        kind, outline, params = _mask_spec(mask.shape, transformed, h, w, mask.feather * short_side)
        plane = limited.index(i) if i in limited else -1
        specs.append((kind, outline, params, bool(mask.invert), plane, (float(mask.stops), float(mask.grade), float(mask.flash))))
    return negpy_core.local_maps((h, w, 3 + len(limited)), specs)


def limited_indices(config: LocalAdjustmentsConfig) -> List[int]:
    """Indices of the tone-limited masks that get a shape plane, in plane order."""
    limited = [i for i, m in enumerate(config.masks) if m.enabled and m.key != MaskKey.OFF and len(m.vertices) >= min_points(m.shape)]
    return limited[:MAX_KEYED_MASKS]


def limited_masks(config: LocalAdjustmentsConfig) -> List[LocalMask]:
    """The tone-limited masks in plane order (plane 3 + k of compute_local_maps)."""
    return [config.masks[i] for i in limited_indices(config)]
