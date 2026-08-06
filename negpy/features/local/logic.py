from typing import List, Tuple

import cv2
import numpy as np

from negpy.features.local.models import LocalAdjustmentsConfig
from negpy.features.geometry.logic import map_coords_to_geometry, smooth_polyline


def _rasterise_mask(
    vertices_img: List[Tuple[float, float]],
    h: int,
    w: int,
    feather_sigma: float,
) -> np.ndarray:
    """
    Rasterise a polygon (in image-pixel coords) to a float32 mask [h, w].
    Feather is a Gaussian sigma in pixels applied to the hard binary fill.
    """
    pts = np.array([[v[0] * w, v[1] * h] for v in vertices_img], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    mask_f = mask.astype(np.float32) / 255.0
    if feather_sigma > 1e-3:
        k = int(feather_sigma * 3) | 1  # odd kernel covering ~3 sigma
        mask_f = cv2.GaussianBlur(mask_f, (k, k), feather_sigma)
    return mask_f


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
        if len(mask.vertices) < 3:
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
            )
            for rx, ry in mask.vertices
        ]

        sigma_px = mask.feather * short_side
        alpha = _rasterise_mask(smooth_polyline(transformed, closed=True), h, w, sigma_px)
        maps[:, :, 0] += mask.stops * alpha
        if mask.grade:
            maps[:, :, 1] += mask.grade * alpha

    return maps
