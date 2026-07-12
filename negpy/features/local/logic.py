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


def compute_local_ev_map(
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
    Build the per-pixel dodge/burn EV map [h, w] float32: ev = sum over masks
    of strength * alpha, where alpha is the feathered polygon mask. Positive =
    dodge, negative = burn. All-zeros when there are no masks.
    """
    ev = np.zeros((h, w), dtype=np.float32)
    if not config.masks:
        return ev

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
        ev += mask.strength * alpha

    return ev
