from typing import List, Tuple

import cv2
import numpy as np

from negpy.features.local.models import LocalAdjustmentsConfig
from negpy.features.geometry.logic import map_coords_to_geometry


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


def compute_local_factor_map(
    config: LocalAdjustmentsConfig,
    h: int,
    w: int,
    orig_shape: Tuple[int, int],
    rotation: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    """
    Build the per-pixel multiplicative dodge/burn factor map [h, w] float32.

    factor = prod over masks of 2^(strength * alpha), where alpha is the
    feathered polygon mask. All-ones when there are no masks. The image is
    adjusted by multiplying each channel by this map.
    """
    factor = np.ones((h, w), dtype=np.float32)
    if not config.masks:
        return factor

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
            )
            for rx, ry in mask.vertices
        ]

        sigma_px = mask.feather * short_side
        alpha = _rasterise_mask(transformed, h, w, sigma_px)
        factor *= np.power(2.0, mask.strength * alpha, dtype=np.float32)

    return factor


def apply_local_adjustments(
    img: np.ndarray,
    config: LocalAdjustmentsConfig,
    orig_shape: Tuple[int, int],
    rotation: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    """
    Apply polygon dodge/burn masks to a linear float32 RGB image [H, W, 3].

    Vertices are in raw-image normalised space; they are mapped into the
    current image's geometry-transformed space before rasterisation.
    Positive strength dodges (brightens), negative burns (darkens).
    Returns the adjusted image clipped to [0, 1].
    """
    if not config.masks:
        return img

    h, w = img.shape[:2]
    factor = compute_local_factor_map(config, h, w, orig_shape, rotation, fine_rotation, flip_horizontal, flip_vertical)
    result = img.astype(np.float32, copy=True) * factor[..., np.newaxis]
    return np.clip(result, 0.0, 1.0)
