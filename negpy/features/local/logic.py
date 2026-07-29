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


def polygon_label_anchor(pts: List[Tuple[float, float]], probes: int = 12) -> Tuple[float, float]:
    """A point inside `pts` to hang a label on: the vertex centroid, snapped to somewhere the
    polygon actually contains.

    A concave mask's centroid can fall outside it — the same trap zone_region_labels solves
    for the zone overlay by snapping to a cell the region owns. Falls back to the centroid
    when the polygon is degenerate and nothing on the lattice is inside.
    """
    if not pts:
        return (0.0, 0.0)
    arr = np.asarray(pts, dtype=np.float32)
    cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    if len(pts) < 3:
        return (cx, cy)

    contour = arr.reshape(-1, 1, 2)
    if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
        return (cx, cy)

    x0, x1 = float(arr[:, 0].min()), float(arr[:, 0].max())
    y0, y1 = float(arr[:, 1].min()), float(arr[:, 1].max())
    best, best_d = None, float("inf")
    for i in range(probes):
        px = x0 + (x1 - x0) * (i + 0.5) / probes
        for j in range(probes):
            py = y0 + (y1 - y0) * (j + 0.5) / probes
            if cv2.pointPolygonTest(contour, (px, py), False) < 0:
                continue
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d < best_d:
                best, best_d = (px, py), d
    return best if best is not None else (cx, cy)


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
