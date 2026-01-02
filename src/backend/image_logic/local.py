import numpy as np
import cv2
from typing import List, Tuple, cast
from src.domain_objects import LocalAdjustment
from src.helpers import get_luminance


def generate_local_mask(
    h: int,
    w: int,
    points: List[Tuple[float, float]],
    radius: float,
    feather: float,
    scale_factor: float,
) -> np.ndarray:
    """
    Generates a grayscale mask from a series of normalized points.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    if not points:
        return mask

    # Calculate pixel radius
    px_radius = int(radius * scale_factor)
    if px_radius < 1:
        px_radius = 1

    # Draw strokes
    for i in range(len(points)):
        p1 = (int(points[i][0] * w), int(points[i][1] * h))
        if i > 0:
            p0 = (int(points[i - 1][0] * w), int(points[i - 1][1] * h))
            cv2.line(mask, p0, p1, 1.0, px_radius * 2)
        cv2.circle(mask, p1, px_radius, 1.0, -1)

    # Apply feathering (Blur)
    if feather > 0:
        # Blur size based on radius and feather intensity
        blur_size = int(px_radius * 2 * feather) | 1
        if blur_size >= 3:
            mask = cast(np.ndarray, cv2.GaussianBlur(mask, (blur_size, blur_size), 0))

    return mask


def calculate_luma_mask(
    img: np.ndarray, luma_range: Tuple[float, float], softness: float
) -> np.ndarray:
    """
    Calculates a mask based on image luminance levels.
    """
    lum = get_luminance(img)

    low, high = luma_range

    if softness <= 0:
        return ((lum >= low) & (lum <= high)).astype(np.float32)

    # Soft thresholds using smoothstep-like logic or simple linear ramps
    # Lower bound ramp
    mask_low = np.clip((lum - (low - softness)) / (softness + 1e-6), 0, 1)
    # Upper bound ramp
    mask_high = np.clip(((high + softness) - lum) / (softness + 1e-6), 0, 1)

    return mask_low * mask_high


def apply_local_adjustments(
    img: np.ndarray, adjustments: List[LocalAdjustment], scale_factor: float
) -> np.ndarray:
    """
    Applies a list of dodge/burn adjustments to the image in linear space.
    """
    if not adjustments:
        return img

    h, w = img.shape[:2]

    for adj in adjustments:
        points = adj.get("points", [])
        if not points:
            continue

        strength = adj.get("strength", 0.0)  # In EV stops
        radius = adj.get("radius", 50)
        feather = adj.get("feather", 0.5)
        luma_range = adj.get("luma_range", (0.0, 1.0))
        luma_softness = adj.get("luma_softness", 0.2)

        # Spatial Mask
        mask = generate_local_mask(h, w, points, radius, feather, scale_factor)

        # Tonal Mask
        luma_mask = calculate_luma_mask(img, luma_range, luma_softness)

        # Combined Mask
        final_mask = mask * luma_mask

        # Exposure math: New_Val = Old_Val * 2^(mask * EV)
        exposure_mult = 2.0 ** (final_mask[:, :, None] * strength)
        img *= exposure_mult

    return np.clip(img, 0, 1)
