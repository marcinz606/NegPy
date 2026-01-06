import numpy as np
import cv2
from typing import List, Tuple
from src.core.types import ImageBuffer
from src.features.retouch.models import LocalAdjustmentConfig
from src.core.validation import ensure_image


def get_luminance(img: ImageBuffer) -> ImageBuffer:
    # Rec.709 luma
    res = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return ensure_image(res)


def apply_dust_removal(
    img: ImageBuffer,
    dust_remove: bool,
    dust_threshold: float,
    dust_size: int,
    manual_spots: List[Tuple[float, float, float]],
    scale_factor: float,
) -> ImageBuffer:
    """
    Applies both automatic and manual dust removal (healing).
    """
    if not (dust_remove or manual_spots):
        return img

    # --- Automatic Detection & Healing ---
    if dust_remove:
        d_size = int(dust_size * 2.0 * scale_factor) | 1
        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        img_median_u8: np.ndarray = cv2.medianBlur(img_uint8, d_size)
        img_median = img_median_u8.astype(np.float32) / 255.0

        gray = get_luminance(img)
        blur_win = int(15 * scale_factor) | 1
        mean = cv2.blur(gray, (blur_win, blur_win))
        sq_mean = cv2.blur(gray**2, (blur_win, blur_win))
        std = np.sqrt(np.clip(sq_mean - mean**2, 0, None))
        flatness = np.clip(1.0 - (std / 0.08), 0, 1)
        flatness_weight = np.sqrt(flatness)
        brightness = np.clip(gray, 0, 1)
        highlight_sens = np.clip((brightness - 0.4) * 1.5, 0, 1)
        detail_boost = (1.0 - flatness) * 0.05
        sens_factor = (1.0 - 0.98 * flatness_weight) * (1.0 - 0.5 * highlight_sens)
        adaptive_thresh = dust_threshold * sens_factor + detail_boost

        diff = np.max(np.abs(img - img_median), axis=2)
        raw_mask = (diff > adaptive_thresh).astype(np.float32)
        exclusion_mask = (std > 0.2).astype(np.float32)
        raw_mask = raw_mask * (1.0 - exclusion_mask)

        if np.any(raw_mask > 0):
            m_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, m_kernel_close)
            m_kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, m_kernel_dilate, iterations=2)
            feather = d_size | 1
            mask = cv2.GaussianBlur(mask, (feather, feather), 0)
            img = img * (1.0 - mask[:, :, None]) + img_median * mask[:, :, None]

    # --- Manual Healing (using Inpainting) ---
    if manual_spots:
        h_img, w_img = img.shape[:2]

        manual_mask_u8 = np.zeros((h_img, w_img), dtype=np.uint8)
        for spot in manual_spots:
            nx, ny, s_size = spot
            radius = int(s_size * scale_factor)

            if radius < 1:
                radius = 1

            px = int(nx * w_img)
            py = int(ny * h_img)
            cv2.circle(manual_mask_u8, (px, py), radius, 255, -1)

        img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        inpaint_rad = int(3 * scale_factor) | 1
        img_inpainted_u8 = ensure_image(
            cv2.inpaint(img_u8, manual_mask_u8, inpaint_rad, cv2.INPAINT_TELEA)
        )

        # Grain Matching (Simple Noise addition to inpainted areas)
        noise_arr = np.random.normal(0, 3.5, img_inpainted_u8.shape)
        noise_f32 = noise_arr.astype(np.float32)
        lum_arr = get_luminance(
            ensure_image(img_inpainted_u8.astype(np.float32) / 255.0)
        )
        mod_arr = 5.0 * lum_arr * (1.0 - lum_arr)
        final_noise = noise_f32 * mod_arr[:, :, None]

        mask_base = manual_mask_u8.astype(np.float32) / 255.0
        mask_3d = mask_base[:, :, None]
        mask_blur = cv2.GaussianBlur(mask_3d, (3, 3), 0)
        if mask_blur.ndim == 2:
            mask_final = mask_blur[:, :, None]
        else:
            mask_final = mask_blur

        img_inpainted_f = img_inpainted_u8.astype(np.float32) + final_noise * mask_final
        img = ensure_image(np.clip(img_inpainted_f, 0, 255) / 255.0)

    return ensure_image(img)


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
            mask_blurred: np.ndarray = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            mask = mask_blurred

    return mask


def calculate_luma_mask(
    img: ImageBuffer, luma_range: Tuple[float, float], softness: float
) -> np.ndarray:
    """
    Calculates a mask based on image luminance levels.
    """
    lum = get_luminance(img)

    low, high = luma_range

    if softness <= 0:
        return ((lum >= low) & (lum <= high)).astype(np.float32)

    # Soft thresholds using smoothstep-like logic or simple linear ramps
    mask_low = np.clip((lum - (low - softness)) / (softness + 1e-6), 0, 1)
    # Upper bound ramp
    mask_high = np.clip(((high + softness) - lum) / (softness + 1e-6), 0, 1)

    return mask_low * mask_high


def apply_local_adjustments(
    img: ImageBuffer, adjustments: List[LocalAdjustmentConfig], scale_factor: float
) -> ImageBuffer:
    """
    Applies a list of dodge/burn adjustments to the image in linear space.
    """
    if not adjustments:
        return img

    h, w = img.shape[:2]

    for adj in adjustments:
        points = adj.points
        if not points:
            continue

        strength = adj.strength  # In EV stops
        radius = adj.radius
        feather = adj.feather
        luma_range = adj.luma_range
        luma_softness = adj.luma_softness

        # Spatial Mask
        mask = generate_local_mask(h, w, points, radius, feather, scale_factor)

        # Tonal Mask
        luma_mask = calculate_luma_mask(img, luma_range, luma_softness)

        # Combined Mask
        final_mask = mask * luma_mask

        # Exposure math: New_Val = Old_Val * 2^(mask * EV)
        exposure_mult = 2.0 ** (final_mask[:, :, None] * strength)
        img *= exposure_mult

    return ensure_image(np.clip(img, 0, 1))
