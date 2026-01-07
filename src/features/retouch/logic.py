import numpy as np
import cv2
from numba import njit, prange  # type: ignore
from typing import List, Tuple, Optional
from src.core.types import ImageBuffer
from src.features.retouch.models import LocalAdjustmentConfig
from src.core.validation import ensure_image
from src.core.performance import time_function


@njit(parallel=True)
def _calculate_luma_mask_jit(
    lum: np.ndarray, low: float, high: float, softness: float
) -> np.ndarray:
    """
    Fast JIT calculation of luma mask with softness.
    """
    h, w = lum.shape
    res = np.empty((h, w), dtype=np.float32)
    soft_eps = softness + 1e-6

    for y in prange(h):
        for x in range(w):
            val = lum[y, x]
            # mask_low
            m_low = (val - (low - softness)) / soft_eps
            if m_low < 0.0:
                m_low = 0.0
            elif m_low > 1.0:
                m_low = 1.0

            # mask_high
            m_high = ((high + softness) - val) / soft_eps
            if m_high < 0.0:
                m_high = 0.0
            elif m_high > 1.0:
                m_high = 1.0

            res[y, x] = m_low * m_high
    return res


@njit(parallel=True)
def _apply_local_exposure_kernel(
    img: np.ndarray, mask: np.ndarray, strength: float
) -> None:
    """
    Fast JIT application of exposure multipliers.
    """
    h, w, c = img.shape
    ln2 = 0.69314718056  # litral hardcoded for jit
    for y in prange(h):
        for x in range(w):
            m_val = mask[y, x]
            if m_val > 0.0:
                mult = np.exp(m_val * strength * ln2)
                for ch in range(3):
                    img[y, x, ch] *= mult


@njit(parallel=True)
def _compute_dust_masks_jit(
    img: np.ndarray,
    img_median: np.ndarray,
    std: np.ndarray,
    sens_factor: np.ndarray,
    detail_boost: np.ndarray,
    dust_threshold: float,
) -> np.ndarray:
    """
    Fuses the dust detection logic.
    """
    h, w, c = img.shape
    raw_mask = np.empty((h, w), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            max_diff = 0.0
            for ch in range(3):
                d = abs(img[y, x, ch] - img_median[y, x, ch])
                if d > max_diff:
                    max_diff = d

            thresh = dust_threshold * sens_factor[y, x] + detail_boost[y, x]
            if max_diff > thresh and std[y, x] <= 0.2:
                raw_mask[y, x] = 1.0
            else:
                raw_mask[y, x] = 0.0
    return raw_mask


@njit(parallel=True)
def _get_luminance_jit(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT luminance calculation.
    """
    h, w, _ = img.shape
    res = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            res[y, x] = (
                0.2126 * img[y, x, 0] + 0.7152 * img[y, x, 1] + 0.0722 * img[y, x, 2]
            )
    return res


def get_luminance(img: ImageBuffer) -> ImageBuffer:
    """
    Calculates relative luminance using Rec. 709 coefficients.
    """
    return ensure_image(_get_luminance_jit(img.astype(np.float32)))


@njit(parallel=True)
def _apply_inpainting_grain_jit(
    img: np.ndarray,
    img_inpainted: np.ndarray,
    mask_final: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """
    Fuses inpainting blending and grain matching.
    """
    h, w, c = img_inpainted.shape
    res = np.empty_like(img_inpainted)

    for y in prange(h):
        for x in range(w):
            # Fused Luminance for noise modulation
            lum = (
                0.2126 * img_inpainted[y, x, 0]
                + 0.7152 * img_inpainted[y, x, 1]
                + 0.0722 * img_inpainted[y, x, 2]
            ) / 255.0

            mod = 5.0 * lum * (1.0 - lum)
            m = mask_final[y, x, 0]

            for ch in range(3):
                # Inpaint + Noise
                val = img_inpainted[y, x, ch] + noise[y, x, ch] * mod * m
                # Blend with original
                res[y, x, ch] = img[y, x, ch] * (1.0 - m) + (val / 255.0) * m

    return res


@time_function
def apply_dust_removal(
    img: ImageBuffer,
    dust_remove: bool,
    dust_threshold: float,
    dust_size: int,
    manual_spots: List[Tuple[float, float, float]],
    scale_factor: float,
) -> ImageBuffer:
    """
    Applies both automatic and manual dust removal (healing/inpainting).

    Automatic detection identifies small, high-contrast irregularities (dust/scratches)
    on the negative and heals them using median statistics. Manual healing uses
    Telea inpainting with fused grain matching to preserve the photographic texture.
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

        # Use JIT for channel diff and thresholding
        raw_mask = _compute_dust_masks_jit(
            img.astype(np.float32),
            img_median.astype(np.float32),
            std.astype(np.float32),
            sens_factor.astype(np.float32),
            detail_boost.astype(np.float32),
            float(dust_threshold),
        )

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

        # Grain Matching & Blending using fused JIT kernel
        noise_arr = np.random.normal(0, 3.5, img_inpainted_u8.shape).astype(np.float32)

        mask_base = manual_mask_u8.astype(np.float32) / 255.0
        mask_3d = mask_base[:, :, None]
        mask_blur = cv2.GaussianBlur(mask_3d, (3, 3), 0)
        mask_final = (
            mask_blur[:, :, None] if mask_blur.ndim == 2 else mask_blur
        ).astype(np.float32)

        img = ensure_image(
            _apply_inpainting_grain_jit(
                img.astype(np.float32),
                img_inpainted_u8.astype(np.float32),
                mask_final,
                noise_arr,
            )
        )

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


@time_function
def calculate_luma_mask(
    img: ImageBuffer,
    luma_range: Tuple[float, float],
    softness: float,
    lum: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculates a luminosity mask based on image luminance levels.

    Luminosity masking (tonal masking) allows for selective adjustments to specific
    zones (e.g., highlights only, deep shadows only) with controllable softness.
    """
    if lum is None:
        lum = get_luminance(img)

    low, high = luma_range

    if softness <= 0:
        return ((lum >= low) & (lum <= high)).astype(np.float32)

    return ensure_image(
        _calculate_luma_mask_jit(
            lum.astype(np.float32), float(low), float(high), float(softness)
        )
    )


@time_function
def apply_local_adjustments(
    img: ImageBuffer, adjustments: List[LocalAdjustmentConfig], scale_factor: float
) -> ImageBuffer:
    """
    Applies a list of 'Dodge and Burn' adjustments to the mask.
    """
    if not adjustments:
        return img

    h, w = img.shape[:2]

    # Calculate base luminance once for all adjustments in this pass
    # to ensure consistency and prevent "drifting" masks.
    base_lum = get_luminance(img)

    for adj in adjustments:
        points = adj.points
        if not points:
            continue

        # Spatial Mask
        mask = generate_local_mask(h, w, points, adj.radius, adj.feather, scale_factor)

        # Tonal Mask (using static base_lum)
        luma_mask = calculate_luma_mask(
            img, adj.luma_range, adj.luma_softness, lum=base_lum
        )

        # Combined Mask
        final_mask = mask * luma_mask

        # Apply using JIT
        _apply_local_exposure_kernel(img, final_mask, float(adj.strength))

    return ensure_image(np.clip(img, 0, 1))
