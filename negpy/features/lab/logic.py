from typing import List, Optional

import cv2
import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working, working_oetf_encode
from negpy.kernel.image.validation import ensure_image


def apply_spectral_crosstalk(img_dens: ImageBuffer, strength: float, matrix: Optional[List[float]]) -> ImageBuffer:
    """
    Mixes channels using calibration matrix.
    """
    if strength == 0.0 or matrix is None:
        return img_dens

    cal_matrix = np.array(matrix).reshape(3, 3)
    identity = np.eye(3)

    applied_matrix = identity * (1.0 - strength) + cal_matrix * strength

    row_sums = np.sum(applied_matrix, axis=1, keepdims=True)
    applied_matrix = applied_matrix / np.maximum(row_sums, 1e-6)

    res = np.einsum("hwc,kc->hwk", img_dens.astype(np.float32, copy=False), applied_matrix.astype(np.float32))

    return ensure_image(res)


def apply_clahe(img: ImageBuffer, strength: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    L-channel Contrast Limited Adaptive Histogram Equalization.
    """
    if strength <= 0:
        return img

    lab = rgb_to_lab_working(img)
    l_chan, a, b = cv2.split(lab)

    l_u16 = (l_chan * (65535.0 / 100.0)).astype(np.uint16)

    clip_limit = strength * 2.5
    grid_dim = max(2, int(8 * scale_factor))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_dim, grid_dim))
    l_enhanced_u16 = clahe.apply(l_u16)

    l_enhanced = l_enhanced_u16.astype(np.float32) * (100.0 / 65535.0)

    l_final = l_chan * (1.0 - strength) + l_enhanced * strength

    lab_enhanced = cv2.merge([l_final, a, b])
    res = lab_to_rgb_working(lab_enhanced)

    return ensure_image(np.clip(res, 0.0, 1.0))


@njit(cache=True, fastmath=True)
def _apply_unsharp_mask_jit(l_chan: np.ndarray, l_blur: np.ndarray, amount: float, threshold: float) -> np.ndarray:
    """
    USM Kernel (Orig + (Orig - Blur) * Amount).
    """
    h, w = l_chan.shape
    res = np.empty((h, w), dtype=np.float32)
    amount_f = amount * 2.5

    for y in range(h):
        for x in range(w):
            orig = l_chan[y, x]
            blur = l_blur[y, x]
            diff = orig - blur
            if abs(diff) > threshold:
                val = orig + diff * amount_f
                if val < 0.0:
                    val = 0.0
                elif val > 100.0:
                    val = 100.0
                res[y, x] = val
            else:
                res[y, x] = orig
    return res


def apply_output_sharpening(img: ImageBuffer, amount: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    LAB Lightness sharpening.
    """
    if amount <= 0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)

    k_size = max(3, int(5 * scale_factor) | 1)
    sigma = 1.0 * scale_factor
    l_blur = cv2.GaussianBlur(l_chan, (k_size, k_size), sigma)

    l_sharpened = _apply_unsharp_mask_jit(
        np.ascontiguousarray(l_chan),
        np.ascontiguousarray(l_blur),
        float(amount),
        2.0,
    )

    res_lab = cv2.merge([l_sharpened, a, b])
    res_rgb = lab_to_rgb_working(res_lab)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


def apply_saturation(img: ImageBuffer, saturation: float) -> ImageBuffer:
    """
    Adjusts saturation by scaling chroma (a*, b*) in CIELAB.
    Preserves perceived lightness, unlike HSV S-scaling which darkens
    already-saturated colors when S clips to 1.0.
    """
    if saturation == 1.0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)
    a_new = a * saturation
    b_new = b * saturation
    res_lab = cv2.merge([l_chan, a_new, b_new])
    res_rgb = lab_to_rgb_working(res_lab)
    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


def apply_chroma_denoise(img: ImageBuffer, radius: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    Smooths A and B channels in LAB space to reduce color noise.
    """
    if radius <= 0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)

    k_radius = radius * scale_factor
    k_size = max(3, int(k_radius * 2 + 1) | 1)
    sigma = k_radius

    a_blur = cv2.GaussianBlur(a, (k_size, k_size), sigma)
    b_blur = cv2.GaussianBlur(b, (k_size, k_size), sigma)

    res_lab = cv2.merge([l_chan, a_blur, b_blur])
    res_rgb = lab_to_rgb_working(res_lab)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


def apply_glow_and_halation(
    img: ImageBuffer,
    glow_amount: float,
    halation_strength: float,
    scale_factor: float = 1.0,
) -> ImageBuffer:
    """
    Glow: all-channel Gaussian bloom of highlights (lens diffusion).
    Halation: red-dominant scatter of highlights (film base reflection).
    """
    if glow_amount == 0.0 and halation_strength == 0.0:
        return img

    # Highlight mask in the display domain (keeps the 0.5 threshold); bloom is linear.
    enc = working_oetf_encode(img)
    luma = enc[:, :, 0] * 0.2126 + enc[:, :, 1] * 0.7152 + enc[:, :, 2] * 0.0722
    threshold = 0.5
    highlight_mask = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0) ** 2

    result = img.copy().astype(np.float32)

    if glow_amount > 0.0:
        base_r = max(3, int(15 * scale_factor))
        k = min((base_r * 2 + 1) | 1, 201)
        sigma = base_r * 0.5
        highlights = (img * highlight_mask[:, :, np.newaxis]).astype(np.float32)
        glow_blur = cv2.GaussianBlur(highlights, (k, k), sigma)
        scaled = glow_blur * glow_amount
        result = 1.0 - (1.0 - result) * (1.0 - scaled)

    if halation_strength > 0.0:
        base_r = max(5, int(25 * scale_factor))
        k = min((base_r * 2 + 1) | 1, 301)
        sigma = base_r * 0.5
        red_hl = np.zeros_like(img, dtype=np.float32)
        red_hl[:, :, 0] = img[:, :, 0] * highlight_mask
        red_hl[:, :, 1] = img[:, :, 0] * highlight_mask * 0.3
        red_hl[:, :, 2] = img[:, :, 0] * highlight_mask * 0.05
        hal_blur = cv2.GaussianBlur(red_hl, (k, k), sigma)
        scaled = hal_blur * halation_strength
        result = 1.0 - (1.0 - result) * (1.0 - scaled)

    return ensure_image(np.clip(result, 0.0, 1.0))


def apply_vibrance(img: ImageBuffer, strength: float) -> ImageBuffer:
    """
    Selectively boosts saturation of muted colors in LAB space.
    """
    if strength == 1.0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)

    chroma = np.sqrt(a**2 + b**2)
    muted_mask = np.clip(1.0 - (chroma / 60.0), 0.0, 1.0)

    boost = (strength - 1.0) * muted_mask
    a_new = a * (1.0 + boost)
    b_new = b * (1.0 + boost)

    res_lab = cv2.merge([l_chan, a_new, b_new])
    res_rgb = lab_to_rgb_working(res_lab)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))
