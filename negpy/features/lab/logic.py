import numpy as np
import cv2
from numba import njit, prange  # type: ignore
from typing import List, Optional
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image


@njit(parallel=True, cache=True, fastmath=True)
def _apply_spectral_crosstalk_jit(img_dens: np.ndarray, applied_matrix: np.ndarray) -> np.ndarray:
    """
    3x3 Matrix multiplication.
    """
    h, w, c = img_dens.shape
    res = np.empty_like(img_dens)
    for y in prange(h):
        for x in range(w):
            r = img_dens[y, x, 0]
            g = img_dens[y, x, 1]
            b = img_dens[y, x, 2]
            res[y, x, 0] = r * applied_matrix[0, 0] + g * applied_matrix[0, 1] + b * applied_matrix[0, 2]
            res[y, x, 1] = r * applied_matrix[1, 0] + g * applied_matrix[1, 1] + b * applied_matrix[1, 2]
            res[y, x, 2] = r * applied_matrix[2, 0] + g * applied_matrix[2, 1] + b * applied_matrix[2, 2]
    return res


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

    res = _apply_spectral_crosstalk_jit(
        np.ascontiguousarray(img_dens.astype(np.float32)),
        np.ascontiguousarray(applied_matrix.astype(np.float32)),
    )

    return ensure_image(res)


def apply_clahe(img: ImageBuffer, strength: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    L-channel Contrast Limited Adaptive Histogram Equalization.
    """
    if strength <= 0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)

    l_u16 = (l_chan * (65535.0 / 100.0)).astype(np.uint16)

    clip_limit = strength * 2.5
    grid_dim = max(2, int(8 * scale_factor))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_dim, grid_dim))
    l_enhanced_u16 = clahe.apply(l_u16)

    l_enhanced = l_enhanced_u16.astype(np.float32) * (100.0 / 65535.0)

    l_final = l_chan * (1.0 - strength) + l_enhanced * strength

    lab_enhanced = cv2.merge([l_final, a, b])
    res = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return ensure_image(np.clip(res, 0.0, 1.0))


@njit(parallel=True, cache=True, fastmath=True)
def _apply_unsharp_mask_jit(l_chan: np.ndarray, l_blur: np.ndarray, amount: float, threshold: float) -> np.ndarray:
    """
    USM Kernel (Orig + (Orig - Blur) * Amount).
    """
    h, w = l_chan.shape
    res = np.empty((h, w), dtype=np.float32)
    amount_f = amount * 2.5

    for y in prange(h):
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

    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
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
    res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


def apply_saturation(img: ImageBuffer, saturation: float) -> ImageBuffer:
    """
    Adjusts saturation via HSV space.
    """
    if saturation == 1.0:
        return img

    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)

    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return ensure_image(np.clip(res, 0.0, 1.0))


def apply_chroma_denoise(img: ImageBuffer, radius: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    Smooths A and B channels in LAB space to reduce color noise.
    """
    if radius <= 0:
        return img

    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)

    k_radius = radius * scale_factor
    k_size = max(3, int(k_radius * 2 + 1) | 1)
    sigma = k_radius

    a_blur = cv2.GaussianBlur(a, (k_size, k_size), sigma)
    b_blur = cv2.GaussianBlur(b, (k_size, k_size), sigma)

    res_lab = cv2.merge([l_chan, a_blur, b_blur])
    res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


def apply_vibrance(img: ImageBuffer, strength: float) -> ImageBuffer:
    """
    Selectively boosts saturation of muted colors in LAB space.
    """
    if strength == 1.0:
        return img

    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)

    chroma = np.sqrt(a**2 + b**2)
    muted_mask = np.clip(1.0 - (chroma / 60.0), 0.0, 1.0)

    boost = (strength - 1.0) * muted_mask
    a_new = a * (1.0 + boost)
    b_new = b * (1.0 + boost)

    res_lab = cv2.merge([l_chan, a_new, b_new])
    res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))
