import numpy as np
import cv2
from numba import njit, prange  # type: ignore
from PIL import Image, ImageFilter
from typing import List, Optional
from src.core.types import ImageBuffer
from src.core.validation import ensure_image
from src.core.performance import time_function


@njit(parallel=True)
def _apply_spectral_crosstalk_jit(
    img_dens: np.ndarray, applied_matrix: np.ndarray
) -> np.ndarray:
    """
    Fast JIT application of 3x3 crosstalk matrix in density space.
    """
    h, w, c = img_dens.shape
    res = np.empty_like(img_dens)
    for y in prange(h):
        for x in range(w):
            r = img_dens[y, x, 0]
            g = img_dens[y, x, 1]
            b = img_dens[y, x, 2]
            res[y, x, 0] = (
                r * applied_matrix[0, 0]
                + g * applied_matrix[0, 1]
                + b * applied_matrix[0, 2]
            )
            res[y, x, 1] = (
                r * applied_matrix[1, 0]
                + g * applied_matrix[1, 1]
                + b * applied_matrix[1, 2]
            )
            res[y, x, 2] = (
                r * applied_matrix[2, 0]
                + g * applied_matrix[2, 1]
                + b * applied_matrix[2, 2]
            )
    return res


@njit(parallel=True)
def _apply_chroma_masking_jit(
    a_chan: np.ndarray,
    b_chan: np.ndarray,
    a_bilat: np.ndarray,
    b_bilat: np.ndarray,
    a_blur: np.ndarray,
    b_blur: np.ndarray,
    l_chan: np.ndarray,
    broad_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuses multiple masking steps for chroma noise removal.
    """
    h, w = a_chan.shape
    res_a = np.empty((h, w), dtype=np.float32)
    res_b = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            l_val = l_chan[y, x]
            deep_m = 1.0 - (l_val / 60.0)
            if deep_m < 0:
                deep_m = 0.0
            elif deep_m > 1:
                deep_m = 1.0
            deep_m = deep_m * deep_m
            a_comb = a_bilat[y, x] * (1.0 - deep_m) + a_blur[y, x] * deep_m
            b_comb = b_bilat[y, x] * (1.0 - deep_m) + b_blur[y, x] * deep_m
            bm = broad_mask[y, x]
            final_a = a_chan[y, x] * (1.0 - bm) + a_comb * bm
            final_b = b_chan[y, x] * (1.0 - bm) + b_comb * bm
            if final_a < 0:
                final_a = 0
            elif final_a > 255:
                final_a = 255
            if final_b < 0:
                final_b = 0
            elif final_b > 255:
                final_b = 255
            res_a[y, x] = final_a
            res_b[y, x] = final_b
    return res_a, res_b


@time_function
def apply_spectral_crosstalk(
    img_dens: ImageBuffer, strength: float, matrix: Optional[List[float]]
) -> ImageBuffer:
    """
    Applies a color crosstalk matrix to an RGB image in density space.
    Input 'img_dens' is expected to be in Density space.
    """
    if strength == 0.0 or matrix is None:
        return img_dens

    cal_matrix = np.array(matrix).reshape(3, 3)
    identity = np.eye(3)

    # Interpolate between identity and calibration matrix
    applied_matrix = identity * (1.0 - strength) + cal_matrix * strength

    # Row-normalization: ensures that neutral grey density is preserved
    row_sums = np.sum(applied_matrix, axis=1, keepdims=True)
    applied_matrix = applied_matrix / np.maximum(row_sums, 1e-6)

    # Use JIT for matrix multiplication
    res = _apply_spectral_crosstalk_jit(
        img_dens.astype(np.float32), applied_matrix.astype(np.float32)
    )

    return ensure_image(res)


@time_function
def apply_hypertone(img: ImageBuffer, strength: float) -> ImageBuffer:
    """
    Applies local contrast enhancement using CLAHE in LAB space.
    """
    if strength <= 0:
        return img

    # RGB to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)

    # CLAHE on uint16 for precision
    l_u16 = (l_chan * (65535.0 / 100.0)).astype(np.uint16)

    clip_limit = strength * 5.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced_u16 = clahe.apply(l_u16)

    l_enhanced = l_enhanced_u16.astype(np.float32) * (100.0 / 65535.0)

    # Blend original and enhanced
    l_final = l_chan * (1.0 - strength) + l_enhanced * strength

    lab_enhanced = cv2.merge([l_final, a, b])
    res = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return ensure_image(np.clip(res, 0.0, 1.0))


@time_function
def apply_chroma_noise_removal(
    img: ImageBuffer, strength_input: float, scale_factor: float = 1.0
) -> ImageBuffer:
    """
    Reduces color noise in deep shadows.
    """
    if strength_input <= 0:
        return img

    strength = strength_input * 100.0
    img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 1. Bilateral Filter
    d_val = int(9 * scale_factor) | 1
    f_strength = float(strength)
    sc_val = f_strength * 2.0
    ss_val = f_strength * 0.75 * scale_factor

    a_bilat = cv2.bilateralFilter(a_chan, d_val, sc_val, ss_val)
    b_bilat = cv2.bilateralFilter(b_chan, d_val, sc_val, ss_val)

    # 2. Strong Blur for Deep Shadows
    base_k = 11 if strength > 50 else 7
    k_size = int(base_k * scale_factor) | 1
    a_blur = cv2.GaussianBlur(a_bilat, (k_size, k_size), 0)
    b_blur = cv2.GaussianBlur(b_bilat, (k_size, k_size), 0)

    l_float = l_chan.astype(np.float32)

    # 3. Broad Masking
    broad_mask = np.clip(1.0 - ((l_float - 150.0) / 80.0), 0.0, 1.0)
    broad_mask = cv2.GaussianBlur(broad_mask, (21, 21), 0)

    # Use JIT for all masking and combinations
    a_final_f32, b_final_f32 = _apply_chroma_masking_jit(
        a_chan.astype(np.float32),
        b_chan.astype(np.float32),
        a_bilat.astype(np.float32),
        b_bilat.astype(np.float32),
        a_blur.astype(np.float32),
        b_blur.astype(np.float32),
        l_float,
        broad_mask.astype(np.float32),
    )

    a_final = a_final_f32.astype(np.uint8)
    b_final = b_final_f32.astype(np.uint8)

    res = cv2.cvtColor(cv2.merge([l_chan, a_final, b_final]), cv2.COLOR_LAB2RGB)
    return ensure_image(res.astype(np.float32) / 255.0)


@time_function
def apply_output_sharpening(img: ImageBuffer, amount: float) -> ImageBuffer:
    """
    Applies Unsharp Mask sharpening to the Lightness channel.
    """
    if amount <= 0:
        return img

    img_u8 = (img * 255).astype(np.uint8)
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(img_lab)

    l_pil = Image.fromarray(l_chan)
    l_sharpened = l_pil.filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=int(amount * 250), threshold=5)
    )

    img_lab_sharpened = cv2.merge([np.array(l_sharpened), a, b])
    result_rgb = cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB)

    return ensure_image(result_rgb.astype(np.float32) / 255.0)
