import numpy as np
import cv2
from PIL import Image, ImageFilter
from src.helpers import ensure_array
from typing import List, Any


def apply_spectral_crosstalk(
    img: np.ndarray, strength: float, matrix: List[float]
) -> np.ndarray:
    """
    Applies a color crosstalk matrix to an RGB image in density space.
    Math: Density_Corrected = Density_Measured @ (Identity * (1 - strength) + Matrix * strength)
    """
    if strength == 0.0:
        return img

    cal_matrix = np.array(matrix).reshape(3, 3)
    identity = np.eye(3)

    # Interpolate between identity and calibration matrix
    applied_matrix = identity * (1.0 - strength) + cal_matrix * strength

    # Row-normalization: ensures that neutral grey density is preserved
    row_sums = np.sum(applied_matrix, axis=1, keepdims=True)
    applied_matrix = applied_matrix / np.maximum(row_sums, 1e-6)

    # Reshape image for matrix multiplication
    orig_shape = img.shape
    img_flat = img.reshape(-1, 3)

    # Apply crosstalk matrix
    res_flat = img_flat @ applied_matrix.T

    return np.array(res_flat.reshape(orig_shape))


def apply_hypertone(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    L channel of the image in LAB color space to enhance local contrast.
    """
    if strength <= 0:
        return img

    # Ensure input is float32 for OpenCV
    img_f32 = img.astype(np.float32)

    # RGB to LAB
    lab = cv2.cvtColor(img_f32, cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)

    # CLAHE works on uint8 or uint16. We'll use uint16 for better precision.
    l_u16 = (l_chan * (65535.0 / 100.0)).astype(np.uint16)

    clip_limit = strength * 5.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced_u16 = clahe.apply(l_u16)

    # Back to float32 [0, 100]
    l_enhanced = l_enhanced_u16.astype(np.float32) * (100.0 / 65535.0)

    # Make the effect granular by interpolating between original and enhanced L
    l_final = l_chan * (1.0 - strength) + l_enhanced * strength

    # Merge and back to RGB
    lab_enhanced = cv2.merge([l_final, a, b])
    res = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return np.clip(res, 0.0, 1.0)


def apply_chroma_noise_removal(
    img: np.ndarray, settings: Any, scale_factor: float = 1.0
) -> np.ndarray:
    """
    Reduces color noise using bilateral filtering and specific deep-shadow blurring in LAB space.
    Targets shadows aggressively where film scan color noise is most prevalent.
    """
    strength_input = getattr(settings, "c_noise_strength", 0.25)
    if not isinstance(strength_input, (int, float)) or strength_input <= 0:
        return img

    # Scale 0.0-1.0 to 0-100
    strength = float(strength_input) * 100.0

    img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
    lab = ensure_array(cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB))
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 1. Bilateral Filter (Base Denoising)
    d_val = int(9 * scale_factor) | 1
    f_strength = float(strength)
    sc_val = f_strength * 2.0
    ss_val = f_strength * 0.75 * scale_factor

    a_bilat = ensure_array(cv2.bilateralFilter(a_chan, d_val, sc_val, ss_val))
    b_bilat = ensure_array(cv2.bilateralFilter(b_chan, d_val, sc_val, ss_val))

    # 2. Strong Blur for Deep Shadows
    base_k = 11 if strength > 50 else 7
    k_size = int(base_k * scale_factor) | 1
    a_blur = ensure_array(cv2.GaussianBlur(a_bilat, (k_size, k_size), 0))
    b_blur = ensure_array(cv2.GaussianBlur(b_bilat, (k_size, k_size), 0))

    l_float = ensure_array(l_chan).astype(np.float32)

    # Deep Shadow Mask: 1.0 at L=0, fades to 0.0 at L=60
    deep_mask = np.clip(1.0 - (l_float / 60.0), 0.0, 1.0)
    deep_mask = deep_mask * deep_mask

    # Blend Blur into Bilateral
    a_combined = (
        a_bilat.astype(np.float32) * (1.0 - deep_mask)
        + a_blur.astype(np.float32) * deep_mask
    )
    b_combined = (
        b_bilat.astype(np.float32) * (1.0 - deep_mask)
        + b_blur.astype(np.float32) * deep_mask
    )

    # 3. Broad Masking
    broad_mask = np.clip(1.0 - ((l_float - 150.0) / 80.0), 0.0, 1.0)
    broad_mask = ensure_array(cv2.GaussianBlur(broad_mask, (21, 21), 0))

    # Final Blend
    a_final = np.clip(
        a_chan.astype(np.float32) * (1.0 - broad_mask) + a_combined * broad_mask, 0, 255
    ).astype(np.uint8)
    b_final = np.clip(
        b_chan.astype(np.float32) * (1.0 - broad_mask) + b_combined * broad_mask, 0, 255
    ).astype(np.uint8)

    img = (
        ensure_array(
            cv2.cvtColor(cv2.merge([l_chan, a_final, b_final]), cv2.COLOR_LAB2RGB)
        ).astype(np.float32)
        / 255.0
    )
    return img


def apply_output_sharpening(img: np.ndarray, amount: float) -> np.ndarray:
    """
    Applies Unsharp Mask sharpening to the Lightness channel.
    """
    if amount <= 0:
        return img

    img_f = img.astype(np.float32)
    # PIL expects 0-255 for RGB
    img_u8 = (img_f * 255).astype(np.uint8)
    pil_working = Image.fromarray(img_u8)

    img_lab = cv2.cvtColor(np.array(pil_working), cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(img_lab)
    l_pil = Image.fromarray(l_chan)

    l_sharpened = l_pil.filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=int(amount * 250), threshold=5)
    )

    img_lab_sharpened = cv2.merge([np.array(l_sharpened), a, b])
    result_rgb = cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB)

    return result_rgb.astype(np.float32) / 255.0
