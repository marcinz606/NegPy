import numpy as np
import cv2
from typing import Tuple, cast
from src.config import ProcessingParams
from src.helpers import get_luminance


def convert_to_monochrome(img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to a 3-channel monochrome (greyscale) image.
    """
    if img.shape[2] != 3:
        return img
    lum = get_luminance(img)
    return cast(np.ndarray, np.stack([lum, lum, lum], axis=2))


def measure_film_base(img: np.ndarray) -> Tuple[float, float, float]:
    """
    Measures the unexposed film base color (brightest part of negative).
    Includes validation to ensure we aren't picking up white objects in a cropped scan.
    """
    if img.ndim != 3:
        return 0.90, 0.65, 0.45

    # Step A: Measure Candidates (99.9th percentile)
    # This rejects hot pixels but captures the "brightest" smooth area.
    cand_r = float(np.percentile(img[:, :, 0], 99.9))
    cand_g = float(np.percentile(img[:, :, 1], 99.9))
    cand_b = float(np.percentile(img[:, :, 2], 99.9))

    # Step B: Validate (Is this an Orange Mask?)
    # 1. Hierarchy Check: Mask should be Red > Green > Blue (Orange)
    is_orange_hierarchy = cand_r > cand_g and cand_g > cand_b
    
    # 2. Separation Check: Red should be significantly brighter than Blue.
    # We require at least 10% separation to rule out Neutral White objects.
    has_color_separation = cand_r > (cand_b * 1.10)

    # Step C: Decide
    if is_orange_hierarchy and has_color_separation:
        return cand_r, cand_g, cand_b
    else:
        # Fallback: Synthetic Base (Generic Orange Mask)
        # Represents typical transmission of developed unexposed C41 film.
        return 0.90, 0.65, 0.45


def calculate_log_medians(img: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculates the Log-Median of each channel.
    This represents the 'Statistical Gray' of the scene.
    Used for 'Gray World' auto-white balance.
    """
    if img.ndim != 3:
        return 0.5, 0.5, 0.5
        
    # Clip to avoid log(0) and log(>1)
    epsilon = 1e-6
    img_log = np.log10(np.clip(img, epsilon, 1.0))
    
    med_r = float(np.median(img_log[:, :, 0]))
    med_g = float(np.median(img_log[:, :, 1]))
    med_b = float(np.median(img_log[:, :, 2]))
    
    return med_r, med_g, med_b


def apply_color_separation(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Increases/decreases color separation (saturation) without shifting luminance.
    Refined: Tapers intensity in shadows to prevent "nuclear" darks.
    """
    if intensity == 1.0:
        return img

    is_float = img.dtype.kind == "f"
    if not is_float:
        img = img.astype(np.float32) / 255.0

    lum = get_luminance(img)

    # Create a luma mask to protect shadows from excessive separation
    luma_mask = np.clip(lum / 0.2, 0.0, 1.0)
    luma_mask = luma_mask * luma_mask

    effective_intensity = 1.0 + (intensity - 1.0) * luma_mask

    lum_3d = lum[:, :, None]
    res = lum_3d + (img - lum_3d) * effective_intensity[:, :, None]
    res = np.clip(res, 0.0, 1.0)

    if not is_float:
        res = (res * 255.0).astype(np.uint8)
    return cast(np.ndarray, res)


def apply_shadow_desaturation(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Automatically reduces saturation in the darkest parts of the image to
    prevent "electric" shadows when lifting them.
    """
    if strength <= 0:
        return img

    luma = get_luminance(img)
    mask = np.clip(1.0 - (luma / 0.3), 0.0, 1.0)
    mask = mask * mask * strength
    luma_3d = luma[:, :, None]
    res = img * (1.0 - mask[:, :, None]) + luma_3d * mask[:, :, None]
    return np.clip(res, 0.0, 1.0)


def calculate_auto_mask_wb(raw_preview: np.ndarray) -> Tuple[float, float, float]:
    """
    Identifies white balance by targeting the physical transparency peak of
    the Red channel. Anchors the result to Red (gain=1.0) so that Cyan
    filtration remains at 0 while Magenta and Yellow are calculated.
    """
    if raw_preview.ndim == 2:
        return 1.0, 1.0, 1.0

    # Resize for fast processing
    h, w = raw_preview.shape[:2]
    small = cv2.resize(raw_preview, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)

    # 1. Filter out clipped pixels
    valid_mask = np.all(small < 0.98, axis=-1)
    valid_pixels = small[valid_mask]
    if len(valid_pixels) < 100:
        valid_pixels = pixels

    # 2. Target the 'Transparency Peak' of the Red channel (Top 0.1%)
    r_vals = cast(np.ndarray, valid_pixels[:, 0])
    r_thresh = np.percentile(r_vals, 99.9)
    mask_pixels = cast(np.ndarray, valid_pixels[r_vals >= r_thresh])

    if len(mask_pixels) > 5:
        # 3. Use the Median color of this physical limit.
        mask_color = np.median(mask_pixels, axis=0)

        # 4. Calculate gains anchored to RED = 1.0
        # This forces Cyan to 0 in the darkroom model.
        r_val = mask_color[0]
        g_gain = r_val / (mask_color[1] + 1e-6)
        b_gain = r_val / (mask_color[2] + 1e-6)

        return 1.0, float(g_gain), float(b_gain)

    # Standard Fallback
    return 1.0, 1.5, 4.0


def apply_manual_color_balance_neg(
    img: np.ndarray, params: ProcessingParams
) -> np.ndarray:
    """
    Applies 'Paper Warmth' (Temperature) in the NEGATIVE domain.
    Inverted Math: warmth (+) decreases neg red, increasing positive red.
    """
    res = img.copy()
    warmth = params.get("temperature", 0.0)

    if warmth != 0.0:
        # Increase warmth (+): decrease neg red, increase neg blue
        res[:, :, 0] *= 1.0 - warmth
        res[:, :, 2] *= 1.0 + warmth

    return np.clip(res, 0, 1)


def apply_shadow_highlight_grading(
    img: np.ndarray, params: ProcessingParams
) -> np.ndarray:
    """
    Applies Split Toning in the NEGATIVE domain.
    Inverted Math: tone (+) decreases neg red, increasing positive red (Amber).
    """
    res = img.copy()
    lum = get_luminance(res)
    # In a negative: High Lum (Clear) = Shadow on Print. Low Lum (Dense) = Highlight on Print.
    w_shadow = lum[:, :, None]
    w_highlight = (1.0 - lum)[:, :, None]

    # Shadow Tone (Amber <-> Blue)
    s_tone = params.get("shadow_temp", 0.0)
    if s_tone != 0.0:
        s_r = 1.0 - s_tone
        s_b = 1.0 + s_tone
        res *= w_shadow * np.array([s_r, 1.0, s_b]) + (1.0 - w_shadow)

    # Highlight Tone (Amber <-> Blue)
    h_tone = params.get("highlight_temp", 0.0)
    if h_tone != 0.0:
        h_r = 1.0 - h_tone
        h_b = 1.0 + h_tone
        res *= w_highlight * np.array([h_r, 1.0, h_b]) + (1.0 - w_highlight)

    return np.clip(res, 0, 1)
