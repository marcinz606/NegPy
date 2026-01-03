import numpy as np
from src.config import ImageSettings, PIPELINE_CONSTANTS
from src.helpers import get_luminance


def convert_to_monochrome(img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to a 3-channel monochrome (greyscale) image.
    """
    if img.shape[2] != 3:
        return img
    lum = get_luminance(img)
    res: np.ndarray = np.stack([lum, lum, lum], axis=2)
    return res


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
    res: np.ndarray = lum_3d + (img - lum_3d) * effective_intensity[:, :, None]
    res = np.clip(res, 0.0, 1.0)

    if not is_float:
        res = (res * 255.0).astype(np.uint8)
    return res


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


def apply_paper_warmth(img: np.ndarray, warmth: float) -> np.ndarray:
    """
    Simulates paper base warmth by shifting the highlight density.
    Warmth (+): shifts print towards Amber/Yellow.
    Applied to the positive image to mimic the physical paper tint.
    """
    if warmth == 0.0:
        return img

    # Use a density-based shift on the positive image
    epsilon = 1e-4
    img_dens = -np.log10(np.clip(img, epsilon, 1.0))

    # Calculate density shift (Amber/Yellow shift)
    d_shift = warmth * PIPELINE_CONSTANTS["paper_warmth_strength"]

    # Highlights-weighted mask based on average density
    avg_dens = np.mean(img_dens, axis=2)
    h_mask = np.clip(1.0 - (avg_dens / 2.0), 0.0, 1.0)

    img_dens[:, :, 0] -= d_shift * h_mask
    img_dens[:, :, 2] += d_shift * h_mask

    res = 10.0 ** -np.clip(img_dens, 0.0, 4.0)
    res_final: np.ndarray = np.clip(res, 0, 1)
    return res_final


def apply_shadow_highlight_grading(
    img: np.ndarray, params: ImageSettings
) -> np.ndarray:
    """
    Applies Split Toning using a density-addition model on the positive image.
    Mimics chemical toning (Shadows) and paper tints (Highlights).
    """
    s_tone = params.shadow_temp
    h_tone = params.highlight_temp

    if s_tone == 0.0 and h_tone == 0.0:
        return img

    epsilon = 1e-4
    img_dens = -np.log10(np.clip(img, epsilon, 1.0))
    avg_dens = np.mean(img_dens, axis=2)
    strength = PIPELINE_CONSTANTS["toning_strength"]

    # Shadow Toning
    if s_tone != 0.0:
        s_mask = np.clip(avg_dens / 2.0, 0.0, 1.0)
        s_mask = s_mask * s_mask
        img_dens[:, :, 0] -= s_tone * strength * s_mask
        img_dens[:, :, 2] += s_tone * strength * s_mask

    # Highlight Toning
    if h_tone != 0.0:
        h_mask = np.clip(1.0 - (avg_dens / 1.5), 0.0, 1.0)
        img_dens[:, :, 0] -= h_tone * strength * h_mask
        img_dens[:, :, 2] += h_tone * strength * h_mask

    res = 10.0 ** -np.clip(img_dens, 0.0, 4.0)
    res_final: np.ndarray = np.clip(res, 0, 1)
    return res_final
