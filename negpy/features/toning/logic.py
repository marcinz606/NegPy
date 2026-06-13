import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R, ImageBuffer
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working
from negpy.kernel.image.validation import ensure_image


@njit(cache=True, fastmath=True)
def _apply_chemical_toning_jit(img: np.ndarray, sel_strength: float, sep_strength: float) -> np.ndarray:
    """
    Selenium (Shadows) & Sepia (Mids) toning.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    sel_color = np.array([0.85, 0.75, 0.85], dtype=np.float32)
    sep_color = np.array([1.1, 0.99, 0.825], dtype=np.float32)

    for y in range(h):
        for x in range(w):
            # Fused Luminance (Rec. 709)
            lum_val = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]

            sel_m = 0.0
            if sel_strength > 0:
                m = 1.0 - lum_val
                if m < 0.0:
                    m = 0.0
                sel_m = m * m * sel_strength

            sep_m = 0.0
            if sep_strength > 0:
                dist = lum_val - 0.6
                sep_m = np.exp(-(dist * dist) / 0.08) * sep_strength

            for ch in range(3):
                pixel = img[y, x, ch]
                if sel_m > 0:
                    pixel = pixel * (1.0 - sel_m) + (pixel * sel_color[ch]) * sel_m
                if sep_m > 0:
                    pixel = pixel * (1.0 - sep_m) + (pixel * sep_color[ch]) * sep_m

                if pixel < 0.0:
                    pixel = 0.0
                elif pixel > 1.0:
                    pixel = 1.0
                res[y, x, ch] = pixel
    return res


def apply_split_toning(
    img: ImageBuffer,
    shadow_hue: float = 0.0,
    shadow_strength: float = 0.0,
    highlight_hue: float = 0.0,
    highlight_strength: float = 0.0,
) -> ImageBuffer:
    """
    Additive Lab-space split toning. Shadow and highlight regions are tinted toward
    the chosen hue angle (0-360°) at the specified strength (0-1). Luminance is preserved.
    """
    if shadow_strength == 0.0 and highlight_strength == 0.0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    L = lab[:, :, 0]  # 0–100 CIELAB (Adobe RGB working space)

    if shadow_strength > 0.0:
        s_mask = np.clip(1.0 - L / 50.0, 0.0, 1.0)
        rad = np.radians(shadow_hue)
        lab[:, :, 1] += np.cos(rad) * 20.0 * shadow_strength * s_mask
        lab[:, :, 2] += np.sin(rad) * 20.0 * shadow_strength * s_mask

    if highlight_strength > 0.0:
        h_mask = np.clip((L - 50.0) / 50.0, 0.0, 1.0)
        rad = np.radians(highlight_hue)
        lab[:, :, 1] += np.cos(rad) * 20.0 * highlight_strength * h_mask
        lab[:, :, 2] += np.sin(rad) * 20.0 * highlight_strength * h_mask

    return ensure_image(np.clip(lab_to_rgb_working(lab), 0.0, 1.0))


def apply_chemical_toning(
    img: ImageBuffer,
    selenium_strength: float = 0.0,
    sepia_strength: float = 0.0,
) -> ImageBuffer:
    """
    Applies split-toning based on luminance.
    """
    if selenium_strength == 0 and sepia_strength == 0:
        return img

    return ensure_image(
        _apply_chemical_toning_jit(
            np.ascontiguousarray(img.astype(np.float32)),
            float(selenium_strength),
            float(sepia_strength),
        )
    )
