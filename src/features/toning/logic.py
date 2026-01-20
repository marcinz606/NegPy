import numpy as np
from numba import njit, prange  # type: ignore
from typing import Dict
from src.domain.types import ImageBuffer, LUMA_R, LUMA_G, LUMA_B
from src.kernel.image.validation import ensure_image
from src.features.toning.models import PaperSubstrate, PaperProfileName


@njit(parallel=True, cache=True, fastmath=True)
def _apply_paper_substrate_jit(
    img: np.ndarray, tint: np.ndarray, dmax_boost: float
) -> np.ndarray:
    """
    Applies tint & density boost.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            for ch in range(3):
                val = img[y, x, ch] * tint[ch]
                if dmax_boost != 1.0:
                    val = val**dmax_boost
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
                res[y, x, ch] = val
    return res


@njit(parallel=True, cache=True, fastmath=True)
def _apply_chemical_toning_jit(
    img: np.ndarray, sel_strength: float, sep_strength: float
) -> np.ndarray:
    """
    Selenium (Shadows) & Sepia (Mids) toning.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    sel_color = np.array([0.85, 0.75, 0.85], dtype=np.float32)
    sep_color = np.array([1.1, 0.99, 0.825], dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            # Fused Luminance (Rec. 709)
            lum_val = (
                LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
            )

            sel_m = 0.0
            if sel_strength > 0:
                sel_m = 1.0 - lum_val
                if sel_m < 0.0:
                    sel_m = 0.0
                sel_m = sel_m * sel_m * sel_strength

            sep_m = 0.0
            if sep_strength > 0:
                sep_m = np.exp(-((lum_val - 0.6) ** 2) / 0.08) * sep_strength

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


PAPER_PROFILES: Dict[str, PaperSubstrate] = {
    PaperProfileName.NONE: PaperSubstrate(PaperProfileName.NONE, (1.0, 1.0, 1.0), 1.0),
    PaperProfileName.NEUTRAL_RC: PaperSubstrate(
        PaperProfileName.NEUTRAL_RC, (0.99, 0.99, 0.99), 1.0
    ),
    PaperProfileName.COOL_GLOSSY: PaperSubstrate(
        PaperProfileName.COOL_GLOSSY, (0.98, 0.99, 1.02), 1.1
    ),
    PaperProfileName.WARM_FIBER: PaperSubstrate(
        PaperProfileName.WARM_FIBER, (1.0, 0.97, 0.92), 1.15
    ),
}


def simulate_paper_substrate(img: ImageBuffer, profile_name: str) -> ImageBuffer:
    """
    Look-up profile -> Apply tint.
    """
    profile = PAPER_PROFILES.get(profile_name, PAPER_PROFILES[PaperProfileName.NONE])
    tint = np.ascontiguousarray(np.array(profile.tint, dtype=np.float32))

    return ensure_image(
        _apply_paper_substrate_jit(
            np.ascontiguousarray(img.astype(np.float32)),
            tint,
            float(profile.dmax_boost),
        )
    )


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
