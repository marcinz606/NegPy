from typing import Any, Dict

import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working
from negpy.kernel.image.validation import ensure_image

TONING_CONSTANTS: Dict[str, Any] = {
    # ── Selenium (silver -> silver selenide, densest silver first) ───────────
    # Density at which selenium conversion saturates (c = strength·(D/this)^power).
    # ↑ conversion spreads to lighter tones more slowly; ↓ shadows convert sooner.
    "sel_d_ref": 2.0,
    # Exponent shaping the density-proportional conversion.
    # ↑ conversion concentrates deeper in the shadows; ↓ creeps into midtones.
    "sel_power": 1.5,
    # Per-channel density multipliers of converted silver: all ≥1 deepens blacks
    # (the Dmax boost selenium is used for); green highest -> eggplant shadow hue.
    "sel_gain": (1.04, 1.10, 1.02),
    # ── Sepia (bleach–redevelop to sulfide, thinnest silver first) ────────────
    # Density above which bleach no longer reaches (c = strength·(1 − D/this)^power).
    # ↑ toning creeps into deeper shadows; ↓ holds toning to the highlights.
    "sep_d_bleach": 1.8,
    # Exponent shaping the highlight-first conversion falloff.
    # ↑ tighter split-sepia (highlights only); ↓ more even toning.
    "sep_power": 2.0,
    # Per-channel density multipliers of converted silver: red < 1 (sulfide's
    # lower covering power lifts/warms), blue > 1 -> yellow-brown hue.
    "sep_gain": (0.82, 0.94, 1.12),
}


@njit(cache=True, fastmath=True)
def _apply_chemical_toning_jit(
    img: np.ndarray,
    sel_strength: float,
    sep_strength: float,
    sel_d_ref: float,
    sel_power: float,
    sel_gain: np.ndarray,
    sep_d_bleach: float,
    sep_power: float,
    sep_gain: np.ndarray,
) -> np.ndarray:
    """
    Density-driven chemical toning on linear reflectance. Silver density
    D = -log10(t); a density-dependent fraction c of it converts to the toner's
    dye, whose per-channel covering power reshapes D: D_ch = D·(1−c) + c·D·gain.
    Selenium converts the densest silver first, sepia the thinnest.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6

    for y in range(h):
        for x in range(w):
            for ch in range(3):
                t = img[y, x, ch]
                if t < eps:
                    t = eps
                elif t > 1.0:
                    t = 1.0
                d = -np.log10(t)

                if sel_strength > 0.0:
                    frac = d / sel_d_ref
                    if frac > 1.0:
                        frac = 1.0
                    # Conversion caps at 1: all the silver is toned (slider > 1 = longer bath).
                    c_sel = sel_strength * frac**sel_power
                    if c_sel > 1.0:
                        c_sel = 1.0
                    d = d * (1.0 - c_sel) + c_sel * d * sel_gain[ch]

                if sep_strength > 0.0:
                    frac = d / sep_d_bleach
                    if frac > 1.0:
                        frac = 1.0
                    c_sep = sep_strength * (1.0 - frac) ** sep_power
                    if c_sep > 1.0:
                        c_sep = 1.0
                    d = d * (1.0 - c_sep) + c_sep * d * sep_gain[ch]

                pixel = 10.0**-d
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
    Selenium / sepia toning of a linear-reflectance print (density domain).
    """
    if selenium_strength == 0 and sepia_strength == 0:
        return img

    c = TONING_CONSTANTS
    return ensure_image(
        _apply_chemical_toning_jit(
            np.ascontiguousarray(img.astype(np.float32)),
            float(selenium_strength),
            float(sepia_strength),
            float(c["sel_d_ref"]),
            float(c["sel_power"]),
            np.array(c["sel_gain"], dtype=np.float32),
            float(c["sep_d_bleach"]),
            float(c["sep_power"]),
            np.array(c["sep_gain"], dtype=np.float32),
        )
    )
