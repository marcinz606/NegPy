from typing import Any, Dict

import numpy as np
from numba import prange  # type: ignore

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.parallel import parallel_njit

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
    # ── Gold (colloidal gold plates onto silver, finest grain first) ──────────
    # Density above which gold no longer deposits (c = strength·(1 − D/this)^power).
    # ↑ toning creeps into deeper shadows; ↓ holds toning to the highlights.
    "gold_d_ref": 1.6,
    # Exponent shaping the highlight-first falloff; gentler than sepia's, so
    # gold creeps further into the midtones.
    "gold_power": 1.5,
    # Per-channel density multipliers on plain silver: all ≥1 (slight
    # intensification), red highest -> cool blue-black hue.
    "gold_gain": (1.08, 1.03, 1.00),
    # Per-channel multipliers where gold plates silver *sulfide* (sepia-toned):
    # the classic gold-over-sepia orange-red shift, redder than sulfide alone.
    "gold_sepia_gain": (0.80, 0.95, 1.20),
}


@parallel_njit(cache=True, fastmath=True)
def _apply_chemical_toning_jit(
    img: np.ndarray,
    sel_strength: float,
    sep_strength: float,
    gold_strength: float,
    sel_d_ref: float,
    sel_power: float,
    sel_gain: np.ndarray,
    sep_d_bleach: float,
    sep_power: float,
    sep_gain: np.ndarray,
    gold_d_ref: float,
    gold_power: float,
    gold_gain: np.ndarray,
    gold_sepia_gain: np.ndarray,
) -> np.ndarray:
    """
    Density-driven chemical toning on linear reflectance. Silver density
    D = -log10(t); a density-dependent fraction c of it converts to the toner's
    dye, whose per-channel covering power reshapes D: D_ch = D·(1−c) + c·D·gain.
    Selenium converts the densest silver first, sepia and gold the thinnest;
    gold runs last (real workflow: sepia first, gold after) and blends its
    covering power by the sulfide fraction — orange-red where sepia toned,
    blue-black on plain silver.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6

    for y in prange(h):
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

                c_sep = 0.0
                if sep_strength > 0.0:
                    frac = d / sep_d_bleach
                    if frac > 1.0:
                        frac = 1.0
                    c_sep = sep_strength * (1.0 - frac) ** sep_power
                    if c_sep > 1.0:
                        c_sep = 1.0
                    d = d * (1.0 - c_sep) + c_sep * d * sep_gain[ch]

                if gold_strength > 0.0:
                    frac = d / gold_d_ref
                    if frac > 1.0:
                        frac = 1.0
                    c_au = gold_strength * (1.0 - frac) ** gold_power
                    if c_au > 1.0:
                        c_au = 1.0
                    gain = gold_gain[ch] * (1.0 - c_sep) + gold_sepia_gain[ch] * c_sep
                    d = d * (1.0 - c_au) + c_au * d * gain

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
    gold_strength: float = 0.0,
) -> ImageBuffer:
    """
    Selenium / sepia / gold toning of a linear-reflectance print (density domain).
    """
    if selenium_strength == 0 and sepia_strength == 0 and gold_strength == 0:
        return img

    c = TONING_CONSTANTS
    return ensure_image(
        _apply_chemical_toning_jit(
            np.ascontiguousarray(img.astype(np.float32)),
            float(selenium_strength),
            float(sepia_strength),
            float(gold_strength),
            float(c["sel_d_ref"]),
            float(c["sel_power"]),
            np.array(c["sel_gain"], dtype=np.float32),
            float(c["sep_d_bleach"]),
            float(c["sep_power"]),
            np.array(c["sep_gain"], dtype=np.float32),
            float(c["gold_d_ref"]),
            float(c["gold_power"]),
            np.array(c["gold_gain"], dtype=np.float32),
            np.array(c["gold_sepia_gain"], dtype=np.float32),
        )
    )
