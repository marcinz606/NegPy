from typing import Any, Dict

import numpy as np
from numba import prange  # type: ignore

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.parallel import parallel_njit

TONING_CONSTANTS: Dict[str, Any] = {
    # Selenium (silver -> silver selenide, densest silver first).
    # Density at which conversion saturates: c = strength*(D/this)^power.
    "sel_d_ref": 2.0,
    # Exponent shaping the density-proportional conversion.
    "sel_power": 1.5,
    # Per-channel density multipliers of converted silver. All above 1, so blacks
    # deepen; green highest gives the eggplant shadow hue.
    "sel_gain": (1.04, 1.10, 1.02),
    # Sepia (bleach and redevelop to sulfide, thinnest silver first).
    # Density above which the bleach no longer reaches: c = strength*(1 - D/this)^power.
    "sep_d_bleach": 1.8,
    # Exponent shaping the highlight-first falloff.
    "sep_power": 2.0,
    # Per-channel multipliers of converted silver: red below 1, because sulfide has
    # lower covering power, and blue above 1, giving the yellow-brown hue.
    "sep_gain": (0.82, 0.94, 1.12),
    # Gold (colloidal gold plates onto silver, finest grain first).
    # Density above which gold no longer deposits: c = strength*(1 - D/this)^power.
    "gold_d_ref": 1.6,
    # Exponent shaping the highlight-first falloff, gentler than sepia's, so gold
    # creeps further into the midtones.
    "gold_power": 1.5,
    # Per-channel multipliers on plain silver: a slight intensification, red highest
    # for the cool blue-black hue.
    "gold_gain": (1.08, 1.03, 1.00),
    # Per-channel multipliers where gold plates silver *sulfide*: the gold-over-sepia
    # orange-red shift, redder than sulfide alone.
    "gold_sepia_gain": (0.80, 0.95, 1.20),
    # Iron blue (silver -> ferric ferrocyanide, silver-proportional).
    # Density at which conversion saturates: c = strength*(D/this)^power. Kept well
    # below Dmax: the hue effect scales with D*c, so a Dmax reference would confine
    # the color to the deep shadows.
    "blue_d_ref": 0.9,
    # Slightly sub-linear, so the blue reaches the mids while shadows still lead.
    "blue_power": 0.85,
    # Prussian blue deposits more coloring matter than the silver it replaces, so the
    # net gain is above 1 and red is absorbed most. The pigment is cyan-leaning, and
    # G at 1.00 is what lets the sepia+blue green split emerge.
    "blue_gain": (1.30, 1.00, 0.80),
    # Copper (silver -> copper ferrocyanide, in-bath bleach).
    # Density at which conversion saturates, low for the same mid-tone visibility
    # reason as blue_d_ref.
    "copper_d_ref": 0.9,
    # Sub-linear exponent, so conversion reaches mids and highlights early.
    "copper_power": 0.6,
    # Net gain below 1: the ferricyanide bleaches while it tones. Red lifted most
    # gives the pink/brick-red hue.
    "copper_gain": (0.72, 0.94, 1.18),
    # Vanadium green (bleach then tone, thinnest silver first).
    # Density above which the bleach no longer reaches.
    "van_d_ref": 1.8,
    # Gentler falloff than sepia's, so the green creeps into the mids while deep
    # shadows keep their black silver.
    "van_power": 1.2,
    # Vanadium yellow plus Prussian blue reads green: R and B absorbed, G spared.
    # Luma-weighted net below 1, so density drops a little.
    "van_gain": (1.12, 0.85, 1.03),
}

# Overrides for a lith-developed print. Lith silver is fine, small-particle silver on
# the steep part of the tone-vs-particle-size curve (Kong & Shore 2007), so the same
# bath moves it much further than normal filamentary silver. Only selenium and gold are
# distinctive on lith, and the sidebar disables the rest.
LITH_TONING_CONSTANTS: Dict[str, Any] = {
    # Reaches much further down the scale than on a normal print and lifts Dmax hard.
    # On some lith papers a short bold selenium is the only way to a real black.
    "sel_d_ref": 1.2,
    # Near-linear: the toner creeps into the midtones instead of hiding in Dmax.
    "sel_power": 1.0,
    # Green highest by a wide margin: the green-black lith shadow goes magenta
    # (Moersch, 1+5 for 20 s). The mean above 1 is the Dmax boost.
    "sel_gain": (1.10, 1.24, 1.12),
    # Gold attacks all densities evenly, unlike selenium, so on lith the bleach-limited
    # shape is replaced by a flat conversion (see gold_flat).
    "gold_gain": (1.16, 1.06, 0.96),
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
    blue_strength: float,
    blue_d_ref: float,
    blue_power: float,
    blue_gain: np.ndarray,
    copper_strength: float,
    copper_d_ref: float,
    copper_power: float,
    copper_gain: np.ndarray,
    van_strength: float,
    van_d_ref: float,
    van_power: float,
    van_gain: np.ndarray,
    gold_flat: float,
) -> np.ndarray:
    """
    Silver-ledger chemical toning on linear reflectance. All baths compete for
    one metallic-silver reservoir: converted silver is locked to later toners
    (Rudman/Ilford — the archival selenium-then-sepia split, "no silver left"
    exhaustion). Each toner's susceptibility c is a pure function of the
    ORIGINAL mean density D0 (grain property); sequence only decides who claims
    silver first via the remaining fraction a: f_i = a·c_i, a -= f_i. Final
    density is the covering-power mix D_ch = D0_ch·(a + Σ f_i·gain_i), per
    channel, so a print that already carries color keeps it. Gold is
    the one lock-out exception: it also plates the sulfide fraction (classic
    gold-over-sepia orange-red), with compounded covering power.

    gold_flat drops gold's bleach-limited density weighting for a flat
    conversion — how it behaves on a lith print, where it "attacks all densities
    evenly" instead of starting in the highlights.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6

    for y in prange(h):
        d3 = np.empty(3, dtype=np.float64)
        for x in range(w):
            d0 = 0.0
            for ch in range(3):
                t = img[y, x, ch]
                if t < eps:
                    t = eps
                elif t > 1.0:
                    t = 1.0
                d3[ch] = -np.log10(t)
                d0 += d3[ch]
            d0 /= 3.0

            # Conversion caps at 1: all the remaining silver is toned
            # (slider > 1 = longer bath).
            c_sel = 0.0
            if sel_strength > 0.0:
                frac = d0 / sel_d_ref
                if frac > 1.0:
                    frac = 1.0
                c_sel = sel_strength * frac**sel_power
                if c_sel > 1.0:
                    c_sel = 1.0

            c_sep = 0.0
            if sep_strength > 0.0:
                frac = d0 / sep_d_bleach
                if frac > 1.0:
                    frac = 1.0
                c_sep = sep_strength * (1.0 - frac) ** sep_power
                if c_sep > 1.0:
                    c_sep = 1.0

            c_au = 0.0
            if gold_strength > 0.0:
                if gold_flat > 0.5:
                    c_au = gold_strength
                else:
                    frac = d0 / gold_d_ref
                    if frac > 1.0:
                        frac = 1.0
                    c_au = gold_strength * (1.0 - frac) ** gold_power
                if c_au > 1.0:
                    c_au = 1.0

            c_blue = 0.0
            if blue_strength > 0.0:
                frac = d0 / blue_d_ref
                if frac > 1.0:
                    frac = 1.0
                c_blue = blue_strength * frac**blue_power
                if c_blue > 1.0:
                    c_blue = 1.0

            c_cu = 0.0
            if copper_strength > 0.0:
                frac = d0 / copper_d_ref
                if frac > 1.0:
                    frac = 1.0
                c_cu = copper_strength * frac**copper_power
                if c_cu > 1.0:
                    c_cu = 1.0

            c_van = 0.0
            if van_strength > 0.0:
                frac = d0 / van_d_ref
                if frac > 1.0:
                    frac = 1.0
                c_van = van_strength * (1.0 - frac) ** van_power
                if c_van > 1.0:
                    c_van = 1.0

            a = 1.0
            f_sel = a * c_sel
            a -= f_sel
            f_sep = a * c_sep
            a -= f_sep
            f_au = a * c_au
            a -= f_au
            f_ausp = f_sep * c_au
            f_sep -= f_ausp
            f_blue = a * c_blue
            a -= f_blue
            f_cu = a * c_cu
            a -= f_cu
            f_van = a * c_van
            a -= f_van

            for ch in range(3):
                # Covering power rides each channel's OWN density, not the mean. The
                # mean is the silver reservoir above, but rebuilding the output from it
                # would throw away the color a lith print hands this stage. Identical
                # for a normal grey B&W print, where d3[ch] == d0.
                d = d3[ch] * (
                    a
                    + f_sel * sel_gain[ch]
                    + f_sep * sep_gain[ch]
                    + f_au * gold_gain[ch]
                    + f_ausp * sep_gain[ch] * gold_sepia_gain[ch]
                    + f_blue * blue_gain[ch]
                    + f_cu * copper_gain[ch]
                    + f_van * van_gain[ch]
                )
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
    blue_strength: float = 0.0,
    copper_strength: float = 0.0,
    vanadium_strength: float = 0.0,
    lith_active: bool = False,
) -> ImageBuffer:
    """
    Selenium / sepia / gold / iron-blue / copper / vanadium-green toning of a
    linear-reflectance print (density domain).

    lith_active swaps in LITH_TONING_CONSTANTS for selenium and gold, the two
    baths that behave differently on lith silver.
    """
    if (
        selenium_strength == 0
        and sepia_strength == 0
        and gold_strength == 0
        and blue_strength == 0
        and copper_strength == 0
        and vanadium_strength == 0
    ):
        return img

    c = {**TONING_CONSTANTS, **LITH_TONING_CONSTANTS} if lith_active else TONING_CONSTANTS
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
            float(blue_strength),
            float(c["blue_d_ref"]),
            float(c["blue_power"]),
            np.array(c["blue_gain"], dtype=np.float32),
            float(copper_strength),
            float(c["copper_d_ref"]),
            float(c["copper_power"]),
            np.array(c["copper_gain"], dtype=np.float32),
            float(vanadium_strength),
            float(c["van_d_ref"]),
            float(c["van_power"]),
            np.array(c["van_gain"], dtype=np.float32),
            1.0 if lith_active else 0.0,
        )
    )
