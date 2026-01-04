import numpy as np
from typing import Dict
from src.domain_objects import PaperSubstrate
from src.helpers import get_luminance

PAPER_PROFILES: Dict[str, PaperSubstrate] = {
    "None": PaperSubstrate(
        name="None",
        tint=(1.0, 1.0, 1.0),
        dmax_boost=1.0,
    ),
    "Neutral RC": PaperSubstrate(
        name="Neutral RC",
        tint=(0.99, 0.99, 0.99),
        dmax_boost=1.0,
    ),
    "Cool Glossy": PaperSubstrate(
        name="Cool Glossy",
        tint=(0.98, 0.99, 1.02),
        dmax_boost=1.1,
    ),
    "Warm Fiber": PaperSubstrate(
        name="Warm Fiber",
        tint=(1.0, 0.97, 0.92),
        dmax_boost=1.15,
    ),
    "Antique Ivory": PaperSubstrate(
        name="Antique Ivory",
        tint=(0.98, 0.94, 0.88),
        dmax_boost=1.05,
    ),
}


def simulate_paper_substrate(img: np.ndarray, profile_name: str) -> np.ndarray:
    """
    Simulates the physics of a photographic paper substrate.
    The paper reflects light through the developed silver/dye image.
    """
    profile = PAPER_PROFILES.get(profile_name, PAPER_PROFILES["None"])
    tint = np.array(profile.tint, dtype=np.float32)

    # Reflectance Simulation: Image * Base_Tint
    res = img * tint

    # D-Max Simulation: Deepen shadows using a power curve
    if profile.dmax_boost != 1.0:
        boosted = np.power(res, profile.dmax_boost)
        res = boosted if isinstance(boosted, np.ndarray) else np.array(boosted)

    return np.clip(res, 0.0, 1.0)


def apply_chemical_toning(
    img: np.ndarray,
    selenium_strength: float = 0.0,
    sepia_strength: float = 0.0,
) -> np.ndarray:
    """
    Simulates chemical reactivity of toners with silver halides.
    - Selenium: Replaces silver with silver selenide, deepening density and cooling shadows.
    - Sepia: Partially bleaches silver and replaces it with silver sulfide, creating a warm glow.
    """
    if selenium_strength == 0 and sepia_strength == 0:
        return img

    res = img.copy()
    lum = get_luminance(img)

    # 1. Selenium (Shadow Focus)
    # Selenium darkens the image (converts Ag to AgSe, increasing density).
    if selenium_strength > 0:
        # Target: Affects darks most
        sel_mask = np.clip(1.0 - lum, 0.0, 1.0)
        sel_mask = sel_mask * sel_mask  # Shadow-weighted

        sel_color = np.array([0.85, 0.75, 0.85], dtype=np.float32)

        # Physics: Selenium increases density (decreases reflectance)
        toned_sel = res * sel_color
        res_mixed_sel = res * (
            1.0 - selenium_strength * sel_mask[:, :, None]
        ) + toned_sel * (selenium_strength * sel_mask[:, :, None])
        res = res_mixed_sel.astype(res.dtype)

    # 2. Sepia (Mid/Highlight Focus)
    # Sepia bleaches density (decreases density, increases reflectance).
    if sepia_strength > 0:
        # Mask: Gaussian bell curve peaking at 0.6 luminance
        sep_mask = np.exp(-((lum - 0.6) ** 2) / (2 * (0.2**2)))

        sep_color = np.array([1.0, 0.9, 0.75], dtype=np.float32)

        # Physics: Sepia bleaches (lightens) the image
        # Blend using weighted mix to simulate partial sulfide replacement
        # 1.1 multiplier simulates the 'bleach' lift
        toned_sep = res * sep_color * 1.1

        res_mixed = res * (1.0 - sepia_strength * sep_mask[:, :, None]) + toned_sep * (
            sepia_strength * sep_mask[:, :, None]
        )
        res = res_mixed.astype(res.dtype)

    return np.clip(res, 0.0, 1.0)
