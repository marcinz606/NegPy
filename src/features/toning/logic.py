import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from src.core.types import ImageBuffer
from src.core.validation import ensure_image


def get_luminance(img: ImageBuffer) -> ImageBuffer:
    res = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return ensure_image(res)


@dataclass
class PaperSubstrate:
    name: str
    tint: Tuple[float, float, float]
    dmax_boost: float


PAPER_PROFILES: Dict[str, PaperSubstrate] = {
    "None": PaperSubstrate("None", (1.0, 1.0, 1.0), 1.0),
    "Neutral RC": PaperSubstrate("Neutral RC", (0.99, 0.99, 0.99), 1.0),
    "Cool Glossy": PaperSubstrate("Cool Glossy", (0.98, 0.99, 1.02), 1.1),
    "Warm Fiber": PaperSubstrate("Warm Fiber", (1.0, 0.97, 0.92), 1.15),
    "Antique Ivory": PaperSubstrate("Antique Ivory", (0.98, 0.94, 0.88), 1.05),
}


def simulate_paper_substrate(img: ImageBuffer, profile_name: str) -> ImageBuffer:
    """
    Simulates the physics of a photographic paper substrate.
    """
    profile = PAPER_PROFILES.get(profile_name, PAPER_PROFILES["None"])
    tint = np.array(profile.tint, dtype=np.float32)

    # Reflectance Simulation
    res = img * tint

    # D-Max Simulation
    if profile.dmax_boost != 1.0:
        boosted = np.power(res, profile.dmax_boost)
        res = boosted if isinstance(boosted, np.ndarray) else np.array(boosted)

    return ensure_image(np.clip(res, 0.0, 1.0))


def apply_chemical_toning(
    img: ImageBuffer,
    selenium_strength: float = 0.0,
    sepia_strength: float = 0.0,
) -> ImageBuffer:
    """
    Simulates chemical reactivity of toners with silver halides.
    """
    if selenium_strength == 0 and sepia_strength == 0:
        return img

    res = img.copy()
    lum = get_luminance(img)

    # 1. Selenium (Shadow Focus)
    if selenium_strength > 0:
        sel_mask = np.clip(1.0 - lum, 0.0, 1.0)
        sel_mask = sel_mask * sel_mask

        sel_color = np.array([0.85, 0.75, 0.85], dtype=np.float32)

        toned_sel = res * sel_color
        res = ensure_image(
            res * (1.0 - selenium_strength * sel_mask[:, :, None])
            + toned_sel * (selenium_strength * sel_mask[:, :, None])
        )

    # 2. Sepia (Mid/Highlight Focus)
    if sepia_strength > 0:
        sep_mask = np.exp(-((lum - 0.6) ** 2) / (2 * (0.2**2)))
        sep_color = np.array([1.0, 0.9, 0.75], dtype=np.float32)

        toned_sep = res * sep_color * 1.1
        res = res * (1.0 - sepia_strength * sep_mask[:, :, None]) + toned_sep * (
            sepia_strength * sep_mask[:, :, None]
        )

    return ensure_image(np.clip(res, 0.0, 1.0))
