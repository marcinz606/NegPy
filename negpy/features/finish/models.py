from dataclasses import dataclass


@dataclass(frozen=True)
class FinishConfig:
    """
    Post-crop finishing effects (vignette).
    """

    vignette_strength: float = 0.0  # [-1.0, 1.0]  0 = off, neg = darken, pos = brighten
    vignette_size: float = 0.5  # [0.0, 1.0]   midpoint of falloff gradient
    border_size: float = 0.0  # [0.0, 10.0] cm
    border_color: str = "#ffffff"  # hex color
