from dataclasses import dataclass


@dataclass(frozen=True)
class ToningConfig:
    """
    Toner params.
    """

    selenium_strength: float = 0.0
    sepia_strength: float = 0.0
    shadow_tint_hue: float = 0.0
    shadow_tint_strength: float = 0.0
    highlight_tint_hue: float = 0.0
    highlight_tint_strength: float = 0.0
