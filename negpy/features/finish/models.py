from dataclasses import dataclass


@dataclass(frozen=True)
class FinishConfig:
    """
    Post-crop finishing effects (vignette).
    """

    vignette_stops: float = 0.0  # [-2.0, 2.0]  0 = off, pos = burn (darken), neg = dodge
    vignette_size: float = 0.5  # [0.0, 1.0]   midpoint of falloff gradient
    vignette_roundness: float = 0.0  # [0.0, 1.0]  0 = radial, 1 = rectangular (follows frame)
    carrier_width: float = 0.0  # [0.0, 5.0] mm on the print; 0 = off, filed-out negative carrier (black rebate frame)
    carrier_rough: float = 0.25  # [0.0, 1.0] inner-edge jitter
    border_size: float = 0.0  # [0.0, 10.0] cm
    border_color: str = "#ffffff"  # hex color
    border_bottom_weight: float = 1.0  # [1.0, 2.0] bottom border × top (window-mat weighting)
    border_match_paper: bool = False  # derive mat colour from toned paper white
