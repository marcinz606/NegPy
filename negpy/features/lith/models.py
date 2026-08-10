from dataclasses import dataclass


@dataclass(frozen=True)
class LithConfig:
    """
    Lith development params. Paper comes from the Exposure panel's profile.
    """

    lith_enabled: bool = False
    lith_exposure: float = 2.0
    lith_snatch: float = 0.55
    lith_abruptness: float = 0.6
