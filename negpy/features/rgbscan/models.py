from dataclasses import dataclass


@dataclass(frozen=True)
class RgbScanConfig:
    """Trichromatic (narrowband RGB) capture: one frame assembled from three exposures.

    The red exposure is the primary source file (the asset itself); the green and
    blue exposures ride along here, the same way the flat-field reference does.
    """

    enabled: bool = False
    green_path: str = ""
    blue_path: str = ""
    align: bool = True  # sub-pixel registration of green/blue to the red exposure
