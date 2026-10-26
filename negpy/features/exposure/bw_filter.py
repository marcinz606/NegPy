from dataclasses import replace
from typing import Optional

import numpy as np

from negpy.features.exposure.models import BwFilter, ExposureConfig
from negpy.features.process.models import ProcessConfig, ProcessMode

# R, G, B weights on the normalized log channels of a color negative: a least-squares fit of a
# panchromatic film (Tri-X/HP5-like) behind each Wratten filter, on Portra 400's layer
# sensitivities under 5500 K. Each row sums to exactly 1, so a gray does not move.
# TODO: the #11 and #58 rows come from modeled curves; check them against Kodak publication B-3.
BW_FILTER_WEIGHTS: dict[BwFilter, tuple[float, float, float]] = {
    BwFilter.NONE: (0.31, 0.39, 0.30),
    BwFilter.YELLOW_8: (0.47, 0.57, -0.04),
    BwFilter.DEEP_YELLOW_15: (0.56, 0.50, -0.06),
    BwFilter.ORANGE_21: (0.72, 0.32, -0.04),
    BwFilter.RED_25: (1.06, -0.06, 0.00),
    BwFilter.YELLOW_GREEN_11: (0.25, 0.78, -0.03),
    BwFilter.GREEN_58: (0.04, 1.08, -0.12),
}


def collapse_weights(process: ProcessConfig, exposure: ExposureConfig) -> tuple[float, float, float]:
    """The B&W print's channel weights. A filter acts only on a color source, where the UI shows it."""
    if process.process_mode == ProcessMode.BW and process.source_color:
        return BW_FILTER_WEIGHTS[exposure.bw_filter]
    return BW_FILTER_WEIGHTS[BwFilter.NONE]


def collapse_matrix(weights: tuple[float, float, float]) -> np.ndarray:
    """3x3 float32 with every row = `weights`: matmul_3x3 then gives three equal channels."""
    return np.tile(np.asarray(weights, dtype=np.float32), (3, 1))


def plane_weights(process: ProcessConfig, exposure: ExposureConfig) -> Optional[tuple[float, float, float]]:
    """The weights the mask and diffusion planes collapse with, or None outside B&W."""
    return collapse_weights(process, exposure) if process.process_mode == ProcessMode.BW else None


_COLOR_FILTRATION = tuple(f"{zone}_{dye}" for zone in ("wb", "shadow", "highlight") for dye in ("cyan", "magenta", "yellow"))


def printed_exposure(process: ProcessConfig, exposure: ExposureConfig) -> ExposureConfig:
    """The exposure the print renders. B&W drops the color filtration, which the Filtration card hides there."""
    if process.process_mode != ProcessMode.BW:
        return exposure
    return replace(exposure, **dict.fromkeys(_COLOR_FILTRATION, 0.0))
