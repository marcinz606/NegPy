import numpy as np
from typing import Tuple, List
from src.core.types import ImageBuffer


class LogNegativeBounds:
    def __init__(
        self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]
    ):
        self.floors = floors
        self.ceils = ceils


def measure_log_negative_bounds(img: ImageBuffer) -> LogNegativeBounds:
    """
    Finds the robust floor and ceiling of each channel in Log10 space.
    Input should be Log10(Linear + epsilon).
    """
    floors: List[float] = []
    ceils: List[float] = []
    for ch in range(3):
        # 1st and 99.5th percentiles capture the usable density range
        f, c = np.percentile(img[:, :, ch], [1.0, 99.5])
        floors.append(float(f))
        ceils.append(float(c))

    return LogNegativeBounds(
        floors=(floors[0], floors[1], floors[2]),
        ceils=(ceils[0], ceils[1], ceils[2]),
    )


def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Normalizes log image using the provided bounds.
    """
    res = np.zeros_like(img_log)
    epsilon = 1e-6
    for ch in range(3):
        f, c = bounds.floors[ch], bounds.ceils[ch]
        # (val - min) / (max - min)
        res[:, :, ch] = np.clip((img_log[:, :, ch] - f) / (max(c - f, epsilon)), 0, 1)
    return res
