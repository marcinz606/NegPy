from typing import Tuple, List
import numpy as np
from numba import njit, prange  # type: ignore
from src.core.types import ImageBuffer
from src.core.performance import time_function
from src.core.validation import ensure_image


@njit(parallel=True)
def _normalize_log_image_jit(
    img_log: np.ndarray, floors: np.ndarray, ceils: np.ndarray
) -> np.ndarray:
    """
    Fast JIT normalization of log images.
    """
    h, w, c = img_log.shape
    res = np.empty_like(img_log)
    epsilon = 1e-6

    for y in prange(h):
        for x in range(w):
            for ch in range(3):
                f = floors[ch]
                c_val = ceils[ch]
                norm = (img_log[y, x, ch] - f) / (max(c_val - f, epsilon))
                if norm < 0.0:
                    norm = 0.0
                elif norm > 1.0:
                    norm = 1.0
                res[y, x, ch] = norm
    return res


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


@time_function
def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Normalizes log image using the provided bounds.
    """
    floors = np.array(bounds.floors, dtype=np.float32)
    ceils = np.array(bounds.ceils, dtype=np.float32)

    return ensure_image(
        _normalize_log_image_jit(img_log.astype(np.float32), floors, ceils)
    )
