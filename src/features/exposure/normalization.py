from typing import Tuple, List
import numpy as np
from numba import njit, prange  # type: ignore
from src.domain.types import ImageBuffer
from src.kernel.system.performance import time_function
from src.kernel.image.validation import ensure_image


@njit(parallel=True, cache=True, fastmath=True)
def _normalize_log_image_jit(
    img_log: np.ndarray, floors: np.ndarray, ceils: np.ndarray
) -> np.ndarray:
    """
    Fast JIT normalization of log-exposure images to a 0.0-1.0 range.
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
    """
    Represents the sensitometric boundaries (black point and white point)
    of a negative emulsion in log-exposure space.
    """

    def __init__(
        self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]
    ):
        self.floors = floors
        self.ceils = ceils


def measure_log_negative_bounds(img: ImageBuffer) -> LogNegativeBounds:
    """
    Finds the robust floor (D-min) and ceiling (D-max) of each emulsion layer.

    This function analyzes the image histogram in log-exposure space to identify
    the usable dynamic range of the latent image. The 'floors' correspond to
    the film base plus fog (B+F), while 'ceils' capture the densest highlights.
    """
    floors: List[float] = []
    ceils: List[float] = []
    for ch in range(3):
        # 0.25th and 99.75th percentiles capture the usable density range
        # but avoiding clipping
        f, c = np.percentile(img[:, :, ch], [0.5, 99.5])
        floors.append(float(f))
        ceils.append(float(c))

    return LogNegativeBounds(
        floors=(floors[0], floors[1], floors[2]),
        ceils=(ceils[0], ceils[1], ceils[2]),
    )


@time_function
def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Normalizes a log-exposure image using the measured sensitometric bounds.

    This ensures that the latent image is correctly 'framed' within the dynamic
    range of the subsequent characteristic curve processing.
    """
    floors = np.ascontiguousarray(np.array(bounds.floors, dtype=np.float32))
    ceils = np.ascontiguousarray(np.array(bounds.ceils, dtype=np.float32))

    return ensure_image(
        _normalize_log_image_jit(
            np.ascontiguousarray(img_log.astype(np.float32)), floors, ceils
        )
    )
