from typing import Tuple, List
import numpy as np
from numba import njit, prange  # type: ignore
from src.domain.types import ImageBuffer
from src.kernel.image.validation import ensure_image


@njit(parallel=True, cache=True, fastmath=True)
def _normalize_log_image_jit(
    img_log: np.ndarray, floors: np.ndarray, ceils: np.ndarray
) -> np.ndarray:
    """
    Log -> 0.0-1.0 (Linear stretch).
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
    D-min / D-max container.
    """

    def __init__(
        self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]
    ):
        self.floors = floors
        self.ceils = ceils


def measure_log_negative_bounds(img: ImageBuffer) -> LogNegativeBounds:
    """
    Detects floor/ceiling (0.5% - 99.5%).
    """
    floors: List[float] = []
    ceils: List[float] = []
    for ch in range(3):
        # 0.5th and 99.5th percentiles capture the usable density range
        # but avoiding clipping
        f, c = np.percentile(img[:, :, ch], [0.5, 99.5])
        floors.append(float(f))
        ceils.append(float(c))

    return LogNegativeBounds(
        floors=(floors[0], floors[1], floors[2]),
        ceils=(ceils[0], ceils[1], ceils[2]),
    )


def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Stretches log-data to fit [0, 1].
    """
    floors = np.ascontiguousarray(np.array(bounds.floors, dtype=np.float32))
    ceils = np.ascontiguousarray(np.array(bounds.ceils, dtype=np.float32))

    return ensure_image(
        _normalize_log_image_jit(
            np.ascontiguousarray(img_log.astype(np.float32)), floors, ceils
        )
    )
