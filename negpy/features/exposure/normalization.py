from typing import Tuple, Optional
import numpy as np
from numba import njit, prange  # type: ignore
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image
from negpy.features.process.models import ProcessMode


@njit(parallel=True, cache=True, fastmath=True)
def _normalize_log_image_jit(img_log: np.ndarray, floors: np.ndarray, ceils: np.ndarray) -> np.ndarray:
    """
    Log -> 0.0-1.0 (Linear stretch).
    Supports both f < c (Negative) and f > c (Positive) mapping.
    """
    h, w, c = img_log.shape
    res = np.empty_like(img_log)
    epsilon = 1e-6

    for y in prange(h):
        for x in range(w):
            for ch in range(3):
                f = floors[ch]
                c_val = ceils[ch]
                delta = c_val - f

                denom = delta
                if abs(delta) < epsilon:
                    if delta >= 0:
                        denom = epsilon
                    else:
                        denom = -epsilon

                norm = (img_log[y, x, ch] - f) / denom
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

    def __init__(self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]):
        self.floors = floors
        self.ceils = ceils


def get_analysis_crop(img: ImageBuffer, buffer_ratio: float) -> ImageBuffer:
    """
    Returns a center crop of the image for analysis purposes.
    The buffer_ratio (0.0 to 0.5) defines how much of the border to exclude.
    """
    if buffer_ratio <= 0:
        return img

    h, w = img.shape[:2]
    safe_buffer = min(max(buffer_ratio, 0.0), 0.3)

    cut_h = int(h * safe_buffer)
    cut_w = int(w * safe_buffer)

    return img[cut_h : h - cut_h, cut_w : w - cut_w]


def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Stretches log-data to fit [0, 1].
    """
    floors = np.ascontiguousarray(np.array(bounds.floors, dtype=np.float32))
    ceils = np.ascontiguousarray(np.array(bounds.ceils, dtype=np.float32))

    return ensure_image(_normalize_log_image_jit(np.ascontiguousarray(img_log.astype(np.float32)), floors, ceils))


def analyze_log_exposure_bounds(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    process_mode: str = ProcessMode.C41,
    e6_normalize: bool = True,
) -> LogNegativeBounds:
    """
    Performs full analysis pass on a linear image to find density floors/ceils.
    """
    epsilon = 1e-6
    img_log = np.log10(np.clip(image, epsilon, 1.0))

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]

    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    p_low, p_high = 0.5, 99.5
    fixed_range = 3.0

    if process_mode == ProcessMode.E6:
        p_low, p_high = 99.9, 0.01
        fixed_range = -3.0

    floors = []
    ceils = []
    for ch in range(3):
        data = img_log[:, :, ch]
        f = np.percentile(data, p_low)
        floors.append(float(f))

        if process_mode != ProcessMode.E6 or e6_normalize:
            c = np.percentile(data, p_high)
            ceils.append(float(c))
        else:
            ceils.append(float(f + fixed_range))

    return LogNegativeBounds(
        (floors[0], floors[1], floors[2]),
        (ceils[0], ceils[1], ceils[2]),
    )
