"""Focus meter for the camera live view: frame sharpness against the best seen so far."""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Mean level (0..255) below which a frame carries no usable contrast: the light is off.
MIN_MEAN_LEVEL = 4.0
#: Weight of the newest frame in the running reading.
SMOOTHING = 0.3


def sharpness(gray: np.ndarray) -> float:
    """Mean squared Laplacian over mean level squared, so a light change does not read as focus.

    0.0 for a frame too dark or too small.
    """
    img = np.asarray(gray, dtype=np.float32)
    if img.ndim != 2 or min(img.shape) < 3:
        return 0.0
    mean = float(img.mean())
    if mean < MIN_MEAN_LEVEL:
        return 0.0
    lap = 4.0 * img[1:-1, 1:-1] - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]
    return float(np.mean(lap * lap)) / (mean * mean)


class FocusMeter:
    """Smoothed sharpness as a fraction of its peak since the last reset."""

    def __init__(self) -> None:
        self._value: Optional[float] = None
        self._peak = 0.0

    def reset(self) -> None:
        self._value = None
        self._peak = 0.0

    def update(self, gray: np.ndarray) -> Optional[float]:
        """Feed one frame. Returns the reading in 0..1 of the peak, or None for a dark frame."""
        score = sharpness(gray)
        if score <= 0.0:
            return None
        self._value = score if self._value is None else self._value + SMOOTHING * (score - self._value)
        self._peak = max(self._peak, self._value)
        return self._value / self._peak
