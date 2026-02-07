from abc import ABC, abstractmethod
import numpy as np
from negpy.features.exposure.normalization import LogNegativeBounds


class NormalizationHeuristic(ABC):
    @abstractmethod
    def calculate_bounds(self, img_log: np.ndarray, low_p: float, high_p: float) -> LogNegativeBounds:
        pass


class NegativeHeuristic(NormalizationHeuristic):
    """Standard independent channel normalization for C-41 and B&W."""

    def calculate_bounds(self, img_log: np.ndarray, low_p: float, high_p: float) -> LogNegativeBounds:
        floors = []
        ceils = []
        for ch in range(3):
            data = img_log[:, :, ch]
            f = np.percentile(data, low_p)
            c = np.percentile(data, high_p)
            floors.append(float(f))
            ceils.append(float(c))

        return LogNegativeBounds((floors[0], floors[1], floors[2]), (ceils[0], ceils[1], ceils[2]))


class SlideHeuristic(NormalizationHeuristic):
    """
    Specialized E-6 logic for Direct Positive processing.
    Expects img_log to be native log10(linear_raw).
    """

    def __init__(self, auto_stretch: bool = True):
        self.auto_stretch = auto_stretch

    def calculate_bounds(self, img_log: np.ndarray, low_p: float, high_p: float) -> LogNegativeBounds:
        raw_shadows = []
        raw_highlights = []
        for ch in range(3):
            data = img_log[:, :, ch]
            s = np.percentile(data, low_p)
            h = np.percentile(data, high_p)
            raw_shadows.append(float(s))
            raw_highlights.append(float(h))

        floors = (raw_highlights[0], raw_highlights[1], raw_highlights[2])

        if not self.auto_stretch:
            return LogNegativeBounds(floors, (floors[0] - 3.0, floors[1] - 3.0, floors[2] - 3.0))

        master_ceil = float(min(raw_shadows))
        linked_ceils = (master_ceil, master_ceil, master_ceil)

        return LogNegativeBounds(floors, linked_ceils)
