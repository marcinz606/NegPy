"""Film strip geometry in millimeters; "along" means along the strip."""

import math
from dataclasses import dataclass
from enum import StrEnum

# 135 (ISO 1007, Kodak KS perforation).
FILM_135_WIDTH = 35.0
FRAME_135_PITCH = 38.0
HALF_FRAME_PITCH = 19.0
PERF_ALONG = 1.98
PERF_ACROSS = 2.80
PERF_RADIUS = 0.5
PERF_FROM_EDGE = 2.0
PERF_PITCH = 4.75
# Half a pitch puts every cut, at a frame-gap center, between two holes.
PERF_PHASE = PERF_PITCH / 2
EDGE_BAND_135 = 2.0

# 120 (ISO 732). FRAME_GAP_120 is typical; the camera sets the real gap.
FILM_120_WIDTH = 61.0
FRAME_120_ACROSS = 56.0
FRAME_GAP_120 = 4.0
FRAME_SIZES_120: dict[str, float] = {
    "6×4.5": 41.5,
    "6×6": 56.0,
    "6×7": 70.0,
    "6×8": 77.0,
    "6×9": 84.0,
    "6×12": 112.0,
    "6×17": 168.0,
}
DEFAULT_FRAME_120 = "6×6"


class SheetFormat(StrEnum):
    FULL_FRAME = "35mm"
    HALF_FRAME = "35mm Half Frame"
    MEDIUM = "120"


@dataclass(frozen=True)
class FilmGeometry:
    format: SheetFormat
    width: float
    frame_along: float
    frame_across: float
    pitch: float
    frame_size: str = ""

    @property
    def perforated(self) -> bool:
        return self.format != SheetFormat.MEDIUM

    @property
    def edge_band(self) -> float:
        """Rebate between a film edge and the perforations (135) or the image (120)."""
        if self.perforated:
            return EDGE_BAND_135
        return (self.width - self.frame_across) / 2

    @property
    def image_top(self) -> float:
        return (self.width - self.frame_across) / 2

    @property
    def window_aspect(self) -> float:
        return self.frame_along / self.frame_across

    def frame_center(self, index: int) -> float:
        return self.pitch * (index + 0.5)


def film_geometry(fmt: SheetFormat, frame_size: str = DEFAULT_FRAME_120) -> FilmGeometry:
    if fmt == SheetFormat.HALF_FRAME:
        return FilmGeometry(fmt, FILM_135_WIDTH, 18.0, 24.0, HALF_FRAME_PITCH)
    if fmt == SheetFormat.MEDIUM:
        size = frame_size if frame_size in FRAME_SIZES_120 else DEFAULT_FRAME_120
        along = FRAME_SIZES_120[size]
        return FilmGeometry(fmt, FILM_120_WIDTH, along, FRAME_120_ACROSS, along + FRAME_GAP_120, size)
    return FilmGeometry(SheetFormat.FULL_FRAME, FILM_135_WIDTH, 36.0, 24.0, FRAME_135_PITCH)


def perforation_centers(x0: float, x1: float) -> list[float]:
    first = math.ceil((x0 - PERF_PHASE) / PERF_PITCH)
    last = math.floor((x1 - PERF_PHASE) / PERF_PITCH)
    return [PERF_PHASE + j * PERF_PITCH for j in range(first, last + 1)]
