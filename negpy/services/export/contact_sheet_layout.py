"""Darkroom contact sheet geometry. Lengths in millimeters; "along" means along the horizontal strip."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Optional, Sequence

MM_PER_INCH = 25.4

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

SHEET_MARGIN = 5.0
STRIP_GAP = 3.0
LABEL_BAND = 8.0
PAPER_MIN = 50.0
PAPER_MAX = 610.0

DPI_CHOICES: tuple[int, ...] = (150, 300, 600)
DEFAULT_DPI = 300
# Pillow's decompression-bomb warning limit.
MAX_SHEET_PIXELS = 89_000_000


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


@dataclass(frozen=True)
class PaperSize:
    label: str
    width: float
    height: float


# Ilford Multigrade sheets, portrait, at their true inch sizes: the cm labels are rounded.
ILFORD_PAPERS: tuple[PaperSize, ...] = (
    PaperSize("12.7 × 17.8 cm", 127.0, 177.8),
    PaperSize("16.5 × 21.6 cm", 165.1, 215.9),
    PaperSize("17.8 × 24 cm", 177.8, 241.3),
    PaperSize("20.3 × 25.4 cm", 203.2, 254.0),
    PaperSize("21 × 29.7 cm", 210.0, 297.0),
    PaperSize("21.6 × 27.9 cm", 215.9, 279.4),
    PaperSize("24 × 30.5 cm", 241.3, 304.8),
    PaperSize("27.9 × 35.6 cm", 279.4, 355.6),
    PaperSize("30.5 × 40.6 cm", 304.8, 406.4),
    PaperSize("40.6 × 50.8 cm", 406.4, 508.0),
    PaperSize("50.8 × 61 cm", 508.0, 609.6),
)
# The smallest sheet that holds a whole 36-exposure roll (37-38 frames).
DEFAULT_PAPER = ILFORD_PAPERS[6]


def paper_preset(width: float, height: float, tolerance: float = 0.05) -> Optional[PaperSize]:
    for paper in ILFORD_PAPERS:
        for w, h in ((paper.width, paper.height), (paper.height, paper.width)):
            if abs(width - w) <= tolerance and abs(height - h) <= tolerance:
                return paper
    return None


def snap_paper(width: float, height: float, tolerance: float = 2.0) -> tuple[float, float]:
    for paper in ILFORD_PAPERS:
        for w, h in ((paper.width, paper.height), (paper.height, paper.width)):
            if abs(width - w) <= tolerance and abs(height - h) <= tolerance:
                return w, h
    return float(round(width)), float(round(height))


def min_paper_size(geometry: FilmGeometry, label: bool = True) -> tuple[float, float]:
    band = LABEL_BAND if label else 0.0
    return geometry.pitch + 2 * SHEET_MARGIN, geometry.width + 2 * SHEET_MARGIN + band


def clamp_paper(width: float, height: float, geometry: FilmGeometry, label: bool = True) -> tuple[float, float]:
    min_w, min_h = min_paper_size(geometry, label)
    return (
        min(PAPER_MAX, max(PAPER_MIN, min_w, width)),
        min(PAPER_MAX, max(PAPER_MIN, min_h, height)),
    )


def sheet_pixels(width: float, height: float, dpi: int) -> tuple[int, int]:
    return round(width / MM_PER_INCH * dpi), round(height / MM_PER_INCH * dpi)


def dpi_allowed(width: float, height: float, dpi: int) -> bool:
    px_w, px_h = sheet_pixels(width, height, dpi)
    return px_w * px_h <= MAX_SHEET_PIXELS


def best_dpi(width: float, height: float, wanted: int) -> int:
    for dpi in sorted(DPI_CHOICES, reverse=True):
        if dpi <= wanted and dpi_allowed(width, height, dpi):
            return dpi
    return min(DPI_CHOICES)


@dataclass(frozen=True)
class StripPlacement:
    """(x, y) is the strip's top-left corner on the paper."""

    x: float
    y: float
    first: int
    count: int
    roll_start: float
    length: float


@dataclass(frozen=True)
class SheetPage:
    strips: tuple[StripPlacement, ...]
    block: tuple[float, float, float, float]


@dataclass(frozen=True)
class SheetPlan:
    paper_width: float
    paper_height: float
    geometry: FilmGeometry
    frame_count: int
    frames_per_strip: int
    strips_per_sheet: int
    pages: tuple[SheetPage, ...]
    label: bool = True
    reason: str = ""

    @property
    def capacity(self) -> int:
        return self.frames_per_strip * self.strips_per_sheet


def plan_capacity(width: float, height: float, geometry: FilmGeometry, label: bool = True) -> tuple[int, int]:
    """(frames per strip, strips per sheet) for this paper."""
    band = LABEL_BAND if label else 0.0
    per_strip = math.floor((width - 2 * SHEET_MARGIN) / geometry.pitch + 1e-9)
    strips = math.floor((height - 2 * SHEET_MARGIN - band + STRIP_GAP) / (geometry.width + STRIP_GAP) + 1e-9)
    return max(0, per_strip), max(0, strips)


def cut_strips(frame_count: int, per_strip: int, breaks: Sequence[int] = ()) -> list[tuple[int, int]]:
    """(first frame, count) per strip; a frame in `breaks` starts a new strip."""
    stops = sorted({b for b in breaks if 0 < b < frame_count}) + [frame_count]
    strips: list[tuple[int, int]] = []
    first = 0
    for stop in stops:
        while first < stop:
            count = min(per_strip, stop - first)
            strips.append((first, count))
            first += count
    return strips


def plan_sheets(
    width: float,
    height: float,
    geometry: FilmGeometry,
    frame_count: int,
    label: bool = True,
    breaks: Sequence[int] = (),
) -> SheetPlan:
    per_strip, strips = plan_capacity(width, height, geometry, label)
    if per_strip == 0 or strips == 0:
        reason = "The paper is too narrow for one frame." if per_strip == 0 else "The paper is too short for one strip."
        return SheetPlan(width, height, geometry, frame_count, per_strip, strips, (), label, reason)
    if frame_count <= 0:
        return SheetPlan(width, height, geometry, frame_count, per_strip, strips, (), label, "No frames to print.")

    band = LABEL_BAND if label else 0.0
    cut = cut_strips(frame_count, per_strip, breaks)
    pages: list[SheetPage] = []
    for page_start in range(0, len(cut), strips):
        rows = cut[page_start : page_start + strips]
        block_w = max(count for _first, count in rows) * geometry.pitch
        block_h = len(rows) * geometry.width + (len(rows) - 1) * STRIP_GAP
        left = (width - block_w) / 2
        top = max((height - block_h) / 2, SHEET_MARGIN + band)
        placed = tuple(
            StripPlacement(left, top + row * (geometry.width + STRIP_GAP), first, count, first * geometry.pitch, count * geometry.pitch)
            for row, (first, count) in enumerate(rows)
        )
        pages.append(SheetPage(placed, (left, top, block_w, block_h)))
    return SheetPlan(width, height, geometry, frame_count, per_strip, strips, tuple(pages), label)


def better_orientation(width: float, height: float, geometry: FilmGeometry, frame_count: int, label: bool = True) -> tuple[float, float]:
    def sheets(w: float, h: float) -> float:
        per_strip, strips = plan_capacity(w, h, geometry, label)
        capacity = per_strip * strips
        return math.inf if capacity == 0 else math.ceil(max(frame_count, 1) / capacity)

    return (height, width) if sheets(height, width) < sheets(width, height) else (width, height)


def frame_turns(upright_aspect: Optional[float], geometry: FilmGeometry, rotation: int, flip_h: bool, flip_v: bool) -> int:
    """`np.rot90` k that lays an upright frame on the strip as it lies on the film.

    A frame scanned upright turns clockwise, as a vertical shot held release-up lies on the film."""
    if upright_aspect is None or abs(math.log(upright_aspect)) < 0.02 or abs(math.log(geometry.window_aspect)) < 0.02:
        return 0
    if (upright_aspect > 1) == (geometry.window_aspect > 1):
        return 0
    k = rotation % 4
    if k % 2:
        return (k if flip_h != flip_v else -k) % 4
    return 3


@dataclass(frozen=True)
class ContactSheetSettings:
    """The film format is not stored: it is read from the frames each time."""

    paper_width: float = DEFAULT_PAPER.width
    paper_height: float = DEFAULT_PAPER.height
    dpi: int = DEFAULT_DPI
    roll_label: bool = True
    edge_print: bool = True
    by_scene: bool = False
    white_paper: bool = False
    film_base: bool = True  # on white paper, print the film base black; off prints it as paper

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ContactSheetSettings":
        if not isinstance(data, Mapping):
            return cls()
        default = cls()

        def length(key: str, fallback: float) -> float:
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return fallback
            return min(PAPER_MAX, max(PAPER_MIN, float(value)))

        def flag(key: str, fallback: bool) -> bool:
            value = data.get(key)
            return value if isinstance(value, bool) else fallback

        dpi = data.get("dpi")
        return cls(
            paper_width=length("paper_width", default.paper_width),
            paper_height=length("paper_height", default.paper_height),
            dpi=dpi if dpi in DPI_CHOICES and not isinstance(dpi, bool) else default.dpi,
            roll_label=flag("roll_label", default.roll_label),
            edge_print=flag("edge_print", default.edge_print),
            by_scene=flag("by_scene", default.by_scene),
            white_paper=flag("white_paper", default.white_paper),
            film_base=flag("film_base", default.film_base),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_width": self.paper_width,
            "paper_height": self.paper_height,
            "dpi": self.dpi,
            "roll_label": self.roll_label,
            "edge_print": self.edge_print,
            "by_scene": self.by_scene,
            "white_paper": self.white_paper,
            "film_base": self.film_base,
        }
