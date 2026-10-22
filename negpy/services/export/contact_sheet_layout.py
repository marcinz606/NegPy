"""Darkroom contact sheet geometry. Lengths in millimeters; "along" means along the horizontal strip."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from negpy.services.film_edge.geometry import FilmGeometry

MM_PER_INCH = 25.4

SHEET_MARGIN = 5.0
STRIP_GAP = 3.0
LABEL_BAND = 8.0
# Film Edges off: each row of upright frames sits on a black band, as on the Light Table.
ROW_BORDER = 0.8
ROW_GAP = 0.5
# A row grows to fill the paper's width only this far; past it, the row stays short.
MAX_STRETCH = 1.5
PAPER_MIN = 50.0
PAPER_MAX = 610.0

DPI_CHOICES: tuple[int, ...] = (150, 300, 600)
DEFAULT_DPI = 300
# Pillow's decompression-bomb warning limit.
MAX_SHEET_PIXELS = 89_000_000


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
class FramePlacement:
    """One upright frame in a row; `slot` indexes the plan's frames."""

    slot: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class RowPlacement:
    """(x, y, width, height) is the row's black band, frames included."""

    x: float
    y: float
    width: float
    height: float
    frames: tuple[FramePlacement, ...]


@dataclass(frozen=True)
class SheetPage:
    strips: tuple[StripPlacement, ...]
    block: tuple[float, float, float, float]
    rows: tuple[RowPlacement, ...] = ()

    def slots(self) -> list[int]:
        """The frame slots this page prints, in order."""
        if self.rows:
            return [frame.slot for row in self.rows for frame in row.frames]
        return [i for strip in self.strips for i in range(strip.first, strip.first + strip.count)]


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


def _cut_rows(
    aspects: Sequence[float], row_height: float, inner: float, max_height: float, breaks: Sequence[int]
) -> list[tuple[list[int], float]]:
    """(slots, height) per row. A row fills `inner` mm unless that stretches it past MAX_STRETCH; the last
    row and a row a break ends stay short. A frame in `breaks` starts a new row."""
    stops = {b for b in breaks if 0 < b < len(aspects)}
    rows: list[tuple[list[int], float]] = []
    slots: list[int] = []
    natural = 0.0

    def close(fill: bool) -> None:
        scale = (inner - ROW_GAP * (len(slots) - 1)) / natural
        if not fill or scale > MAX_STRETCH:
            scale = min(1.0, scale)
        rows.append((slots, row_height * min(scale, max_height / row_height)))

    for slot, aspect in enumerate(aspects):
        width = row_height * aspect
        if slots and (slot in stops or natural + width + ROW_GAP * len(slots) > inner):
            close(fill=slot not in stops)
            slots, natural = [], 0.0
        slots.append(slot)
        natural += width
    if slots:
        close(fill=False)
    return rows


def plan_rows(
    width: float,
    height: float,
    geometry: FilmGeometry,
    aspects: Sequence[float],
    label: bool = True,
    breaks: Sequence[int] = (),
) -> SheetPlan:
    """Film Edges off: each frame upright at its own aspect (width over height), in rows as tall as the
    film's frame. Rows run left to right and pages fill top to bottom."""
    band = LABEL_BAND if label else 0.0
    inner = width - 2 * SHEET_MARGIN - 2 * ROW_BORDER
    room = height - 2 * SHEET_MARGIN - band - 2 * ROW_BORDER
    row_height = geometry.frame_across
    if inner <= 0 or room < row_height:
        reason = "The paper is too narrow for one frame." if inner <= 0 else "The paper is too short for one row."
        return SheetPlan(width, height, geometry, len(aspects), 0, 0, (), label, reason)
    if not aspects:
        return SheetPlan(width, height, geometry, 0, 0, 0, (), label, "No frames to print.")

    paged: list[list[tuple[list[int], float]]] = [[]]
    used = 0.0
    for slots, h in _cut_rows(aspects, row_height, inner, room, breaks):
        need = h + 2 * ROW_BORDER + (STRIP_GAP if paged[-1] else 0.0)
        if paged[-1] and used + need > room + 2 * ROW_BORDER:
            paged.append([])
            need = h + 2 * ROW_BORDER
            used = 0.0
        paged[-1].append((slots, h))
        used += need

    pages: list[SheetPage] = []
    for rows in paged:
        widths = [sum(h * aspects[i] for i in slots) + ROW_GAP * (len(slots) - 1) + 2 * ROW_BORDER for slots, h in rows]
        block_w = max(widths)
        block_h = sum(h + 2 * ROW_BORDER for _slots, h in rows) + STRIP_GAP * (len(rows) - 1)
        left = (width - block_w) / 2
        y = max((height - block_h) / 2, SHEET_MARGIN + band)
        placed = []
        for (slots, h), row_w in zip(rows, widths):
            x = left + ROW_BORDER
            frames = []
            for slot in slots:
                frames.append(FramePlacement(slot, x, y + ROW_BORDER, h * aspects[slot], h))
                x += h * aspects[slot] + ROW_GAP
            placed.append(RowPlacement(left, y, row_w, h + 2 * ROW_BORDER, tuple(frames)))
            y += h + 2 * ROW_BORDER + STRIP_GAP
        pages.append(SheetPage((), (left, placed[0].y, block_w, block_h), tuple(placed)))
    per_row = max(len(row.frames) for page in pages for row in page.rows)
    rows_per_page = max(len(page.rows) for page in pages)
    return SheetPlan(width, height, geometry, len(aspects), per_row, rows_per_page, tuple(pages), label)


def better_orientation(
    width: float,
    height: float,
    geometry: FilmGeometry,
    frame_count: int,
    label: bool = True,
    aspects: Optional[Sequence[float]] = None,
) -> tuple[float, float]:
    """The paper turned the way that needs fewer sheets; `aspects` plans rows (Film Edges off)."""

    def sheets(w: float, h: float) -> float:
        if aspects is not None:
            plan = plan_rows(w, h, geometry, aspects, label)
            return len(plan.pages) if plan.pages else math.inf
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
    film_edges: bool = True  # off prints the frames upright in rows, as the Light Table shows them

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
            film_edges=flag("film_edges", default.film_edges),
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
            "film_edges": self.film_edges,
        }
