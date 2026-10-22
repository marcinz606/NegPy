"""Maker edge print layouts, in roll millimeters. A new maker style is one `LAYOUTS` entry."""

from dataclasses import dataclass
from enum import StrEnum

from negpy.services.film_edge.barcodes import KEYKODE_MODULE, KEYKODE_MODULES
from negpy.services.film_edge.geometry import PERF_PHASE, PERF_PITCH

# A mark sits this many cap heights clear of its text.
MARK_GAP = 0.3
# SMPTE ST 254 zero-frame dot, 0.64-0.76 mm across.
KEY_DOT = 0.7


class Band(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class EdgeRun:
    """One repeating line of edge print.

    `on`: "number" at each frame number's label, "half" at each A label (135 only), or "pitch" every `pitch` mm from
    `at`; a label run prints where {n} % `every` == `phase`. `text` is a template over {n} {label} {maker} {text}
    {code} {roll} {batch} {lot}; an item that renders empty is not printed.
    `lead`, `trail` and `under` name a mark: "triangle", "wedge", "outline", "arrow", "dots" or "dot". `char_width` sets a
    monospaced advance per character. `barcode` ("keykode", "keykode+32") follows each item, `barcode_at` mm on;
    "index" is a frame-index bar centered on the item's position."""

    on: str
    band: Band
    text: str
    cap: float
    offset: float = 1.0
    at: float = 0.0
    pitch: float = 0.0
    count: int = 0  # 0: to the roll end
    first: int = 1  # {n} of the first "pitch" item
    align: str = "center"
    max_width: float = 0.0
    lead: str = ""
    trail: str = ""
    under: str = ""
    char_width: float = 0.0
    barcode: str = ""
    barcode_at: float = 0.0
    every: int = 1
    phase: int = 0


@dataclass(frozen=True)
class EdgeLayout:
    runs_135: tuple[EdgeRun, ...]
    runs_120: tuple[EdgeRun, ...]
    # Palettes ("color", "bw") whose 135 edge carries the DX code; slides never do.
    dx: tuple[str, ...] = ("color",)


TOP, BOTTOM = Band.TOP, Band.BOTTOM
# 120 numbers run at the maker's own pitch, not in step with the camera's frames.
_CAP_120, _OFFSET_120 = 1.8, 1.3


def _numbers_120(lead: str) -> tuple[EdgeRun, ...]:
    return (
        EdgeRun("pitch", TOP, "{maker} {text}", _CAP_120, _OFFSET_120, at=30.0, pitch=86.0, count=19, max_width=60.0),
        EdgeRun("pitch", BOTTOM, "{n}", _CAP_120, _OFFSET_120, at=8.0, pitch=43.0, count=19, lead=lead),
    )


# SMPTE ST 254 KEYKODE: a key every 64 perforations (one foot) and a smaller "+32" key half a foot on, each ending in a
# zero-frame dot centered on a perforation; the barcode after it ends 6 perforations past the dot.
_FOOT = 64 * PERF_PITCH
_KEY_DOT_X = PERF_PHASE + 6 * PERF_PITCH
_DOT_TO_BAR = 6 * PERF_PITCH - KEYKODE_MODULES * KEYKODE_MODULE


def _key(text: str, cap: float, char_width: float, dot_x: float, barcode: str) -> EdgeRun:
    text_to_dot = MARK_GAP * cap + KEY_DOT / 2
    return EdgeRun(
        "pitch",
        BOTTOM,
        text,
        cap,
        at=dot_x - text_to_dot,
        pitch=_FOOT,
        first=1000,
        align="right",
        trail="dot",
        char_width=char_width,
        barcode=barcode,
        barcode_at=text_to_dot + _DOT_TO_BAR,
    )


_KEYKODE_135 = (
    _key("{code} {roll} {n:04d}", 1.52, 1.52, _KEY_DOT_X, "keykode"),
    _key("{code} {roll} {n:04d}+32", 1.0, 0.68, _KEY_DOT_X + _FOOT / 2, "keykode+32"),
    # Frame-index bars every 4 perforations of each half foot, skipping the ones under a key and its barcode.
    *(EdgeRun("pitch", BOTTOM, "", 1.0, at=_KEY_DOT_X + m * PERF_PITCH, pitch=_FOOT / 2, barcode="index") for m in (8, 12, 16, 20, 24)),
)

_KODAK_LABELS_135 = (
    EdgeRun("number", BOTTOM, "{label}", 1.4),
    EdgeRun("half", BOTTOM, "{label}", 0.9, offset=1.4, under="arrow"),
)
_KODAK_TOP_135 = (
    EdgeRun("number", TOP, "{n}", 1.2),
    EdgeRun("number", TOP, "{maker} {text}", 1.25, at=15.5, max_width=26.0),
)
_KODAK_120 = (
    EdgeRun("pitch", TOP, "{n}", _CAP_120, _OFFSET_120, at=8.0, pitch=44.5, count=15, first=41),
    EdgeRun("pitch", TOP, "{maker} {text}", _CAP_120, _OFFSET_120, at=30.25, pitch=89.0, count=8, max_width=36.0),
    EdgeRun("pitch", BOTTOM, "{n}", _CAP_120, _OFFSET_120, at=20.0, pitch=60.0, count=12, trail="triangle"),
)
# The name once a frame near its number, and the emulsion batch ("5048-12", the lot smaller) every third frame.
_ILFORD_BATCH = (
    EdgeRun("number", TOP, "{batch}", 1.0, at=20.0, align="right", every=3, phase=2),
    EdgeRun("number", TOP, "-{lot}", 0.7, offset=1.15, at=20.05, align="left", every=3, phase=2),
)
_ILFORD_135 = (
    EdgeRun("number", BOTTOM, "{label}", 1.5),
    EdgeRun("half", BOTTOM, "{label}", 1.1, lead="triangle"),
    EdgeRun("number", TOP, "{maker} {text}", 1.5, at=2.5),
    *_ILFORD_BATCH,
)
# A monospaced name, numbers near the film edge, a long wedge before each A number.
_KENTMERE_135 = (
    EdgeRun("number", BOTTOM, "{label}", 1.55, 0.85),
    EdgeRun("half", BOTTOM, "{label}", 1.05, 0.85, at=1.0, lead="wedge"),
    EdgeRun("number", TOP, "{maker} {text}", 1.45, 0.95, at=2.5, char_width=1.32),
    *_ILFORD_BATCH,
)

LAYOUTS: dict[str, EdgeLayout] = {
    "kodak": EdgeLayout(runs_135=(*_KODAK_LABELS_135, *_KODAK_TOP_135), runs_120=_KODAK_120),
    # Kodak Gold: the name's word order swaps every frame ("KODAK GB 200-7", "GB 200-7 KODAK").
    "kodak-alternating": EdgeLayout(
        runs_135=(
            *_KODAK_LABELS_135,
            _KODAK_TOP_135[0],
            EdgeRun("number", TOP, "{maker} {text}", 1.25, at=15.5, max_width=26.0, every=2, phase=0),
            EdgeRun("number", TOP, "{text} {maker}", 1.25, at=15.5, max_width=26.0, every=2, phase=1),
        ),
        runs_120=_KODAK_120,
    ),
    "fuji": EdgeLayout(
        runs_135=(
            EdgeRun("number", BOTTOM, "{label}", 1.5),
            EdgeRun("half", BOTTOM, "{label}", 1.5, lead="arrow"),
            EdgeRun("number", TOP, "{n}", 1.5),
            EdgeRun("number", TOP, "{code}", 1.0, at=9.5, max_width=7.0),
            EdgeRun("number", TOP, "{n}A", 1.5, at=19.0),
            EdgeRun("number", TOP, "FUJI", 1.0, at=28.5),
        ),
        runs_120=(
            EdgeRun("pitch", TOP, "FUJI {code}", _CAP_120, _OFFSET_120, at=30.0, pitch=84.0, count=19, max_width=50.0),
            EdgeRun("pitch", BOTTOM, "{code}", _CAP_120, _OFFSET_120, at=6.4, pitch=42.0, count=19, align="right", max_width=24.0),
            EdgeRun("pitch", BOTTOM, "{n}", _CAP_120, _OFFSET_120, at=8.0, pitch=42.0, count=19, align="left", lead="outline"),
        ),
    ),
    "ilford": EdgeLayout(runs_135=_ILFORD_135, runs_120=_numbers_120("dots"), dx=("color", "bw")),
    # Ilford's batch numbers, no DX barcode.
    "kentmere": EdgeLayout(runs_135=_KENTMERE_135, runs_120=_numbers_120("dots"), dx=()),
    "cine": EdgeLayout(
        runs_135=_KEYKODE_135,
        runs_120=_numbers_120("triangle"),
        dx=(),
    ),
    "generic": EdgeLayout(
        runs_135=(
            EdgeRun("number", BOTTOM, "{label}", 1.4),
            EdgeRun("half", BOTTOM, "{label}", 1.1, lead="triangle"),
            *_KODAK_TOP_135,
        ),
        runs_120=_numbers_120("triangle"),
    ),
}
