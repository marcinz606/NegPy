from dataclasses import dataclass
from enum import StrEnum


class ScanMode(StrEnum):
    NEGATIVE = "Negative"
    POSITIVE = "Positive"
    TRANSPARENCY = "Transparency"


class MultiExposureMode(StrEnum):
    """Whether a short+long colour pass pair is merged for extended dynamic range.

    OFF: one exposure, the fast path. ADAPTIVE: short+long merged, long exposure picked per
    frame from image content — pyopticfilm's only multi-exposure behavior.

    Orthogonal to ``ScanParams.n_passes``: this picks *whether* to merge a second exposure,
    while ``n_passes`` repeats whichever exposure(s) are chosen for a same-exposure SNR stack —
    the two compose (pyopticfilm's "Adaptive Multi-Pass") rather than being alternatives.
    """

    OFF = "off"
    ADAPTIVE = "adaptive"


#: Repeats of the same exposure to stack for an SNR gain (pyopticfilm's `n_passes`); 1 = no
#: stacking. This module must not import pyopticfilm directly (only plustek_backend.py may,
#: see test_only_adapter_imports_plustek_driver), so this mirrors — rather than imports —
#: pyopticfilm's own `Scanner.scan()` bound; plustek_backend.py's own MAX_N_PASSES import
#: keeps the two in sync at the one place that already reaches into pyopticfilm.
MIN_N_PASSES = 1
MAX_N_PASSES = 9
#: Starting point when a user first turns Passes above the off position — comfortably past
#: the floor without defaulting to the slow end.
DEFAULT_N_PASSES = 3


@dataclass(frozen=True)
class ScanParams:
    dpi: int
    depth: int
    capture_ir: bool
    multi_exposure_mode: MultiExposureMode = MultiExposureMode.OFF
    # Same-exposure repeats to stack for an SNR gain (1-9); independent of multi_exposure_mode.
    n_passes: int = MIN_N_PASSES
    # Normalized (x1,y1,x2,y2) window 0..1; backend maps to device units (coolscan3 int px).
    window: tuple[float, float, float, float] | None = None
    # coolscan3 `subframe` (mm), applied to every frame. 0 = scanner default.
    frame_offset_mm: float = 0.0
    autofocus: bool = True
    # Select a frame on a roll-fed scanner (coolscan3) before scanning. If a frame is
    # requested and the device has no frame option, the scan fails rather than reading
    # whatever frame is under the sensor.
    frame: int | None = None
    # Hardware auto-exposure (SANE `ae`), distinct from NegPy's rendering auto-exposure. An
    # explicit request fails if the option is unavailable.
    auto_exposure: bool = False
    # Hardware scan exposure time in microseconds (SANE `scan-exposure-time`). None lets the
    # scanner use its default, and it is ignored when the device has no such option.
    exposure_time_us: int | None = None
    # Let the transport remove dust itself, baked into what it returns. Requires
    # ScannerCapabilities.hw_clean; a request against a device without it fails.
    clean: bool = False
    # Repeated reads of one line for the transport to average, not binning. 1 = single read.
    samples: int = 1
    # One line per pass: slower, and it owes the host no line registration.
    superfine: bool = False
    # Film format the transport cannot measure itself ("135", "66", ...). None = let it decide.
    film_format: str | None = None
    # What is on the film. Reversal stock reads the opposite way against the unexposed film
    # between frames, which is what finding the frames on a strip goes by, and silver blocks
    # infrared, so this decides whether an IR pass means anything.
    film_type: str = "negative"


class FilmType(StrEnum):
    """What is on the film, as the transports spell it."""

    NEGATIVE = "negative"
    MONO = "mono"
    POSITIVE = "positive"
    KODACHROME = "kodachrome"


#: Label for each film type, and whether infrared can see dust through it. Silver grain stops
#: infrared as it stops light, and Kodachrome's dyes do the same, so the mask comes back as the
#: picture rather than the dust on it.
#: Plain `str` keys, not the enum: these cross into Qt as combo data, and a QVariant holding a
#: StrEnum does not compare equal to the string a caller looks it up with.
FILM_TYPES: dict[str, tuple[str, bool]] = {
    FilmType.NEGATIVE.value: ("Color negative", True),
    FilmType.MONO.value: ("B&W negative", False),
    FilmType.POSITIVE.value: ("Slide", True),
    FilmType.KODACHROME.value: ("Kodachrome", False),
}


def film_reads_positive(film_type: str) -> bool:
    """Reversal stock develops its unexposed film to maximum density, negatives to their base."""
    return film_type in (FilmType.POSITIVE.value, FilmType.KODACHROME.value)


def film_passes_infrared(film_type: str) -> bool:
    return FILM_TYPES.get(film_type, ("", True))[1]


MIN_FRAME_EXTENT_MM = 1.0  # below this a capped scan is a useless sliver

CANONICAL_DPI_STOPS = (75, 150, 300, 600, 1200, 2400, 3600, 4800, 6400, 7200, 9600)


def dpi_stops_in_range(lo: float, hi: float) -> tuple[int, ...]:
    """Canonical stops a (min, max) hardware range covers, or every stop when it covers none.

    A transport that reports a continuous range offers no ladder of its own, and a UI needs
    one — the scanner rounds an off-ladder request rather than refusing it.
    """
    stops = tuple(s for s in CANONICAL_DPI_STOPS if lo <= s <= hi)
    return stops or CANONICAL_DPI_STOPS


ScanArea = tuple[float, float, float, float]


def clamp_scan_area(area: ScanArea) -> ScanArea:
    """Clamp a normalized TA rect to ``0..1`` with a tiny positive size."""
    x1, y1, x2, y2 = (float(v) for v in area)
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-3)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-3)
    return (x1, y1, x2, y2)


def crop_to_scan_window(crop: ScanArea, *, mirror_x: bool) -> ScanArea:
    """Map Prescan widget coords ↔ TA ``area`` for ``ScanParams.window``.

    Self-inverse when ``mirror_x`` is fixed: image-left is sensor-right on mirrored
    scanners, so trimming left chrome on the Prescan must crop the opposite TA side.

    Mirrors the pre-#958 behaviour that PR #958 accidentally flattened to a plain
    clamp, which displaced Prescan crops on mirror_x Plustek devices.
    """
    area = clamp_scan_area(crop)
    if mirror_x:
        x1, y1, x2, y2 = area
        return (1.0 - x2, y1, 1.0 - x1, y2)
    return area


def clamp_frame_offset_mm(offset_mm: float, pitch_mm: float) -> float:
    """Effective feed-axis offset, floored at 0 and held short of one frame pitch.

    The transport cannot back up, and the scan blacks out one pitch past the frame
    start — at `offset >= pitch` the window collapses to zero height and the scan
    comes back empty. Pitch 0 means unknown: floor only.
    """
    offset = max(0.0, offset_mm)
    if pitch_mm <= 0:
        return offset
    return min(offset, max(0.0, pitch_mm - MIN_FRAME_EXTENT_MM))


def scan_window_to_area(
    rect: tuple[float, float, float, float] | None,
    max_area_mm: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """Normalized (x1,y1,x2,y2) window → approximate mm for the UI readout only.

    The scan maps the fraction to device units in the backend; this is display-only.
    """
    if rect is None or len(rect) != 4:
        return None
    x1, y1, x2, y2 = rect
    w, h = max_area_mm
    return (x1 * w, y1 * h, x2 * w, y2 * h)
