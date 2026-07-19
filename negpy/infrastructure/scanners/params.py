from dataclasses import dataclass
from enum import StrEnum


class ScanMode(StrEnum):
    NEGATIVE = "Negative"
    POSITIVE = "Positive"
    TRANSPARENCY = "Transparency"


@dataclass(frozen=True)
class ScanParams:
    dpi: int
    depth: int
    capture_ir: bool
    # Normalized (x1,y1,x2,y2) window 0..1; backend maps to device units (coolscan3 int px).
    window: tuple[float, float, float, float] | None = None
    # coolscan3 `subframe` (mm), applied to every frame. 0 = scanner default.
    frame_offset_mm: float = 0.0
    autofocus: bool = True
    # Select a frame on a roll-fed scanner (coolscan3) before scanning. If a
    # frame is requested and the device has no frame option, the scan fails
    # rather than reading whatever frame is under the sensor.
    frame: int | None = None
    # Hardware auto-exposure (SANE `ae`), distinct from NegPy's rendering
    # auto-exposure. An explicit request fails if the option is unavailable.
    auto_exposure: bool = False


MIN_FRAME_EXTENT_MM = 1.0  # below this a capped scan is a useless sliver


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
