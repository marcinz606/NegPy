from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class ScanMode(StrEnum):
    NEGATIVE = "Negative"
    POSITIVE = "Positive"
    TRANSPARENCY = "Transparency"


@dataclass(frozen=True)
class ScannerCaptureState:
    """Replayable hardware state shared by adjacent source windows.

    The LS-5000 meters RGB exposure times and chooses a focus position before
    acquisition.  A full negative can span two feeder windows, so its second
    source must reuse those exact values instead of independently focusing or
    metering a different part of the photograph.
    """

    focus_position: int
    exposure_multiplier: float
    red_exposure_us: float
    green_exposure_us: float
    blue_exposure_us: float

    def __post_init__(self) -> None:
        if type(self.focus_position) is not int or self.focus_position < 0:
            raise ValueError("focus_position must be a non-negative integer")
        for field in (
            "exposure_multiplier",
            "red_exposure_us",
            "green_exposure_us",
            "blue_exposure_us",
        ):
            value = getattr(self, field)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")


@dataclass(frozen=True)
class RegisteredScanGeometry:
    """Coupled transport position and scan window for one registered frame.

    ``br_y_device_px`` is the inclusive bottom coordinate in the scanner's
    native device-pixel grid, not a height and not an output-resolution row.
    ``frame`` may be omitted for a single-frame carrier or the current adapter
    position.
    """

    subframe_mm: float
    br_y_device_px: int
    frame: int | None = None

    def __post_init__(self) -> None:
        if self.frame is not None and (type(self.frame) is not int or self.frame < 1):
            raise ValueError("registered frame must be a positive integer or None")
        if type(self.subframe_mm) not in (int, float) or not isfinite(self.subframe_mm) or self.subframe_mm < 0:
            raise ValueError("registered subframe must be a finite non-negative distance")
        if type(self.br_y_device_px) is not int or self.br_y_device_px < 0:
            raise ValueError("registered bottom coordinate must be a non-negative integer")


def parse_registration_manifest(data: object, *, frame: int) -> RegisteredScanGeometry:
    """Extract one frame's window from an already-parsed registration manifest.

    Schema matches the roll-scan CLI's ``--registration-json``
    (``negpy.infrastructure.scanners.roll_scan_runner``):
    ``{"frames": [{"frame": N, "subframe_mm": ..., "br_y": ...}, ...]}``.

    Fails loud (``ValueError``) on anything missing, ambiguous, or malformed.
    Numeric/geometric validation of the matched entry is delegated to
    ``RegisteredScanGeometry.__post_init__`` rather than duplicated here.
    File I/O is the caller's concern — this only walks already-decoded JSON.
    """
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise ValueError("registration manifest must contain a 'frames' array")
    # `type(...) is int` (not `isinstance`) so a JSON `true`/`false` — which
    # Python bools satisfy `== 1`/`== 0` for — can never masquerade as frame 1/0.
    matches = [item for item in data["frames"] if isinstance(item, dict) and type(item.get("frame")) is int and item["frame"] == frame]
    if not matches:
        raise ValueError(f"registration manifest has no entry for frame {frame}")
    if len(matches) > 1:
        raise ValueError(f"registration manifest has duplicate entries for frame {frame}")
    entry = matches[0]
    if "subframe_mm" not in entry or "br_y" not in entry:
        raise ValueError(f"registration entry for frame {frame} must contain subframe_mm and br_y")
    return RegisteredScanGeometry(subframe_mm=entry["subframe_mm"], br_y_device_px=entry["br_y"], frame=frame)


@dataclass(frozen=True)
class ScanParams:
    dpi: int
    depth: int
    capture_ir: bool
    area: tuple[float, float, float, float] | None = None
    # Autofocus before the scan where the device supports it. Film is never
    # perfectly flat and some scanners (Coolscan LS-5000) power up at an
    # uncalibrated focus position, so this defaults on.
    autofocus: bool = True
    # Hardware multi-sampling passes per line (1 = off). Archival intent:
    # if a value > 1 cannot be honored, the scan fails rather than silently
    # degrading to a single pass.
    samples_per_scan: int = 1
    # Select a specific frame on a roll-fed scanner (e.g. coolscan3) before
    # scanning. Archival intent: if a frame is requested and the device has
    # no frame-selection option, the scan fails rather than silently reading
    # whatever frame is currently under the sensor.
    frame: int | None = None
    # Registration geometry is opt-in and keeps the fine transport shift and
    # shortened scan window inseparable. ``frame`` above remains supported for
    # callers that only need legacy frame selection.
    registered_geometry: RegisteredScanGeometry | None = None
    # Ask the scanner to meter this positioned frame before acquisition.
    # This is hardware auto-exposure, distinct from NegPy's later rendering
    # auto-exposure. Explicit requests fail if the SANE option is unavailable.
    auto_exposure: bool = False
    # Exact focus/exposure state captured from an earlier source window.  It
    # is mutually exclusive with autofocus/auto-exposure: replay means replay,
    # not a hint the backend may silently overwrite.
    capture_state: ScannerCaptureState | None = None
    # Full-negative capture keeps every RGB sample and carries an explicit IR
    # validity mask instead of cropping both planes to their common overlap.
    preserve_full_canvas: bool = False

    def __post_init__(self) -> None:
        if self.capture_state is not None and (self.autofocus or self.auto_exposure):
            raise ValueError("locked capture state cannot be combined with autofocus or auto-exposure")
        if self.preserve_full_canvas and not (self.capture_ir and self.samples_per_scan > 1):
            raise ValueError("full-canvas IR preservation requires split RGB+IR capture")
