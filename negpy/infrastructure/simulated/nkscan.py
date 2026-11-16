"""A stand-in for the nkscan extension: Coolscans whose adapters find frames with no strip pass.

`NkscanBackend` runs on it unchanged. One unit per nkscan framing mechanism that has no
thumbnail: an LS-9000 with a mounted-slide holder ("published": the holder gives every rect)
and an LS-5000 with an SF-210 slide feeder ("address": one rect per slide, and load() takes
the next slide from the stack).
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from negpy.infrastructure.simulated.images import negative

_MAX_EDGE_PX = 2000
_PROGRESS_STEPS = 10
_STEP_S = 0.05

# Stage addresses at 4000 dpi: a 36 x 24 mm frame is 5669 along the feed by 3780 across.
_FRAME_ALONG = 5669
_FRAME_ACROSS = 3780
# ponytail: FH-835M geometry guessed from the 6x9 strip page in nkscan's caps/frames.rs (top 2236,
# left 518, width 8964) and a 50 mm mount pitch. Replace with a real C8h page when one is captured.
_HOLDER_TOP = 2236
_HOLDER_LEFT = 518 + (8964 - _FRAME_ACROSS) // 2
_MOUNT_PITCH = 7874
_HOLDER_SLIDES = 5
_FEEDER_SLIDES = 6


class ScannerError(RuntimeError): ...


class TransientError(ScannerError): ...


class MediaError(ScannerError): ...


class ScanCancelled(ScannerError): ...


class UnsupportedError(ScannerError):
    op = "scan"
    reason = "not offered"


@dataclass(frozen=True)
class Capabilities:
    product: str
    model: str
    framing: str
    max_frames: int
    vendor: str = "Nikon"
    x_dpi_range: tuple[int, int] = (500, 4000)
    optical_dpi: int = 4000
    max_samples: int = 16
    thumbnail: bool = False
    film_format: str | None = None
    film_formats: tuple[str, ...] = ()
    multi_line: bool = True
    eject: bool = True


@dataclass(frozen=True)
class Device:
    location: str


@dataclass(frozen=True)
class Discovery:
    frames: list[tuple[int, int, int, int]]
    thumbnail: None = None
    column_addresses: None = None
    thumbnail_complete: None = None
    thumbnail_blocks: None = None
    base_density: None = None


@dataclass(frozen=True)
class ScanResult:
    colors: dict[str, np.ndarray]
    ir: np.ndarray | None
    dpi: int
    exposures: dict[str, int]
    cleaned: int | None
    complete: bool = True
    blocks: int = 1


@dataclass
class _Unit:
    """The unit's own state, which outlives every session on it."""

    caps: Capabilities
    frames: list[tuple[int, int, int, int]]
    stack: int  # slides the feeder has not loaded yet; 0 on a holder
    seed: int = 0
    loaded: bool = True

    def load(self) -> bool:
        if not self.loaded and self.stack:
            self.stack -= 1
            self.seed += 1
            self.loaded = True
        return self.loaded

    def eject(self) -> bool:
        ejected = self.loaded
        self.loaded = False
        return ejected


def _units() -> dict[str, _Unit]:
    holder = [
        (top, _HOLDER_LEFT, top + _FRAME_ALONG, _HOLDER_LEFT + _FRAME_ACROSS)
        for top in range(_HOLDER_TOP, _HOLDER_TOP + _HOLDER_SLIDES * _MOUNT_PITCH, _MOUNT_PITCH)
    ]
    return {
        "sim:ls9000": _Unit(
            Capabilities(product="LS-9000 ED (FH-835M)", model="LS-9000 (FH-835M)", framing="published", max_frames=_HOLDER_SLIDES),
            holder,
            stack=0,
        ),
        "sim:sf210": _Unit(
            Capabilities(product="LS-5000 ED (SF-210)", model="LS-5000 (SF-210)", framing="address", max_frames=1),
            [(0, 0, _FRAME_ALONG, _FRAME_ACROSS)],
            stack=_FEEDER_SLIDES - 1,
        ),
    }


class Session:
    _units: dict[str, _Unit]

    def __init__(self, location: str) -> None:
        self._unit = self._units[location]

    @classmethod
    def open(cls, device: Device) -> "Session":
        return cls(device.location)

    @property
    def capabilities(self) -> Capabilities:
        return self._unit.caps

    def media_loaded(self) -> bool:
        return self._unit.loaded

    def load(self) -> bool:
        return self._unit.load()

    def stage(self) -> None:
        pass

    def eject(self) -> bool:
        return self._unit.eject()

    def close(self) -> None:
        pass

    def nikon_profile(self, film: str) -> bytes | None:
        return None

    def discover_frames(self, format: str | None = None, film: str | None = None, progress: Callable[..., Any] | None = None) -> Discovery:  # noqa: A002 - the binding's own name
        self._require_media()
        return Discovery(frames=list(self._unit.frames))

    def meter_frame(self, frame: tuple[int, int, int, int], infrared: bool = False, **_: Any) -> dict[str, int]:
        self._require_media()
        exposures = {"red": 1200, "green": 1800, "blue": 2600}
        return {**exposures, "infrared": 900} if infrared else exposures

    def scan_frame(
        self,
        frame: tuple[int, int, int, int],
        dpi: int | None = None,
        infrared: bool = False,
        clean: bool = False,
        exposures: dict[str, int] | None = None,
        progress: Callable[..., Any] | None = None,
        **_: Any,
    ) -> ScanResult:
        self._require_media()
        for step in range(1, _PROGRESS_STEPS + 1):
            threading.Event().wait(_STEP_S)
            if progress is not None and progress("scan", 0, step, _PROGRESS_STEPS) is False:
                raise ScanCancelled("cancelled by the progress callback")
        dpi = int(dpi or self._unit.caps.optical_dpi)
        top, left, bottom, right = frame
        scale = dpi / self._unit.caps.optical_dpi
        rows, cols = (bottom - top) * scale, (right - left) * scale
        scale = min(1.0, _MAX_EDGE_PX / max(rows, cols, 1.0))
        rgb, ir = negative(max(1, int(rows * scale)), max(1, int(cols * scale)), seed=self._unit.seed * 100 + top)
        return ScanResult(
            colors={name: rgb[..., i] for i, name in enumerate(("red", "green", "blue"))},
            ir=ir if infrared or clean else None,
            dpi=dpi,
            exposures=dict(exposures or {"red": 1200, "green": 1800, "blue": 2600}),
            cleaned=0 if clean else None,
        )

    def _require_media(self) -> None:
        if not self._unit.loaded:
            raise MediaError("no medium loaded")


class Module:
    """The attributes `NkscanBackend` reads off the nkscan module."""

    ScannerError = ScannerError
    TransientError = TransientError
    MediaError = MediaError
    ScanCancelled = ScanCancelled
    UnsupportedError = UnsupportedError

    def __init__(self) -> None:
        units = _units()

        class _Session(Session):
            _units = units

        self.Session = _Session
        self._locations = list(units)

    def list_devices(self) -> list[Device]:
        return [Device(location) for location in self._locations]
