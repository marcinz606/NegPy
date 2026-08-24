"""A stand-in for the nkscan extension module, so the backend is testable without hardware.

The real module is a compiled PyO3 extension. `NkscanBackend` keeps it on `self._nk` for
exactly this reason: `make_backend` builds the backend without running its import.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest import mock

import numpy as np

from negpy.infrastructure.scanners.nkscan_backend import NkscanBackend

DEVICE_ID = "usb:1-3.2"
# Rects an LS-50 measured on a real 135 strip: 4000 addresses to the inch, so a frame is
# 5668 tall by 3945 across, and the pitch between them is a little over its height.
FRAMES = ((4387, 0, 10055, 3945), (10742, 0, 16410, 3945), (17753, 0, 23421, 3945))


class ScannerError(RuntimeError): ...


class TransientError(ScannerError): ...


class TransportError(TransientError): ...


class DeviceBusy(TransientError): ...


class DeviceNotFound(ScannerError): ...


class MediaError(ScannerError): ...


class ScanCancelled(ScannerError): ...


class UnsupportedError(ScannerError):
    def __init__(self, message: str, op: str = "scan", reason: str = "not offered") -> None:
        super().__init__(message)
        self.op = op
        self.reason = reason


# What the real Capabilities.locks_white_balance answers: only a colour negative is metered
# per channel, and the binding refuses a film it does not know.
_LOCKED_FILMS = {"negative": False, "mono": True, "monochrome": True, "positive": True, "slide": True, "kodachrome": True}


@dataclass(frozen=True)
class FakeCapabilities:
    vendor: str = "Nikon"
    product: str = "LS-50 ED"
    revision: str = "1.00"
    model: str | None = "LS-50"
    x_dpi_range: tuple[int, int] = (500, 4000)
    y_dpi_range: tuple[int, int] = (500, 4000)
    optical_dpi: int = 4000
    max_frames: int = 6
    thumbnail_dpi: tuple[int, int] = (250, 250)
    focus_range: tuple[int, int] = (0, 255)
    max_samples: int = 16
    framing: str = "thumbnail"
    thumbnail: bool = True
    multi_line: bool = True
    eject: bool = True
    autofocus: bool = True
    hardware_metering: bool = False
    interleavings: tuple[str, ...] = ("LINE_WITHOUT_DISTANCE", "MULTILINE_SIMULTANEOUS")

    @staticmethod
    def locks_white_balance(film: str) -> bool:
        try:
            return _LOCKED_FILMS[film.lower()]
        except KeyError:
            raise RuntimeError(f"unknown film type {film!r}") from None


@dataclass(frozen=True)
class FakeDevice:
    location: str
    name: str = "LS-50"


@dataclass(frozen=True)
class FakeDiscovery:
    frames: list[tuple[int, int, int, int]]
    thumbnail: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class FakeScanResult:
    colors: dict[str, np.ndarray]
    ir: np.ndarray | None
    dpi: int
    rows: int
    cols: int
    exposures: dict[str, int]
    cleaned: int | None


@dataclass
class FakeNkscanModule:
    """Every knob a scenario needs, and a record of what the backend asked for."""

    locations: tuple[str, ...] = (DEVICE_ID,)
    frames: tuple[tuple[int, int, int, int], ...] = FRAMES
    caps: FakeCapabilities = field(default_factory=FakeCapabilities)
    media_loaded_at_open: bool = True
    with_eject: bool = False
    progress_steps: int = 0
    rows: int = 8
    cols: int = 6
    thumbnail: bool = True
    strip_slack: int = 4  # columns past the last frame; negative pushes frames off the pass
    scan_error: Exception | None = None
    discover_error: Exception | None = None
    open_error: Exception | None = None
    probe_error: Exception | None = None
    opened: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        module = self

        class Session(FakeSession):
            _module = module

        self.Session = Session
        self.Capabilities = FakeCapabilities
        for name in (
            "ScannerError",
            "TransientError",
            "TransportError",
            "DeviceBusy",
            "DeviceNotFound",
            "MediaError",
            "ScanCancelled",
            "UnsupportedError",
        ):
            setattr(self, name, globals()[name])

    def list_devices(self) -> list[FakeDevice]:
        return [FakeDevice(location=loc) for loc in self.locations]

    def strip_pass(self) -> dict[str, np.ndarray] | None:
        """The whole-strip pass, laid out the way the unit delivers one.

        Columns are feed addresses from the axis start, at the same resolution as the rows,
        which span the adapter opening. Each frame's band carries its own slot number, so a
        test can tell which part of the strip a tile was cut from.
        """
        if not self.thumbnail or not self.frames:
            return None
        top, left, _bottom, right = self.frames[0]
        scale = (right - left) / self.rows
        cols = int(max(f[2] for f in self.frames) / scale) + self.strip_slack
        plane = np.zeros((self.rows, cols), np.uint16)
        for slot, (top, _l, bottom, _r) in enumerate(self.frames, 1):
            plane[:, int(top / scale) : int(bottom / scale)] = slot
        return {c: plane.copy() for c in ("red", "green", "blue")}

    @property
    def sessions(self) -> list[Any]:
        return self.opened


class FakeSession:
    _module: FakeNkscanModule

    def __init__(self, location: str) -> None:
        module = self._module
        if location not in module.locations:
            raise DeviceNotFound(f"no scanner at {location}")
        if module.open_error is not None:
            raise module.open_error
        self.location = location
        self.closed = False
        self.staged = 0
        self.loads = 0
        self.ejects = 0
        self.discoveries: list[str | None] = []
        self.polarities: list[bool] = []
        self.scans: list[dict[str, Any]] = []
        module.opened.append(self)

    @classmethod
    def open(cls, device: FakeDevice) -> FakeSession:
        if cls._module.probe_error is not None:
            raise cls._module.probe_error
        return cls(device.location)

    @property
    def capabilities(self) -> FakeCapabilities:
        return self._module.caps

    def media_loaded(self) -> bool:
        return self._module.media_loaded_at_open

    def load(self) -> bool:
        self.loads += 1
        return True

    def stage(self) -> None:
        self.staged += 1

    def eject(self) -> bool:
        self.ejects += 1
        return self._module.with_eject

    def discover_frames(
        self,
        format: str | None = None,  # noqa: A002 - the binding's own name
        positive: bool = False,
        progress: Callable[..., Any] | None = None,
    ) -> FakeDiscovery:
        module = self._module
        self.discoveries.append(format)
        self.polarities.append(positive)
        if progress is not None:
            progress("discover", 0, 1, 1)
        if module.discover_error is not None:
            raise module.discover_error
        return FakeDiscovery(frames=list(module.frames), thumbnail=module.strip_pass())

    def scan_frame(
        self,
        frame: tuple[int, int, int, int],
        dpi: int | None = None,
        samples: int = 1,
        superfine: bool = False,
        infrared: bool = False,
        clean: bool = False,
        lock_white_balance: bool = True,
        exposures: dict[str, int] | None = None,
        progress: Callable[..., Any] | None = None,
    ) -> FakeScanResult:
        module = self._module
        # A unit whose CCD reads one line at a time refuses the fast ordering, in the recipe
        # check before the stage moves.
        if not module.caps.multi_line and not superfine:
            raise UnsupportedError(
                "color interleaving: this unit does not read the CCD three rows at once",
                op="color interleaving",
                reason="only ColorInterleaving(LINE_WITHOUT_DISTANCE)",
            )
        self.scans.append(
            {
                "frame": frame,
                "dpi": dpi,
                "samples": samples,
                "superfine": superfine,
                "infrared": infrared,
                "clean": clean,
                "lock_white_balance": lock_white_balance,
                "exposures": exposures,
            }
        )
        for step in range(1, module.progress_steps + 1):
            if progress is not None and progress("scan", 0, step, module.progress_steps) is False:
                raise ScanCancelled("cancelled by the progress callback")
        if module.scan_error is not None:
            raise module.scan_error
        rows, cols = module.rows, module.cols
        colors = {name: np.full((rows, cols), value, np.uint16) for name, value in (("red", 10), ("green", 20), ("blue", 30))}
        return FakeScanResult(
            colors=colors,
            ir=np.full((rows, cols), 40, np.uint16) if infrared else None,
            dpi=int(dpi or module.caps.optical_dpi),
            rows=rows,
            cols=cols,
            exposures={"red": 1, "green": 2, "blue": 3},
            cleaned=7 if clean else None,
        )

    def close(self) -> None:
        self.closed = True


def make_backend(module: FakeNkscanModule | None = None, **kwargs: Any) -> tuple[NkscanBackend, FakeNkscanModule]:
    """A backend wired to a fake module.

    Constructed through the real `__init__` with the fake standing in for the extension, so the
    backend's own state cannot drift from what a test set up by hand.
    """
    module = module or FakeNkscanModule(**kwargs)
    with mock.patch.dict(sys.modules, {"nkscan": module}):
        return NkscanBackend(), module
