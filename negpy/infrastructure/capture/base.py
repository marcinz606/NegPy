"""Capture-layer value types and hardware interfaces (Qt-free, injectable)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class Channel(Enum):
    """One narrowband exposure of a trichromatic capture."""

    RED = "R"
    GREEN = "G"
    BLUE = "B"

    @property
    def letter(self) -> str:
        return self.value

    def rgb(self, level: int) -> tuple[int, int, int]:
        """The (r, g, b) Scanlight tuple that lights *only* this channel at `level`."""
        return (
            level if self is Channel.RED else 0,
            level if self is Channel.GREEN else 0,
            level if self is Channel.BLUE else 0,
        )


#: Capture order — red first, blue last (matches TriRGB and NegPy RGB-scan).
CAPTURE_ORDER: tuple[Channel, Channel, Channel] = (Channel.RED, Channel.GREEN, Channel.BLUE)


@runtime_checkable
class LightSource(Protocol):
    """Minimal Scanlight interface the capture service depends on."""

    def set_color(self, r: int = 0, g: int = 0, b: int = 0, w: int = 0, save: bool = False) -> None: ...

    def off(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class Camera(Protocol):
    """Minimal camera interface: download one RAW to `out_path`, return its path."""

    def capture(self, out_path: str, shutter: Optional[str] = None) -> str: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CaptureSettings:
    """Everything needed to capture one R/G/B triplet.

    `min_raw_bytes` defaults to 8 MB — a permissive floor that still catches a missing
    or truncated file while accepting any body: a 12 MP sensor's compressed RAWs run
    ~12-15 MB, well under a floor tuned for a 33 MP one. `shutters` (per-channel
    exposure) is a hook — None means the operator's fixed camera shutter is used.
    """

    roll_name: str
    frame_number: int
    output_folder: str
    levels: tuple[int, int, int]  # (r, g, b) LED levels, 0-255
    settle_s: float = 0.4
    min_raw_bytes: int = 8 * 1024 * 1024
    max_raw_bytes: int = 200 * 1024 * 1024
    shutters: Optional[tuple[Optional[str], Optional[str], Optional[str]]] = None


@dataclass(frozen=True)
class CaptureResult:
    """The three RAW files written for one frame (red is the primary asset)."""

    frame_number: int
    red_path: str
    green_path: str
    blue_path: str

    @property
    def paths(self) -> list[str]:
        return [self.red_path, self.green_path, self.blue_path]
