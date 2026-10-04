from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class LensCorrections:
    distortion: bool = False
    ca: bool = False

    def __bool__(self) -> bool:
        return self.distortion or self.ca


class LensWarp(Protocol):
    @property
    def has_distortion(self) -> bool: ...

    @property
    def has_ca(self) -> bool: ...

    def remap(
        self,
        lens: LensMetadata,
        shape: tuple[int, ...],
        start: int,
        stop: int,
        channel: int,
        corrections: LensCorrections = LensCorrections(True, True),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return float32 inverse x/y maps for one channel and rows [start, stop)."""
        ...


@dataclass(frozen=True)
class LensMetadata:
    source: str = ""
    warps: tuple[LensWarp, ...] = ()
    reason: str = "No embedded lens correction data."
    # DNG opcodes use the active image, before DefaultCrop and EXIF orientation.
    active_area: tuple[int, int, int, int] | None = None
    buffer_area: tuple[int, int, int, int] | None = None
    # Scale about the warp's center applied to output coordinates before the map read; apply_lens sets it.
    fill_scale: float = 1.0

    @property
    def distortion(self) -> bool:
        return any(warp.has_distortion for warp in self.warps)

    @property
    def ca(self) -> bool:
        return any(warp.has_ca for warp in self.warps)

    @property
    def available(self) -> bool:
        return self.distortion or self.ca

    @property
    def description(self) -> str:
        if not self.available:
            return self.reason
        corrections = " + ".join(label for enabled, label in ((self.distortion, "distortion"), (self.ca, "lateral CA")) if enabled)
        return f"{self.source}: {corrections}"
