"""Inverse lens maps in unrotated sensor coordinates."""

from dataclasses import dataclass
from functools import lru_cache

import negpy_core
import numpy as np

from negpy.features.lens.models import LensCorrections, LensMetadata

IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@lru_cache(maxsize=64)
def _frame(lens: LensMetadata, shape: tuple[int, ...], center: tuple[float, float]) -> tuple:
    """(buffer origin, buffer-to-image scale, warp center, radius, fill scale) of the frame."""
    h, w = shape[:2]
    t, left, b, r = lens.active_area or (0, 0, h, w)
    bt, bl, bb, br = lens.buffer_area or (t, left, b, r)
    sx, sy = (br - bl) / w, (bb - bt) / h
    cx, cy = left + center[0] * (r - left - 1), t + center[1] * (b - t - 1)
    radius = np.hypot(max(cx - left, r - 1 - cx), max(cy - t, b - 1 - cy))
    return (float(bl), float(bt)), (sx, sy), (cx, cy), float(radius), float(lens.fill_scale)


@dataclass(frozen=True)
class RectilinearWarp:
    coefficients: tuple[tuple[float, ...], ...]
    center: tuple[float, float] = (0.5, 0.5)

    @property
    def has_distortion(self) -> bool:
        return self.coefficients[0 if len(self.coefficients) == 1 else 1] != IDENTITY

    @property
    def has_ca(self) -> bool:
        return len(self.coefficients) == 3 and (
            self.coefficients[0] != self.coefficients[1] or self.coefficients[2] != self.coefficients[1]
        )

    def remap(
        self,
        lens: LensMetadata,
        shape: tuple[int, ...],
        start: int,
        stop: int,
        channel: int,
        corrections: LensCorrections = LensCorrections(True, True),
    ) -> tuple[np.ndarray, np.ndarray]:
        common = self.coefficients[0 if len(self.coefficients) == 1 else 1]
        selected = self.coefficients[channel] if corrections.ca and len(self.coefficients) == 3 else common
        if corrections.distortion:
            frame = _frame(lens, tuple(shape[:2]), self.center)
            return negpy_core.rectilinear_map(int(shape[1]), (start, stop), [float(v) for v in selected], frame)
        if selected == common:
            my, mx = np.mgrid[start:stop, : shape[1]].astype(np.float32)
            return mx, my
        # CA-only maps to the original green geometry: channel(green_inverse(x, y)); outside the
        # calibrated radius the edge's relative CA displacement extends.
        frame = _frame(lens, tuple(shape[:2]), self.center)
        coefficients = ([float(v) for v in selected], [float(v) for v in common])
        return negpy_core.rectilinear_ca_map(int(shape[1]), (start, stop), coefficients, frame)


@dataclass(frozen=True)
class PanasonicWarp:
    scale: float
    a: float
    b: float
    c: float

    @property
    def has_distortion(self) -> bool:
        return self.scale != 1.0 or self.a != 0.0 or self.b != 0.0 or self.c != 0.0

    @property
    def has_ca(self) -> bool:
        return False

    def remap(
        self,
        lens: LensMetadata,
        shape: tuple[int, ...],
        start: int,
        stop: int,
        channel: int,
        corrections: LensCorrections = LensCorrections(True, True),
    ) -> tuple[np.ndarray, np.ndarray]:
        # Model from darktable's embedded-metadata reader (GPL-3.0+) and https://github.com/trou/panasonic-rw2:
        # Ru = Rd + scale*(a*Rd^3 + b*Rd^5 + c*Rd^7), Rd (source) and Ru (corrected) normalized to the half-diagonal.
        if not corrections.distortion:
            my, mx = np.mgrid[start:stop, : shape[1]].astype(np.float32)
            return mx, my
        # Ru -> Rd by fixed-point iteration; Rd/Ru is the source-sampling multiplier.
        model = (float(self.scale), float(self.a), float(self.b), float(self.c))
        return negpy_core.panasonic_map((int(shape[0]), int(shape[1])), (start, stop), model, float(lens.fill_scale))


@dataclass(frozen=True)
class SonyWarp:
    distortion: tuple[float, ...] = ()
    ca_red: tuple[float, ...] = ()
    ca_blue: tuple[float, ...] = ()

    @property
    def has_distortion(self) -> bool:
        return any(self.distortion)

    @property
    def has_ca(self) -> bool:
        return any(self.ca_red) or any(self.ca_blue)

    def remap(
        self,
        lens: LensMetadata,
        shape: tuple[int, ...],
        start: int,
        stop: int,
        channel: int,
        corrections: LensCorrections = LensCorrections(True, True),
    ) -> tuple[np.ndarray, np.ndarray]:
        # Sony's knot positions and units follow darktable's embedded-metadata model (GPL-3.0+).
        # https://github.com/darktable-org/darktable/blob/master/src/iop/lens.cc
        n = len(self.distortion) or len(self.ca_red) or len(self.ca_blue)
        factors = np.ones(n)
        if corrections.distortion and self.distortion:
            factors += np.asarray(self.distortion) / 16384.0
        ca = self.ca_red if channel == 0 else self.ca_blue if channel == 2 else ()
        if corrections.ca and ca:
            factors *= 1 + np.asarray(ca) / 2097152.0
        return negpy_core.sony_map((int(shape[0]), int(shape[1])), (start, stop), factors.tolist(), float(lens.fill_scale))
