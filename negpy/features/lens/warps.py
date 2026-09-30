"""Inverse lens maps in unrotated sensor coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np

from negpy.features.lens.models import LensCorrections, LensMetadata

IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _rectilinear_xy(x: np.ndarray, y: np.ndarray, coefficients: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    k0, k1, k2, k3, t0, t1 = coefficients
    r2 = x * x + y * y
    factor = k0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return x * factor + 2 * t0 * x * y + t1 * (r2 + 2 * x * x), y * factor + 2 * t1 * x * y + t0 * (r2 + 2 * y * y)


def _coordinates(lens: LensMetadata, shape: tuple[int, ...], start: int, stop: int, center: tuple[float, float]) -> tuple:
    h, w = shape[:2]
    t, left, b, r = lens.active_area or (0, 0, h, w)
    bt, bl, bb, br = lens.buffer_area or (t, left, b, r)
    sx, sy = (br - bl) / w, (bb - bt) / h
    cx, cy = left + center[0] * (r - left - 1), t + center[1] * (b - t - 1)
    radius = np.hypot(max(cx - left, r - 1 - cx), max(cy - t, b - 1 - cy))
    x = (bl + (np.arange(w, dtype=np.float32)[None, :] + 0.5) * sx - 0.5 - cx) / radius * lens.fill_scale
    y = (bt + (np.arange(start, stop, dtype=np.float32)[:, None] + 0.5) * sy - 0.5 - cy) / radius * lens.fill_scale
    return x, y, cx, cy, radius, sx, sy, bl, bt


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
        x, y, cx, cy, radius, sx, sy, left, top = _coordinates(lens, shape, start, stop, self.center)
        common = self.coefficients[0 if len(self.coefficients) == 1 else 1]
        selected = self.coefficients[channel] if corrections.ca and len(self.coefficients) == 3 else common
        target_x, target_y = x, y
        if not corrections.distortion:
            if selected == common:
                my, mx = np.mgrid[start:stop, : shape[1]].astype(np.float32)
                return mx, my
            # CA-only maps to the original green geometry: channel(green_inverse(x, y)).
            g0, g1, g2, g3, t0, t1 = common
            points = np.stack(np.broadcast_arrays(x, y), axis=-1)
            unwarped = cv2.undistortPointsIter(
                points.reshape(-1, 1, 2),
                np.diag([g0, g0, 1.0]),
                np.array([g1, g2, t0, t1, g3]) / g0,
                None,
                None,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-8),
            ).reshape(points.shape)
            x, y = unwarped[..., 0], unwarped[..., 1]
            # Outside the calibrated radius, extend the edge's relative CA displacement.
            limit = np.maximum(np.hypot(x, y), 1.0)
            x, y = x / limit, y / limit
        wx, wy = _rectilinear_xy(x, y, selected)
        if not corrections.distortion:
            gx, gy = _rectilinear_xy(x, y, common)
            scale = np.hypot(target_x, target_y) / np.maximum(np.hypot(gx, gy), 1e-8)
            wx, wy = target_x + (wx - gx) * scale, target_y + (wy - gy) * scale
        mx = (cx + radius * wx - left + 0.5) / sx - 0.5
        my = (cy + radius * wy - top + 0.5) / sy - 0.5
        return mx.astype(np.float32), my.astype(np.float32)


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
        h, w = shape[:2]
        x = (np.arange(w, dtype=np.float32)[None, :] - w * 0.5) * lens.fill_scale
        y = (np.arange(start, stop, dtype=np.float32)[:, None] - h * 0.5) * lens.fill_scale
        radius = np.hypot(x, y) / np.hypot(w * 0.5, h * 0.5)
        n = len(self.distortion) or len(self.ca_red) or len(self.ca_blue)
        knots = (np.arange(n) + 0.5) / (n - 1)
        factors = np.ones(n)
        if corrections.distortion and self.distortion:
            factors += np.asarray(self.distortion) / 16384.0
        ca = self.ca_red if channel == 0 else self.ca_blue if channel == 2 else ()
        if corrections.ca and ca:
            factors *= 1 + np.asarray(ca) / 2097152.0
        factor = np.interp(radius, knots, factors).astype(np.float32)
        return (x * factor + w * 0.5).astype(np.float32), (y * factor + h * 0.5).astype(np.float32)
