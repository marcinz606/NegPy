"""Inverse lens maps in the scanning camera's linear RGB coordinates."""

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.lens.models import LensMetadata, RectilinearWarp, SonyWarp
from negpy.kernel.image.logic import apply_exif_orientation


def _coordinates(lens: LensMetadata, shape: tuple[int, ...], start: int, stop: int, center: tuple[float, float]) -> tuple:
    h, w = shape[:2]
    t, left, b, r = lens.active_area or (0, 0, h, w)
    bt, bl, bb, br = lens.buffer_area or (t, left, b, r)
    sx, sy = (br - bl) / w, (bb - bt) / h
    cx, cy = left + center[0] * (r - left - 1), t + center[1] * (b - t - 1)
    radius = np.hypot(max(cx - left, r - 1 - cx), max(cy - t, b - 1 - cy))
    x = (bl + (np.arange(w, dtype=np.float32)[None, :] + 0.5) * sx - 0.5 - cx) / radius
    y = (bt + (np.arange(start, stop, dtype=np.float32)[:, None] + 0.5) * sy - 0.5 - cy) / radius
    return x, y, cx, cy, radius, sx, sy, bl, bt


def _dng_maps(
    lens: LensMetadata, warp: RectilinearWarp, shape: tuple[int, ...], start: int, stop: int, channel: int
) -> tuple[np.ndarray, np.ndarray]:
    x, y, cx, cy, radius, sx, sy, left, top = _coordinates(lens, shape, start, stop, warp.center)
    k0, k1, k2, k3, t0, t1 = warp.coefficients[0 if len(warp.coefficients) == 1 else channel]
    r2 = x * x + y * y
    factor = k0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    mx = (cx + radius * (x * factor + 2 * t0 * x * y + t1 * (r2 + 2 * x * x)) - left + 0.5) / sx - 0.5
    my = (cy + radius * (y * factor + 2 * t1 * x * y + t0 * (r2 + 2 * y * y)) - top + 0.5) / sy - 0.5
    return mx.astype(np.float32), my.astype(np.float32)


def _sony_maps(warp: SonyWarp, shape: tuple[int, ...], start: int, stop: int, channel: int) -> tuple[np.ndarray, np.ndarray]:
    # Sony's knot positions and units follow darktable's embedded-metadata model (GPL-3.0+).
    # https://github.com/darktable-org/darktable/blob/master/src/iop/lens.cc
    h, w = shape[:2]
    x = np.arange(w, dtype=np.float32)[None, :] - w * 0.5
    y = np.arange(start, stop, dtype=np.float32)[:, None] - h * 0.5
    radius = np.hypot(x, y) / np.hypot(w * 0.5, h * 0.5)
    n = len(warp.distortion) or len(warp.ca_red)
    knots = (np.arange(n) + 0.5) / (n - 1)
    factors = np.ones(n)
    if warp.distortion:
        factors += np.asarray(warp.distortion) / 16384.0
    ca = warp.ca_red if channel == 0 else warp.ca_blue if channel == 2 else ()
    if ca:
        factors *= 1 + np.asarray(ca) / 2097152.0
    factor = np.interp(radius, knots, factors).astype(np.float32)
    return (x * factor + w * 0.5).astype(np.float32), (y * factor + h * 0.5).astype(np.float32)


def apply_lens(img: ImageBuffer, lens: LensMetadata, orientation: int = 1) -> ImageBuffer:
    """Apply all supported embedded warps, with bounded temporary map memory."""
    if not lens.available or min(img.shape[:2]) < 2:
        return img
    inverse_orientation = {6: 8, 8: 6}.get(orientation, orientation)
    source = np.ascontiguousarray(apply_exif_orientation(img, inverse_orientation))
    for warp in lens.warps:
        h, w = source.shape[:2]
        result = np.empty_like(source)
        for channel in range(3):
            plane = np.ascontiguousarray(source[..., channel])
            for start in range(0, h, 256):
                stop = min(start + 256, h)
                if isinstance(warp, SonyWarp):
                    mx, my = _sony_maps(warp, source.shape, start, stop, channel)
                else:
                    mx, my = _dng_maps(lens, warp, source.shape, start, stop, channel)
                result[start:stop, :, channel] = cv2.remap(plane, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        source = np.clip(result, 0.0, 1.0, out=result)
    return np.ascontiguousarray(apply_exif_orientation(source, orientation))
