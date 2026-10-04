"""Inverse lens maps in the scanning camera's linear RGB coordinates."""

from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.lens.models import LensCorrections, LensMetadata, LensWarp
from negpy.kernel.image.logic import apply_exif_orientation

# Rows sampled down the side edges in the fill-scale search; the top and bottom rows are read whole.
_FILL_EDGE_ROWS = 64
_FILL_STEPS = 24
# Below this the profile is not a lens correction; the edge is left replicated instead.
_FILL_MIN = 0.5


def _applied(lens: LensMetadata, corrections: LensCorrections) -> tuple[LensWarp, ...]:
    return tuple(w for w in lens.warps if corrections.distortion and w.has_distortion or corrections.ca and w.has_ca)


def _lookup(
    warp: LensWarp, lens: LensMetadata, shape: tuple[int, int], channel: int, corrections: LensCorrections, pts: np.ndarray
) -> np.ndarray:
    """One map read at the nearest pixel of each (x, y) point."""
    h, w = shape
    rows = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    cols = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    out = np.empty_like(pts)
    for row in np.unique(rows):
        mx, my = warp.remap(lens, shape, int(row), int(row) + 1, channel, corrections)
        sel = rows == row
        out[sel, 0] = mx[0, cols[sel]]
        out[sel, 1] = my[0, cols[sel]]
    return out


def _edge_reads(
    lens: LensMetadata, shape: tuple[int, int], corrections: LensCorrections, applied: tuple[LensWarp, ...], scale: float
) -> bool:
    """True when every output edge pixel reads inside the image at every stage of the warp chain.

    Each stage is checked before the next lookup clamps its points: apply_lens replicates the edge
    of every intermediate image, not only of the source.
    """
    h, w = shape

    def outside(pts: np.ndarray) -> bool:
        return bool(pts[:, 0].min() < 0 or pts[:, 0].max() > w - 1 or pts[:, 1].min() < 0 or pts[:, 1].max() > h - 1)

    step = max(1, h // _FILL_EDGE_ROWS)
    side_rows = range(step, h - 1, step)
    last = replace(lens, fill_scale=scale)
    for channel in range(3) if corrections.ca else (1,):
        parts = []
        for row in (0, h - 1):
            mx, my = applied[-1].remap(last, shape, row, row + 1, channel, corrections)
            parts.append(np.stack([mx[0], my[0]], axis=1))
        for row in side_rows:
            mx, my = applied[-1].remap(last, shape, row, row + 1, channel, corrections)
            parts.append(np.array([[mx[0, 0], my[0, 0]], [mx[0, -1], my[0, -1]]]))
        pts = np.concatenate(parts).astype(np.float64)
        if outside(pts):
            return False
        for warp in reversed(applied[:-1]):
            pts = _lookup(warp, lens, shape, channel, corrections, pts)
            if outside(pts):
                return False
    return True


def _search_fill_scale(lens: LensMetadata, shape: tuple[int, int], corrections: LensCorrections, applied: tuple[LensWarp, ...]) -> float:
    if _edge_reads(lens, shape, corrections, applied, 1.0):
        return 1.0
    if not _edge_reads(lens, shape, corrections, applied, _FILL_MIN):
        return 1.0
    lo, hi = _FILL_MIN, 1.0
    for _ in range(_FILL_STEPS):
        mid = 0.5 * (lo + hi)
        if _edge_reads(lens, shape, corrections, applied, mid):
            lo = mid
        else:
            hi = mid
    return lo


_cached_fill_scale = lru_cache(maxsize=16)(_search_fill_scale)


def fill_scale(lens: LensMetadata, shape: tuple[int, ...], corrections: LensCorrections) -> float:
    """Largest scale up to 1 at which the corrected frame reads no pixel from past the source edge.

    Never above 1, so a profile is not extrapolated past the frame it was calibrated on. A CA-only
    correction keeps 1: its green map is the identity, and scaling red and blue alone would misregister them.
    """
    applied = _applied(lens, corrections)
    if not corrections.distortion or not any(w.has_distortion for w in applied):
        return 1.0
    size = (int(shape[0]), int(shape[1]))
    try:
        hash((lens, applied))
    except TypeError:  # an unhashable warp cannot key the cache
        return _search_fill_scale(lens, size, corrections, applied)
    return _cached_fill_scale(lens, size, corrections, applied)


def apply_lens(
    img: ImageBuffer,
    lens: LensMetadata,
    orientation: int = 1,
    corrections: LensCorrections = LensCorrections(True, True),
) -> ImageBuffer:
    """Apply selected embedded warps, with bounded temporary map memory."""
    if not corrections or not lens.available or min(img.shape[:2]) < 2:
        return img
    inverse_orientation = {6: 8, 8: 6}.get(orientation, orientation)
    source = np.ascontiguousarray(apply_exif_orientation(img, inverse_orientation))
    applied = _applied(lens, corrections)
    # The last warp's map is read first, at the output pixel, so the fill scale belongs to it.
    scale = fill_scale(lens, source.shape, corrections)
    filled = replace(lens, fill_scale=scale) if scale != 1.0 else lens
    for i, warp in enumerate(applied):
        warp_lens = filled if i == len(applied) - 1 else lens
        h, w = source.shape[:2]
        result = np.empty_like(source)
        for channel in range(3):
            plane = np.ascontiguousarray(source[..., channel])
            for start in range(0, h, 256):
                stop = min(start + 256, h)
                mx, my = warp.remap(warp_lens, source.shape, start, stop, channel, corrections)
                result[start:stop, :, channel] = cv2.remap(plane, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # Flat-field gains can exceed white before sensor unmix.
        source = np.maximum(result, 0.0, out=result)
    return np.ascontiguousarray(apply_exif_orientation(source, orientation))
