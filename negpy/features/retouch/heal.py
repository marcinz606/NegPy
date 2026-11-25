"""Content-aware manual heal strokes (RetouchConfig.manual_heal_strokes and legacy
manual_dust_spots), applied in order on the linear source via multi-scale PatchMatch
and screened-Poisson image melding in log-transmittance space."""

import math
from typing import Iterable, List, Optional, Tuple

import negpy_core
import numpy as np

from negpy.features.geometry.logic import smooth_polyline
from negpy.features.retouch.models import HEAL_SIZE_REF

_MAX_BRUSH_PX = 4096


def _prepare_stroke(points, size: float, w: int, h: int) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray, int]]:
    """Convert normalized stroke ``(points, size)`` to ``(rect, crop_path_i32, brush_px)``."""
    if not points or h < 2 or w < 2:
        return None
    fsize = float(size)
    if not math.isfinite(fsize) or fsize <= 0.0:
        return None
    chain: List[Tuple[float, float]] = []
    for p in points:
        px, py = float(p[0]) * w, float(p[1]) * h
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        chain.append((px, py))
    radius = max(1.0, fsize * max(w, h) / HEAL_SIZE_REF * 0.5)
    brush = max(1, min(_MAX_BRUSH_PX, int(round(radius))))
    if len(chain) >= 3:
        chain = smooth_polyline(chain, closed=False)
    xs = [c[0] for c in chain]
    ys = [c[1] for c in chain]
    min_span = min(max(xs) - min(xs), max(ys) - min(ys))
    pad = int(brush + max(20.0, min(384.0, 7.6 * brush), min(120.0, 0.4 * min_span + 2.0 * brush)))
    x0 = max(0, int(math.floor(min(xs))) - pad)
    y0 = max(0, int(math.floor(min(ys))) - pad)
    x1 = min(w, int(math.ceil(max(xs))) + pad + 1)
    y1 = min(h, int(math.ceil(max(ys))) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    path = np.round(np.array(chain, dtype=np.float64) - (x0, y0)).clip(-1_000_000, 1_000_000).astype(np.int32)
    if len(path) > 1:
        keep = np.empty(len(path), dtype=bool)
        keep[0] = True
        keep[1:] = (path[1:] != path[:-1]).any(axis=1)
        path = path[keep]
    path = np.ascontiguousarray(path)
    return (x0, y0, x1, y1), path, brush


def apply_heal_stroke(out: np.ndarray, points, size: float) -> None:
    """One stroke in place on ``out``: multi-scale PatchMatch + screened-Poisson image melding
    inside the brush footprint, preserving unaveraged film grain from clean donor patches."""
    if out.ndim != 3 or out.shape[2] != 3 or out.dtype != np.float32 or not out.flags.c_contiguous:
        raise ValueError("needs a C-contiguous float32 HxWx3 image")
    h, w = out.shape[:2]
    prep = _prepare_stroke(points, size, w, h)
    if prep is None:
        return
    rect, path, brush = prep
    negpy_core.heal_stroke(out, rect, path, brush)


def apply_heal_strokes(
    img: np.ndarray,
    strokes: Iterable[Tuple],
    legacy_spots: Iterable[Tuple[float, float, float]] = (),
) -> np.ndarray:
    """A new buffer with legacy dust spots and manual heal strokes applied in order."""
    out = np.array(img, dtype=np.float32, copy=True, order="C")
    h, w = out.shape[:2]
    batch: List[Tuple[Tuple[int, int, int, int], np.ndarray, int]] = []
    for nx, ny, size in legacy_spots:
        prep = _prepare_stroke([[nx, ny]], float(size), w, h)
        if prep is not None:
            batch.append(prep)
    for stroke in strokes:
        prep = _prepare_stroke(stroke[0], float(stroke[1]), w, h)
        if prep is not None:
            batch.append(prep)
    if batch:
        negpy_core.heal_strokes(out, batch)
    return out
