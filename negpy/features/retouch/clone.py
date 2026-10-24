"""Clone strokes (RetouchConfig.clone_strokes), applied in order on the linear source, so a later
stroke can copy from an earlier one."""

import hashlib
from typing import Iterable

import negpy_core
import numpy as np

from negpy.features.geometry.logic import smooth_polyline
from negpy.features.retouch.models import HEAL_SIZE_REF


def clone_token(retouch) -> str:
    """Config identity of the clone strokes, folded into source_hash."""
    strokes = getattr(retouch, "clone_strokes", [])
    if not strokes:
        return ""
    return "|clone" + hashlib.sha1(repr(strokes).encode()).hexdigest()[:12]


def _path(points, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """The stroke's path, smoothed past two points, in crop pixels."""
    chain = [(float(p[0]) * w, float(p[1]) * h) for p in points]
    if len(chain) >= 3:
        chain = smooth_polyline(chain, closed=False)
    return np.ascontiguousarray(np.round(np.array(chain, dtype=np.float32) - (x0, y0)).astype(np.int32))


def apply_clone_stroke(out: np.ndarray, stroke) -> None:
    """One stroke in place: the source patch over the brush footprint (a thick path with round
    caps), feathered by distance inside it. With match_tone the source keeps its texture and
    takes the destination's low frequencies, both read only outside the brush, as the healing
    brush does, so the covered defect never tints its patch."""
    points, size, dx, dy, strength, feather, match_tone = stroke
    if not points or strength <= 0.0:
        return
    h, w = out.shape[:2]
    radius = max(1.0, float(size) * max(w, h) / HEAL_SIZE_REF * 0.5)
    sx, sy = int(round(float(dx) * w)), int(round(float(dy) * h))
    if sx == 0 and sy == 0:
        return
    sigma = max(2.0, radius)
    pad = int(radius + 3.0 * sigma) + 2
    xs = [float(p[0]) * w for p in points]
    ys = [float(p[1]) * h for p in points]
    x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x1, y1 = min(w, int(max(xs)) + pad + 1), min(h, int(max(ys)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    brush = max(1, int(round(radius)))
    path = _path(points, x0, y0, w, h)
    negpy_core.clone_stroke(
        out, (x0, y0, x1, y1), path, brush, (radius, sigma), (sx, sy), (float(feather), float(strength), bool(match_tone))
    )


def apply_clone_strokes(img: np.ndarray, strokes: Iterable) -> np.ndarray:
    """A new buffer with every stroke applied in order (buffers are read-only)."""
    out = np.array(img, dtype=np.float32, copy=True)
    for stroke in strokes:
        apply_clone_stroke(out, stroke)
    return out
