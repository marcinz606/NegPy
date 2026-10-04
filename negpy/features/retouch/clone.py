"""Clone strokes (RetouchConfig.clone_strokes), applied in order on the linear source, so a later
stroke can copy from an earlier one."""

import hashlib
from typing import Iterable, Tuple

import cv2
import numpy as np

from negpy.features.geometry.logic import smooth_polyline
from negpy.features.retouch.models import HEAL_SIZE_REF

# Tone match gain, six stops either way; also bounds the ratio where the source surround is near black.
_MATCH_GAIN_MIN = 1.0 / 64.0
_MATCH_GAIN_MAX = 64.0
_EPS = 1e-6


def clone_token(retouch) -> str:
    """Config identity of the clone strokes, folded into source_hash."""
    strokes = getattr(retouch, "clone_strokes", [])
    if not strokes:
        return ""
    return "|clone" + hashlib.sha1(repr(strokes).encode()).hexdigest()[:12]


def _cover(points, radius: float, x0: int, y0: int, shape: Tuple[int, int], w: int, h: int) -> np.ndarray:
    chain = [(float(p[0]) * w, float(p[1]) * h) for p in points]
    if len(chain) >= 3:
        chain = smooth_polyline(chain, closed=False)
    local = np.round(np.array(chain, dtype=np.float32) - (x0, y0)).astype(np.int32)
    cover = np.zeros(shape, dtype=np.uint8)
    r = max(1, int(round(radius)))
    if len(local) > 1:
        cv2.polylines(cover, [local], False, 1, thickness=2 * r)
    for cx, cy in local:
        cv2.circle(cover, (int(cx), int(cy)), r, 1, -1)
    return cover


def _local_mean(plane: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian mean of ``plane`` over the pixels ``weight`` admits (normalized convolution)."""
    num = cv2.GaussianBlur(plane * weight[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
    return num / np.maximum(den, _EPS)


def apply_clone_stroke(out: np.ndarray, stroke) -> None:
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

    cover = _cover(points, radius, x0, y0, (y1 - y0, x1 - x0), w, h)
    if not cover.any():
        return
    # Source pixels off the frame clamp to the edge and carry no weight.
    yy = np.arange(y0, y1) + sy
    xx = np.arange(x0, x1) + sx
    valid = ((yy >= 0) & (yy < h))[:, None] & ((xx >= 0) & (xx < w))[None, :]
    src = out[np.clip(yy, 0, h - 1)[:, None], np.clip(xx, 0, w - 1)[None, :]].astype(np.float32)
    dst = out[y0:y1, x0:x1].astype(np.float32)

    if feather > 0.0:
        # The ramp ends at the brush core at most, so full feather still covers the center.
        depth = cv2.distanceTransform(cover, cv2.DIST_L2, 3)
        alpha = np.clip(depth / max(float(feather) * min(radius, float(depth.max())), _EPS), 0.0, 1.0)
    else:
        alpha = cover.astype(np.float32)
    alpha *= float(strength) * valid

    patch = src
    if match_tone:
        # Healing-brush tone match: the source keeps its texture and takes the destination's low frequencies.
        # Both means read only pixels outside the brush, so the covered defect never tints its patch.
        ring = ((cover == 0) & valid).astype(np.float32)
        if ring.any():
            gain = _local_mean(dst, ring, sigma) / np.maximum(_local_mean(src, ring, sigma), _EPS)
            patch = src * np.clip(gain, _MATCH_GAIN_MIN, _MATCH_GAIN_MAX)

    a = alpha[..., None]
    out[y0:y1, x0:x1] = dst * (1.0 - a) + patch * a


def apply_clone_strokes(img: np.ndarray, strokes: Iterable) -> np.ndarray:
    """A new buffer with every stroke applied in order (buffers are read-only)."""
    out = np.array(img, dtype=np.float32, copy=True)
    for stroke in strokes:
        apply_clone_stroke(out, stroke)
    return out
