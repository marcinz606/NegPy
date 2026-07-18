"""Half-frame scans: one file holds two frames side by side.

A half asset is a normal asset dict plus ``half`` (1 = left, 2 = right) and
``split_x`` (normalized gutter position). Its identity is the file hash
suffixed with ``#<half>``, so every hash-keyed store (edits, history, marks,
thumbnails) is per-frame automatically. Decode caches key on the unsuffixed
hash so both halves share one decode.
"""

from typing import Any, Dict, Optional

import numpy as np

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_SEP = "#"


def half_hash(file_hash: str, half: int) -> str:
    return f"{file_hash}{_SEP}{half}"


def base_hash(file_hash: Optional[str]) -> Optional[str]:
    """The unsuffixed file hash — the decode-cache identity shared by both halves."""
    return file_hash.split(_SEP, 1)[0] if file_hash else file_hash


def half_name(name: str, half: int) -> str:
    return f"{name} [{half}]"


def slice_half(buf: np.ndarray, half: int, split_x: float) -> np.ndarray:
    """View of one half of a decoded buffer, split at the normalized gutter x."""
    w = buf.shape[1]
    xs = min(max(int(round(w * split_x)), 1), w - 1)
    return buf[:, :xs] if half == 1 else buf[:, xs:]


def slice_for_asset(buf: np.ndarray, file_info: Dict[str, Any]) -> np.ndarray:
    """Apply the asset's half slice; no-op for whole-frame assets."""
    half = int(file_info.get("half") or 0)
    if not half:
        return buf
    return slice_half(buf, half, float(file_info.get("split_x") or 0.5))


def detect_split_x(buf: np.ndarray) -> float:
    """Normalized x of the unexposed gutter between the two frames.

    The gutter is a narrow column extremal against its surroundings in either
    polarity (bright film base on negatives, dark on positives), so pick the
    column whose smoothed luma deviates most from a local running-median
    background — a window much wider than the gutter, so broad brightness
    differences between the two frames don't register. Returns 0.5 when no
    clear gutter stands out in the central band.
    """
    # ponytail: 1-D local-deviation heuristic; upgrade to variance+edge profile if it misses
    a = np.asarray(buf)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32, copy=False)
    h, w = a.shape[:2]
    if w < 64 or h < 8:
        return 0.5
    peak_val = float(a.max())
    if peak_val <= 0:
        return 0.5
    sub = a[:: max(1, h // 512)] / peak_val
    col = sub.mean(axis=0)
    k = max(3, w // 150)
    sm = np.convolve(col, np.ones(k, np.float32) / k, mode="same")
    win = max(9, (w // 8) | 1)
    padded = np.pad(sm, win // 2, mode="edge")
    bg = np.median(np.lib.stride_tricks.sliding_window_view(padded, win), axis=1)
    dev = np.abs(sm - bg)
    lo, hi = int(w * 0.35), int(w * 0.65)
    peak = lo + int(np.argmax(dev[lo:hi]))
    # Take the deviating band's center so the ±delta taps below land outside the gutter.
    thr = 0.5 * dev[peak]
    i0 = peak
    while i0 > 0 and dev[i0 - 1] >= thr:
        i0 -= 1
    i1 = peak
    while i1 < w - 1 and dev[i1 + 1] >= thr:
        i1 += 1
    center = (i0 + i1) // 2
    # A gutter is extremal against BOTH sides; a step edge (up one side, down the
    # other) is in-scene — reject it.
    delta = max(3, int(w * 0.05))
    d1 = float(sm[center] - sm[max(0, center - delta)])
    d2 = float(sm[center] - sm[min(w - 1, center + delta)])
    if min(abs(d1), abs(d2)) < 0.04 or d1 * d2 <= 0:
        return 0.5
    # Unexposed film is uniform top to bottom; a bright/dark in-scene feature isn't.
    if float(sub[:, center].std()) > 0.10:
        return 0.5
    return center / w


def detect_split_x_for_file(file_path: str) -> float:
    """Gutter position from a small decode of the file; 0.5 on any failure."""
    try:
        from negpy.services.assets.thumbnails import decode_source_image

        img = decode_source_image(file_path)
        if img is None:
            return 0.5
        img.thumbnail((1024, 1024))
        return detect_split_x(np.asarray(img))
    except Exception as e:
        logger.warning("Half-frame split detection failed for %s: %s", file_path, e)
        return 0.5
