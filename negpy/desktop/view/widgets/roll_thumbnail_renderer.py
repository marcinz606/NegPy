"""Display-only rendering for low-resolution roll-index thumbnails."""

from __future__ import annotations

import numpy as np


def render_roll_thumbnail_rgb8(thumbnail: object) -> np.ndarray:
    """Turn one scanner-linear negative thumbnail into an upright RGB8 preview.

    The roll index is an alignment aid, not an archival render. Per-channel
    percentile levels make the orange mask readable, while a direct linear
    inversion preserves highlight detail. Applying another 1/2.2 curve here
    made the already display-ready contact sheet look roughly one stop too
    bright even though the 16-bit scanner data was not clipped.
    """

    raw = np.asarray(thumbnail)
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("roll thumbnail must be an HxWx3 RGB array")
    if raw.dtype != np.uint16:
        raise ValueError("roll thumbnail must contain uint16 scanner samples")

    pixels = np.rot90(raw.astype(np.float32), k=1, axes=(0, 1))
    low = np.percentile(pixels, 1.0, axis=(0, 1), keepdims=True)
    high = np.percentile(pixels, 99.0, axis=(0, 1), keepdims=True)
    positive = np.clip(1.0 - (pixels - low) / np.maximum(high - low, 1.0), 0.0, 1.0)
    return np.ascontiguousarray(np.rint(positive * 255.0).astype(np.uint8))


__all__ = ["render_roll_thumbnail_rgb8"]
