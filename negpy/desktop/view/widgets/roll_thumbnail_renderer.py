"""Display-only rendering for low-resolution roll-index thumbnails."""

from __future__ import annotations

import numpy as np

from negpy.infrastructure.scanners.settings import (
    DEFAULT_PREVIEW_METER_INSET_PERCENT,
    MAX_PREVIEW_METER_INSET_PERCENT,
)


def render_roll_thumbnail_rgb8(
    thumbnail: object,
    *,
    meter_inset_percent: int = DEFAULT_PREVIEW_METER_INSET_PERCENT,
) -> np.ndarray:
    """Turn one scanner-linear negative thumbnail into an upright RGB8 preview.

    The roll index is an alignment aid, not an archival render. Per-channel
    percentile levels from the configured center region make the orange mask
    readable without letting clear rebate or transport edges set the display
    exposure. The default 10% inset meters the center 80%. The whole crop
    remains visible for alignment. A direct linear inversion preserves
    highlight detail; applying another 1/2.2 curve made the contact sheet look
    roughly one stop too bright even though the scanner data was not clipped.
    """

    raw = np.asarray(thumbnail)
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("roll thumbnail must be an HxWx3 RGB array")
    if raw.dtype != np.uint16:
        raise ValueError("roll thumbnail must contain uint16 scanner samples")
    if type(meter_inset_percent) is not int or not 0 <= meter_inset_percent <= MAX_PREVIEW_METER_INSET_PERCENT:
        raise ValueError("preview meter inset must be an integer from 0 to 30 percent")

    pixels = np.rot90(raw.astype(np.float32), k=1, axes=(0, 1))
    meter_inset_fraction = meter_inset_percent / 100
    row_inset = round(pixels.shape[0] * meter_inset_fraction)
    column_inset = round(pixels.shape[1] * meter_inset_fraction)
    if row_inset * 2 >= pixels.shape[0] or column_inset * 2 >= pixels.shape[1]:
        meter = pixels
    else:
        meter = pixels[
            row_inset : pixels.shape[0] - row_inset,
            column_inset : pixels.shape[1] - column_inset,
        ]
    low = np.percentile(meter, 1.0, axis=(0, 1), keepdims=True)
    high = np.percentile(meter, 99.0, axis=(0, 1), keepdims=True)
    positive = np.clip(1.0 - (pixels - low) / np.maximum(high - low, 1.0), 0.0, 1.0)
    return np.ascontiguousarray(np.rint(positive * 255.0).astype(np.uint8))


__all__ = ["render_roll_thumbnail_rgb8"]
