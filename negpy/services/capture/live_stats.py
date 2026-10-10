"""Clipping masks for the camera live view."""

from __future__ import annotations

import numpy as np

#: The preview range's ends, in 8-bit levels: the bins the main panel's histogram counts as
#: clipped. The preview JPEG clips before the RAW does, so a zebra is an early warning.
ZEBRA_HIGH = 253
ZEBRA_LOW = 2


def clip_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(highlights, blacks)` boolean `HxW` masks for an `HxWx3` uint8 frame: any channel at
    the top of the range, every channel at the bottom."""
    top = np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    return top >= ZEBRA_HIGH, top <= ZEBRA_LOW
