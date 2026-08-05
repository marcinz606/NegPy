"""Hue Trim: undoes the hue rotation an unusual scanning light imposes.

A rotation, not a cast. Measured on one negative under two lights, |ΔH| held 16-21° across
CIELAB chroma 4 to 60+; a cast shrinks with chroma, a channel mix grows. Rotating about the
neutral axis therefore fixes neutrals, so it cannot fight the cast removal in normalization.

Runs on the scene-linear print, and inside RenderIntent.FLAT: a light's hue error is a capture
defect like the sensor unmix, not a look.
"""

from __future__ import annotations

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working
from negpy.kernel.image.validation import ensure_image


def apply_hue_trim(img: ImageBuffer, degrees: float) -> ImageBuffer:
    """Rotate every hue by `degrees` in the working CIELAB a*b* plane. Identity at 0."""
    if degrees == 0.0:
        return img
    lab = rgb_to_lab_working(np.asarray(img[:, :, :3], dtype=np.float32))
    rad = float(np.radians(degrees))
    cos, sin = float(np.cos(rad)), float(np.sin(rad))
    a = lab[:, :, 1].copy()
    b = lab[:, :, 2].copy()
    lab[:, :, 1] = a * cos - b * sin
    lab[:, :, 2] = a * sin + b * cos
    return ensure_image(np.clip(lab_to_rgb_working(lab), 0.0, 1.0))
