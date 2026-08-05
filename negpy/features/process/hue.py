"""Hue Trim — corrects the hue rotation an unusual scanning light imposes on every colour.

A narrowband or odd-phosphor light source rotates hues by a roughly *constant* angle rather than
casting them: measured on one negative scanned under two lights, |ΔH| stayed within 16-21° across
CIELAB chroma 4 to 60+. A colour cast would have shrunk with chroma (a fixed a*b* offset barely
turns a saturated colour) and a general channel mix would have grown, so the correction is a single
rotation about the neutral axis. That also leaves neutrals exactly where the base-anchored cast
removal put them, so this cannot fight normalization.

Applied to the print (scene-linear positive, after the print curve) because the reference
measurement was taken there, and inside the flat render intent because a light's hue error is a
capture defect, not a look — a digital-intermediate master should already be free of it.
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
