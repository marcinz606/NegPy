import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image


def apply_luminance_vignette(img: ImageBuffer, strength: float, feather: float, roundness: float = 0.0) -> ImageBuffer:
    """
    Luminance vignette in scene-linear: a scalar gain on every channel moves
    luma, not color. Mirrors vignette.wgsl so CPU and GPU stay in parity.
    """
    if strength == 0.0:
        return img
    return ensure_image(
        negpy_core.apply_luminance_vignette(
            np.ascontiguousarray(img, dtype=np.float32),
            float(strength),
            float(feather),
            float(roundness),
        )
    )
