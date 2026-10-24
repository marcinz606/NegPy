from typing import Any, Dict

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.lab.logic import gaussian_kernel_1d

SABATTIER_CONSTANTS: Dict[str, Any] = {
    # Softness of the fold at the re-exposure density, in D. The softplus below turns the
    # hard "tones lighter than R reverse" rule into a rounded knee of this width.
    "fold_width": 0.08,
    # Width of the dense/light boundary the bromide mask is cut at, in D: the silver that
    # counts as developed, and so as a bromide source, when the re-exposure comes.
    "edge_width": 0.10,
    # The bromide's reach in a still bath, in per-cent of the frame's short side. Agitation
    # sweeps the bromide off the edges and shortens it to nothing.
    "still_reach": 1.0,
}


def apply_sabattier(
    img: ImageBuffer,
    d_max: float,
    enabled: bool = False,
    strength: float = 1.3,
    reexposure: float = 0.45,
    line_sigma_px: float = 0.0,
) -> ImageBuffer:
    """Sabattier on a linear-reflectance print: developed silver leaves little fresh halide, so the scale folds at
    R = reexposure * d_max and lighter tones reverse past strength 1. Bromide from dense areas leaves a light Mackie
    line along each edge, `line_sigma_px` wide (0: none). The shaders read the same `gaussian_kernel_1d` taps."""
    if not enabled:
        return img

    c = SABATTIER_CONSTANTS
    f32 = np.float32
    scalars = (f32(reexposure * d_max), f32(c["fold_width"]), f32(c["edge_width"]), f32(strength), f32(d_max))
    k = gaussian_kernel_1d(float(line_sigma_px)) if line_sigma_px > 0.0 else np.zeros(0, np.float32)
    return negpy_core.sabattier(np.ascontiguousarray(img, dtype=np.float32), scalars, k)


def line_sigma_px(agitation: float, shape: tuple) -> float:
    """The Mackie line's blur sigma in render pixels, a fraction of the short side so preview
    and export draw the same line: the still-bath reach at agitation 0, none at 1."""
    reach = SABATTIER_CONSTANTS["still_reach"] * min(max(1.0 - float(agitation), 0.0), 1.0)
    return reach * 0.01 * float(min(shape[0], shape[1]))
