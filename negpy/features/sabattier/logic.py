from typing import Any, Dict

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.lab.logic import gaussian_kernel_1d
from negpy.kernel.image.validation import ensure_image

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
    """
    Sabattier effect on a linear-reflectance print: white light part-way through
    development. Where silver has developed the re-exposure finds little fresh halide, so
    dense tones hold; the lighter the tone, the more fresh halide develops after the flash,
    so the scale folds at the re-exposure density R = reexposure * d_max: a tone D below it
    prints at D + strength (R - D), paper white at strength * R. Strength 1 flattens the
    light tones onto the fold; above 1 they reverse, the lighter the darker.

    Bromide released by the developing dense areas diffuses into the adjoining light ones
    and restrains their re-development, which leaves a light line along every edge between
    dark and light: the Mackie line. `line_sigma_px` is the bromide's reach in render
    pixels; 0 draws no lines. The blur's taps are `gaussian_kernel_1d`'s, the same array
    the shaders read, so both engines draw the same line. The result is neutral silver.
    """
    if not enabled:
        return img

    c = SABATTIER_CONSTANTS
    arr = np.clip(img.astype(np.float32), 1e-6, 1.0)
    luma = arr[:, :, 0] * np.float32(0.2126) + arr[:, :, 1] * np.float32(0.7152) + arr[:, :, 2] * np.float32(0.0722)
    dens = -np.log10(np.clip(luma, 1e-6, 1.0))

    re = np.float32(reexposure * d_max)
    w = np.float32(c["fold_width"])
    shortfall = w * np.log1p(np.exp(np.clip((re - dens) / w, -30.0, 30.0)))
    reversal = np.float32(strength) * shortfall
    if line_sigma_px > 0.0:
        developed = 1.0 / (1.0 + np.exp(-np.clip((dens - re) / np.float32(c["edge_width"]), -30.0, 30.0)))
        k = gaussian_kernel_1d(float(line_sigma_px))
        bromide = cv2.sepFilter2D(developed.astype(np.float32), -1, k, k, borderType=cv2.BORDER_REPLICATE)
        reversal = reversal * (np.float32(1.0) - bromide)

    grey = np.power(np.float32(10.0), -np.minimum(dens + reversal, np.float32(d_max))).astype(np.float32)
    return ensure_image(np.clip(np.dstack([grey, grey, grey]), 0.0, 1.0))


def line_sigma_px(agitation: float, shape: tuple) -> float:
    """The Mackie line's blur sigma in render pixels, a fraction of the short side so preview
    and export draw the same line: the still-bath reach at agitation 0, none at 1."""
    reach = SABATTIER_CONSTANTS["still_reach"] * min(max(1.0 - float(agitation), 0.0), 1.0)
    return reach * 0.01 * float(min(shape[0], shape[1]))
