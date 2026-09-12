"""Inverse lens maps in the scanning camera's linear RGB coordinates."""

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.lens.models import LensMetadata
from negpy.kernel.image.logic import apply_exif_orientation


def apply_lens(img: ImageBuffer, lens: LensMetadata, orientation: int = 1) -> ImageBuffer:
    """Apply all supported embedded warps, with bounded temporary map memory."""
    if not lens.available or min(img.shape[:2]) < 2:
        return img
    inverse_orientation = {6: 8, 8: 6}.get(orientation, orientation)
    source = np.ascontiguousarray(apply_exif_orientation(img, inverse_orientation))
    for warp in lens.warps:
        h, w = source.shape[:2]
        result = np.empty_like(source)
        for channel in range(3):
            plane = np.ascontiguousarray(source[..., channel])
            for start in range(0, h, 256):
                stop = min(start + 256, h)
                mx, my = warp.remap(lens, source.shape, start, stop, channel)
                result[start:stop, :, channel] = cv2.remap(plane, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        source = np.clip(result, 0.0, 1.0, out=result)
    return np.ascontiguousarray(apply_exif_orientation(source, orientation))
