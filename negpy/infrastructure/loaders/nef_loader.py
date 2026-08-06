import os
from typing import Any, ContextManager, Optional, Tuple

import numpy as np
import tifffile

from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, read_orientation
from negpy.kernel.image.logic import uint8_to_float32, uint16_to_float32
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _find_rgb_subifd(tif: tifffile.TiffFile) -> Optional[Any]:
    """Return the largest RGB SubIFD (Coolscan NEF stores full-res here via tag 0x014A)."""
    page0 = tif.pages[0]
    best = None
    best_pixels = 0
    for sub in page0.pages or []:
        tags = getattr(sub, "tags", None)
        if tags is None:
            continue
        spp_tag = tags.get("SamplesPerPixel")
        photo_tag = tags.get("PhotometricInterpretation")
        if spp_tag is None or photo_tag is None:
            continue
        spp = int(spp_tag.value)
        photo = int(photo_tag.value)
        if spp < 3 or photo != 2:
            continue
        pixels = sub.shape[0] * sub.shape[1]
        if pixels > best_pixels:
            best = sub
            best_pixels = pixels
    return best


def _has_cfa_subifd(tif: tifffile.TiffFile) -> bool:
    """True if any SubIFD carries Bayer/CFA data (camera NEF indicator).

    Camera NEFs embed an RGB preview SubIFD alongside the Bayer mosaic data,
    so "has RGB SubIFD" alone is not enough to detect a scanner NEF — we must
    also confirm there's no CFA data.
    """
    page0 = tif.pages[0]
    for sub in page0.pages or []:
        tags = getattr(sub, "tags", None)
        if tags is None:
            continue
        photo_tag = tags.get("PhotometricInterpretation")
        comp_tag = tags.get("Compression")
        spp_tag = tags.get("SamplesPerPixel")
        if photo_tag is not None and int(photo_tag.value) == 32803:
            return True
        if comp_tag is not None and int(comp_tag.value) == 34713:
            return True
        spp = int(spp_tag.value) if spp_tag is not None else 1
        if spp == 1 and sub.shape[0] * sub.shape[1] > 100_000:
            return True
    return False


def is_coolscan_nef(file_path: str) -> bool:
    """True if this NEF is a Nikon Coolscan scanner file (RGB SubIFDs, no CFA data)."""
    if os.path.splitext(file_path)[1].lower() != ".nef":
        return False
    try:
        with tifffile.TiffFile(file_path) as tif:
            if _has_cfa_subifd(tif):
                return False
            return _find_rgb_subifd(tif) is not None
    except Exception:
        return False


class NefLoader(IImageLoader):
    """Loader for Nikon Coolscan scanner NEF files.

    These are TIFF-structured files with the full-res processed RGB image in a
    SubIFD chain (tag 0x014A). The data is Nikon Scan's output — curves, gain,
    and optionally DigitalICE are already applied — not raw sensor data.

    Data is returned as-is — no color-space assumptions or linearization.
    """

    def load(self, file_path: str, linear_raw: bool = False) -> Tuple[ContextManager[Any], dict]:
        with tifffile.TiffFile(file_path) as tif:
            sub = _find_rgb_subifd(tif)
            if sub is None:
                raise ValueError(f"No RGB SubIFD in {file_path}")
            arr = sub.asarray()

        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = np.ascontiguousarray(arr[:, :, :3])
        elif arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        if arr.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(arr))
        elif arr.dtype == np.uint16:
            f32 = uint16_to_float32(np.ascontiguousarray(arr))
        else:
            f32 = np.clip(arr.astype(np.float32), 0, 1)

        metadata = {
            "orientation": read_orientation(file_path),
        }
        return NonStandardFileWrapper(f32), metadata
