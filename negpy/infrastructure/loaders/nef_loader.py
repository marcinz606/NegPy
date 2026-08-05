import os
from typing import Any, ContextManager, Optional, Tuple

import numpy as np
import tifffile

from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, identify_color_space_from_icc, read_orientation
from negpy.infrastructure.loaders.ir_planes import normalize_ir_to_float32
from negpy.kernel.image.logic import srgb_to_linear, uint8_to_float32, uint16_to_float32
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


def is_coolscan_nef(file_path: str) -> bool:
    """True if this NEF is a Nikon Coolscan scanner file (already-processed RGB in SubIFDs)."""
    if os.path.splitext(file_path)[1].lower() != ".nef":
        return False
    try:
        with tifffile.TiffFile(file_path) as tif:
            return _find_rgb_subifd(tif) is not None
    except Exception:
        return False


class NefLoader(IImageLoader):
    """Loader for Nikon Coolscan scanner NEF files.

    These are TIFF-structured files with the full-res processed RGB image in a
    SubIFD chain (tag 0x014A). The data is Nikon Scan's output — curves, gain,
    and optionally DigitalICE are already applied — not raw sensor data.

    Color space handling follows TiffLoader: ICC profile → identify space →
    linearise if sRGB. Untagged 16-bit is assumed linear; untagged 8-bit is
    assumed sRGB.
    """

    def load(self, file_path: str, linear_raw: bool = False) -> Tuple[ContextManager[Any], dict]:
        with tifffile.TiffFile(file_path) as tif:
            sub = _find_rgb_subifd(tif)
            if sub is None:
                raise ValueError(f"No RGB SubIFD in {file_path}")
            arr = sub.asarray()

            icc_bytes: Optional[bytes] = None
            for page in (sub, tif.pages[0]):
                tags = getattr(page, "tags", None)
                if tags is None:
                    continue
                tag = tags.get("InterColorProfile")
                if tag is not None and tag.value:
                    icc_bytes = bytes(tag.value)
                    break

        ir: Optional[np.ndarray] = None
        if arr.ndim == 3 and arr.shape[2] == 4:
            ir = normalize_ir_to_float32(arr[:, :, 3])
            arr = np.ascontiguousarray(arr[:, :, :3])
        elif arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        if arr.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(arr))
        elif arr.dtype == np.uint16:
            f32 = uint16_to_float32(np.ascontiguousarray(arr))
        else:
            f32 = np.clip(arr.astype(np.float32), 0, 1)

        color_space = None
        if not linear_raw:
            color_space = identify_color_space_from_icc(icc_bytes)
            if color_space is None and arr.dtype == np.uint8:
                color_space = ColorSpace.SRGB.value
            if color_space == ColorSpace.SRGB.value:
                f32 = srgb_to_linear(f32)

        metadata = {
            "orientation": read_orientation(file_path),
            "color_space": color_space,
            "icc_profile": icc_bytes,
            "ir": ir,
        }
        return NonStandardFileWrapper(f32), metadata
