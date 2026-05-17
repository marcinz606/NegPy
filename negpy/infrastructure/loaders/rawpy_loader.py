import os
from typing import Any, ContextManager, Optional, Tuple

import numpy as np
import rawpy
import tifffile

from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, detect_color_space_from_raw
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# DNG PhotometricInterpretation value for LinearRaw (TIFF/EP §6.10.4).
_LINEAR_RAW = 34892


def _peek_linearraw_4ch(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Inspect a DNG. If it carries 4 linear samples (RGB + IR), return (rgb, ir) as float32 [0,1].

    NegPy's own `write_dng_linear` produces exactly this format; we close the loop here.
    Returns None for camera DNGs (Bayer, 3-channel, etc.) so rawpy can handle them.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".dng":
        return None
    try:
        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0]
            tags = getattr(page, "tags", None)
            if tags is None:
                return None
            spp_tag = tags.get("SamplesPerPixel")
            photo_tag = tags.get("PhotometricInterpretation")
            spp = int(spp_tag.value) if spp_tag is not None else 0
            photo = int(photo_tag.value) if photo_tag is not None else 0
            if spp != 4 or photo != _LINEAR_RAW:
                return None
            arr = page.asarray()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"DNG peek failed for {file_path}: {e}")
        return None

    if arr.ndim != 3 or arr.shape[2] != 4:
        return None

    if arr.dtype == np.uint16:
        scale = 1.0 / 65535.0
    elif arr.dtype == np.uint8:
        scale = 1.0 / 255.0
    else:
        scale = 1.0
    full = np.clip(arr.astype(np.float32) * scale, 0.0, 1.0)
    rgb = np.ascontiguousarray(full[:, :, :3])
    ir = np.ascontiguousarray(full[:, :, 3])
    return rgb, ir


class RawpyLoader(IImageLoader):
    """
    Standard RAW loader (libraw). For LinearRaw 4-channel DNGs (RGB + IR), bypasses
    rawpy and reads via tifffile so the IR plane is preserved.
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        peeked = _peek_linearraw_4ch(file_path)
        if peeked is not None:
            rgb, ir = peeked
            metadata = {
                "orientation": 0,
                "raw_flip": 0,
                "color_space": ColorSpace.ADOBE_RGB.value,
                "ir": ir,
            }
            return NonStandardFileWrapper(rgb), metadata

        raw = rawpy.imread(file_path)

        metadata = {
            "orientation": 0,
            "raw_flip": 0,
            "color_space": detect_color_space_from_raw(raw) or "Adobe RGB",
            "ir": None,
        }

        return raw, metadata
