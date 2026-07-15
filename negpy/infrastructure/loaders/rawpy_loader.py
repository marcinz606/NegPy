import os
from typing import Any, ContextManager, Optional, Tuple

import numpy as np
import rawpy
import tifffile

from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, read_orientation
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# DNG PhotometricInterpretation value for LinearRaw (TIFF/EP §6.10.4).
_LINEAR_RAW = 34892


def _find_linearraw_page(tif: "tifffile.TiffFile") -> Optional[Any]:
    """Return the page carrying 4-sample LinearRaw data: page 0 itself (NegPy's own
    single-IFD DNGs) or one of its SubIFDs (VueScan/Adobe-style thumbnail + SubIFD DNGs)."""
    page0 = tif.pages[0]
    for page in (page0, *(page0.pages or [])):
        tags = getattr(page, "tags", None)
        if tags is None:
            continue
        spp_tag = tags.get("SamplesPerPixel")
        photo_tag = tags.get("PhotometricInterpretation")
        spp = int(spp_tag.value) if spp_tag is not None else 0
        photo = int(photo_tag.value) if photo_tag is not None else 0
        if spp == 4 and photo == _LINEAR_RAW:
            return page
    return None


def _peek_linearraw_4ch(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Inspect a DNG. If it carries 4 linear samples (RGB + IR), return (rgb, ir) as float32 [0,1].

    NegPy's own `write_dng_linear` produces a single-IFD DNG; VueScan and Adobe-style DNGs
    put the full-res data in a SubIFD behind a reduced-resolution thumbnail IFD0 — both are
    checked. Returns None for camera DNGs (Bayer, 3-channel, etc.) so rawpy can handle them.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".dng":
        return None
    try:
        with tifffile.TiffFile(file_path) as tif:
            page = _find_linearraw_page(tif)
            if page is None:
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
                "orientation": read_orientation(file_path),
                "raw_flip": 0,
                # Sensor-native linear samples; no ColorSpace names them.
                "color_space": None,
                "ir": ir,
            }
            return NonStandardFileWrapper(rgb), metadata

        raw = rawpy.imread(file_path)

        metadata = {
            "orientation": read_orientation(file_path),
            "raw_flip": 0,
            # Decoded output_color=raw, so the file's own tags characterise nothing here.
            "color_space": None,
            "ir": None,
        }

        return raw, metadata
