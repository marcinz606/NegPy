import os
import plistlib
import re
from typing import Any, ContextManager, Optional, Tuple

import numpy as np
import tifffile

from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, read_orientation
from negpy.infrastructure.loaders.logluv import (
    decode_logluv_strips,
)
from negpy.kernel.image.logic import uint8_to_float32, uint16_to_float32
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


_FILM_TYPES = {0: "positive", 1: "negative", 2: "b&w"}


def _parse_fff_plist(raw: bytes) -> dict:
    """Extract FlexColor metadata from tag 50457.

    Returns a flat dict with scanner-relevant fields or {} on failure.
    The plist may have a 4-byte length prefix (FlexColor 4.8.10+) and is
    always null-padded to a fixed block size.
    """
    try:
        xml_start = raw.find(b"<?xml")
        end = raw.find(b"</plist>")
        if xml_start < 0 or end < 0:
            return {}
        plist = plistlib.loads(raw[xml_start : end + len(b"</plist>")])
        settings = plist.get("ImageSettings", [{}])[0]
        ic = settings.get("ImageCorrection", {})
        desc = settings.get("ImageDescription", {})
        created = settings.get("Created", {})

        result: dict = {}
        film_name = settings.get("Name")
        if film_name:
            result["film_stock"] = film_name
        film_type = ic.get("FilmType")
        if film_type is not None:
            result["film_type"] = _FILM_TYPES.get(film_type, str(film_type))
        gamma = ic.get("Gamma")
        if gamma is not None:
            result["flexcolor_gamma"] = round(float(gamma), 2)
        res = desc.get("Resolution")
        if res:
            result["scan_dpi"] = int(res)
        if created.get("Year"):
            result["scan_date"] = f"{created['Year']:04d}-{created.get('Month', 0):02d}-{created.get('Day', 0):02d}"
        return result
    except Exception:
        return {}


def _parse_fff_firmware(raw: bytes) -> dict:
    """Extract FlexColor version and scanner serial from tag 46279."""
    try:
        text = raw.decode("latin1", errors="replace")
        result: dict = {}
        ver = re.search(r"(\d+\.\d+[\.\d]* \w+)", text)
        if ver:
            result["flexcolor_version"] = ver.group(1)
        ser = re.search(r"(FX\d+)", text)
        if ser:
            result["scanner_serial"] = ser.group(1)
        return result
    except Exception:
        return {}


_PHOTOMETRIC_RGB = 2
_PHOTOMETRIC_LOGLUV = 32845


def _find_full_res_ifd(tif: tifffile.TiffFile) -> Optional[Any]:
    """Return the largest image IFD by pixel count.

    FFF files can have multiple IFDs flagged as full-resolution (the SubfileType
    tag is unreliable — e.g. a small secondary image tagged full-res). Pixel
    count is the reliable signal, matching the approach in flexcolor-tool and
    the reference loader. Accepts both RGB and LogLuv photometric.
    """
    best = None
    best_pixels = 0
    for page in tif.pages:
        tags = getattr(page, "tags", None)
        if tags is None:
            continue
        spp_tag = tags.get("SamplesPerPixel")
        photo_tag = tags.get("PhotometricInterpretation")
        bps_tag = tags.get("BitsPerSample")
        if spp_tag is None or photo_tag is None:
            continue
        spp = int(spp_tag.value) if not hasattr(spp_tag.value, "__len__") else int(spp_tag.value[0])
        photo = int(photo_tag.value)
        if photo == _PHOTOMETRIC_LOGLUV:
            pixels = page.shape[0] * page.shape[1] if hasattr(page, "shape") else 0
            if pixels == 0:
                w_tag = tags.get("ImageWidth")
                h_tag = tags.get("ImageLength")
                if w_tag and h_tag:
                    pixels = int(w_tag.value) * int(h_tag.value)
            if pixels > best_pixels:
                best = page
                best_pixels = pixels
            continue
        if spp < 3 or photo != _PHOTOMETRIC_RGB:
            continue
        bits = int(bps_tag.value) if bps_tag and not hasattr(bps_tag.value, "__len__") else (int(bps_tag.value[0]) if bps_tag else 8)
        if bits < 16:
            continue
        pixels = page.shape[0] * page.shape[1]
        if pixels > best_pixels:
            best = page
            best_pixels = pixels
    return best


_SGILOG_COMPRESSIONS = {34676, 34677}


def _has_sgilog_ifd(tif: tifffile.TiffFile) -> bool:
    for page in tif.pages:
        tags = getattr(page, "tags", None)
        if tags is None:
            continue
        comp_tag = tags.get("Compression")
        if comp_tag is not None and int(comp_tag.value) in _SGILOG_COMPRESSIONS:
            return True
    return False


def is_flextight_fff(file_path: str) -> bool:
    """True if this FFF is an Imacon/Hasselblad Flextight scanner file."""
    if os.path.splitext(file_path)[1].lower() != ".fff":
        return False
    try:
        with tifffile.TiffFile(file_path) as tif:
            return _find_full_res_ifd(tif) is not None
    except Exception:
        return False


class FffLoader(IImageLoader):
    """Loader for Imacon/Hasselblad Flextight FFF scanner files.

    Handles two variants:
    - Uncompressed 16-bit RGB (standard FFF from FlexColor export)
    - SGI LogLuv compressed (raw .3fr/.fff from the scanner hardware)

    Data is returned as-is — no color-space assumptions or linearization.
    """

    def load(self, file_path: str, linear_raw: bool = False) -> Tuple[ContextManager[Any], dict]:
        with tifffile.TiffFile(file_path) as tif:
            page = _find_full_res_ifd(tif)
            if page is None:
                raise ValueError(f"No full-res image IFD in {file_path}")

            page_tags = getattr(page, "tags", None)
            comp_tag = page_tags.get("Compression") if page_tags else None
            is_logluv = comp_tag is not None and int(comp_tag.value) in _SGILOG_COMPRESSIONS

            if is_logluv:
                w_tag = page_tags.get("ImageWidth")
                h_tag = page_tags.get("ImageLength")
                w = int(w_tag.value) if w_tag else page.shape[1]
                h = int(h_tag.value) if h_tag else page.shape[0]
                with open(file_path, "rb") as fh:
                    raw_data = fh.read()
                byte_order = "<" if tif.byteorder == "<" else ">"
                f32 = decode_logluv_strips([page], w, h, raw_data, byte_order)
            else:
                arr = page.asarray()
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

            fff_meta: dict = {}
            p0_tags = getattr(tif.pages[0], "tags", None)
            if p0_tags is not None:
                plist_tag = p0_tags.get(50457)
                if plist_tag is not None and isinstance(plist_tag.value, bytes):
                    fff_meta.update(_parse_fff_plist(plist_tag.value))
                fw_tag = p0_tags.get(46279)
                if fw_tag is not None and isinstance(fw_tag.value, bytes):
                    fff_meta.update(_parse_fff_firmware(fw_tag.value))

        metadata = {
            "orientation": read_orientation(file_path),
            **fff_meta,
        }
        return NonStandardFileWrapper(f32), metadata
