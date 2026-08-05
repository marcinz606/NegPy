import os
import plistlib
import re
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


def _find_full_res_ifd(tif: tifffile.TiffFile) -> Optional[Any]:
    """Return the largest RGB IFD by pixel count.

    FFF files can have multiple IFDs flagged as full-resolution (the SubfileType
    tag is unreliable — e.g. a small secondary image tagged full-res). Pixel
    count is the reliable signal, matching the approach in flexcolor-tool and
    the reference loader.
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
        if spp < 3 or photo != 2:
            continue
        bits = int(bps_tag.value) if bps_tag and not hasattr(bps_tag.value, "__len__") else (int(bps_tag.value[0]) if bps_tag else 8)
        if bits < 16:
            continue
        pixels = page.shape[0] * page.shape[1]
        if pixels > best_pixels:
            best = page
            best_pixels = pixels
    return best


def is_flextight_fff(file_path: str) -> bool:
    """True if this FFF is an Imacon/Hasselblad Flextight scanner file (16-bit RGB in a top-level IFD)."""
    if os.path.splitext(file_path)[1].lower() != ".fff":
        return False
    try:
        with tifffile.TiffFile(file_path) as tif:
            return _find_full_res_ifd(tif) is not None
    except Exception:
        return False


class FffLoader(IImageLoader):
    """Loader for Imacon/Hasselblad Flextight FFF scanner files.

    These are big-endian TIFFs with the full-res 16-bit linear RGB image in a
    top-level IFD (picked by pixel count, not SubfileType tag). The data is
    uninverted scanner output — linear, no gamma applied.

    Color space handling follows TiffLoader: ICC profile → identify space →
    linearise if sRGB. Untagged 16-bit is assumed linear.
    """

    def load(self, file_path: str, linear_raw: bool = False) -> Tuple[ContextManager[Any], dict]:
        with tifffile.TiffFile(file_path) as tif:
            page = _find_full_res_ifd(tif)
            if page is None:
                raise ValueError(f"No full-res RGB IFD in {file_path}")
            arr = page.asarray()

            icc_bytes: Optional[bytes] = None
            fff_meta: dict = {}
            p0_tags = getattr(tif.pages[0], "tags", None)
            for p in (page, tif.pages[0]):
                tags = getattr(p, "tags", None)
                if tags is None:
                    continue
                tag = tags.get("InterColorProfile")
                if tag is not None and tag.value:
                    icc_bytes = bytes(tag.value)
                    break
            if p0_tags is not None:
                plist_tag = p0_tags.get(50457)
                if plist_tag is not None and isinstance(plist_tag.value, bytes):
                    fff_meta.update(_parse_fff_plist(plist_tag.value))
                fw_tag = p0_tags.get(46279)
                if fw_tag is not None and isinstance(fw_tag.value, bytes):
                    fff_meta.update(_parse_fff_firmware(fw_tag.value))

        # Imacon/Flextight scanners have no IR hardware — no 4th channel exists.
        # 4-channel branch kept for defensive consistency with TiffLoader.
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
            **fff_meta,
        }
        return NonStandardFileWrapper(f32), metadata
