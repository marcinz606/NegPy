import io
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np
import rawpy
from PIL import ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def read_exif_from_file(file_path: str) -> Optional[dict]:
    """Read EXIF data from a file as a piexif-format dict. Returns None on failure."""
    import piexif

    # Try piexif first (works for JPEG, TIFF)
    try:
        return piexif.load(file_path)
    except Exception:
        pass

    # Fallback: try to read EXIF via PIL from RAW by opening the file
    try:
        from PIL import Image

        with Image.open(file_path) as img:
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                return piexif.load(exif_bytes)
    except Exception:
        pass

    return None


def read_orientation(file_path: str) -> int:
    """Read the EXIF orientation tag (1-8) from a file. Returns 1 (normal) when absent."""
    import piexif

    exif = read_exif_from_file(file_path)
    if not exif:
        return 1
    try:
        val = exif.get("0th", {}).get(piexif.ImageIFD.Orientation)
    except Exception:
        return 1
    if isinstance(val, int) and 1 <= val <= 8:
        return val
    return 1


def identify_color_space_from_icc(icc_bytes: Optional[bytes]) -> Optional[str]:
    """
    Resolve a ColorSpace enum value from an embedded ICC profile's description.
    Returns None when bytes are missing or the description doesn't match a known space.
    """
    if not icc_bytes:
        return None
    try:
        profile = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
        desc = (ImageCms.getProfileDescription(profile) or "").lower()
    except Exception as e:
        logger.warning(f"Could not parse embedded ICC profile: {e}")
        return None

    # Order matters — more specific matches first.
    if "prophoto" in desc:
        return ColorSpace.PROPHOTO.value
    if "rec. 2020" in desc or "rec2020" in desc or "bt.2020" in desc:
        return ColorSpace.REC2020.value
    if "display p3" in desc or "p3 d65" in desc:
        return ColorSpace.P3_D65.value
    if "aces" in desc:
        return ColorSpace.ACES.value
    if "adobe rgb" in desc or "adobe compat" in desc:
        return ColorSpace.ADOBE_RGB.value
    if "srgb" in desc or "iec 61966" in desc or "iec61966" in desc:
        return ColorSpace.SRGB.value
    return None


def detect_color_space_from_raw(raw: Any) -> Optional[str]:
    """
    Try to read the color space declared in a RAW file's embedded JPEG thumbnail EXIF.
    Returns a ColorSpace.value string or None if detection fails.
    EXIF tag 0xa001: 1=sRGB, 65535=Adobe RGB (manufacturer convention).
    """
    import io
    from PIL import Image

    try:
        thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            with Image.open(io.BytesIO(thumb.data)) as img:
                cs_tag = img.getexif().get(0xA001)
                if cs_tag == 1:
                    return ColorSpace.SRGB.value
                if cs_tag == 65535:
                    return ColorSpace.ADOBE_RGB.value
    except Exception:
        pass
    return None


class NonStandardFileWrapper:
    """
    numpy -> rawpy-like interface.
    """

    def __init__(self, data: np.ndarray, full_output_hw: Optional[Tuple[int, int]] = None) -> None:
        self.data = data
        # If set, `sizes` reports this (h, w) for full image; else derived from `data` shape.
        self._full_output_hw: Optional[Tuple[int, int]] = full_output_hw

    @property
    def sizes(self) -> Any:
        if self._full_output_hw is not None:
            h, w = self._full_output_hw
        else:
            h, w = self.data.shape[0], self.data.shape[1]
        return SimpleNamespace(raw_height=int(h), raw_width=int(w))

    def __enter__(self) -> "NonStandardFileWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        bps = kwargs.get("output_bps", 8)
        half_size = kwargs.get("half_size", False)
        data = self.data
        if half_size:
            data = data[::2, ::2]

        if bps == 16:
            return (data * 65535.0).astype(np.uint16)
        return (data * 255.0).astype(np.uint8)


def get_best_demosaic_algorithm(raw: Any, for_preview: bool = False) -> Any:
    """
    Selects optimal demosaicing algorithm based on sensor type and CFA pattern.
    Exclusively uses algorithms packaged in the standard permissive (LGPL) rawpy build.

    When for_preview=True, selects faster algorithms appropriate for downsampled
    preview rendering (PPG for Bayer, LINEAR for X-Trans).
    """
    selected_algo = rawpy.DemosaicAlgorithm.LINEAR

    if isinstance(raw, NonStandardFileWrapper):
        return selected_algo

    try:
        # Stacked sensors (Linear DNG, Foveon, sRAW)
        if raw.raw_type == rawpy.RawType.Stack:
            selected_algo = rawpy.DemosaicAlgorithm.LINEAR

        # Flat sensors (Bayer, X-Trans)
        elif raw.raw_type == rawpy.RawType.Flat:
            cfa_block_size = raw.raw_pattern.shape[0]

            if cfa_block_size == 6:
                # 6x6 block means it's a Fujifilm X-Trans sensor.
                # LINEAR is ~4.6x faster than DHT; artifacts are invisible at preview scale.
                # VNG was used previously, but it produces dot/maze artifacts on X-Trans's
                # 6x6 pattern in high-contrast regions (see #272). DHT was added to dcraw/
                # LibRaw specifically to handle X-Trans correctly and is also LGPL-clean.
                selected_algo = rawpy.DemosaicAlgorithm.LINEAR if for_preview else rawpy.DemosaicAlgorithm.DHT

            elif cfa_block_size == 2:
                # 2x2 block means it's a standard Bayer sensor (Canon, Nikon, Sony, etc.)
                # PPG is ~60% faster than AHD with negligible quality difference at preview scale.
                selected_algo = rawpy.DemosaicAlgorithm.PPG if for_preview else rawpy.DemosaicAlgorithm.AHD

    except (AttributeError, ValueError) as e:
        logger.exception(f"Failed to determine sensor CFA pattern: {e}. Falling back to LINEAR.")
        selected_algo = rawpy.DemosaicAlgorithm.LINEAR

    return selected_algo


def get_supported_raw_wildcards() -> str:
    """
    Returns raw formats as string for file dialogs.
    """
    wildcards = []
    for ext in sorted(SUPPORTED_RAW_EXTENSIONS):
        base = ext.lstrip(".")
        wildcards.append(f"*.{base}")
        wildcards.append(f"*.{base.upper()}")

    return " ".join(wildcards)
