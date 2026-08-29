"""Container encoding for every export path.

Integer pixels in, file bytes out. No color management, no sizing, no naming and
no disk access: callers own those. Every writer in the app routes its final
encode through here so a format setting cannot reach one path and miss another.
"""

import io
import struct
import zlib
from typing import Any, Dict, Optional

import imagecodecs
import numpy as np
import tifffile
from PIL import Image, PngImagePlugin

from negpy.domain.models import TiffCompression
from negpy.features.metadata.resolution import Resolution

_PNG_METRE = 1  # pHYs unit specifier
_INCH_PER_METRE = 39.3701


def encode_tiff(
    arr: np.ndarray,
    *,
    icc: Optional[bytes] = None,
    resolution: Optional[Resolution] = None,
    compression: TiffCompression = TiffCompression.ZIP,
    **extra: Any,
) -> bytes:
    """Encode uint8/uint16 pixels as TIFF. ``extra`` passes through to imwrite."""
    compression = TiffCompression(compression)
    buf = io.BytesIO()
    kwargs: Dict[str, Any] = {
        "photometric": "rgb" if arr.ndim == 3 else "minisblack",
        # tifffile raises on a predictor without compression, and differencing only
        # pays off once something downstream packs the result.
        "compression": None if compression == TiffCompression.NONE else compression.value,
        "predictor": compression != TiffCompression.NONE,
    }
    if icc:
        kwargs["iccprofile"] = icc
    if resolution is not None:
        kwargs["resolution"] = (resolution.x, resolution.y)
        kwargs["resolutionunit"] = resolution.unit
    kwargs.update(extra)
    tifffile.imwrite(buf, arr, **kwargs)
    return buf.getvalue()


def encode_png(
    arr: np.ndarray,
    *,
    icc: Optional[bytes] = None,
    resolution: Optional[Resolution] = None,
    level: int = 6,
    exif: Optional[bytes] = None,
    xmp: Optional[bytes] = None,
) -> bytes:
    """Encode uint8/uint16 pixels as PNG.

    uint16 goes through imagecodecs and gets its ancillary chunks spliced in,
    because PIL has no 16-bit RGB mode.
    """
    if arr.dtype == np.uint16 and arr.ndim == 3:
        return _encode_png_u16(arr, icc=icc, resolution=resolution, level=level, exif=exif, xmp=xmp)

    buf = io.BytesIO()
    kwargs: Dict[str, Any] = {"format": "PNG", "compress_level": level}
    if resolution is not None:
        kwargs["dpi"] = (resolution.x_dpi, resolution.y_dpi)
    if icc:
        kwargs["icc_profile"] = icc
    if exif:
        kwargs["exif"] = exif
    if xmp:
        info = PngImagePlugin.PngInfo()
        info.add_itxt("XML:com.adobe.xmp", xmp.decode("utf-8"), zip=False)
        kwargs["pnginfo"] = info
    Image.fromarray(arr).save(buf, **kwargs)
    return buf.getvalue()


def encode_jpeg(
    arr: np.ndarray,
    *,
    icc: Optional[bytes] = None,
    resolution: Optional[Resolution] = None,
    quality: int = 90,
    progressive: bool = False,
) -> bytes:
    """Encode uint8 pixels as JPEG. Chroma subsampling stays 4:4:4 (#224)."""
    buf = io.BytesIO()
    kwargs: Dict[str, Any] = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0,
        "optimize": True,
        "progressive": progressive,
    }
    if icc:
        kwargs["icc_profile"] = icc
    if resolution is not None:
        kwargs["dpi"] = (resolution.x_dpi, resolution.y_dpi)
    Image.fromarray(arr).save(buf, **kwargs)
    return buf.getvalue()


def encode_webp(
    arr: np.ndarray,
    *,
    icc: Optional[bytes] = None,
    lossless: bool = False,
    quality: int = 90,
    method: int = 4,
) -> bytes:
    """Encode uint8 pixels as WebP."""
    img = Image.fromarray(arr)
    if max(img.size) > 16383:
        raise ValueError("WebP max dimension is 16383 px; use TIFF/PNG for larger exports.")
    buf = io.BytesIO()
    kwargs: Dict[str, Any] = {"format": "WEBP", "lossless": lossless, "quality": quality, "method": method}
    if icc:
        kwargs["icc_profile"] = icc
    img.save(buf, **kwargs)
    return buf.getvalue()


def encode_jxl(
    arr: np.ndarray,
    *,
    photometric: str,
    primaries: Optional[str],
    transfer: Optional[str],
    lossless: bool = True,
    distance: Optional[float] = None,
    effort: int = 7,
) -> bytes:
    """Encode uint8/uint16 pixels as JPEG XL. libjxl tags the color enumeratively,
    so no ICC profile is embedded."""
    bits = imagecodecs.jpegxl_encode(
        np.ascontiguousarray(arr),
        bitspersample=8 if arr.dtype == np.uint8 else 16,
        photometric=photometric,
        primaries=primaries,
        transfer=transfer,
        lossless=lossless,
        distance=None if lossless else distance,
        effort=effort,
        numthreads=0,  # all cores; single-threaded otherwise (~7x slower)
    )
    return bytes(bits)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _encode_png_u16(
    arr: np.ndarray,
    *,
    icc: Optional[bytes],
    resolution: Optional[Resolution],
    level: int,
    exif: Optional[bytes],
    xmp: Optional[bytes],
) -> bytes:
    png = bytes(imagecodecs.png_encode(np.ascontiguousarray(arr), level=level))
    chunks = []
    if icc:
        chunks.append(_png_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc, 9)))
    if resolution is not None:
        ppm_x = max(1, int(round(resolution.x_dpi * _INCH_PER_METRE)))
        ppm_y = max(1, int(round(resolution.y_dpi * _INCH_PER_METRE)))
        chunks.append(_png_chunk(b"pHYs", struct.pack(">IIB", ppm_x, ppm_y, _PNG_METRE)))
    if exif:
        # eXIf carries the bare TIFF structure; the APP1 marker piexif emits is a
        # JPEG framing detail and makes readers reject the chunk.
        chunks.append(_png_chunk(b"eXIf", exif[6:] if exif.startswith(b"Exif\x00\x00") else exif))
    if xmp:
        chunks.append(_png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp))
    if not chunks:
        return png
    # Ancillary chunks must precede the image data.
    cut = png.index(b"IDAT") - 4
    return png[:cut] + b"".join(chunks) + png[cut:]
