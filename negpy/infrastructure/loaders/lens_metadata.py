"""Bounded readers for Sony ARW coefficients and DNG OpcodeList3."""

import os
import struct
from dataclasses import replace
from functools import lru_cache
from itertools import islice
from typing import Any

import numpy as np
import tifffile

from negpy.features.lens.models import IDENTITY, LensMetadata, RectilinearWarp, SonyWarp

_MAX_OPCODE_BYTES = 4 * 1024 * 1024


def parse_opcodes(data: bytes) -> tuple[RectilinearWarp, ...]:
    """Read DNG 1.3 rectilinear warps; reject incomplete or unsupported lists."""
    if len(data) < 4 or len(data) > _MAX_OPCODE_BYTES:
        raise ValueError("Invalid DNG correction data size.")
    count = struct.unpack_from(">I", data)[0]
    if count > 32:
        raise ValueError("Too many DNG correction instructions.")
    offset = 4
    warps = []
    for _ in range(count):
        if offset + 16 > len(data):
            raise ValueError("Incomplete DNG correction header.")
        opcode, version, flags, size = struct.unpack_from(">4I", data, offset)
        offset += 16
        end = offset + size
        if end > len(data) or flags & ~3:
            raise ValueError("Invalid DNG correction instruction.")
        if opcode != 1 or version > 0x01030000:
            raise ValueError("Unsupported DNG correction instruction (requires WarpRectilinear).")
        if size < 4:
            raise ValueError("Incomplete DNG warp.")
        planes = struct.unpack_from(">I", data, offset)[0]
        if planes not in (1, 3) or size != 4 + 48 * planes + 16:
            raise ValueError("Unsupported DNG warp plane count or size.")
        values = struct.unpack_from(f">{6 * planes + 2}d", data, offset + 4)
        if not all(np.isfinite(values)) or not all(0 <= v <= 1 for v in values[-2:]):
            raise ValueError("Invalid DNG warp coefficients or optical center.")
        coeffs = tuple(tuple(values[i * 6 : (i + 1) * 6]) for i in range(planes))
        # A folding radial map is not a lens correction. Tangential folds are checked below.
        r2 = np.linspace(0, 1, 257)
        for k0, k1, k2, k3, t0, t1 in coeffs:
            if max(abs(v) for v in (k0, k1, k2, k3, t0, t1)) > 16:
                raise ValueError("DNG warp coefficients are out of range.")
            derivative = k0 + r2 * (3 * k1 + r2 * (5 * k2 + 7 * k3 * r2))
            if np.min(derivative) <= 6 * (abs(t0) + abs(t1)):
                raise ValueError("DNG warp may fold the image.")
        if any(k != IDENTITY for k in coeffs):
            warps.append(RectilinearWarp(coeffs, (values[-2], values[-1])))
        offset = end
    if offset != len(data):
        raise ValueError("Trailing data in DNG correction list.")
    return tuple(warps)


def _sony_values(tags: Any, code: int, channels: int) -> tuple[float, ...]:
    tag = tags.get(code)
    if tag is None:
        return ()
    if int(tag.dtype) != 8 or tag.count > 33:
        raise ValueError("Unsupported Sony correction data type.")
    values = tuple(tag.value)
    n = values[0] if values else 0
    if n < 2 * channels or n > 16 * channels or n % channels or len(values) not in (n + 1, 16 * channels + 1):
        raise ValueError("Invalid Sony correction coefficient count.")
    return tuple(float(v) for v in values[1 : n + 1])


def _read_sony(page: Any) -> LensMetadata:
    distortion = _sony_values(page.tags, 0x7037, 1)
    ca = _sony_values(page.tags, 0x7035, 2)
    for code, values in ((0x7036, distortion), (0x7034, ca)):
        status = page.tags.get(code)
        if values and status is not None and status.value == 255:
            raise ValueError("Sony marks the correction coefficients as unavailable.")
    if distortion and ca and len(ca) != 2 * len(distortion):
        raise ValueError("Sony correction arrays have different lengths.")
    warp = SonyWarp(distortion, ca[: len(ca) // 2], ca[len(ca) // 2 :])
    if any(1 + v / 16384 <= 0 for v in distortion):
        raise ValueError("Invalid Sony distortion coefficients.")
    return LensMetadata("Sony ARW", (warp,), "No nonzero Sony lens correction coefficients.")


def _read_dng(page: Any) -> LensMetadata:
    opcode = page.tags.get(51022)
    if opcode is None:
        return LensMetadata(reason="No embedded DNG lens warp (OpcodeList3).")
    if opcode.count > _MAX_OPCODE_BYTES or int(opcode.dtype) not in (1, 7):
        raise ValueError("Invalid DNG correction data size or type.")
    # Non-square source pixels need a separate coordinate transform.
    scale = page.tags.get(50718)
    if scale is not None and tuple(scale.value) not in ((1, 1, 1, 1), (1, 1)):
        raise ValueError("DNG lens correction requires square source pixels.")
    colors = page.tags.get(50710)
    if colors is not None and tuple(colors.value) != (0, 1, 2):
        raise ValueError("Unsupported DNG color plane order.")
    area_tag = page.tags.get(50829)
    area = tuple(int(v) for v in area_tag.value) if area_tag is not None else (0, 0, page.imagelength, page.imagewidth)
    if len(area) != 4 or not (0 <= area[0] < area[2] <= page.imagelength and 0 <= area[1] < area[3] <= page.imagewidth):
        raise ValueError("Invalid DNG active image area.")
    buffer_area = area
    origin_tag, size_tag = page.tags.get(50719), page.tags.get(50720)
    if origin_tag is not None and size_tag is not None:

        def numbers(tag: Any) -> tuple[int, ...]:
            values = tag.value
            if int(tag.dtype) in (5, 10):
                values = tuple(n / d for n, d in zip(values[::2], values[1::2]))
            return tuple(round(v) for v in values)

        ox, oy = numbers(origin_tag)
        width, height = numbers(size_tag)
        buffer_area = (area[0] + oy, area[1] + ox, area[0] + oy + height, area[1] + ox + width)
        if not (area[0] <= buffer_area[0] < buffer_area[2] <= area[2] and area[1] <= buffer_area[1] < buffer_area[3] <= area[3]):
            raise ValueError("Invalid DNG default crop.")
    return LensMetadata("DNG WarpRectilinear", parse_opcodes(bytes(opcode.value)), "Embedded DNG warp is an identity.", area, buffer_area)


def read_lens_metadata(file_path: str | None) -> LensMetadata:
    """Inspect source metadata without decoding pixels; cache against the file revision."""
    if not file_path:
        return LensMetadata(reason="Load a Sony ARW or DNG file.")
    if os.path.splitext(file_path)[1].lower() not in (".arw", ".dng"):
        return LensMetadata(reason="Embedded correction supports Sony ARW and DNG WarpRectilinear.")
    try:
        stat = os.stat(file_path)
        return _read_cached(os.path.abspath(file_path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return LensMetadata(reason="Cannot read source lens metadata.")


@lru_cache(maxsize=128)
def _read_cached(file_path: str, mtime_ns: int, size: int) -> LensMetadata:
    try:
        with tifffile.TiffFile(file_path) as tif:
            pages = list(islice(tif.pages, 16))
            for parent in tuple(pages):
                if parent.pages is not None:
                    pages.extend(islice(parent.pages, 16))
            raw_pages = [p for p in pages if int(p.photometric) in (32803, 34892) and p.samplesperpixel in (1, 3)]
            if not raw_pages:
                return LensMetadata(reason="No supported RAW image plane; rendered images are not corrected again.")
            page = max(raw_pages, key=lambda p: p.imagewidth * p.imagelength)
            return _read_dng(page) if file_path.lower().endswith(".dng") else _read_sony(page)
    except ValueError as exc:
        return LensMetadata(reason=str(exc))
    except (OSError, TypeError, struct.error, IndexError, OverflowError, ZeroDivisionError):
        return LensMetadata(reason="Cannot read embedded lens correction data.")


def bind_decode(lens: LensMetadata, raw: Any, fallback: bool = False) -> LensMetadata:
    """Locate LibRaw's visible pixels within the DNG active image."""
    if not lens.available or lens.active_area is None:
        return lens
    sizes = raw.sizes
    if fallback:
        area = lens.buffer_area or lens.active_area
        if (sizes.raw_height, sizes.raw_width) != (area[2] - area[0], area[3] - area[1]):
            return LensMetadata(reason="DNG fallback crop does not match the lens correction area.")
        return replace(lens, buffer_area=area)
    area = (sizes.top_margin, sizes.left_margin, sizes.top_margin + sizes.height, sizes.left_margin + sizes.width)
    t, left, b, r = lens.active_area
    if not (t <= area[0] < area[2] <= b and left <= area[1] < area[3] <= r):
        return LensMetadata(reason="Decoded DNG area does not match the lens correction area.")
    return replace(lens, buffer_area=area)
