from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import tifffile
from PIL import Image


@dataclass(frozen=True)
class PreviewMemoryEstimate:
    cached_bytes: int
    temporary_bytes: int
    source_dimensions: tuple[int, int] | None


_PROFILE_BYTES_PER_PIXEL = {
    "display": 24,
    "scan": 48,
    "raw": 64,
}
_PROFILE_FILE_MULTIPLIER = {
    "display": 4,
    "scan": 12,
    "raw": 24,
}
_PROFILE_FLOOR_BYTES = {
    "display": 64 * 1024 * 1024,
    "scan": 256 * 1024 * 1024,
    "raw": 512 * 1024 * 1024,
}


def _tiff_dimensions(file_path: str) -> tuple[int, int] | None:
    try:
        with open(file_path, "rb") as source:
            marker = source.read(4)
        if marker not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            return None
        best: tuple[int, int] | None = None
        with tifffile.TiffFile(file_path) as tif:
            pages: list[Any] = list(tif.pages)
            seen: set[int] = set()
            while pages and len(seen) < 64:
                page = pages.pop()
                offset = int(getattr(page, "offset", id(page)))
                if offset in seen:
                    continue
                seen.add(offset)
                width = int(getattr(page, "imagewidth", 0))
                height = int(getattr(page, "imagelength", 0))
                if width > 0 and height > 0:
                    candidate = (width, height)
                    if best is None or candidate[0] * candidate[1] > best[0] * best[1]:
                        best = candidate
                pages.extend(list(page.pages or ()))
        return best
    except Exception:
        return None


def _image_dimensions(file_path: str) -> tuple[int, int] | None:
    dimensions = _tiff_dimensions(file_path)
    if dimensions is not None:
        return dimensions
    try:
        with Image.open(file_path) as image:
            width, height = image.size
        return (int(width), int(height)) if width > 0 and height > 0 else None
    except Exception:
        return None


def estimate_preview_memory(file_path: str, max_edge: int, profile: str) -> PreviewMemoryEstimate:
    """Estimate preview storage and peak loader workspace from headers and file size."""
    dimensions = _image_dimensions(file_path)
    edge = max(1, int(max_edge))
    if dimensions is None:
        cached_bytes = edge * edge * 3 * 4
        source_pixels = 0
    else:
        width, height = dimensions
        scale = min(1.0, edge / max(width, height))
        cached_width = max(1, round(width * scale))
        cached_height = max(1, round(height * scale))
        cached_bytes = cached_width * cached_height * 3 * 4
        source_pixels = width * height

    try:
        file_bytes = os.path.getsize(file_path)
    except OSError:
        file_bytes = 0
    bytes_per_pixel = _PROFILE_BYTES_PER_PIXEL.get(profile, _PROFILE_BYTES_PER_PIXEL["raw"])
    file_multiplier = _PROFILE_FILE_MULTIPLIER.get(profile, _PROFILE_FILE_MULTIPLIER["raw"])
    floor_bytes = _PROFILE_FLOOR_BYTES.get(profile, _PROFILE_FLOOR_BYTES["raw"])
    encoded_bytes = file_bytes * 2 if dimensions is not None else file_bytes * file_multiplier
    temporary_bytes = max(source_pixels * bytes_per_pixel, encoded_bytes, floor_bytes)
    # Decoded previews can retain a separate detection plane and the source IR plane.
    if dimensions is None or source_pixels * 3 * 4 > cached_bytes:
        cached_bytes *= 2
        if profile != "display":
            cached_bytes += cached_width * cached_height * 4 if dimensions is not None else edge * edge * 4
    if profile != "display":
        cached_bytes += source_pixels * 4 if dimensions is not None else temporary_bytes
    return PreviewMemoryEstimate(cached_bytes, temporary_bytes, dimensions)
