"""Read scan-rig metadata from source-file EXIF (piexif dict)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import piexif


def safe_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00")
    if value is None:
        return ""
    return str(value).strip("\x00")


def rational_to_float(value: Any) -> Optional[float]:
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if den == 0:
            return None
        return num / den
    if isinstance(value, (int, float)):
        return float(value)
    return None


def rational_to_int(value: Any) -> Optional[int]:
    f = rational_to_float(value)
    return int(round(f)) if f is not None else None


def format_exposure(exif_tags: dict) -> str:
    """Format scan exposure time, f-number, and ISO from an Exif IFD dict."""
    parts: list[str] = []

    exposure_time = exif_tags.get(piexif.ExifIFD.ExposureTime)
    if exposure_time is not None:
        if isinstance(exposure_time, tuple) and len(exposure_time) == 2:
            num, den = exposure_time
            if num == 1:
                parts.append(f"1/{den}s")
            elif num:
                parts.append(f"{num}/{den}s")
        elif isinstance(exposure_time, (int, float)):
            parts.append(f"{exposure_time}s")

    f_number = exif_tags.get(piexif.ExifIFD.FNumber)
    f_val = rational_to_float(f_number)
    if f_val is not None:
        parts.append(f"f/{f_val:.1f}")

    iso = exif_tags.get(piexif.ExifIFD.ISOSpeedRatings)
    if iso is not None:
        if isinstance(iso, tuple) and len(iso) == 2:
            iso = iso[0] // iso[1] if iso[1] else 0
        parts.append(f"ISO {iso}")

    return "  ".join(parts)


def format_exif_datetime(value: Any) -> str:
    """`YYYY:MM:DD HH:MM:SS` as read from EXIF, shown with ISO date separators."""
    text = safe_str(value).strip()
    if len(text) >= 10 and text[4] == ":" and text[7] == ":":
        return f"{text[:4]}-{text[5:7]}-{text[8:]}"
    return text


def gps_decimal(dms: Any, ref: Any) -> Optional[float]:
    """Signed decimal degrees from an EXIF GPS DMS triplet and its hemisphere reference."""
    if not isinstance(dms, (tuple, list)) or len(dms) != 3:
        return None
    parts = [rational_to_float(part) for part in dms]
    if any(part is None for part in parts):
        return None
    value = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    return -value if safe_str(ref).upper() in ("S", "W") else value


@dataclass(frozen=True)
class ScanExif:
    """DSLR / scanner rig metadata read from the source file before export overwrite."""

    camera_make: str = ""
    camera_model: str = ""
    lens_make: str = ""
    lens_model: str = ""
    focal_length_mm: Optional[float] = None
    aperture: Optional[float] = None
    iso: Optional[int] = None
    exposure: str = ""
    datetime_original: str = ""
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None

    def has_any(self) -> bool:
        return bool(
            self.camera_make
            or self.camera_model
            or self.lens_make
            or self.lens_model
            or self.focal_length_mm is not None
            or self.aperture is not None
            or self.iso is not None
            or self.exposure
            or self.datetime_original
            or self.gps_latitude is not None
        )


# Tags that describe the digitization rig; strip when writing film capture to standard EXIF.
_SCAN_RESIDUAL_EXIF_TAGS = frozenset(
    {
        piexif.ExifIFD.FocalLengthIn35mmFilm,
        piexif.ExifIFD.LensSerialNumber,
        piexif.ExifIFD.BodySerialNumber,
        piexif.ExifIFD.LensSpecification,
        piexif.ExifIFD.SubjectDistance,
        piexif.ExifIFD.DigitalZoomRatio,
    }
)


def strip_scan_gps(exif_dict: dict) -> None:
    """Drop the source GPS block whole: a picked location must not keep the scan's altitude,
    timestamp or heading beside its own coordinates."""
    if isinstance(exif_dict.get("GPS"), dict):
        exif_dict["GPS"] = {}


_SCAN_EXPOSURE_EXIF_TAGS = frozenset(
    {
        piexif.ExifIFD.ExposureTime,
        piexif.ExifIFD.FNumber,
        piexif.ExifIFD.ISOSpeedRatings,
    }
)


def strip_scan_exif_for_capture(exif_dict: dict) -> None:
    """Remove scan-rig EXIF so film capture fields are not mixed with digitization data."""
    exif = exif_dict.get("Exif")
    if not isinstance(exif, dict):
        return
    for tag in _SCAN_RESIDUAL_EXIF_TAGS:
        exif.pop(tag, None)
    for tag in _SCAN_EXPOSURE_EXIF_TAGS:
        exif.pop(tag, None)


def extract_scan_from_exif(source_exif: dict | None) -> ScanExif:
    if not source_exif or not isinstance(source_exif, dict):
        return ScanExif()

    zeroth = source_exif.get("0th", {}) or {}
    exif_tags = source_exif.get("Exif", {}) or {}
    gps_tags = source_exif.get("GPS", {}) or {}

    iso_raw = exif_tags.get(piexif.ExifIFD.ISOSpeedRatings)
    iso: Optional[int] = None
    if isinstance(iso_raw, int):
        iso = iso_raw
    elif iso_raw is not None:
        iso = rational_to_int(iso_raw)

    return ScanExif(
        camera_make=safe_str(zeroth.get(piexif.ImageIFD.Make, "")),
        camera_model=safe_str(zeroth.get(piexif.ImageIFD.Model, "")),
        lens_make=safe_str(exif_tags.get(piexif.ExifIFD.LensMake, "")),
        lens_model=safe_str(exif_tags.get(piexif.ExifIFD.LensModel, "")),
        focal_length_mm=rational_to_float(exif_tags.get(piexif.ExifIFD.FocalLength)),
        aperture=rational_to_float(exif_tags.get(piexif.ExifIFD.FNumber)),
        iso=iso,
        exposure=format_exposure(exif_tags),
        datetime_original=format_exif_datetime(exif_tags.get(piexif.ExifIFD.DateTimeOriginal) or zeroth.get(piexif.ImageIFD.DateTime, "")),
        gps_latitude=gps_decimal(gps_tags.get(piexif.GPSIFD.GPSLatitude), gps_tags.get(piexif.GPSIFD.GPSLatitudeRef)),
        gps_longitude=gps_decimal(gps_tags.get(piexif.GPSIFD.GPSLongitude), gps_tags.get(piexif.GPSIFD.GPSLongitudeRef)),
    )
