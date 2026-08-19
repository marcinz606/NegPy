"""Capture time and place: parse, validate, and convert for EXIF/XMP."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import piexif

PRECISIONS = ("year", "month", "day", "minute", "second")

CAPTURE_DATE_HINT = "YYYY, YYYY-MM, YYYY-MM-DD or YYYY-MM-DD HH:MM"

_DATE_RE = re.compile(
    r"^(?P<y>\d{4})"
    r"(?:-(?P<mo>\d{1,2})"
    r"(?:-(?P<d>\d{1,2})"
    r"(?:[ T](?P<h>\d{1,2}):(?P<mi>\d{2})"
    r"(?::(?P<s>\d{2}))?"
    r"\s*(?P<tz>Z|[+-]\d{2}:?\d{2})?)?)?)?$"
)

_MIN_YEAR = 1800
_MAX_YEAR = 2999


@dataclass(frozen=True)
class CaptureDate:
    """A capture instant known only to some precision. `text` is ISO-8601, truncated."""

    text: str
    precision: str
    tz_offset: str = ""

    @property
    def year(self) -> int:
        return int(self.text[:4])

    def xmp_text(self) -> str:
        return f"{self.text}{self.tz_offset}"

    def exif_text(self) -> str:
        """EXIF cannot hold a partial date, so the unknown parts are padded."""
        y, mo, d, h, mi, s = _parts(self)
        return f"{y:04d}:{mo:02d}:{d:02d} {h:02d}:{mi:02d}:{s:02d}"

    def compact(self) -> str:
        y, mo, d, _h, _mi, _s = _parts(self)
        return f"{y:04d}{mo:02d}{d:02d}"


def _parts(cd: CaptureDate) -> tuple[int, int, int, int, int, int]:
    m = _DATE_RE.match(cd.text)
    if m is None:
        return int(cd.text[:4]), 1, 1, 0, 0, 0

    def num(key: str, default: int) -> int:
        raw = m.group(key)
        return int(raw) if raw else default

    return num("y", 1), num("mo", 1), num("d", 1), num("h", 0), num("mi", 0), num("s", 0)


def _normalize_offset(raw: str) -> str:
    if raw == "Z":
        return "+00:00"
    body = raw.replace(":", "")
    return f"{body[:3]}:{body[3:]}"


def parse_capture_date(text: str) -> Optional[CaptureDate]:
    """A CaptureDate from a truncated ISO-8601 string, or None when it is not a real instant."""
    cleaned = text.strip().replace("/", "-")
    if not cleaned:
        return None

    m = _DATE_RE.match(cleaned)
    if m is None:
        return None

    year = int(m.group("y"))
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None

    month, day = m.group("mo"), m.group("d")
    hour, minute, second = m.group("h"), m.group("mi"), m.group("s")

    try:
        datetime(
            year,
            int(month) if month else 1,
            int(day) if day else 1,
            int(hour) if hour else 0,
            int(minute) if minute else 0,
            int(second) if second else 0,
        )
    except ValueError:
        return None

    normalized = f"{year:04d}"
    precision = "year"
    if month:
        normalized += f"-{int(month):02d}"
        precision = "month"
    if day:
        normalized += f"-{int(day):02d}"
        precision = "day"
    if hour:
        normalized += f" {int(hour):02d}:{minute}"
        precision = "minute"
    if second:
        normalized += f":{second}"
        precision = "second"

    tz = m.group("tz")
    return CaptureDate(normalized, precision, _normalize_offset(tz) if tz and hour else "")


_OSM_HASH_RE = re.compile(r"map=\d+(?:\.\d+)?/(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)")
_MARKER_RE = re.compile(r"mlat=(-?\d+(?:\.\d+)?)[^0-9-]+mlon=(-?\d+(?:\.\d+)?)")
_AT_RE = re.compile(r"[@=](-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)")
_PAIR_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_coords(text: str) -> Optional[tuple[float, float]]:
    """Latitude/longitude from a typed pair or a pasted OpenStreetMap / Google Maps link."""
    stripped = text.strip()
    if not stripped:
        return None

    for pattern in (_OSM_HASH_RE, _MARKER_RE, _AT_RE, _PAIR_RE):
        m = pattern.search(stripped)
        if m is None:
            continue
        lat, lon = float(m.group(1)), float(m.group(2))
        if abs(lat) <= 90.0 and abs(lon) <= 180.0:
            return lat, lon
    return None


def format_coords(lat: float, lon: float) -> str:
    return f"{lat:.5f}, {lon:.5f}"


def place_summary(city: str, state: str, country: str, lat: Optional[float], lon: Optional[float]) -> str:
    """One-line place label: names when known, otherwise the coordinates."""
    names = [part.strip() for part in (city, state, country) if part.strip()]
    if names:
        return ", ".join(names)
    if lat is not None and lon is not None:
        return format_coords(lat, lon)
    return ""


def _dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    total = abs(value)
    degrees = int(total)
    minutes_full = (total - degrees) * 60.0
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60.0 * 100.0)
    return (degrees, 1), (minutes, 1), (int(seconds), 100)


def exif_gps_rationals(lat: float, lon: float) -> dict:
    """A piexif GPS IFD for a WGS-84 position."""
    return {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: _dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _dms(lon),
        piexif.GPSIFD.GPSMapDatum: b"WGS-84",
    }


def xmp_gps(lat: float, lon: float) -> tuple[str, str]:
    """XMP exif:GPSLatitude / exif:GPSLongitude in the spec's `DDD,MM.mmK` form."""

    def one(value: float, positive: str, negative: str) -> str:
        total = abs(value)
        degrees = int(total)
        minutes = (total - degrees) * 60.0
        return f"{degrees},{minutes:.4f}{positive if value >= 0 else negative}"

    return one(lat, "N", "S"), one(lon, "E", "W")


def deg2tile(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Fractional Web Mercator tile coordinates."""
    lat = max(-85.05112878, min(85.05112878, lat))
    n = float(2**zoom)
    x = (lon + 180.0) / 360.0 * n
    rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(rad)) / math.pi) / 2.0 * n
    return x, min(max(y, 0.0), n)


def tile2deg(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = float(2**zoom)
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon
