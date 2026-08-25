"""A file's resolution, and where to read it from.

Kept as the rationals and unit the file actually carries, not a rounded DPI: a
source can declare centimetres, a fraction, or different densities per axis, and
Protect original metadata has to copy all three through untouched.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional, TypeGuard

import piexif

_INCH = 2
_CENTIMETRE = 3
_MAX_DENOMINATOR = 1000


@dataclass(frozen=True)
class Resolution:
    """XResolution/YResolution as rationals, with the TIFF ResolutionUnit they use."""

    x: tuple[int, int]
    y: tuple[int, int]
    unit: int = _INCH

    @staticmethod
    def from_dpi(dpi: float) -> "Resolution":
        r = _to_rational(dpi)
        return Resolution(r, r, _INCH)

    @property
    def x_dpi(self) -> float:
        return _in_inches(self.x, self.unit)

    @property
    def y_dpi(self) -> float:
        return _in_inches(self.y, self.unit)

    @property
    def dpi(self) -> int:
        """Single rounded value, for the callers that need one number."""
        return max(1, int(round(self.x_dpi)))


def _in_inches(rational: tuple[int, int], unit: int) -> float:
    value = rational[0] / rational[1]
    return value * 2.54 if unit == _CENTIMETRE else value


def _to_rational(value: float) -> tuple[int, int]:
    if float(value).is_integer():
        return (int(value), 1)
    frac = Fraction(value).limit_denominator(_MAX_DENOMINATOR)
    return (frac.numerator, frac.denominator)


def from_exif(exif_dict: Optional[dict]) -> Optional[Resolution]:
    """Resolution declared in an EXIF/TIFF IFD0, or None.

    ResolutionUnit 1 means "no absolute unit", so the pair is an aspect ratio and
    not a resolution; anything but inches or centimetres is treated the same way.
    """
    if not exif_dict:
        return None
    zeroth = exif_dict.get("0th") or {}
    x = zeroth.get(piexif.ImageIFD.XResolution)
    y = zeroth.get(piexif.ImageIFD.YResolution) or x
    unit = zeroth.get(piexif.ImageIFD.ResolutionUnit)
    if unit not in (_INCH, _CENTIMETRE) or not _is_rational(x) or not _is_rational(y):
        return None
    return Resolution((int(x[0]), int(x[1])), (int(y[0]), int(y[1])), int(unit))


def _is_rational(value: Any) -> TypeGuard[tuple[int, int]]:
    return isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value) and value[0] > 0 and value[1] > 0


def from_container(path: str) -> Optional[Resolution]:
    """Resolution held by the file format itself rather than by EXIF: a JPEG's JFIF
    density, a PNG's pHYs, a TIFF's baseline tags. Returns None for anything Pillow
    cannot open, which includes every RAW format."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            dpi = img.info.get("dpi")
    except Exception:
        return None
    if not dpi or len(dpi) != 2:
        return None
    try:
        x, y = float(dpi[0]), float(dpi[1])
    except (TypeError, ValueError):
        return None
    if x <= 0 or y <= 0:
        return None
    return Resolution(_to_rational(round(x, 2)), _to_rational(round(y, 2)), _INCH)


def read_source(path: Optional[str], exif_dict: Optional[dict] = None) -> Optional[Resolution]:
    """The source's declared resolution: its EXIF first, then the container's own
    record. ``exif_dict`` is the cached read when there is one; it is only populated
    for files the user has selected, so the file is still consulted without it."""
    found = from_exif(exif_dict)
    if found is not None:
        return found
    if not path:
        return None
    found = from_exif(_read_exif(path))
    return found if found is not None else from_container(path)


def _read_exif(path: str) -> Optional[dict]:
    try:
        from negpy.infrastructure.loaders.helpers import read_exif_from_file

        return read_exif_from_file(path)
    except Exception:
        return None
