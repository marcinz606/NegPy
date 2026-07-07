"""Domain models for the analog gear library (cameras, lenses, film stocks, presets)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FilmFormat(str, Enum):
    FORMAT_35MM = "35mm"
    FORMAT_120 = "120"
    FORMAT_110 = "110"
    FORMAT_4X5 = "4×5"
    FORMAT_8X10 = "8×10"
    OTHER = "Other"

    @classmethod
    def from_storage(cls, value: str) -> "FilmFormat":
        """Parse a format string from gear JSON (native value or legacy alias)."""
        if value in {e.value for e in cls}:
            return cls(value)
        legacy = {
            "Format35mm": cls.FORMAT_35MM,
            "Format120": cls.FORMAT_120,
            "Format110": cls.FORMAT_110,
            "LargeFormat": cls.FORMAT_4X5,
        }
        return legacy.get(value, cls.OTHER)

    def to_storage(self) -> str:
        return self.value


class FilmColorType(str, Enum):
    COLOR_NEGATIVE = "ColorNegative"
    BW_NEGATIVE = "B&W Negative"
    COLOR_SLIDE = "ColorSlide"
    BW_SLIDE = "B&W Slide"
    OTHER = "Other"

    @classmethod
    def from_storage(cls, value: str) -> "FilmColorType":
        """Parse a color-type string from gear JSON (native value or legacy alias)."""
        if value in {e.value for e in cls}:
            return cls(value)
        legacy = {
            "ColorNegative": cls.COLOR_NEGATIVE,
            "BlackAndWhiteNegative": cls.BW_NEGATIVE,
            "ColorSlide": cls.COLOR_SLIDE,
            "BlackAndWhiteSlide": cls.BW_SLIDE,
        }
        return legacy.get(value, cls.OTHER)

    def to_storage(self) -> str:
        return self.value


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Camera:
    id: str = field(default_factory=_new_id)
    make: str = ""
    model: str = ""
    display_name: str = ""
    serial_number: str = ""
    notes: str = ""
    is_bundled: bool = False

    @property
    def resolved_display_name(self) -> str:
        if self.display_name.strip():
            return self.display_name.strip()
        return f"{self.make} {self.model}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "make": self.make,
            "model": self.model,
            "displayName": self.display_name,
            "serialNumber": self.serial_number or None,
            "notes": self.notes or None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Camera":
        return cls(
            id=str(data.get("id") or _new_id()),
            make=str(data.get("make") or ""),
            model=str(data.get("model") or ""),
            display_name=str(data.get("displayName") or data.get("display_name") or ""),
            serial_number=str(data.get("serialNumber") or data.get("serial_number") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class Lens:
    id: str = field(default_factory=_new_id)
    lens_model: str = ""
    make: str = ""
    display_name: str = ""
    focal_length_mm: Optional[float] = None
    max_aperture: Optional[float] = None
    serial_number: str = ""
    notes: str = ""
    is_bundled: bool = False

    @property
    def resolved_display_name(self) -> str:
        if self.display_name.strip():
            return self.display_name.strip()
        if self.lens_model.strip():
            return self.lens_model.strip()
        return "Unnamed lens"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "lensModel": self.lens_model,
            "make": self.make or None,
            "displayName": self.display_name,
            "serialNumber": self.serial_number or None,
            "notes": self.notes or None,
        }
        if self.focal_length_mm is not None:
            d["focalLength"] = self.focal_length_mm
        if self.max_aperture is not None:
            d["maxAperture"] = self.max_aperture
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lens":
        fl = data.get("focalLength", data.get("focal_length_mm"))
        ap = data.get("maxAperture", data.get("max_aperture"))
        return cls(
            id=str(data.get("id") or _new_id()),
            lens_model=str(data.get("lensModel") or data.get("lens_model") or ""),
            make=str(data.get("make") or ""),
            display_name=str(data.get("displayName") or data.get("display_name") or ""),
            focal_length_mm=float(fl) if fl is not None else None,
            max_aperture=float(ap) if ap is not None else None,
            serial_number=str(data.get("serialNumber") or data.get("serial_number") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class FilmStock:
    id: str = field(default_factory=_new_id)
    manufacturer: str = ""
    stock_name: str = ""
    display_name: str = ""
    iso: int = 100
    format: FilmFormat = FilmFormat.FORMAT_35MM
    color_type: FilmColorType = FilmColorType.COLOR_NEGATIVE
    notes: str = ""
    is_bundled: bool = False

    @property
    def resolved_display_name(self) -> str:
        if self.display_name.strip():
            return self.display_name.strip()
        if self.manufacturer.strip():
            return f"{self.manufacturer} {self.stock_name}".strip()
        return self.stock_name.strip()

    @property
    def full_film_label(self) -> str:
        if self.manufacturer.strip():
            return f"{self.manufacturer} {self.stock_name}".strip()
        return self.stock_name.strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "manufacturer": self.manufacturer,
            "stockName": self.stock_name,
            "iso": self.iso,
            "format": self.format.to_storage(),
            "colorType": self.color_type.to_storage(),
            "notes": self.notes or None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilmStock":
        fmt_raw = data.get("format", "35mm")
        if isinstance(fmt_raw, FilmFormat):
            fmt = fmt_raw
        else:
            fmt = FilmFormat.from_storage(str(fmt_raw))

        ct_raw = data.get("colorType", data.get("color_type", "ColorNegative"))
        if isinstance(ct_raw, FilmColorType):
            ct = ct_raw
        else:
            ct = FilmColorType.from_storage(str(ct_raw))

        return cls(
            id=str(data.get("id") or _new_id()),
            manufacturer=str(data.get("manufacturer") or ""),
            stock_name=str(data.get("stockName") or data.get("stock_name") or ""),
            display_name=str(data.get("displayName") or data.get("display_name") or ""),
            iso=int(data.get("iso") or 100),
            format=fmt,
            color_type=ct,
            notes=str(data.get("notes") or ""),
        )


@dataclass
class GearPreset:
    id: str = field(default_factory=_new_id)
    display_name: str = ""
    camera_id: str = ""
    lens_id: str = ""
    film_stock_id: str = ""
    notes: str = ""
    is_bundled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "cameraId": self.camera_id or None,
            "lensId": self.lens_id or None,
            "filmStockId": self.film_stock_id or None,
            "notes": self.notes or None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GearPreset":
        return cls(
            id=str(data.get("id") or _new_id()),
            display_name=str(data.get("displayName") or data.get("display_name") or ""),
            camera_id=str(data.get("cameraId") or data.get("camera_id") or ""),
            lens_id=str(data.get("lensId") or data.get("lens_id") or ""),
            film_stock_id=str(data.get("filmStockId") or data.get("film_stock_id") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class GearLibrary:
    """In-memory snapshot of all gear library collections."""

    cameras: list[Camera] = field(default_factory=list)
    lenses: list[Lens] = field(default_factory=list)
    film_stocks: list[FilmStock] = field(default_factory=list)
    gear_presets: list[GearPreset] = field(default_factory=list)

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        return next((c for c in self.cameras if c.id == camera_id), None)

    def get_lens(self, lens_id: str) -> Optional[Lens]:
        return next((lens for lens in self.lenses if lens.id == lens_id), None)

    def get_film_stock(self, film_stock_id: str) -> Optional[FilmStock]:
        return next((f for f in self.film_stocks if f.id == film_stock_id), None)

    def get_gear_preset(self, preset_id: str) -> Optional[GearPreset]:
        return next((p for p in self.gear_presets if p.id == preset_id), None)
