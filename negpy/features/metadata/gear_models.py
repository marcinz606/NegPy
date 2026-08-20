"""Domain models for the analog gear library (cameras, lenses, film stocks, chemistry, rigs, presets)."""

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


class DevProcess(str, Enum):
    BW = "B&W"
    C41 = "C-41"
    E6 = "E-6"
    ECN2 = "ECN-2"
    OTHER = "Other"

    @classmethod
    def from_storage(cls, value: str) -> "DevProcess":
        """Parse a process string from gear JSON (native value or legacy alias)."""
        if value in {e.value for e in cls}:
            return cls(value)
        legacy = {
            "BlackAndWhite": cls.BW,
            "BW": cls.BW,
            "C41": cls.C41,
            "E6": cls.E6,
            "ECN2": cls.ECN2,
        }
        return legacy.get(value, cls.OTHER)

    def to_storage(self) -> str:
        return self.value


class ScanMethod(str, Enum):
    COPY_STAND = "Copy stand"
    FLATBED = "Flatbed"
    FILM_SCANNER = "Film scanner"
    DRUM = "Drum"
    LAB = "Lab scan"
    OTHER = "Other"

    @classmethod
    def from_storage(cls, value: str) -> "ScanMethod":
        """Parse a scan-method string from gear JSON (native value or legacy alias)."""
        if value in {e.value for e in cls}:
            return cls(value)
        legacy = {
            "CopyStand": cls.COPY_STAND,
            "DSLR": cls.COPY_STAND,
            "FilmScanner": cls.FILM_SCANNER,
            "Lab": cls.LAB,
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
class DeveloperRecipe:
    """A development recipe: chemistry, dilution and the conditions it was run at."""

    id: str = field(default_factory=_new_id)
    display_name: str = ""
    developer: str = ""
    dilution: str = ""
    time: str = ""
    temperature_c: Optional[float] = None
    process: DevProcess = DevProcess.BW
    lab: str = ""
    notes: str = ""
    is_bundled: bool = False

    @property
    def resolved_display_name(self) -> str:
        if self.display_name.strip():
            return self.display_name.strip()
        name = " ".join(p for p in (self.developer.strip(), self.dilution.strip()) if p)
        return name or self.lab.strip() or "Unnamed developer"

    @property
    def full_developer_label(self) -> str:
        """The metadata string: ``D-76 1+1, 9:30 @ 20 °C``."""
        head = " ".join(p for p in (self.developer.strip(), self.dilution.strip()) if p)
        if not head:
            head = self.display_name.strip()
        tail: list[str] = []
        if self.time.strip():
            tail.append(self.time.strip())
        if self.temperature_c is not None:
            tail.append(f"{self.temperature_c:g} °C")
        conditions = " @ ".join(tail) if len(tail) == 2 else (tail[0] if tail else "")
        parts = [p for p in (head, conditions) if p]
        label = ", ".join(parts)
        if self.lab.strip():
            label = f"{label} ({self.lab.strip()})" if label else self.lab.strip()
        return label

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "displayName": self.display_name,
            "developer": self.developer,
            "dilution": self.dilution or None,
            "time": self.time or None,
            "process": self.process.to_storage(),
            "lab": self.lab or None,
            "notes": self.notes or None,
        }
        if self.temperature_c is not None:
            d["temperatureC"] = self.temperature_c
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeveloperRecipe":
        temp = data.get("temperatureC", data.get("temperature_c"))
        proc_raw = data.get("process", "B&W")
        proc = proc_raw if isinstance(proc_raw, DevProcess) else DevProcess.from_storage(str(proc_raw))
        return cls(
            id=str(data.get("id") or _new_id()),
            display_name=str(data.get("displayName") or data.get("display_name") or ""),
            developer=str(data.get("developer") or ""),
            dilution=str(data.get("dilution") or ""),
            time=str(data.get("time") or ""),
            temperature_c=float(temp) if temp is not None else None,
            process=proc,
            lab=str(data.get("lab") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class ScanSetup:
    """A digitization rig: how the frame was scanned, and with what."""

    id: str = field(default_factory=_new_id)
    display_name: str = ""
    method: ScanMethod = ScanMethod.COPY_STAND
    device: str = ""
    light_source: str = ""
    holder: str = ""
    software: str = ""
    notes: str = ""
    is_bundled: bool = False

    @property
    def resolved_display_name(self) -> str:
        if self.display_name.strip():
            return self.display_name.strip()
        if self.device.strip():
            return f"{self.method.value} — {self.device.strip()}"
        return self.method.value

    @property
    def full_scan_label(self) -> str:
        """The metadata string: ``Copy stand — Sony A7RIV, Scanlight narrowband``."""
        detail = ", ".join(p for p in (self.device.strip(), self.light_source.strip(), self.holder.strip(), self.software.strip()) if p)
        if detail:
            return f"{self.method.value} — {detail}"
        return self.method.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "method": self.method.to_storage(),
            "device": self.device or None,
            "lightSource": self.light_source or None,
            "holder": self.holder or None,
            "software": self.software or None,
            "notes": self.notes or None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanSetup":
        method_raw = data.get("method", "Copy stand")
        method = method_raw if isinstance(method_raw, ScanMethod) else ScanMethod.from_storage(str(method_raw))
        return cls(
            id=str(data.get("id") or _new_id()),
            display_name=str(data.get("displayName") or data.get("display_name") or ""),
            method=method,
            device=str(data.get("device") or ""),
            light_source=str(data.get("lightSource") or data.get("light_source") or ""),
            holder=str(data.get("holder") or ""),
            software=str(data.get("software") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class GearPreset:
    id: str = field(default_factory=_new_id)
    display_name: str = ""
    camera_id: str = ""
    lens_id: str = ""
    film_stock_id: str = ""
    developer_id: str = ""
    scan_setup_id: str = ""
    notes: str = ""
    is_bundled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "cameraId": self.camera_id or None,
            "lensId": self.lens_id or None,
            "filmStockId": self.film_stock_id or None,
            "developerId": self.developer_id or None,
            "scanSetupId": self.scan_setup_id or None,
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
            developer_id=str(data.get("developerId") or data.get("developer_id") or ""),
            scan_setup_id=str(data.get("scanSetupId") or data.get("scan_setup_id") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class ProcessScanPreset:
    """A development recipe paired with the rig it was scanned on."""

    id: str = field(default_factory=_new_id)
    display_name: str = ""
    developer_id: str = ""
    scan_setup_id: str = ""
    notes: str = ""
    is_bundled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "developerId": self.developer_id or None,
            "scanSetupId": self.scan_setup_id or None,
            "notes": self.notes or None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessScanPreset":
        return cls(
            id=str(data.get("id") or _new_id()),
            display_name=str(data.get("displayName") or data.get("display_name") or ""),
            developer_id=str(data.get("developerId") or data.get("developer_id") or ""),
            scan_setup_id=str(data.get("scanSetupId") or data.get("scan_setup_id") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class GearLibrary:
    """In-memory snapshot of all gear library collections."""

    cameras: list[Camera] = field(default_factory=list)
    lenses: list[Lens] = field(default_factory=list)
    film_stocks: list[FilmStock] = field(default_factory=list)
    developers: list[DeveloperRecipe] = field(default_factory=list)
    scan_setups: list[ScanSetup] = field(default_factory=list)
    gear_presets: list[GearPreset] = field(default_factory=list)
    process_scan_presets: list[ProcessScanPreset] = field(default_factory=list)

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        return next((c for c in self.cameras if c.id == camera_id), None)

    def get_lens(self, lens_id: str) -> Optional[Lens]:
        return next((lens for lens in self.lenses if lens.id == lens_id), None)

    def get_film_stock(self, film_stock_id: str) -> Optional[FilmStock]:
        return next((f for f in self.film_stocks if f.id == film_stock_id), None)

    def get_developer(self, developer_id: str) -> Optional[DeveloperRecipe]:
        return next((d for d in self.developers if d.id == developer_id), None)

    def get_scan_setup(self, scan_setup_id: str) -> Optional[ScanSetup]:
        return next((s for s in self.scan_setups if s.id == scan_setup_id), None)

    def get_gear_preset(self, preset_id: str) -> Optional[GearPreset]:
        return next((p for p in self.gear_presets if p.id == preset_id), None)

    def get_process_scan_preset(self, preset_id: str) -> Optional[ProcessScanPreset]:
        return next((p for p in self.process_scan_presets if p.id == preset_id), None)
