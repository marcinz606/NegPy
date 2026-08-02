"""Resolve MetadataConfig + gear library into export-ready metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from negpy.features.metadata.exif_read import ScanExif, extract_scan_from_exif
from negpy.features.metadata.gear_models import GearLibrary
from negpy.features.metadata.models import (
    MetadataConfig,
    PUSH_PULL_LABELS,
    normalize_description_fields,
    DEFAULT_DESCRIPTION_FIELDS,
    resolve_description_fields,
)

_NEGPY_SOFTWARE = "NegPy"
NEGPY_SOFTWARE = _NEGPY_SOFTWARE


@dataclass(frozen=True)
class ExifWriteFlags:
    """Which standard EXIF fields to write from film capture (vs leave source scan EXIF)."""

    camera: bool = False
    lens: bool = False
    film_iso: bool = False
    exposure: bool = False

    @property
    def strip_scan_residuals(self) -> bool:
        return self.camera or self.lens or self.film_iso or self.exposure


def has_capture_gear(config: MetadataConfig) -> bool:
    """True when the user provided gear/exposure data that should replace scan EXIF."""
    if config.camera_id or config.lens_id or config.film_stock_id:
        return True
    if config.camera_make or config.camera_model or config.lens_make or config.lens_model:
        return True
    if config.film or config.film_iso is not None:
        return True
    if config.focal_length_mm is not None or config.max_aperture is not None:
        return True
    if config.exposure_override.strip():
        return True
    return False


def compute_exif_write_flags(config: MetadataConfig, payload: "MetadataPayload") -> ExifWriteFlags:
    """Decide which standard EXIF tags to overwrite with film capture data."""
    if not has_capture_gear(config):
        return ExifWriteFlags()

    return ExifWriteFlags(
        camera=bool(payload.camera_make or payload.camera_model or config.camera_id),
        lens=bool(
            payload.lens_make
            or payload.lens_model
            or config.lens_id
            or payload.focal_length_mm is not None
            or payload.max_aperture is not None
        ),
        film_iso=payload.iso is not None,
        exposure=bool(payload.capture_exposure),
    )


@dataclass(frozen=True)
class MetadataPayload:
    """Resolved metadata that will be written to exported files."""

    # Original analog capture (standard EXIF when exif_flags permit; negpy:Capture* in XMP)
    camera_make: str = ""
    camera_model: str = ""
    lens_make: str = ""
    lens_model: str = ""
    focal_length_mm: Optional[float] = None
    max_aperture: Optional[float] = None
    iso: Optional[int] = None
    film_stock: str = ""
    film_manufacturer: str = ""
    film_format: str = ""
    film_color_type: str = ""
    capture_exposure: str = ""
    capture_roll: str = ""
    capture_frame: Optional[int] = None

    # Digitization rig (negpy:Scan* XMP only; source EXIF when capture gear not set)
    scan_camera_make: str = ""
    scan_camera_model: str = ""
    scan_lens_make: str = ""
    scan_lens_model: str = ""
    scan_focal_length_mm: Optional[float] = None
    scan_aperture: Optional[float] = None
    scan_iso: Optional[int] = None
    scan_exposure: str = ""
    scan_method: str = ""

    image_description: str = ""
    developer: str = ""
    push_pull: str = ""
    notes: str = ""
    exif_flags: ExifWriteFlags = ExifWriteFlags()

    def camera_display(self) -> str:
        return f"{self.camera_make} {self.camera_model}".strip()

    def lens_display(self) -> str:
        return self.lens_model or self.lens_make

    def scan_camera_display(self) -> str:
        return f"{self.scan_camera_make} {self.scan_camera_model}".strip()

    def to_preview_sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Grouped preview: original capture, scan rig, process."""
        sections: list[tuple[str, list[tuple[str, str]]]] = []

        capture: list[tuple[str, str]] = []
        if self.camera_make:
            capture.append(("Camera make", self.camera_make))
        if self.camera_model:
            capture.append(("Camera model", self.camera_model))
        if self.lens_make:
            capture.append(("Lens make", self.lens_make))
        if self.lens_model:
            capture.append(("Lens model", self.lens_model))
        if self.focal_length_mm is not None:
            capture.append(("Focal length", f"{self.focal_length_mm:g} mm"))
        if self.max_aperture is not None:
            capture.append(("Max aperture", f"f/{self.max_aperture:g}"))
        if self.capture_exposure:
            capture.append(("Exposure", self.capture_exposure))
        if self.iso is not None:
            capture.append(("Film ISO", str(self.iso)))
        if self.film_stock:
            capture.append(("Film stock", self.film_stock))
        if self.film_manufacturer:
            capture.append(("Film manufacturer", self.film_manufacturer))
        if self.film_format:
            capture.append(("Film format", self.film_format))
        if self.film_color_type:
            capture.append(("Film type", self.film_color_type))
        if capture:
            sections.append(("Original capture", capture))

        scan: list[tuple[str, str]] = []
        if self.scan_camera_make:
            scan.append(("Camera make", self.scan_camera_make))
        if self.scan_camera_model:
            scan.append(("Camera model", self.scan_camera_model))
        if self.scan_lens_make:
            scan.append(("Lens make", self.scan_lens_make))
        if self.scan_lens_model:
            scan.append(("Lens model", self.scan_lens_model))
        if self.scan_focal_length_mm is not None:
            scan.append(("Focal length", f"{self.scan_focal_length_mm:g} mm"))
        if self.scan_aperture is not None:
            scan.append(("Aperture", f"f/{self.scan_aperture:g}"))
        if self.scan_exposure:
            scan.append(("Exposure", self.scan_exposure))
        if self.scan_iso is not None:
            scan.append(("ISO", str(self.scan_iso)))
        if self.scan_method:
            scan.append(("Scan method", self.scan_method))
        if self.capture_roll:
            scan.append(("Roll", self.capture_roll))
        if self.capture_frame is not None:
            scan.append(("Frame", str(self.capture_frame)))
        if scan:
            sections.append(("Scan", scan))

        process: list[tuple[str, str]] = []
        if self.film_format and not capture:
            process.append(("Format", self.film_format))
        if self.developer:
            process.append(("Developer", self.developer))
        if self.push_pull and self.push_pull != "Normal":
            process.append(("Push / pull", self.push_pull))
        if process:
            sections.append(("Process", process))

        summary: list[tuple[str, str]] = []
        if self.image_description:
            summary.append(("Image description", self.image_description))
        summary.append(("Software", _NEGPY_SOFTWARE))
        if self.exif_flags.strip_scan_residuals:
            summary.append(("EXIF capture tags", "Written to standard EXIF"))
        elif self.scan_camera_make or self.scan_camera_model:
            summary.append(("EXIF camera/lens", "Source scan device (capture not set)"))
        if summary:
            sections.append(("File", summary))

        return sections

    def to_preview_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for _title, rows in self.to_preview_sections():
            pairs.extend(rows)
        return pairs

    def has_any_data(self) -> bool:
        return bool(self.to_preview_pairs())


def _apex_from_f_number(f_number: float) -> float:
    return 2.0 * math.log(f_number, 2.0)


def build_image_description(payload: MetadataPayload, fields: object = None) -> str:
    """Human-readable EXIF ImageDescription from the selected field set."""
    enabled = frozenset(normalize_description_fields(fields if fields is not None else DEFAULT_DESCRIPTION_FIELDS))
    parts: list[str] = []
    if "camera" in enabled:
        camera = payload.camera_display()
        if camera:
            parts.append(camera)
    if "lens" in enabled:
        lens = payload.lens_display()
        if lens:
            parts.append(lens)
    if "film" in enabled and payload.film_stock:
        parts.append(payload.film_stock)
    if "iso" in enabled and payload.iso is not None:
        parts.append(f"ISO {payload.iso}")
    if "format" in enabled and payload.film_format:
        parts.append(payload.film_format)
    if "developer" in enabled and payload.developer:
        parts.append(payload.developer)
    if "push_pull" in enabled and payload.push_pull and payload.push_pull != "Normal":
        parts.append(payload.push_pull)
    if "scanning" in enabled and payload.scan_method:
        parts.append(payload.scan_method)
    if parts:
        return " • ".join(parts)
    if "film" in enabled:
        return payload.film_stock or ""
    return ""


def build_metadata_payload(
    config: MetadataConfig,
    gear: Optional[GearLibrary] = None,
    source_exif: Optional[dict[str, Any]] = None,
) -> MetadataPayload:
    """Resolve gear IDs, config fields, and source scan EXIF into an export payload."""
    camera_make = config.camera_make
    camera_model = config.camera_model
    lens_make = config.lens_make
    lens_model = config.lens_model
    focal_length = config.focal_length_mm
    max_aperture = config.max_aperture
    iso = config.film_iso
    film_manufacturer = config.film_manufacturer
    film_format = config.format_other if config.format == "Other" else config.format
    film_color_type = config.film_color_type
    film_stock = config.film

    if gear is not None:
        if config.camera_id:
            cam = gear.get_camera(config.camera_id)
            if cam:
                if not camera_make:
                    camera_make = cam.make
                if not camera_model:
                    camera_model = cam.model
        if config.lens_id:
            lens = gear.get_lens(config.lens_id)
            if lens:
                if not lens_make:
                    lens_make = lens.make
                if not lens_model:
                    lens_model = lens.lens_model or lens.resolved_display_name
                if focal_length is None:
                    focal_length = lens.focal_length_mm
                if max_aperture is None:
                    max_aperture = lens.max_aperture
        if config.film_stock_id:
            stock = gear.get_film_stock(config.film_stock_id)
            if stock:
                if not film_stock:
                    film_stock = stock.full_film_label
                if not film_manufacturer:
                    film_manufacturer = stock.manufacturer
                if iso is None:
                    iso = stock.iso
                if not film_format:
                    film_format = stock.format.value
                if not film_color_type:
                    film_color_type = stock.color_type.value

    scan: ScanExif = extract_scan_from_exif(source_exif)
    push_pull = PUSH_PULL_LABELS.get(config.push_pull, "Normal")
    capture_exposure = config.exposure_override.strip()

    draft = MetadataPayload(
        camera_make=camera_make.strip(),
        camera_model=camera_model.strip(),
        lens_make=lens_make.strip(),
        lens_model=lens_model.strip(),
        focal_length_mm=focal_length,
        max_aperture=max_aperture,
        iso=iso,
        film_stock=film_stock.strip(),
        film_manufacturer=film_manufacturer.strip(),
        film_format=film_format.strip(),
        film_color_type=film_color_type.strip(),
        capture_exposure=capture_exposure,
        capture_roll=config.capture_roll.strip(),
        capture_frame=config.capture_frame,
        scan_camera_make=scan.camera_make,
        scan_camera_model=scan.camera_model,
        scan_lens_make=scan.lens_make,
        scan_lens_model=scan.lens_model,
        scan_focal_length_mm=scan.focal_length_mm,
        scan_aperture=scan.aperture,
        scan_iso=scan.iso,
        scan_exposure=scan.exposure,
        scan_method=config.scanning.strip(),
        developer=config.developer.strip(),
        push_pull=push_pull,
    )

    desc_fields = resolve_description_fields(config.description_fields)
    desc = build_image_description(draft, desc_fields)
    if not desc and config.film and "film" in desc_fields:
        desc = config.film.strip()

    exif_flags = compute_exif_write_flags(config, draft)

    return MetadataPayload(
        camera_make=draft.camera_make,
        camera_model=draft.camera_model,
        lens_make=draft.lens_make,
        lens_model=draft.lens_model,
        focal_length_mm=draft.focal_length_mm,
        max_aperture=draft.max_aperture,
        iso=draft.iso,
        film_stock=draft.film_stock,
        film_manufacturer=draft.film_manufacturer,
        film_format=draft.film_format,
        film_color_type=draft.film_color_type,
        capture_exposure=draft.capture_exposure,
        capture_roll=draft.capture_roll,
        capture_frame=draft.capture_frame,
        scan_camera_make=draft.scan_camera_make,
        scan_camera_model=draft.scan_camera_model,
        scan_lens_make=draft.scan_lens_make,
        scan_lens_model=draft.scan_lens_model,
        scan_focal_length_mm=draft.scan_focal_length_mm,
        scan_aperture=draft.scan_aperture,
        scan_iso=draft.scan_iso,
        scan_exposure=draft.scan_exposure,
        scan_method=draft.scan_method,
        image_description=desc,
        developer=draft.developer,
        push_pull=draft.push_pull,
        exif_flags=exif_flags,
    )
