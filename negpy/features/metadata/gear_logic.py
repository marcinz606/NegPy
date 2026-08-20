"""Resolve gear library selections into MetadataConfig updates."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

from negpy.features.metadata.gear_models import (
    Camera,
    DeveloperRecipe,
    FilmStock,
    GearLibrary,
    GearPreset,
    Lens,
    ProcessScanPreset,
    ScanSetup,
)
from negpy.features.metadata.models import MetadataConfig

GearItem = Union[Camera, Lens, FilmStock, DeveloperRecipe, ScanSetup, GearPreset, ProcessScanPreset]


def _process_scan_search_parts(item: Union[GearPreset, ProcessScanPreset], library: GearLibrary) -> list[str]:
    """Searchable text contributed by a preset's developer and scan-setup slots."""
    parts: list[str] = []
    dev = library.get_developer(item.developer_id)
    rig = library.get_scan_setup(item.scan_setup_id)
    if dev:
        parts.extend([dev.resolved_display_name, dev.developer, dev.lab])
    if rig:
        parts.extend([rig.resolved_display_name, rig.method.value, rig.device])
    return parts


def gear_search_text(item: GearItem, library: Optional[GearLibrary] = None) -> str:
    """Lowercase searchable text for substring filtering."""
    parts: list[str] = []

    if isinstance(item, Camera):
        parts = [item.display_name, item.make, item.model, item.notes]
    elif isinstance(item, Lens):
        parts = [item.display_name, item.make, item.lens_model, item.notes]
        if item.focal_length_mm is not None:
            parts.append(f"{item.focal_length_mm:g}")
        if item.max_aperture is not None:
            parts.append(f"{item.max_aperture:g}")
    elif isinstance(item, FilmStock):
        parts = [
            item.display_name,
            item.manufacturer,
            item.stock_name,
            item.notes,
            str(item.iso),
            item.format.value,
            item.color_type.value,
        ]
    elif isinstance(item, DeveloperRecipe):
        parts = [
            item.display_name,
            item.developer,
            item.dilution,
            item.time,
            item.process.value,
            item.lab,
            item.notes,
        ]
        if item.temperature_c is not None:
            parts.append(f"{item.temperature_c:g}")
    elif isinstance(item, ScanSetup):
        parts = [
            item.display_name,
            item.method.value,
            item.device,
            item.light_source,
            item.holder,
            item.software,
            item.notes,
        ]
    elif isinstance(item, GearPreset):
        parts = [item.display_name, item.notes]
        if library is not None:
            cam = library.get_camera(item.camera_id)
            lens = library.get_lens(item.lens_id)
            stock = library.get_film_stock(item.film_stock_id)
            if cam:
                parts.extend([cam.resolved_display_name, cam.make, cam.model])
            if lens:
                parts.extend([lens.resolved_display_name, lens.make, lens.lens_model])
            if stock:
                parts.extend([stock.resolved_display_name, stock.manufacturer, stock.stock_name])
            parts.extend(_process_scan_search_parts(item, library))
    elif isinstance(item, ProcessScanPreset):
        parts = [item.display_name, item.notes]
        if library is not None:
            parts.extend(_process_scan_search_parts(item, library))

    return " ".join(p.strip() for p in parts if p and str(p).strip()).lower()


def matches_gear_filter(item: GearItem, query: str, library: Optional[GearLibrary] = None) -> bool:
    needle = query.strip().lower()
    if not needle:
        return True
    return needle in gear_search_text(item, library)


def metadata_from_gear(
    config: MetadataConfig,
    library: GearLibrary,
    *,
    gear_preset_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    lens_id: Optional[str] = None,
    film_stock_id: Optional[str] = None,
    process_scan_preset_id: Optional[str] = None,
    developer_id: Optional[str] = None,
    scan_setup_id: Optional[str] = None,
    clear_preset: bool = False,
    clear_process_scan_preset: bool = False,
) -> MetadataConfig:
    """Build updated MetadataConfig from gear library selections.

    Pass ``None`` (default) to leave an id unchanged; pass ``""`` to clear it.
    """
    if clear_preset:
        preset_id = ""
    elif gear_preset_id is not None:
        preset_id = gear_preset_id
    else:
        preset_id = config.gear_preset_id

    if clear_process_scan_preset:
        ps_preset_id = ""
    elif process_scan_preset_id is not None:
        ps_preset_id = process_scan_preset_id
    else:
        ps_preset_id = config.process_scan_preset_id

    cam_id = config.camera_id if camera_id is None else camera_id
    lens_id_val = config.lens_id if lens_id is None else lens_id
    film_id = config.film_stock_id if film_stock_id is None else film_stock_id
    dev_id = config.developer_id if developer_id is None else developer_id
    rig_id = config.scan_setup_id if scan_setup_id is None else scan_setup_id

    # The free-text fields are re-derived on a deliberate pick only. An id merely
    # carried over from the config leaves whatever the user typed in place.
    derive_developer = bool(developer_id)
    derive_scanning = bool(scan_setup_id)

    if preset_id:
        preset = library.get_gear_preset(preset_id)
        if preset:
            # Preset is authoritative: empty preset slots clear manual picks.
            cam_id = preset.camera_id
            lens_id_val = preset.lens_id
            film_id = preset.film_stock_id
            # Chemistry and rig are additive here, so a gear pick never wipes them.
            # Only the Process & Scan preset owns those two slots outright.
            derive_developer = derive_developer or bool(preset.developer_id)
            derive_scanning = derive_scanning or bool(preset.scan_setup_id)
            dev_id = preset.developer_id or dev_id
            rig_id = preset.scan_setup_id or rig_id

    if ps_preset_id:
        ps_preset = library.get_process_scan_preset(ps_preset_id)
        if ps_preset:
            derive_developer = True
            derive_scanning = True
            dev_id = ps_preset.developer_id
            rig_id = ps_preset.scan_setup_id

    camera_make = ""
    camera_model = ""
    lens_make = ""
    lens_model = ""
    focal_length: Optional[float] = None
    max_aperture: Optional[float] = None
    film = config.film
    film_manufacturer = ""
    film_iso: Optional[int] = None
    film_format = config.format
    film_color_type = ""
    developer = config.developer
    scanning = config.scanning

    if cam_id:
        cam = library.get_camera(cam_id)
        if cam:
            camera_make = cam.make
            camera_model = cam.model

    if lens_id_val:
        lens = library.get_lens(lens_id_val)
        if lens:
            lens_make = lens.make
            lens_model = lens.lens_model or lens.resolved_display_name
            focal_length = lens.focal_length_mm
            max_aperture = lens.max_aperture

    if film_id:
        stock = library.get_film_stock(film_id)
        if stock:
            film = stock.full_film_label
            film_manufacturer = stock.manufacturer
            film_iso = stock.iso
            film_format = stock.format.value
            film_color_type = stock.color_type.value

    if dev_id and (derive_developer or not config.developer.strip()):
        recipe = library.get_developer(dev_id)
        if recipe:
            developer = recipe.full_developer_label

    if rig_id and (derive_scanning or not config.scanning.strip()):
        rig = library.get_scan_setup(rig_id)
        if rig:
            scanning = rig.full_scan_label

    return replace(
        config,
        gear_preset_id=preset_id,
        camera_id=cam_id,
        lens_id=lens_id_val,
        film_stock_id=film_id,
        process_scan_preset_id=ps_preset_id,
        developer_id=dev_id,
        scan_setup_id=rig_id,
        developer=developer,
        scanning=scanning,
        camera_make=camera_make,
        camera_model=camera_model,
        lens_make=lens_make,
        lens_model=lens_model,
        focal_length_mm=focal_length,
        max_aperture=max_aperture,
        film=film,
        film_manufacturer=film_manufacturer,
        film_iso=film_iso,
        format=film_format if film_id else config.format,
        film_color_type=film_color_type,
    )
