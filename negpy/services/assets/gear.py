"""JSON-backed gear library (cameras, lenses, film stocks, chemistry, rigs, presets)."""

from __future__ import annotations

import json
import os
from typing import TypeVar

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
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.paths import get_resource_path

T = TypeVar("T")

_CAMERAS_FILE = "cameras.json"
_LENSES_FILE = "lenses.json"
_FILM_STOCKS_FILE = "film_stocks.json"
_DEVELOPERS_FILE = "developers.json"
_SCAN_SETUPS_FILE = "scan_setups.json"
_GEAR_PRESETS_FILE = "gear_presets.json"
_PROCESS_SCAN_PRESETS_FILE = "process_scan_presets.json"


class GearProfiles:
    """
    JSON I/O for analog gear libraries under APP_CONFIG.gear_dir.
    Disk I/O on dropdown/dialog open and on save — never per render.
    """

    @staticmethod
    def _gear_dir() -> str:
        path = APP_CONFIG.gear_dir
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _read_list(path: str, factory: type[T]) -> list[T]:
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                return []
            return [factory.from_dict(item) for item in raw if isinstance(item, dict)]  # type: ignore[attr-defined]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    @staticmethod
    def _write_list(path: str, items: list) -> None:
        tmp = path + ".tmp"
        data = [item.to_dict() for item in items]
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)

    @staticmethod
    def _read_bundled_list(fname: str, factory: type[T]) -> list[T]:
        items = GearProfiles._read_list(os.path.join(get_resource_path("gear"), fname), factory)
        for item in items:
            item.is_bundled = True  # type: ignore[attr-defined]
        return items

    @staticmethod
    def _merge(bundled: list[T], user: list[T]) -> list[T]:
        """Bundled ∪ user, deduped by id, bundled wins."""
        bundled_ids = {item.id for item in bundled}  # type: ignore[attr-defined]
        return bundled + [item for item in user if item.id not in bundled_ids]  # type: ignore[attr-defined]

    @staticmethod
    def _load_merged(fname: str, factory: type[T]) -> list[T]:
        bundled = GearProfiles._read_bundled_list(fname, factory)
        user = GearProfiles._read_list(os.path.join(GearProfiles._gear_dir(), fname), factory)
        return GearProfiles._merge(bundled, user)

    @staticmethod
    def load_library() -> GearLibrary:
        return GearLibrary(
            cameras=GearProfiles._load_merged(_CAMERAS_FILE, Camera),
            lenses=GearProfiles._load_merged(_LENSES_FILE, Lens),
            film_stocks=GearProfiles._load_merged(_FILM_STOCKS_FILE, FilmStock),
            developers=GearProfiles._load_merged(_DEVELOPERS_FILE, DeveloperRecipe),
            scan_setups=GearProfiles._load_merged(_SCAN_SETUPS_FILE, ScanSetup),
            gear_presets=GearProfiles._load_merged(_GEAR_PRESETS_FILE, GearPreset),
            process_scan_presets=GearProfiles._load_merged(_PROCESS_SCAN_PRESETS_FILE, ProcessScanPreset),
        )

    @staticmethod
    def save_cameras(cameras: list[Camera]) -> None:
        GearProfiles._write_list(os.path.join(GearProfiles._gear_dir(), _CAMERAS_FILE), cameras)

    @staticmethod
    def save_lenses(lenses: list[Lens]) -> None:
        GearProfiles._write_list(os.path.join(GearProfiles._gear_dir(), _LENSES_FILE), lenses)

    @staticmethod
    def save_film_stocks(film_stocks: list[FilmStock]) -> None:
        GearProfiles._write_list(os.path.join(GearProfiles._gear_dir(), _FILM_STOCKS_FILE), film_stocks)

    @staticmethod
    def save_developers(developers: list[DeveloperRecipe]) -> None:
        GearProfiles._write_list(os.path.join(GearProfiles._gear_dir(), _DEVELOPERS_FILE), developers)

    @staticmethod
    def save_scan_setups(scan_setups: list[ScanSetup]) -> None:
        GearProfiles._write_list(os.path.join(GearProfiles._gear_dir(), _SCAN_SETUPS_FILE), scan_setups)

    @staticmethod
    def save_gear_presets(presets: list[GearPreset]) -> None:
        GearProfiles._write_list(os.path.join(GearProfiles._gear_dir(), _GEAR_PRESETS_FILE), presets)

    @staticmethod
    def save_process_scan_presets(presets: list[ProcessScanPreset]) -> None:
        GearProfiles._write_list(os.path.join(GearProfiles._gear_dir(), _PROCESS_SCAN_PRESETS_FILE), presets)

    @staticmethod
    def save_library(library: GearLibrary) -> None:
        GearProfiles.save_cameras([c for c in library.cameras if not c.is_bundled])
        GearProfiles.save_lenses([lens for lens in library.lenses if not lens.is_bundled])
        GearProfiles.save_film_stocks([f for f in library.film_stocks if not f.is_bundled])
        GearProfiles.save_developers([d for d in library.developers if not d.is_bundled])
        GearProfiles.save_scan_setups([s for s in library.scan_setups if not s.is_bundled])
        GearProfiles.save_gear_presets([p for p in library.gear_presets if not p.is_bundled])
        GearProfiles.save_process_scan_presets([p for p in library.process_scan_presets if not p.is_bundled])

    @staticmethod
    def ensure_user_dir() -> None:
        """Make sure the user's gear directory exists; no seeding."""
        GearProfiles._gear_dir()
