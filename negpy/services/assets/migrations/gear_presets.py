"""One-time conversion of gear presets into metadata presets.

Gear presets were library records holding three ids (camera, lens, film stock).
Metadata presets store the resolved field values instead, so each one is read
through the library once and written to the metadata preset store under its
display name, with its notes. Bundled presets convert too — the metadata store
has no bundled tier, so the shipped combinations become the user's own, editable
copies.

Best-effort and idempotent: guarded by a done flag, an existing preset of the
same name is never overwritten, and nothing raises into app startup. The flag is
written only when every source file was read and every preset was stored, so a
damaged file or a refused write retries on the next launch instead of stranding
the presets behind it.
"""

from __future__ import annotations

import json
import os
from itertools import count

from negpy.features.metadata.gear_logic import metadata_from_gear
from negpy.features.metadata.models import GEAR_FIELDS, MetadataConfig
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.paths import get_resource_path
from negpy.services.assets.gear import GearProfiles
from negpy.services.assets.presets import MetadataPresets, sanitize_preset_name, with_preset_notes

logger = get_logger(__name__)

_DONE_FLAG = "gear_presets_migrated"
_GEAR_PRESETS_FILE = "gear_presets.json"
_UNNAMED = "Unnamed preset"


def _read_presets(path: str) -> tuple[bool, list[dict]]:
    """(readable, presets). A missing file is readable and empty; a damaged one is not,
    and must leave the migration pending so a repaired file still converts."""
    if not os.path.isfile(path):
        return True, []
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return False, []
    if not isinstance(raw, list):
        return False, []
    return True, [item for item in raw if isinstance(item, dict)]


def _safe_name(name: str) -> str:
    """A name the preset store accepts, with the gaps a stripped separator leaves
    collapsed: "AE-1P / FD 50 f/1.4" names two things, not two blank fields."""
    return " ".join(sanitize_preset_name(name).split())


def _unique_name(base: str, taken: set[str]) -> str:
    """``base``, or its first free numbered variant: two unnamed presets are two presets."""
    if base.casefold() not in taken:
        return base
    return next(f"{base} {i}" for i in count(2) if f"{base} {i}".casefold() not in taken)


def migrate_gear_presets(repo) -> None:
    """Write every gear preset to the metadata preset store, once."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    unstored = False
    try:
        bundled_ok, bundled = _read_presets(os.path.join(get_resource_path("gear"), _GEAR_PRESETS_FILE))
        user_ok, user = _read_presets(os.path.join(APP_CONFIG.gear_dir, _GEAR_PRESETS_FILE))
        if not (bundled_ok and user_ok):
            logger.warning("Gear preset file unreadable; migration stays pending")
            return
        seen_ids = {str(p.get("id")) for p in bundled}
        presets = bundled + [p for p in user if str(p.get("id")) not in seen_ids]

        library = GearProfiles.load_library()
        # Case-folded: on macOS and Windows a differently-cased name is the same file.
        existing = {n.casefold() for n in MetadataPresets.list_presets()}
        for preset in presets:
            name = _safe_name(str(preset.get("displayName") or preset.get("display_name") or ""))
            if not name:
                name = _unique_name(_UNNAMED, existing)
            if name.casefold() in existing:
                continue
            resolved = metadata_from_gear(
                MetadataConfig(),
                library,
                camera_id=str(preset.get("cameraId") or preset.get("camera_id") or ""),
                lens_id=str(preset.get("lensId") or preset.get("lens_id") or ""),
                film_stock_id=str(preset.get("filmStockId") or preset.get("film_stock_id") or ""),
            )
            fields = {f: getattr(resolved, f) for f in GEAR_FIELDS}
            try:
                MetadataPresets.save_preset(name, with_preset_notes(fields, str(preset.get("notes") or "")))
            except (OSError, ValueError) as e:
                # One preset the store refuses must not strand every preset after it.
                logger.warning("Gear preset %r not converted: %s", name, e)
                unstored = True
                continue
            existing.add(name.casefold())
    except Exception as e:  # startup must survive a broken gear file
        logger.warning("Gear preset migration skipped: %s", e)
        return
    if not unstored:
        repo.save_global_setting(_DONE_FLAG, True)
