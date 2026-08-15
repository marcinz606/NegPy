"""One-time migration of legacy DB-backed flat-field profiles to the npz file store.

Before the file store, profiles lived in a ``flatfield_profiles`` table (name, path,
k1) and each per-image edit stored the resolved reference *path*. This bakes every
legacy profile's gain from its reference (if the file is still present), rewrites the
edits to reference the new profile by opaque id, remaps the active-profile setting,
then drops the old table. Best-effort and idempotent — guarded by a done flag, and it
never raises into app startup.
"""

import json
import sqlite3
from typing import Dict

from negpy.kernel.system.logging import get_logger
from negpy.services.assets.flatfield import FlatFieldProfiles

logger = get_logger(__name__)

_DONE_FLAG = "flatfield_migrated_v2"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def _rewrite_configs(conn: sqlite3.Connection, table: str, key_col: str, path_to_id: Dict[str, str]) -> None:
    """Swap each edit's legacy ``reference_path`` for the new ``profile_id``."""
    rows = conn.execute(f"SELECT {key_col}, settings_json FROM {table}").fetchall()
    for key, settings_json in rows:
        if not settings_json or "reference_path" not in settings_json:
            continue
        try:
            data = json.loads(settings_json)
        except (ValueError, TypeError):
            continue
        if "reference_path" not in data:
            continue
        path = data.pop("reference_path") or ""
        data["profile_id"] = path_to_id.get(path, "")
        conn.execute(
            f"UPDATE {table} SET settings_json = ? WHERE {key_col} = ?",
            (json.dumps(data, default=str), key),
        )


def migrate_legacy_flatfield_profiles(repo) -> None:
    """Bake legacy DB profiles into the npz store and repoint edits/settings at them."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        with sqlite3.connect(repo.edits_db_path) as conn:
            if _table_exists(conn, "flatfield_profiles"):
                legacy = conn.execute("SELECT name, path, k1 FROM flatfield_profiles").fetchall()

                name_to_id: Dict[str, str] = {}
                path_to_id: Dict[str, str] = {}
                for name, path, k1 in legacy:
                    new_id = FlatFieldProfiles.create(str(name), str(path or ""), float(k1 or 0.0))
                    if new_id is None:
                        # The reference file is gone, so there is nothing to bake and the correction was already
                        # broken. Leave the name and path unmapped, so edits fall back to inactive.
                        logger.warning("Flat-field migration: could not bake profile %r (reference missing)", name)
                        continue
                    name_to_id[str(name)] = new_id
                    if path:
                        path_to_id[str(path)] = new_id

                _rewrite_configs(conn, "file_settings", "file_hash", path_to_id)
                if _table_exists(conn, "edit_history"):
                    _rewrite_configs(conn, "edit_history", "rowid", path_to_id)

                conn.execute("DROP TABLE flatfield_profiles")

                active_name = repo.get_global_setting("flatfield_active_profile")
                if active_name:
                    repo.save_global_setting("flatfield_active_profile", name_to_id.get(str(active_name), ""))
    except Exception:
        logger.exception("Flat-field migration failed; continuing without it")

    repo.save_global_setting(_DONE_FLAG, True)
