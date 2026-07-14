import sqlite3
import json
import os
from contextlib import contextmanager
from typing import Any, List, Optional
from negpy.domain.models import ExportPreset, WorkspaceConfig
from negpy.domain.interfaces import IRepository


class StorageRepository(IRepository):
    """
    SQLite backend for settings.
    """

    def __init__(self, edits_db_path: str, settings_db_path: str) -> None:
        self.edits_db_path = edits_db_path
        self.settings_db_path = settings_db_path

    @contextmanager
    def _connect(self, path: str):
        """Connection context manager that actually closes the connection (sqlite3's own doesn't)."""
        conn = sqlite3.connect(path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        """
        Ensures DB tables exist.
        """
        os.makedirs(os.path.dirname(self.edits_db_path), exist_ok=True)

        with self._connect(self.edits_db_path) as conn:
            # WAL persists on the DB file; per-call connections inherit it
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_settings (
                    file_hash TEXT PRIMARY KEY,
                    settings_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS normalization_rolls (
                    name TEXT PRIMARY KEY,
                    floors_json TEXT,
                    ceils_json TEXT,
                    cast_json TEXT
                )
            """)
            # Migration: add cast_json if not exists
            try:
                conn.execute("ALTER TABLE normalization_rolls ADD COLUMN cast_json TEXT")
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS flatfield_profiles (
                    name TEXT PRIMARY KEY,
                    path TEXT
                )
            """)
            # Migration: add radial distortion coefficient (rig-level, like the flat frame).
            try:
                conn.execute("ALTER TABLE flatfield_profiles ADD COLUMN k1 REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS edit_history (
                    file_hash TEXT,
                    step_index INTEGER,
                    settings_json TEXT,
                    PRIMARY KEY (file_hash, step_index)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_marks (
                    file_hash TEXT PRIMARY KEY,
                    mark TEXT NOT NULL
                )
            """)

            # Migration: add file_path column for path-based settings recovery
            try:
                conn.execute("ALTER TABLE file_settings ADD COLUMN file_path TEXT")
            except sqlite3.OperationalError:
                pass  # already exists

            # Migration: index on file_path for path-based fallback queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_settings_path ON file_settings(file_path)")

        with self._connect(self.settings_db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT
                )
            """)

    def save_normalization_roll(self, name: str, floors: tuple, ceils: tuple, cast: tuple = (0.0, 0.0, 0.0)) -> None:
        """
        Persists a named normalization baseline (roll).
        """
        with self._connect(self.edits_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO normalization_rolls (name, floors_json, ceils_json, cast_json) VALUES (?, ?, ?, ?)",
                (name, json.dumps(floors), json.dumps(ceils), json.dumps(cast)),
            )

    def load_normalization_roll(self, name: str) -> Optional[tuple[tuple, tuple]]:
        """
        Retrieves a named normalization baseline.
        """
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT floors_json, ceils_json FROM normalization_rolls WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                floors = tuple(json.loads(row[0]))
                ceils = tuple(json.loads(row[1]))
                return floors, ceils
        return None

    def list_normalization_rolls(self) -> list[str]:
        """
        Returns names of all saved normalization rolls.
        """
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT name FROM normalization_rolls ORDER BY name")
            return [row[0] for row in cursor.fetchall()]

    def delete_normalization_roll(self, name: str) -> None:
        """
        Deletes a named normalization baseline.
        """
        with self._connect(self.edits_db_path) as conn:
            conn.execute("DELETE FROM normalization_rolls WHERE name = ?", (name,))

    def save_flatfield_profile(self, name: str, path: str, k1: float = 0.0) -> None:
        """Persists a named flat-field reference profile (reference path + rig distortion)."""
        with self._connect(self.edits_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO flatfield_profiles (name, path, k1) VALUES (?, ?, ?)",
                (name, path, float(k1)),
            )

    def get_flatfield_profile(self, name: str) -> Optional[tuple[str, float]]:
        """Returns (reference path, distortion k1) for a named flat-field profile."""
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT path, k1 FROM flatfield_profiles WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return str(row[0]), float(row[1] or 0.0)
        return None

    def list_flatfield_profiles(self) -> list[str]:
        """Returns names of all saved flat-field profiles."""
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT name FROM flatfield_profiles ORDER BY name")
            return [row[0] for row in cursor.fetchall()]

    def delete_flatfield_profile(self, name: str) -> None:
        """Deletes a named flat-field profile."""
        with self._connect(self.edits_db_path) as conn:
            conn.execute("DELETE FROM flatfield_profiles WHERE name = ?", (name,))

    def save_file_mark(self, file_hash: str, mark: Optional[str]) -> None:
        """Persists a triage mark ('keeper'/'excluded'); None clears it."""
        with self._connect(self.edits_db_path) as conn:
            if mark:
                conn.execute("INSERT OR REPLACE INTO file_marks (file_hash, mark) VALUES (?, ?)", (file_hash, mark))
            else:
                conn.execute("DELETE FROM file_marks WHERE file_hash = ?", (file_hash,))

    def load_file_marks(self) -> dict[str, str]:
        """Returns all triage marks as {file_hash: mark}."""
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT file_hash, mark FROM file_marks")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def save_file_settings(self, file_hash: str, settings: WorkspaceConfig, file_path: str = "") -> None:
        with self._connect(self.edits_db_path) as conn:
            settings_json = json.dumps(settings.to_dict(), default=str)
            conn.execute(
                "INSERT OR REPLACE INTO file_settings (file_hash, settings_json, file_path) VALUES (?, ?, ?)",
                (file_hash, settings_json, file_path),
            )

    def load_file_settings(self, file_hash: str) -> Optional[WorkspaceConfig]:
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT settings_json FROM file_settings WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return WorkspaceConfig.from_flat_dict(data)
        return None

    def load_file_settings_by_path(self, file_path: str) -> Optional[tuple[str, WorkspaceConfig]]:
        """Look up settings by file path (fallback for when hash changed due to EXIF edits).
        Returns (old_hash, config) if found, or None."""
        if not file_path:
            return None
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT file_hash, settings_json FROM file_settings WHERE file_path = ?",
                (file_path,),
            )
            row = cursor.fetchone()
            if row:
                return str(row[0]), WorkspaceConfig.from_flat_dict(json.loads(row[1]))
        return None

    def rehome_file_settings(self, old_hash: str, new_hash: str, file_path: str) -> None:
        """Copy settings from old_hash to new_hash (with updated path), then delete old entry."""
        if old_hash == new_hash:
            return
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT settings_json FROM file_settings WHERE file_hash = ?",
                (old_hash,),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    "INSERT OR REPLACE INTO file_settings (file_hash, settings_json, file_path) VALUES (?, ?, ?)",
                    (new_hash, row[0], file_path),
                )
                conn.execute("DELETE FROM file_settings WHERE file_hash = ?", (old_hash,))

    def save_history_step(self, file_hash: str, index: int, settings: WorkspaceConfig) -> None:
        with self._connect(self.edits_db_path) as conn:
            settings_json = json.dumps(settings.to_dict(), default=str)
            conn.execute(
                "INSERT OR REPLACE INTO edit_history (file_hash, step_index, settings_json) VALUES (?, ?, ?)",
                (file_hash, index, settings_json),
            )

    def load_history_step(self, file_hash: str, index: int) -> Optional[WorkspaceConfig]:
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT settings_json FROM edit_history WHERE file_hash = ? AND step_index = ?",
                (file_hash, index),
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return WorkspaceConfig.from_flat_dict(data)
        return None

    def load_all_history(self, file_hash: str) -> List[tuple[int, WorkspaceConfig]]:
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT step_index, settings_json FROM edit_history WHERE file_hash = ? ORDER BY step_index",
                (file_hash,),
            )
            return [(int(idx), WorkspaceConfig.from_flat_dict(json.loads(js))) for idx, js in cursor.fetchall()]

    def get_max_history_index(self, file_hash: str) -> int:
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT MAX(step_index) FROM edit_history WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
        return 0

    def clear_history(self, file_hash: str) -> None:
        with self._connect(self.edits_db_path) as conn:
            conn.execute("DELETE FROM edit_history WHERE file_hash = ?", (file_hash,))

    def truncate_history_above(self, file_hash: str, index: int) -> None:
        """Deletes all history steps with step_index > index (orphaned future branch)."""
        with self._connect(self.edits_db_path) as conn:
            conn.execute(
                "DELETE FROM edit_history WHERE file_hash = ? AND step_index > ?",
                (file_hash, index),
            )

    def prune_history(self, file_hash: str, max_steps: int = 10) -> None:
        with self._connect(self.edits_db_path) as conn:
            # Delete steps that are older than (current_max_index - max_steps)
            # Find current max index for this file
            cursor = conn.execute("SELECT MAX(step_index) FROM edit_history WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                max_idx = row[0]
                conn.execute(
                    "DELETE FROM edit_history WHERE file_hash = ? AND step_index <= ?",
                    (file_hash, max_idx - max_steps),
                )

    def save_global_setting(self, key: str, value: Any) -> None:
        with self._connect(self.settings_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO global_settings (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value, default=str)),
            )

    def save_global_settings(self, settings: dict[str, Any]) -> None:
        """Writes many global settings in one transaction (one connection, one commit)."""
        with self._connect(self.settings_db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO global_settings (key, value_json) VALUES (?, ?)",
                [(k, json.dumps(v, default=str)) for k, v in settings.items()],
            )

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        with self._connect(self.settings_db_path) as conn:
            cursor = conn.execute("SELECT value_json FROM global_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return default

    def save_export_presets(self, presets: List[ExportPreset]) -> None:
        self.save_global_setting("export_presets", [p.to_dict() for p in presets])

    def load_export_presets(self) -> List[ExportPreset]:
        from negpy.domain.models import ExportFormat, ExportResolutionMode

        raw = self.get_global_setting("export_presets", default=None)
        if raw is None:
            return [
                ExportPreset(
                    name="JPEG",
                    enabled=True,
                    export_fmt=ExportFormat.JPEG,
                    jpeg_quality=90,
                    export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
                ),
                ExportPreset(
                    name="TIFF", enabled=False, export_fmt=ExportFormat.TIFF, export_resolution_mode=ExportResolutionMode.ORIGINAL.value
                ),
                ExportPreset(
                    name="PNG", enabled=False, export_fmt=ExportFormat.PNG, export_resolution_mode=ExportResolutionMode.ORIGINAL.value
                ),
            ]
        result = []
        for d in raw:
            try:
                result.append(ExportPreset.from_dict(d))
            except Exception:
                pass
        return result
