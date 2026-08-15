import sqlite3
import json
import os
import time
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
                CREATE TABLE IF NOT EXISTS edit_history (
                    file_hash TEXT,
                    step_index INTEGER,
                    settings_json TEXT,
                    PRIMARY KEY (file_hash, step_index)
                )
            """)

            # Named versions of one frame's edit. Deliberately not a column on edit_history: an edit
            # after stepping back truncates the future branch, so a named step would be deleted by
            # the very thing it is meant to survive.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS work_prints (
                    file_hash TEXT,
                    name TEXT,
                    created_at REAL,
                    settings_json TEXT,
                    PRIMARY KEY (file_hash, name)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_marks (
                    file_hash TEXT PRIMARY KEY,
                    mark TEXT NOT NULL
                )
            """)

            # Migration: add file_path so a mark resolves without the file's hash. Library search
            # joins by path and never hashes, so it would otherwise be blind to triage marks on
            # frames it has not loaded.
            try:
                conn.execute("ALTER TABLE file_marks ADD COLUMN file_path TEXT")
            except sqlite3.OperationalError:
                pass  # already exists

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

    def save_file_mark(self, file_hash: str, mark: Optional[str], file_path: str = "") -> None:
        """Persists a triage mark ('keeper'/'excluded'); None clears it."""
        with self._connect(self.edits_db_path) as conn:
            if mark:
                conn.execute(
                    "INSERT OR REPLACE INTO file_marks (file_hash, mark, file_path) VALUES (?, ?, ?)",
                    (file_hash, mark, file_path),
                )
            else:
                conn.execute("DELETE FROM file_marks WHERE file_hash = ?", (file_hash,))

    def load_file_marks(self) -> dict[str, str]:
        """Returns all triage marks as {file_hash: mark}."""
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT file_hash, mark FROM file_marks")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def load_file_marks_by_path(self) -> dict[str, str]:
        """Triage marks as {file_path: mark}, for lookups that have no hash in hand.
        Marks written before the path column exists are absent, not wrong."""
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT file_path, mark FROM file_marks WHERE file_path IS NOT NULL AND file_path != ''")
            return {str(row[0]): str(row[1]) for row in cursor.fetchall()}

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

    def load_file_settings_many(self, hashes: List[str]) -> dict[str, WorkspaceConfig]:
        """Saved edits for many hashes in one connection — the search facts for a whole
        roll cost one round trip, not one per frame. Hashes with no saved edit are absent
        from the result, which is what marks a frame as never edited."""
        out: dict[str, WorkspaceConfig] = {}
        if not hashes:
            return out
        with self._connect(self.edits_db_path) as conn:
            # Chunked to stay under SQLite's bound-variable limit on long rolls.
            for start in range(0, len(hashes), 500):
                chunk = hashes[start : start + 500]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT file_hash, settings_json FROM file_settings WHERE file_hash IN ({placeholders})",
                    chunk,
                )
                for file_hash, settings_json in cursor.fetchall():
                    out[str(file_hash)] = WorkspaceConfig.from_flat_dict(json.loads(settings_json))
        return out

    def load_settings_by_path(self) -> dict[str, WorkspaceConfig]:
        """Every saved edit that knows its file path, as {file_path: config}.

        The bridge that lets a search join edit metadata onto files it has not opened —
        and the reason library search never needs a hash. Only edited files have a row,
        so this stays small next to the size of an archive.
        """
        out: dict[str, WorkspaceConfig] = {}
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT file_path, settings_json FROM file_settings WHERE file_path IS NOT NULL AND file_path != ''")
            for file_path, settings_json in cursor.fetchall():
                out[str(file_path)] = WorkspaceConfig.from_flat_dict(json.loads(settings_json))
        return out

    def load_file_settings_by_path(self, file_path: str) -> Optional[tuple[str, WorkspaceConfig]]:
        """Look up settings by file path (fallback for when hash changed due to EXIF edits).
        Returns (old_hash, config) if found, or None."""
        if not file_path:
            return None
        with self._connect(self.edits_db_path) as conn:
            # Half-frame rows ('<hash>#<n>') share the scan's path, and rehoming one onto the
            # whole-file identity would steal that half's edit, so exclude them.
            cursor = conn.execute(
                "SELECT file_hash, settings_json FROM file_settings WHERE file_path = ? AND file_hash NOT LIKE '%#%'",
                (file_path,),
            )
            row = cursor.fetchone()
            if row:
                return str(row[0]), WorkspaceConfig.from_flat_dict(json.loads(row[1]))
        return None

    def rehome_file_settings(self, old_hash: str, new_hash: str, file_path: str) -> None:
        """Move settings, undo history and triage mark from old_hash to new_hash (with
        updated path). No-op if new_hash already holds an edit — a live edit is never
        clobbered by a rehome."""
        if old_hash == new_hash:
            return
        with self._connect(self.edits_db_path) as conn:
            if conn.execute("SELECT 1 FROM file_settings WHERE file_hash = ?", (new_hash,)).fetchone():
                return
            row = conn.execute(
                "SELECT settings_json FROM file_settings WHERE file_hash = ?",
                (old_hash,),
            ).fetchone()
            if not row:
                return
            conn.execute(
                "INSERT OR REPLACE INTO file_settings (file_hash, settings_json, file_path) VALUES (?, ?, ?)",
                (new_hash, row[0], file_path),
            )
            conn.execute("DELETE FROM file_settings WHERE file_hash = ?", (old_hash,))
            conn.execute("UPDATE OR REPLACE edit_history SET file_hash = ? WHERE file_hash = ?", (new_hash, old_hash))
            conn.execute("UPDATE OR REPLACE work_prints SET file_hash = ? WHERE file_hash = ?", (new_hash, old_hash))
            conn.execute("UPDATE OR REPLACE file_marks SET file_hash = ? WHERE file_hash = ?", (new_hash, old_hash))

    def save_work_print(self, file_hash: str, name: str, settings: WorkspaceConfig) -> None:
        """Store (or replace) a named version of this frame's edit."""
        with self._connect(self.edits_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO work_prints (file_hash, name, created_at, settings_json) VALUES (?, ?, ?, ?)",
                (file_hash, name, time.time(), json.dumps(settings.to_dict(), default=str)),
            )

    def list_work_prints(self, file_hash: str) -> List[str]:
        """This frame's work-print names, newest first."""
        with self._connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM work_prints WHERE file_hash = ? ORDER BY created_at DESC, rowid DESC",
                (file_hash,),
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def load_work_print(self, file_hash: str, name: str) -> Optional[WorkspaceConfig]:
        with self._connect(self.edits_db_path) as conn:
            row = conn.execute(
                "SELECT settings_json FROM work_prints WHERE file_hash = ? AND name = ?",
                (file_hash, name),
            ).fetchone()
        return WorkspaceConfig.from_flat_dict(json.loads(row[0])) if row else None

    def rename_work_print(self, file_hash: str, name: str, new_name: str) -> None:
        with self._connect(self.edits_db_path) as conn:
            conn.execute(
                "UPDATE OR REPLACE work_prints SET name = ? WHERE file_hash = ? AND name = ?",
                (new_name, file_hash, name),
            )

    def delete_work_print(self, file_hash: str, name: str) -> None:
        with self._connect(self.edits_db_path) as conn:
            conn.execute("DELETE FROM work_prints WHERE file_hash = ? AND name = ?", (file_hash, name))

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
            # Delete steps older than (current_max_index - max_steps). Find the current max index
            # for this file first.
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

    # Database management (view and clear), behind the DB Management dialog.

    @staticmethod
    def _count(conn: sqlite3.Connection, table: str) -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.OperationalError:
            return 0  # table not created yet

    @staticmethod
    def _db_size_bytes(path: str) -> int:
        """On-disk footprint including the WAL/SHM sidecars (uncheckpointed writes
        live in -wal, so the bare .db size understates real usage)."""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += os.path.getsize(path + suffix)
            except OSError:
                pass
        return total

    def database_stats(self) -> dict[str, int]:
        """Row counts per category plus on-disk sizes, for the management dialog.

        ``export_presets`` is one JSON row inside global_settings, so it's counted
        from that list and excluded from ``app_preferences`` to avoid double-counting.
        """
        with self._connect(self.edits_db_path) as conn:
            file_settings = self._count(conn, "file_settings")
            edit_history = self._count(conn, "edit_history")
            work_prints = self._count(conn, "work_prints")
            file_marks = self._count(conn, "file_marks")
            normalization_rolls = self._count(conn, "normalization_rolls")

        with self._connect(self.settings_db_path) as conn:
            global_settings = self._count(conn, "global_settings")

        raw_presets = self.get_global_setting("export_presets", default=None)
        export_presets = len(raw_presets) if isinstance(raw_presets, list) else 0
        has_presets_row = raw_presets is not None

        return {
            "file_settings": file_settings,
            "edit_history": edit_history,
            "work_prints": work_prints,
            "file_marks": file_marks,
            "normalization_rolls": normalization_rolls,
            "export_presets": export_presets,
            # global_settings rows minus the single export_presets row (if present).
            "app_preferences": max(0, global_settings - (1 if has_presets_row else 0)),
            "edits_db_bytes": self._db_size_bytes(self.edits_db_path),
            "settings_db_bytes": self._db_size_bytes(self.settings_db_path),
        }

    def _wipe(self, path: str, tables: list[str]) -> None:
        """Empty the given tables, then checkpoint + VACUUM so the disk footprint
        actually shrinks (WAL retains freed pages until checkpointed; VACUUM
        rebuilds the file). VACUUM must run outside a transaction — a fresh
        connection with no prior DML has none open."""
        with self._connect(path) as conn:
            for table in tables:
                try:
                    conn.execute(f"DELETE FROM {table}")
                except sqlite3.OperationalError:
                    pass  # table absent — nothing to clear
        with self._connect(path) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")

    def clear_saved_edits(self) -> None:
        """Drop per-image looks: saved edits, their undo history, work prints, and
        keep/reject marks. Rig calibration (normalization rolls), export presets, and app
        preferences are left intact — so a reloaded image starts from defaults
        without losing the user's tooling. Flat-field profiles live in the file
        store (APP_CONFIG.flatfield_dir), not here, so they are untouched too."""
        self._wipe(self.edits_db_path, ["file_settings", "edit_history", "work_prints", "file_marks"])

    def reset_everything(self) -> None:
        """Full clean slate: every table in both databases. Export presets, rig
        profiles, and all app preferences go too. Schema is preserved (rows only),
        so the app keeps working against the emptied databases without re-init.
        File-store assets (flat-field profiles, sensor/crosstalk matrices) are on
        disk, not in these databases, so they survive — as with a fresh install."""
        self._wipe(
            self.edits_db_path,
            ["file_settings", "edit_history", "work_prints", "file_marks", "normalization_rolls"],
        )
        self._wipe(self.settings_db_path, ["global_settings"])
