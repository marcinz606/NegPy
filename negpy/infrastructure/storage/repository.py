import sqlite3
import json
import os
from typing import Any, Optional
from negpy.domain.models import WorkspaceConfig
from negpy.domain.interfaces import IRepository


class StorageRepository(IRepository):
    """
    SQLite backend for settings.
    """

    def __init__(self, edits_db_path: str, settings_db_path: str) -> None:
        self.edits_db_path = edits_db_path
        self.settings_db_path = settings_db_path

    def initialize(self) -> None:
        """
        Ensures DB tables exist.
        """
        os.makedirs(os.path.dirname(self.edits_db_path), exist_ok=True)

        with sqlite3.connect(self.edits_db_path) as conn:
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

        with sqlite3.connect(self.settings_db_path) as conn:
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
        with sqlite3.connect(self.edits_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO normalization_rolls (name, floors_json, ceils_json, cast_json) VALUES (?, ?, ?, ?)",
                (name, json.dumps(floors), json.dumps(ceils), json.dumps(cast)),
            )

    def load_normalization_roll(self, name: str) -> Optional[tuple[tuple, tuple, tuple]]:
        """
        Retrieves a named normalization baseline.
        """
        with sqlite3.connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT floors_json, ceils_json, cast_json FROM normalization_rolls WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                floors = tuple(json.loads(row[0]))
                ceils = tuple(json.loads(row[1]))
                cast = tuple(json.loads(row[2])) if row[2] else (0.0, 0.0, 0.0)
                return floors, ceils, cast
        return None

    def list_normalization_rolls(self) -> list[str]:
        """
        Returns names of all saved normalization rolls.
        """
        with sqlite3.connect(self.edits_db_path) as conn:
            cursor = conn.execute("SELECT name FROM normalization_rolls ORDER BY name")
            return [row[0] for row in cursor.fetchall()]

    def delete_normalization_roll(self, name: str) -> None:
        """
        Deletes a named normalization baseline.
        """
        with sqlite3.connect(self.edits_db_path) as conn:
            conn.execute("DELETE FROM normalization_rolls WHERE name = ?", (name,))

    def save_file_settings(self, file_hash: str, settings: WorkspaceConfig) -> None:
        with sqlite3.connect(self.edits_db_path) as conn:
            settings_json = json.dumps(settings.to_dict(), default=str)
            conn.execute(
                "INSERT OR REPLACE INTO file_settings (file_hash, settings_json) VALUES (?, ?)",
                (file_hash, settings_json),
            )

    def load_file_settings(self, file_hash: str) -> Optional[WorkspaceConfig]:
        with sqlite3.connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT settings_json FROM file_settings WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return WorkspaceConfig.from_flat_dict(data)
        return None

    def save_global_setting(self, key: str, value: Any) -> None:
        with sqlite3.connect(self.settings_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO global_settings (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value, default=str)),
            )

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        with sqlite3.connect(self.settings_db_path) as conn:
            cursor = conn.execute("SELECT value_json FROM global_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return default
