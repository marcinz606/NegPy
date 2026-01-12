import sqlite3
import json
import os
from typing import Any, Optional
from src.domain.models import WorkspaceConfig
from src.domain.interfaces import IRepository


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

        with sqlite3.connect(self.settings_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT
                )
            """)

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
            cursor = conn.execute(
                "SELECT value_json FROM global_settings WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return default
