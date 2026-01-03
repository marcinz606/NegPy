import sqlite3
import json
import os
from typing import Any, Optional
from src.config import APP_CONFIG
from src.domain_objects import ImageSettings


class SettingsRepository:
    """
    Handles the persistence of image and global settings using SQLite.
    """

    def __init__(self) -> None:
        self.edits_db_path = APP_CONFIG.edits_db_path
        self.settings_db_path = APP_CONFIG.settings_db_path
        self._init_db()

    def _init_db(self) -> None:
        """
        Initializes the SQLite databases.
        """
        os.makedirs(os.path.dirname(self.edits_db_path), exist_ok=True)

        # 1. Edits DB
        with sqlite3.connect(self.edits_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_settings (
                    file_hash TEXT PRIMARY KEY,
                    settings_json TEXT
                )
            """)

        # 2. Global Settings DB
        with sqlite3.connect(self.settings_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT
                )
            """)

    def save_file_settings(self, file_hash: str, settings: ImageSettings) -> None:
        """
        Saves all settings for a specific file.
        """
        with sqlite3.connect(self.edits_db_path) as conn:
            # dataclasses.asdict handled by ImageSettings.to_dict()
            settings_json = json.dumps(settings.to_dict())
            conn.execute(
                "INSERT OR REPLACE INTO file_settings (file_hash, settings_json) VALUES (?, ?)",
                (file_hash, settings_json),
            )

    def load_file_settings(self, file_hash: str) -> Optional[ImageSettings]:
        """
        Loads settings for a specific file.
        """
        with sqlite3.connect(self.edits_db_path) as conn:
            cursor = conn.execute(
                "SELECT settings_json FROM file_settings WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return ImageSettings.from_dict(data)
        return None

    def save_global_setting(self, key: str, value: Any) -> None:
        """
        Saves a global configuration setting.
        """
        with sqlite3.connect(self.settings_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO global_settings (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a global configuration setting.
        """
        with sqlite3.connect(self.settings_db_path) as conn:
            cursor = conn.execute(
                "SELECT value_json FROM global_settings WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return default


# Global instance for easy access
repository = SettingsRepository()


# Legacy compatibility wrappers
def init_db() -> None:
    pass  # Managed by class init now


def db_save_file_settings(file_hash: str, settings: ImageSettings) -> None:
    repository.save_file_settings(file_hash, settings)


def db_load_file_settings(file_hash: str) -> Optional[ImageSettings]:
    return repository.load_file_settings(file_hash)


def db_save_global_setting(key: str, value: Any) -> None:
    repository.save_global_setting(key, value)


def db_get_global_setting(key: str, default: Any = None) -> Any:
    return repository.get_global_setting(key, default)
