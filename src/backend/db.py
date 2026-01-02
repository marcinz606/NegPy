import sqlite3
import json
import os
from typing import Dict, Any, Optional
from src.config import APP_CONFIG

def init_db() -> None:
    """
    Initializes the SQLite databases for persistent file settings and global settings.
    Creates necessary tables if they do not already exist.
    """
    os.makedirs(os.path.dirname(APP_CONFIG['edits_db_path']), exist_ok=True)
    
    # 1. Edits DB
    with sqlite3.connect(APP_CONFIG['edits_db_path']) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_settings (
                filename TEXT PRIMARY KEY,
                settings_json TEXT
            )
        """)
    
    # 2. Global Settings DB
    with sqlite3.connect(APP_CONFIG['settings_db_path']) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS global_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT
            )
        """)

def db_save_file_settings(filename: str, settings: Dict[str, Any]) -> None:
    """
    Saves all settings for a specific file to the database.
    
    Args:
        filename (str): The name of the file (key).
        settings (Dict[str, Any]): Dictionary containing all image parameters.
    """
    with sqlite3.connect(APP_CONFIG['edits_db_path']) as conn:
        settings_json = json.dumps(settings)
        conn.execute(
            "INSERT OR REPLACE INTO file_settings (filename, settings_json) VALUES (?, ?)",
            (filename, settings_json)
        )

def db_load_file_settings(filename: str) -> Optional[Dict[str, Any]]:
    """
    Loads settings for a specific file from the database.
    
    Args:
        filename (str): The name of the file to look up.
        
    Returns:
        Optional[Dict[str, Any]]: The settings dictionary if found, else None.
    """
    with sqlite3.connect(APP_CONFIG['edits_db_path']) as conn:
        cursor = conn.execute("SELECT settings_json FROM file_settings WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
    return None

def db_save_global_setting(key: str, value: Any) -> None:
    """
    Saves a global configuration setting.
    """
    with sqlite3.connect(APP_CONFIG['settings_db_path']) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO global_settings (key, value_json) VALUES (?, ?)",
            (key, json.dumps(value))
        )

def db_get_global_setting(key: str, default: Any = None) -> Any:
    """
    Retrieves a global configuration setting.
    """
    with sqlite3.connect(APP_CONFIG['settings_db_path']) as conn:
        cursor = conn.execute("SELECT value_json FROM global_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
    return default
