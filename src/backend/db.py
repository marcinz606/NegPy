import sqlite3
import json
import os
from typing import Dict, Any, Optional
from src.backend.config import APP_CONFIG

def init_db() -> None:
    """
    Initializes the SQLite database for persistent file settings.
    Creates the 'file_settings' table if it does not already exist.
    """
    with sqlite3.connect(APP_CONFIG['db_path']) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_settings (
                filename TEXT PRIMARY KEY,
                settings_json TEXT
            )
        """)

def db_save_file_settings(filename: str, settings: Dict[str, Any]) -> None:
    """
    Saves all settings for a specific file to the database.
    
    Args:
        filename (str): The name of the file (key).
        settings (Dict[str, Any]): Dictionary containing all image parameters.
    """
    with sqlite3.connect(APP_CONFIG['db_path']) as conn:
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
    with sqlite3.connect(APP_CONFIG['db_path']) as conn:
        cursor = conn.execute("SELECT settings_json FROM file_settings WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
    return None
