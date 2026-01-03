import streamlit as st
from typing import List, Dict, Optional, Any
from src.domain_objects import ImageSettings
from src.backend.db import repository
from src.backend.engine import DarkroomEngine
from src.backend.assets import AssetManager
from src.config import DEFAULT_SETTINGS


class DarkroomSession:
    """
    Manages the application state and orchestrates interaction between
    the UI, the Engine, and the Database.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.repository = repository
        self.engine = DarkroomEngine()
        self.asset_manager = AssetManager

        # State
        self.uploaded_files: List[Dict[str, str]] = []
        self.file_settings: Dict[str, ImageSettings] = {}
        self.thumbnails: Dict[str, Any] = {}
        self.selected_file_idx: int = 0
        self.clipboard: Optional[Dict[str, Any]] = None
        self.icc_profile_path: Optional[str] = None
        self.show_curve: bool = False

    def sync_files(self, current_uploaded_names: set, raw_files: list) -> None:
        """
        Synchronizes the session's file list with the uploader widget.
        """
        last_names = {f["name"] for f in self.uploaded_files}
        new_names = current_uploaded_names - last_names

        if new_names:
            for f in raw_files:
                if f.name in new_names:
                    cached_path, f_hash = self.asset_manager.persist(f, self.session_id)
                    if cached_path and f_hash:
                        self.uploaded_files.append(
                            {"name": f.name, "path": cached_path, "hash": f_hash}
                        )

        removed_from_widget = last_names - current_uploaded_names
        if removed_from_widget:
            for f_meta in self.uploaded_files:
                if f_meta["name"] in removed_from_widget:
                    self.asset_manager.remove(f_meta["path"])

            self.uploaded_files = [
                f for f in self.uploaded_files if f["name"] not in removed_from_widget
            ]
            if self.selected_file_idx >= len(self.uploaded_files):
                self.selected_file_idx = max(0, len(self.uploaded_files) - 1)

    def load_active_settings(self) -> None:
        """
        Loads settings for the currently selected file into the session state.
        """
        if not self.uploaded_files:
            return

        f_hash = self.uploaded_files[self.selected_file_idx]["hash"]
        if f_hash not in self.file_settings:
            settings = self.repository.load_file_settings(f_hash)
            if settings is None:
                # Create default from global state if available
                settings = ImageSettings.from_dict(DEFAULT_SETTINGS.to_dict())
            self.file_settings[f_hash] = settings

        # Apply to session state for widgets
        for key, value in self.file_settings[f_hash].to_dict().items():
            st.session_state[key] = value

    def save_active_settings(self) -> None:
        """
        Saves current widget values to the active file's settings object and DB.
        """
        if not self.uploaded_files:
            return

        f_hash = self.uploaded_files[self.selected_file_idx]["hash"]
        settings = self.file_settings[f_hash]

        for field in ImageSettings.__dataclass_fields__.keys():
            if field in st.session_state:
                setattr(settings, field, st.session_state[field])

        self.repository.save_file_settings(f_hash, settings)

    @property
    def current_file(self) -> Optional[Dict[str, str]]:
        if not self.uploaded_files:
            return None
        return self.uploaded_files[self.selected_file_idx]
