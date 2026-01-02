import streamlit as st
from typing import cast
from src.config import DEFAULT_SETTINGS
from src.domain_objects import ProcessingParams
from src.backend.db import (
    db_save_file_settings,
    db_load_file_settings,
    db_save_global_setting,
    db_get_global_setting,
)

# Keys that should persist globally across all files if no specific edits exist
GLOBAL_PERSIST_KEYS = {
    "process_mode",
    "wb_cyan",
    "wb_magenta",
    "wb_yellow",
    "temperature",
    "shadow_temp",
    "highlight_temp",
    "toe",
    "shoulder",
    "export_fmt",
    "export_color_space",
    "export_size",
    "export_dpi",
    "export_add_border",
    "export_border_size",
    "export_border_color",
    "export_path",
    "sharpen",
}


def init_session_state() -> None:
    """
    Initializes all necessary session state variables for the Streamlit app.
    Ensures that dictionaries and lists are present to avoid KeyErrors.
    """
    if "file_settings" not in st.session_state:
        st.session_state.file_settings = {}

    # Load Global Settings from DB into Session State
    for key in GLOBAL_PERSIST_KEYS:
        if key not in st.session_state:
            # Fallback to DEFAULT_SETTINGS if not in DB
            default_val = DEFAULT_SETTINGS.get(key)
            st.session_state[key] = db_get_global_setting(key, default_val)

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "last_uploaded_names" not in st.session_state:
        st.session_state.last_uploaded_names = set()
    if "thumbnails" not in st.session_state:
        st.session_state.thumbnails = {}
    if "selected_file_idx" not in st.session_state:
        st.session_state.selected_file_idx = 0
    if "clipboard" not in st.session_state:
        st.session_state.clipboard = None
    if "last_dust_click" not in st.session_state:
        st.session_state.last_dust_click = None
    if "dust_start_point" not in st.session_state:
        st.session_state.dust_start_point = None
    if "icc_profile_path" not in st.session_state:
        st.session_state.icc_profile_path = None


def load_settings(file_hash: str) -> bool:
    """
    Loads settings for the given file into session state widgets.
    Attempts to pull from local session cache first, then the SQLite database.

    Returns:
        bool: True if this is a first-time load (no previous settings), False otherwise.
    """
    is_new = False
    if file_hash in st.session_state.file_settings:
        settings = cast(ProcessingParams, st.session_state.file_settings[file_hash])
    else:
        db_settings = db_load_file_settings(file_hash)
        if db_settings:
            settings = DEFAULT_SETTINGS.copy()
            settings.update(db_settings)
        else:
            settings = DEFAULT_SETTINGS.copy()
            # Apply current global settings for new files
            for key in GLOBAL_PERSIST_KEYS:
                if key in st.session_state:
                    settings[key] = st.session_state[key]  # type: ignore[literal-required]

            settings["manual_dust_spots"] = []
            settings["local_adjustments"] = []
            is_new = True
        st.session_state.file_settings[file_hash] = settings

    # Ensure independent list objects for dust spots and local adjustments
    if "manual_dust_spots" in settings:
        settings["manual_dust_spots"] = list(settings["manual_dust_spots"])
    else:
        settings["manual_dust_spots"] = []

    if "local_adjustments" in settings:
        settings["local_adjustments"] = list(settings["local_adjustments"])
    else:
        settings["local_adjustments"] = []

    for key, value in settings.items():
        st.session_state[key] = value

    return is_new


def save_settings(file_hash: str) -> None:
    """
    Saves current widget values to the file's settings dict and persists to the SQLite DB.
    Also updates global settings DB.

    Args:
        file_hash (str): The content hash of the file to save settings for.
    """
    if file_hash not in st.session_state.file_settings:
        init_settings = DEFAULT_SETTINGS.copy()
        init_settings["manual_dust_spots"] = []
        init_settings["local_adjustments"] = []
        st.session_state.file_settings[file_hash] = init_settings

    for key in DEFAULT_SETTINGS.keys():
        if key in st.session_state:
            val = st.session_state[key]
            if key == "manual_dust_spots" or key == "local_adjustments":
                st.session_state.file_settings[file_hash][key] = list(val)
            else:
                st.session_state.file_settings[file_hash][key] = val

            # Persist global settings
            if key in GLOBAL_PERSIST_KEYS:
                db_save_global_setting(key, val)

    # Handle export keys that might not be in DEFAULT_SETTINGS
    for key in GLOBAL_PERSIST_KEYS:
        if key not in DEFAULT_SETTINGS and key in st.session_state:
            db_save_global_setting(key, st.session_state[key])

    db_save_file_settings(
        file_hash, cast(ProcessingParams, st.session_state.file_settings[file_hash])
    )


def copy_settings() -> None:
    """
    Copies current file settings to the session clipboard.
    Excludes image-specific manual dust spots and local adjustments.
    """
    if not st.session_state.uploaded_files:
        return
    current_file = st.session_state.uploaded_files[st.session_state.selected_file_idx]
    f_hash = current_file["hash"]
    save_settings(f_hash)
    settings = st.session_state.file_settings[f_hash].copy()
    # Remove image-specific retouching and rotation from clipboard to avoid unwanted resets
    if "manual_dust_spots" in settings:
        del settings["manual_dust_spots"]
    if "local_adjustments" in settings:
        del settings["local_adjustments"]
    if "rotation" in settings:
        del settings["rotation"]

    st.session_state.clipboard = settings
    st.toast("Settings copied to clipboard!")


def paste_settings() -> None:
    """
    Pastes settings from the session clipboard to the currently selected file.
    """
    if st.session_state.clipboard and st.session_state.uploaded_files:
        current_file = st.session_state.uploaded_files[
            st.session_state.selected_file_idx
        ]
        f_hash = current_file["hash"]
        # Update instead of replace to preserve local settings (rotation, retouching)
        st.session_state.file_settings[f_hash].update(st.session_state.clipboard)
        load_settings(f_hash)
        # Explicitly save to DB after paste
        db_save_file_settings(f_hash, st.session_state.file_settings[f_hash])
        st.toast("Settings pasted!")


def reset_file_settings(file_name: str) -> None:
    """
    Resets settings for the given file to default values and updates the database.

    Args:
        file_name (str): The name of the file to reset.
    """
    st.session_state.file_settings[file_name] = DEFAULT_SETTINGS.copy()
    st.session_state.file_settings[file_name]["manual_dust_spots"] = []
    st.session_state.file_settings[file_name]["local_adjustments"] = []
    db_save_file_settings(file_name, st.session_state.file_settings[file_name])
    load_settings(file_name)
    st.toast(f"Reset settings for {file_name}")
