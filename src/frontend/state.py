import streamlit as st
from typing import Dict, Any, Optional
from src.backend.config import DEFAULT_SETTINGS
from src.backend.db import db_save_file_settings, db_load_file_settings

def init_session_state() -> None:
    """
    Initializes all necessary session state variables for the Streamlit app.
    Ensures that dictionaries and lists are present to avoid KeyErrors.
    """
    if 'file_settings' not in st.session_state:
        st.session_state.file_settings = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'last_uploaded_names' not in st.session_state:
        st.session_state.last_uploaded_names = set()
    if 'thumbnails' not in st.session_state:
        st.session_state.thumbnails = {}
    if 'selected_file_idx' not in st.session_state:
        st.session_state.selected_file_idx = 0
    if 'clipboard' not in st.session_state:
        st.session_state.clipboard = None
    if 'last_wb_click' not in st.session_state:
        st.session_state.last_wb_click = None
    if 'last_dust_click' not in st.session_state:
        st.session_state.last_dust_click = None
    if 'dust_start_point' not in st.session_state:
        st.session_state.dust_start_point = None
    if 'pick_wb' not in st.session_state:
        st.session_state.pick_wb = False

def load_settings(file_name: str) -> None:
    """
    Loads settings for the given file into session state widgets.
    Attempts to pull from local session cache first, then the SQLite database.
    
    Args:
        file_name (str): The name of the file to load settings for.
    """
    if file_name in st.session_state.file_settings:
        settings = st.session_state.file_settings[file_name]
    else:
        db_settings = db_load_file_settings(file_name)
        if db_settings:
            settings = db_settings
        else:
            settings = DEFAULT_SETTINGS.copy()
            settings['manual_dust_spots'] = []
            settings['local_adjustments'] = []
        st.session_state.file_settings[file_name] = settings
    
    # Ensure independent list objects for dust spots and local adjustments
    if 'manual_dust_spots' in settings:
        settings['manual_dust_spots'] = list(settings['manual_dust_spots'])
    else:
        settings['manual_dust_spots'] = []
        
    if 'local_adjustments' in settings:
        settings['local_adjustments'] = list(settings['local_adjustments'])
    else:
        settings['local_adjustments'] = []

    for key, value in settings.items():
        if key == 'black_point' or key == 'white_point':
            continue
        st.session_state[key] = value
    
    st.session_state.bw_points = (settings.get('black_point', 0.0), settings.get('white_point', 1.0))

def save_settings(file_name: str) -> None:
    """
    Saves current widget values to the file's settings dict and persists to the SQLite DB.
    
    Args:
        file_name (str): The name of the file to save settings for.
    """
    if file_name not in st.session_state.file_settings:
        init_settings = DEFAULT_SETTINGS.copy()
        init_settings['manual_dust_spots'] = []
        init_settings['local_adjustments'] = []
        st.session_state.file_settings[file_name] = init_settings
    
    for key in DEFAULT_SETTINGS.keys():
        if key == 'black_point':
            if 'bw_points' in st.session_state:
                st.session_state.file_settings[file_name][key] = st.session_state['bw_points'][0]
            continue
        if key == 'white_point':
            if 'bw_points' in st.session_state:
                st.session_state.file_settings[file_name][key] = st.session_state['bw_points'][1]
            continue
            
        if key in st.session_state:
            val = st.session_state[key]
            if key == 'manual_dust_spots' or key == 'local_adjustments':
                st.session_state.file_settings[file_name][key] = list(val)
            else:
                st.session_state.file_settings[file_name][key] = val
    
    db_save_file_settings(file_name, st.session_state.file_settings[file_name])

def copy_settings() -> None:
    """
    Copies current file settings to the session clipboard.
    Excludes image-specific manual dust spots and local adjustments.
    """
    if not st.session_state.uploaded_files:
        return
    current_file = st.session_state.uploaded_files[st.session_state.selected_file_idx].name
    save_settings(current_file)
    settings = st.session_state.file_settings[current_file].copy()
    # Remove image-specific retouching and rotation from clipboard to avoid unwanted resets
    if 'manual_dust_spots' in settings:
        del settings['manual_dust_spots']
    if 'local_adjustments' in settings:
        del settings['local_adjustments']
    if 'rotation' in settings:
        del settings['rotation']
        
    st.session_state.clipboard = settings
    st.toast("Settings copied to clipboard!")

def paste_settings() -> None:
    """
    Pastes settings from the session clipboard to the currently selected file.
    """
    if st.session_state.clipboard and st.session_state.uploaded_files:
        current_file = st.session_state.uploaded_files[st.session_state.selected_file_idx].name
        # Update instead of replace to preserve local settings (rotation, retouching)
        st.session_state.file_settings[current_file].update(st.session_state.clipboard)
        load_settings(current_file)
        # Explicitly save to DB after paste
        db_save_file_settings(current_file, st.session_state.file_settings[current_file])
        st.toast("Settings pasted!")

def reset_file_settings(file_name: str) -> None:
    """
    Resets settings for the given file to default values and updates the database.
    
    Args:
        file_name (str): The name of the file to reset.
    """
    st.session_state.file_settings[file_name] = DEFAULT_SETTINGS.copy()
    st.session_state.file_settings[file_name]['manual_dust_spots'] = []
    st.session_state.file_settings[file_name]['local_adjustments'] = []
    db_save_file_settings(file_name, st.session_state.file_settings[file_name])
    load_settings(file_name)
    st.toast(f"Reset settings for {file_name}")
