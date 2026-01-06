import streamlit as st
import uuid
from src.core.session.manager import WorkspaceSession
from src.infrastructure.persistence.sqlite_repository import SQLiteRepository
from src.infrastructure.persistence.local_asset_store import LocalAssetStore
from src.orchestration.engine import DarkroomEngine
from src.config import APP_CONFIG, DEFAULT_SETTINGS
from src.domain_objects import ImageSettings

# Keys that should persist globally across all files if no specific edits exist
GLOBAL_PERSIST_KEYS = {
    "process_mode",
    "paper_profile",
    "selenium_strength",
    "sepia_strength",
    "export_fmt",
    "export_color_space",
    "export_size",
    "export_dpi",
    "export_add_border",
    "export_border_size",
    "export_border_color",
    "export_path",
    "sharpen",
    "hypertone_strength",
    "color_separation",
    "c_noise_strength",
    "working_copy_size",
    "hot_folder_mode",
    "last_picker_dir",
}


def init_session_state() -> None:
    """
    Initializes the WorkspaceSession and core infrastructure.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "session" not in st.session_state:
        # Instantiate Infrastructure
        repo = SQLiteRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)
        repo.initialize()

        store = LocalAssetStore(APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir)
        store.initialize()

        engine = DarkroomEngine()

        # Create Domain Session
        session = WorkspaceSession(st.session_state.session_id, repo, store, engine)
        st.session_state.session = session

        # 1. First, populate state with hardcoded defaults
        defaults = DEFAULT_SETTINGS.to_dict()
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

        if "working_copy_size" not in st.session_state:
            st.session_state.working_copy_size = APP_CONFIG.preview_render_size

        # 2. Then, override with Global Settings from DB if they exist
        for key in GLOBAL_PERSIST_KEYS:
            val = repo.get_global_setting(key)
            if val is not None:
                st.session_state[key] = val

    if "last_dust_click" not in st.session_state:
        st.session_state.last_dust_click = None

    if "dust_start_point" not in st.session_state:
        st.session_state.dust_start_point = None


def load_settings() -> None:
    """
    Loads settings for the current file.
    If file has no edits, it populates st.session_state with current global values.
    """
    session: WorkspaceSession = st.session_state.session
    settings = session.get_active_settings()

    if settings:
        settings_dict = settings.to_dict()

        # If this is a NEW file (no edits in DB), we want to keep current global values
        # instead of overwriting with DEFAULT_SETTINGS values.
        f_hash = session.uploaded_files[session.selected_file_idx]["hash"]
        has_edits = session.repository.load_file_settings(f_hash) is not None

        for key, value in settings_dict.items():
            if not has_edits and key in GLOBAL_PERSIST_KEYS:
                # Keep what is already in st.session_state (the global/last used value)
                continue
            st.session_state[key] = value


def save_settings() -> None:
    """
    Saves file settings AND updates global persistent settings.
    """
    session: WorkspaceSession = st.session_state.session

    # Save Global Persistent Settings (even if no file is selected)
    for key in GLOBAL_PERSIST_KEYS:
        if key in st.session_state:
            session.repository.save_global_setting(key, st.session_state[key])

    if not session.uploaded_files:
        return

    # Extract current UI state into ImageSettings
    from src.presentation.app import get_processing_params

    settings = get_processing_params(st.session_state)
    session.update_active_settings(settings)


def copy_settings() -> None:
    save_settings()
    session: WorkspaceSession = st.session_state.session
    current_file = session.current_file
    if current_file:
        f_hash = current_file["hash"]
        settings = session.file_settings[f_hash]
        settings_dict = settings.to_dict()

        # Strip image-specifics
        for key in ["manual_dust_spots", "local_adjustments", "rotation"]:
            if key in settings_dict:
                del settings_dict[key]

        session.clipboard = settings_dict
        st.toast("Settings copied to clipboard!")


def paste_settings() -> None:
    session: WorkspaceSession = st.session_state.session
    if session.clipboard and session.current_file:
        f_hash = session.current_file["hash"]
        current_settings = session.file_settings[f_hash]
        current_dict = current_settings.to_dict()
        current_dict.update(session.clipboard)

        session.file_settings[f_hash] = ImageSettings.from_dict(current_dict)
        load_settings()
        save_settings()
        st.toast("Settings pasted!")


def reset_file_settings() -> None:
    session: WorkspaceSession = st.session_state.session
    if not session.current_file:
        return

    f_hash = session.current_file["hash"]
    new_settings = ImageSettings.from_dict(DEFAULT_SETTINGS.to_dict())
    new_settings.manual_dust_spots = []
    new_settings.local_adjustments = []

    session.file_settings[f_hash] = new_settings
    session.repository.save_file_settings(f_hash, new_settings)
    load_settings()
    st.toast("Reset settings for this file")
