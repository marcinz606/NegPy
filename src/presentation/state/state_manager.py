import streamlit as st
import uuid
from src.core.session.manager import WorkspaceSession
from src.core.session.models import WorkspaceConfig
from src.infrastructure.persistence.sqlite_repository import SQLiteRepository
from src.infrastructure.persistence.local_asset_store import LocalAssetStore
from src.orchestration.engine import DarkroomEngine
from src.config import APP_CONFIG

# Keys that should persist globally across all files if no specific edits exist
GLOBAL_PERSIST_KEYS = {
    "process_mode",
    "paper_profile",
    "selenium_strength",
    "sepia_strength",
    "export_fmt",
    "export_color_space",
    "export_print_size",
    "export_dpi",
    "export_add_border",
    "export_border_size",
    "export_border_color",
    "export_path",
    "apply_icc",
    "sharpen",
    "hypertone_strength",
    "color_separation",
    "c_noise_strength",
    "working_copy_size",
    "working_copy_size_vertical",
    "working_copy_size_horizontal",
    "hot_folder_mode",
    "last_picker_dir",
    "autocrop",
    "autocrop_ratio",
}


def init_session_state() -> None:
    """
    Initializes the WorkspaceSession and core infrastructure.
    Forcefully seeds defaults on first run to ensure environment variables are respected.
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

        # Restore Global Settings from DB if they exist (Fresh Session Only)
        for key in GLOBAL_PERSIST_KEYS:
            val = repo.get_global_setting(key)
            if val is not None:
                st.session_state[key] = val

    # 1. Always ensure session state is populated with defaults for any missing keys
    # This guards against stale state or new keys being added during development
    session = st.session_state.session
    defaults = session.create_default_config().to_dict()

    for key, val in defaults.items():
        if st.session_state.get(key) is None:
            st.session_state[key] = val

    if "working_copy_size" not in st.session_state:
        st.session_state.working_copy_size = APP_CONFIG.preview_render_size

    if "working_copy_size_vertical" not in st.session_state:
        st.session_state.working_copy_size_vertical = APP_CONFIG.preview_render_size

    if "working_copy_size_horizontal" not in st.session_state:
        st.session_state.working_copy_size_horizontal = APP_CONFIG.preview_render_size

    if "last_dust_click" not in st.session_state:
        st.session_state.last_dust_click = None

    if "dust_start_point" not in st.session_state:
        st.session_state.dust_start_point = None


def load_settings() -> None:
    """
    Loads settings for the current file.
    """
    session: WorkspaceSession = st.session_state.session
    settings = session.get_active_settings()

    if settings:
        settings_dict = settings.to_dict()

        # Check if this file has existing edits in DB
        f_hash = session.uploaded_files[session.selected_file_idx]["hash"]
        has_edits = session.repository.load_file_settings(f_hash) is not None

        for key, value in settings_dict.items():
            # If the file has NO EDITS, we want to respect current global UI state
            # for keys in GLOBAL_PERSIST_KEYS.
            if not has_edits and key in GLOBAL_PERSIST_KEYS:
                # If key already exists and is not None, don't overwrite it with default
                if st.session_state.get(key) is not None:
                    continue

            # Apply value from settings object
            st.session_state[key] = value


def save_settings(persist: bool = False) -> None:
    """
    Saves file settings. Defaults to memory-only (persist=False) for performance.
    Set persist=True for commit points (file switch, export).
    """
    session: WorkspaceSession = st.session_state.session

    # Save Global Persistent Settings (Only if persisting)
    if persist:
        for key in GLOBAL_PERSIST_KEYS:
            if key in st.session_state:
                session.repository.save_global_setting(key, st.session_state[key])

    if not session.uploaded_files:
        return

    # Extract current UI state into WorkspaceConfig
    from src.presentation.app import get_processing_params_composed

    settings = get_processing_params_composed(st.session_state)
    session.update_active_settings(settings, persist=persist)


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

        session.file_settings[f_hash] = WorkspaceConfig.from_flat_dict(current_dict)
        load_settings()
        save_settings(persist=True)
        st.toast("Settings pasted!")


def reset_file_settings() -> None:
    session: WorkspaceSession = st.session_state.session
    if not session.current_file:
        return

    f_hash = session.current_file["hash"]

    # Use centralized factory to ensure correct defaults (including export paths)
    new_settings = session.create_default_config()

    session.file_settings[f_hash] = new_settings
    session.repository.save_file_settings(f_hash, new_settings)
    load_settings()
    st.toast("Reset settings for this file")
