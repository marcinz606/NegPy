import streamlit as st
import uuid
from typing import Dict, Any
from src.domain.session import WorkspaceSession
from src.domain.models import WorkspaceConfig
from src.infrastructure.storage.repository import StorageRepository
from src.infrastructure.storage.local_asset_store import LocalAssetStore
from src.services.rendering.engine import DarkroomEngine
from src.kernel.system.config import APP_CONFIG

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
    "filename_pattern",
    "apply_icc",
    "sharpen",
    "clahe_strength",
    "color_separation",
    "working_copy_size",
    "working_copy_size_vertical",
    "working_copy_size_horizontal",
    "hot_folder_mode",
    "last_picker_dir",
    "autocrop",
    "autocrop_ratio",
    "keep_full_frame",
}


def init_session_state() -> None:
    """
    Setup WorkspaceSession and persistent UI state.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "session" not in st.session_state:
        repo = StorageRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)
        repo.initialize()

        store = LocalAssetStore(APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir)
        store.initialize()

        engine = DarkroomEngine()

        session = WorkspaceSession(st.session_state.session_id, repo, store, engine)
        st.session_state.session = session

        # Restore Global Settings (Fresh Session)
        for key in GLOBAL_PERSIST_KEYS:
            val = repo.get_global_setting(key)
            if val is not None:
                st.session_state[key] = val

        # Restore watched folders
        watched = repo.get_global_setting("watched_folders")
        if watched and isinstance(watched, list):
            session.watched_folders = set(watched)

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


def _to_ui_state_dict(config: WorkspaceConfig) -> Dict[str, Any]:
    """
    Flatten WorkspaceConfig for Streamlit session_state.
    """
    data = config.to_dict()
    data["local_adjustments"] = config.retouch.local_adjustments
    return data


def load_settings(force: bool = False) -> None:
    """
    Hydrates st.session_state from DB or memory.
    """
    session: WorkspaceSession = st.session_state.session
    settings = session.get_active_settings()

    if settings:
        settings_dict = _to_ui_state_dict(settings)

        # Check if this file has existing edits in DB
        f_hash = session.uploaded_files[session.selected_file_idx]["hash"]
        has_edits = session.repository.load_file_settings(f_hash) is not None

        for key, value in settings_dict.items():
            # Respect global UI state if file has no previous edits
            if not force and not has_edits and key in GLOBAL_PERSIST_KEYS:
                if st.session_state.get(key) is not None:
                    continue

            st.session_state[key] = value


def save_settings(persist: bool = False) -> None:
    """
    Syncs UI state back to Session/DB.
    """
    session: WorkspaceSession = st.session_state.session

    if persist:
        for key in GLOBAL_PERSIST_KEYS:
            if key in st.session_state:
                session.repository.save_global_setting(key, st.session_state[key])

        # Save watched folders
        session.repository.save_global_setting(
            "watched_folders", list(session.watched_folders)
        )

    if not session.uploaded_files:
        return

    # Extract current UI state into WorkspaceConfig
    from src.ui.app import get_processing_params_composed

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

    new_settings = session.create_default_config()

    session.file_settings[f_hash] = new_settings
    session.repository.save_file_settings(f_hash, new_settings)
    load_settings()
    st.toast("Reset settings for this file")
