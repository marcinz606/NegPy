import streamlit as st
import uuid
from src.core.session.manager import WorkspaceSession
from src.infrastructure.persistence.sqlite_repository import SQLiteRepository
from src.infrastructure.persistence.local_asset_store import LocalAssetStore
from src.orchestration.engine import DarkroomEngine
from src.config import APP_CONFIG, DEFAULT_SETTINGS
from src.domain_objects import ImageSettings


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
        st.session_state.session = WorkspaceSession(
            st.session_state.session_id, repo, store, engine
        )

    if "last_dust_click" not in st.session_state:
        st.session_state.last_dust_click = None

    if "dust_start_point" not in st.session_state:
        st.session_state.dust_start_point = None

    if "working_copy_size" not in st.session_state:
        st.session_state.working_copy_size = 1800


def load_settings() -> None:
    session: WorkspaceSession = st.session_state.session
    settings = session.get_active_settings()
    if settings:
        for key, value in settings.to_dict().items():
            st.session_state[key] = value


def save_settings() -> None:
    session: WorkspaceSession = st.session_state.session
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
