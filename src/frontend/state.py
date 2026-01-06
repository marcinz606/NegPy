import streamlit as st
import uuid
from src.backend.session import DarkroomSession

# Keys that should persist globally across all files if no specific edits exist
GLOBAL_PERSIST_KEYS = {
    "process_mode",
    "wb_cyan",
    "wb_magenta",
    "wb_yellow",
    "paper_profile",
    "selenium_strength",
    "sepia_strength",
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
    "hypertone_strength",
    "color_separation",
}


def init_session_state() -> None:
    """
    Initializes the DarkroomSession and core Streamlit state.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "session" not in st.session_state:
        st.session_state.session = DarkroomSession(st.session_state.session_id)

    if "last_dust_click" not in st.session_state:
        st.session_state.last_dust_click = None

    if "dust_start_point" not in st.session_state:
        st.session_state.dust_start_point = None

    if "working_copy_size" not in st.session_state:
        st.session_state.working_copy_size = 1800


def load_settings() -> None:
    """
    Delegates to the session to load settings for the current file.
    """
    st.session_state.session.load_active_settings()


def save_settings() -> None:
    """
    Delegates to the session to save settings.
    """
    st.session_state.session.save_active_settings()


def copy_settings() -> None:
    """
    Copies current settings to the session clipboard.
    """
    session: DarkroomSession = st.session_state.session
    session.save_active_settings()

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
    """
    Pastes clipboard settings to the active file.
    """
    session: DarkroomSession = st.session_state.session
    if session.clipboard and session.current_file:
        f_hash = session.current_file["hash"]
        current_settings = session.file_settings[f_hash]
        current_dict = current_settings.to_dict()
        current_dict.update(session.clipboard)

        from src.domain_objects import ImageSettings

        session.file_settings[f_hash] = ImageSettings.from_dict(current_dict)
        session.load_active_settings()
        session.save_active_settings()
        st.toast("Settings pasted!")


def reset_file_settings() -> None:
    """
    Resets the current file to default settings.
    """
    session: DarkroomSession = st.session_state.session
    if not session.current_file:
        return

    f_hash = session.current_file["hash"]
    from src.config import DEFAULT_SETTINGS
    from src.domain_objects import ImageSettings

    new_settings = ImageSettings.from_dict(DEFAULT_SETTINGS.to_dict())
    new_settings.manual_dust_spots = []
    new_settings.local_adjustments = []

    session.file_settings[f_hash] = new_settings
    session.repository.save_file_settings(f_hash, new_settings)
    session.load_active_settings()
    st.toast("Reset settings for this file")
