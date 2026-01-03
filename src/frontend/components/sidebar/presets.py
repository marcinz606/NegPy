import streamlit as st
from src.backend.utils import list_presets, load_preset, save_preset
from src.frontend.state import save_settings, load_settings
from src.backend.session import DarkroomSession


def render_presets() -> None:
    """
    Renders the Presets expander.
    """
    session: DarkroomSession = st.session_state.session
    if not session.current_file:
        return

    f_hash = session.current_file["hash"]

    with st.expander(":material/pages: Presets"):
        presets = list_presets()
        c1, c2 = st.columns([2, 1])
        selected_p = c1.selectbox(
            "Select Preset", presets, label_visibility="collapsed"
        )
        if c2.button("Load", width="stretch", disabled=not presets):
            p_settings = load_preset(selected_p)
            if p_settings:
                current_settings = session.file_settings[f_hash]
                current_dict = current_settings.to_dict()
                current_dict.update(p_settings)

                from src.domain_objects import ImageSettings

                session.file_settings[f_hash] = ImageSettings.from_dict(current_dict)
                load_settings()
                st.toast(f"Loaded preset: {selected_p}")
                st.rerun()

        st.divider()
        c1, c2 = st.columns([2, 1])
        preset_name = c1.text_input(
            "Preset Name", label_visibility="collapsed", placeholder="New Preset Name"
        )
        if c2.button("Save", width="stretch", disabled=not preset_name):
            save_settings()
            save_preset(preset_name, session.file_settings[f_hash])
            st.toast(f"Saved preset: {preset_name}")
            st.rerun()
