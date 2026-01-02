import streamlit as st
from src.backend.utils import list_presets, load_preset, save_preset
from src.frontend.state import save_settings, load_settings


def render_presets(current_file_hash: str) -> None:
    """
    Renders the Presets expander.
    """
    with st.expander(":material/pages: Presets"):
        presets = list_presets()
        c1, c2 = st.columns([2, 1])
        selected_p = c1.selectbox(
            "Select Preset", presets, label_visibility="collapsed"
        )
        if c2.button("Load", width="stretch", disabled=not presets):
            p_settings = load_preset(selected_p)
            if p_settings:
                st.session_state.file_settings[current_file_hash].update(p_settings)
                load_settings(current_file_hash)
                st.toast(f"Loaded preset: {selected_p}")
                st.rerun()

        st.divider()
        c1, c2 = st.columns([2, 1])
        preset_name = c1.text_input(
            "Preset Name", label_visibility="collapsed", placeholder="New Preset Name"
        )
        if c2.button("Save", width="stretch", disabled=not preset_name):
            save_settings(current_file_hash)
            save_preset(preset_name, st.session_state.file_settings[current_file_hash])
            st.toast(f"Saved preset: {preset_name}")
            st.rerun()
