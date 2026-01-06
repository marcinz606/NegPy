import streamlit as st
from src.features.presets.service import PresetService
from src.presentation.state.state_manager import save_settings, load_settings
from src.core.session.manager import WorkspaceSession


def load_preset_callback() -> None:
    """
    Callback to load the selected preset into the current file settings.
    """
    session: WorkspaceSession = st.session_state.session
    if not session.current_file:
        return

    f_hash = session.current_file["hash"]
    selected_p = st.session_state.get("selected_preset_name")

    if not selected_p:
        return

    p_settings = PresetService.load_preset(selected_p)
    if p_settings:
        current_settings = session.file_settings[f_hash]
        current_dict = current_settings.to_dict()
        current_dict.update(p_settings)

        from src.domain_objects import ImageSettings

        session.file_settings[f_hash] = ImageSettings.from_dict(current_dict)
        load_settings()
        st.toast(f"Loaded preset: {selected_p}")


def save_preset_callback() -> None:
    """
    Callback to save the current settings as a new preset.
    """
    session: WorkspaceSession = st.session_state.session
    if not session.current_file:
        return

    f_hash = session.current_file["hash"]
    preset_name = st.session_state.get("new_preset_name")

    if not preset_name:
        return

    save_settings()
    PresetService.save_preset(preset_name, session.file_settings[f_hash])
    st.toast(f"Saved preset: {preset_name}")


def render_presets() -> None:
    """
    Renders the Presets expander.
    """
    session: WorkspaceSession = st.session_state.session
    if not session.current_file:
        return

    with st.expander(":material/pages: Presets"):
        presets = PresetService.list_presets()
        c1, c2 = st.columns([2, 1])

        c1.selectbox(
            "Select Preset",
            presets,
            label_visibility="collapsed",
            key="selected_preset_name",
        )

        c2.button(
            "Load",
            width="stretch",
            disabled=not presets,
            on_click=load_preset_callback,
        )

        st.divider()
        c1, c2 = st.columns([2, 1])

        c1.text_input(
            "Preset Name",
            label_visibility="collapsed",
            placeholder="New Preset Name",
            key="new_preset_name",
        )

        c2.button(
            "Save",
            width="stretch",
            on_click=save_preset_callback,
        )
