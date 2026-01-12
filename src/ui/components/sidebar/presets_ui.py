import streamlit as st
from src.services.assets.presets import Presets
from src.ui.state.state_manager import save_settings, load_settings
from src.domain.session import WorkspaceSession
from src.ui.components.sidebar.helpers import (
    render_control_selectbox,
    render_control_text_input,
)


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

    p_settings = Presets.load_preset(selected_p)
    if p_settings:
        current_settings = session.file_settings[f_hash]
        current_dict = current_settings.to_dict()
        current_dict.update(p_settings)

        from src.domain.models import WorkspaceConfig

        session.file_settings[f_hash] = WorkspaceConfig.from_flat_dict(current_dict)
        load_settings(force=True)
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
    Presets.save_preset(preset_name, session.file_settings[f_hash])
    st.toast(f"Saved preset: {preset_name}")


def render_presets() -> None:
    session: WorkspaceSession = st.session_state.session
    if not session.current_file:
        return

    with st.expander(":material/pages: Presets"):
        presets = Presets.list_presets()
        c1, c2 = st.columns([2, 1])

        with c1:
            render_control_selectbox(
                "Select Preset",
                presets,
                default_val=presets[0] if presets else None,
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

        with c1:
            render_control_text_input(
                "Preset Name",
                default_val="",
                label_visibility="collapsed",
                placeholder="New Preset Name",
                key="new_preset_name",
            )

        c2.button(
            "Save",
            width="stretch",
            on_click=save_preset_callback,
        )
