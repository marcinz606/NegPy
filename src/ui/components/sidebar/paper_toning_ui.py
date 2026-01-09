import streamlit as st
from src.ui.state.view_models import ToningViewModel
from src.ui.components.sidebar.helpers import (
    render_control_slider,
    render_control_selectbox,
)
from src.config import DEFAULT_WORKSPACE_CONFIG


def render_paper_section() -> None:
    """
    Renders the 'Color & Toning' section of the sidebar.
    """
    vm = ToningViewModel()

    with st.expander(":material/colorize: Paper & Toning", expanded=False):
        # 1. Paper Substrate Selection
        render_control_selectbox(
            "Paper Profile",
            ["None", "Neutral RC", "Cool Glossy", "Warm Fiber", "Antique Ivory"],
            default_val=DEFAULT_WORKSPACE_CONFIG.toning.paper_profile,
            key=vm.get_key("paper_profile"),
            help_text="Simulates the specific spectral reflectance and D-max of analog paper bases.",
        )

        # 2. Chemical Toning (B&W Simulation)
        if st.session_state.get("process_mode") == "B&W":
            st.subheader("Chemical Toning")
            render_control_slider(
                label="Selenium",
                min_val=0.0,
                max_val=2.0,
                default_val=0.0,
                step=0.01,
                key=vm.get_key("selenium_strength"),
                help_text="Deepens D-max and shifts shadows towards purple/red-black by converting silver to silver selenide.",
            )
            render_control_slider(
                label="Sepia",
                min_val=0.0,
                max_val=2.0,
                default_val=0.0,
                step=0.01,
                key=vm.get_key("sepia_strength"),
                help_text="Adds a warm, orange-brown glow to mid-tones and highlights by converting silver to silver sulfide.",
            )
