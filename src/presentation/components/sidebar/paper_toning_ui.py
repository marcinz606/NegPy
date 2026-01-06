import streamlit as st
from src.presentation.state.view_models import ToningViewModel
from src.presentation.components.sidebar.helpers import render_control_slider


def render_paper_section() -> None:
    """
    Renders the 'Color & Toning' section of the sidebar.
    """
    vm = ToningViewModel()

    with st.expander(":material/colorize: Paper & Toning", expanded=True):
        # 1. Paper Substrate Selection
        st.selectbox(
            "Paper Profile",
            ["None", "Neutral RC", "Cool Glossy", "Warm Fiber", "Antique Ivory"],
            key=vm.get_key("paper_profile"),
            help="Simulates the specific spectral reflectance and D-max of analog paper bases.",
        )

        # 2. Chemical Toning (B&W Simulation)
        if st.session_state.get(vm.get_key("process_mode")) == "B&W":
            st.subheader("Chemical Toning")
            render_control_slider(
                "Selenium",
                0.0,
                4.0,
                0.0,
                0.01,
                vm.get_key("selenium_strength"),
                help_text="Deepens D-max and shifts shadows towards purple/red-black by converting silver to silver selenide.",
            )
            render_control_slider(
                "Sepia",
                0.0,
                4.0,
                0.0,
                0.01,
                vm.get_key("sepia_strength"),
                help_text="Adds a warm, orange-brown glow to mid-tones and highlights by converting silver to silver sulfide.",
            )
