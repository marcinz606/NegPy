import streamlit as st
from src.ui.state.view_models import ToningViewModel
from src.ui.components.sidebar.helpers import (
    render_control_slider,
    render_control_selectbox,
)
from src.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def render_paper_section() -> None:
    vm = ToningViewModel()

    with st.expander(":material/colorize: Paper & Toning", expanded=False):
        render_control_selectbox(
            "Paper Profile",
            ["None", "Neutral RC", "Cool Glossy", "Warm Fiber", "Antique Ivory"],
            default_val=DEFAULT_WORKSPACE_CONFIG.toning.paper_profile,
            key=vm.get_key("paper_profile"),
            help_text="Tint & D-max of specific paper stocks.",
        )

        if st.session_state.get("process_mode") == "B&W":
            st.subheader("Chemical Toning")
            render_control_slider(
                label="Selenium",
                min_val=0.0,
                max_val=2.0,
                default_val=0.0,
                step=0.01,
                key=vm.get_key("selenium_strength"),
                help_text="Shadow tone shift (Silver Selenide simulation).",
            )
            render_control_slider(
                label="Sepia",
                min_val=0.0,
                max_val=2.0,
                default_val=0.0,
                step=0.01,
                key=vm.get_key("sepia_strength"),
                help_text="Mid-tone warm shift (Silver Sulfide simulation).",
            )
