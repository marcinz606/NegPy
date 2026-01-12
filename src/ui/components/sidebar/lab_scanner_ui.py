import streamlit as st
from src.ui.state.view_models import LabViewModel
from src.ui.components.sidebar.helpers import render_control_slider


def render_lab_scanner_section() -> None:
    vm = LabViewModel()
    is_bw = st.session_state.get("process_mode") == "B&W"

    with st.expander(":material/scanner: Lab Scanner Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            if not is_bw:
                render_control_slider(
                    label="Color Separation",
                    min_val=1.0,
                    max_val=3.0,
                    default_val=2.0,
                    step=0.05,
                    key=vm.get_key("color_separation"),
                    format="%.2f",
                    help_text="Color matrix strength (un-mixing dyes).",
                )
            else:
                st.write("")

        with c2:
            render_control_slider(
                label="CLAHE",
                min_val=0.0,
                max_val=1.0,
                default_val=0.0,
                step=0.05,
                key=vm.get_key("clahe_strength"),
                format="%.2f",
                help_text="Increases local contrast.",
            )

        with c3:
            render_control_slider(
                label="Luma Sharpening",
                min_val=0.0,
                max_val=1.0,
                default_val=0.25,
                step=0.05,
                key=vm.get_key("sharpen"),
                help_text="Unsharp mask on L channel.",
            )
