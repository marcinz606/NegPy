import streamlit as st
from src.ui.state.view_models import LabViewModel
from src.ui.components.sidebar.helpers import render_control_slider


def render_lab_scanner_section() -> None:
    """
    Renders the 'Lab Scanner Parameters' section of the sidebar.
    """
    vm = LabViewModel()
    is_bw = st.session_state.get("process_mode") == "B&W"

    with st.expander(":material/scanner: Lab Scanner Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            if not is_bw:
                render_control_slider(
                    label="Color Separation",
                    min_val=1.0,
                    max_val=4.0,
                    default_val=1.0,
                    step=0.05,
                    key=vm.get_key("color_separation"),
                    format="%.2f",
                    help_text=(
                        "Corrects spectral overlap between film dyes and scanner sensors. "
                        "Mathematically 'un-mixes' muddy colors to restore pure, distinct hues "
                        "mimicking the spectral response of the human eye."
                    ),
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
                key=vm.get_key("hypertone_strength"),
                format="%.2f",
                help_text=(
                    "Contrast Limited Adaptive Histogram Equalization. "
                    "Simulates the 'Hyper-Tone' engine of Fuji Frontier scanners. "
                    "Intelligently compresses high-contrast negatives to fit the dynamic range "
                    "of print paper, recovering shadow detail without flattening the image."
                ),
            )

        with c3:
            render_control_slider(
                label="Luma Sharpening",
                min_val=0.0,
                max_val=1.0,
                default_val=0.25,
                step=0.05,
                key=vm.get_key("sharpen"),
                help_text=(
                    "Applies Unsharp Masking only to the Luminance channel. "
                    "Crispens details and grain structure without introducing "
                    "color halos or enhancing chromatic noise."
            ),
        )
