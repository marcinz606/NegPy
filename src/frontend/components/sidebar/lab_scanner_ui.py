import streamlit as st
from src.frontend.components.sidebar.helpers import render_control_slider


def render_lab_scanner_section() -> None:
    """
    Renders the 'Lab Scanner Parameters' section of the sidebar.
    Consolidates advanced lab-scanner-style adjustments.
    """
    is_bw = st.session_state.get("process_mode") == "B&W"

    with st.expander(":material/scanner: Lab Scanner Parameters", expanded=True):
        c1, c2 = st.columns(2)

        with c1:
            # 1. Spectral Crosstalk (Color Only)
            if not is_bw:
                render_control_slider(
                    "Color Separation",
                    1.0,
                    4.0,
                    1.0,
                    0.05,
                    "color_separation",
                    format="%.2f",
                    help_text=(
                        "Corrects spectral overlap between film dyes and scanner sensors. "
                        "Mathematically 'un-mixes' muddy colors to restore pure, distinct hues "
                        "mimicking the spectral response of the human eye."
                    ),
                )
            else:
                st.write("")  # Spacer for alignment

        with c2:
            # 2. Hypertone (CLAHE)
            render_control_slider(
                "CLAHE",
                0.0,
                1.0,
                0.0,
                0.05,
                "hypertone_strength",
                format="%.2f",
                help_text=(
                    "Contrast Limited Adaptive Histogram Equalization."
                    "Simulates the 'Hyper-Tone' engine of Fuji Frontier scanners. "
                    "Intelligently compresses high-contrast negatives to fit the dynamic range "
                    "of print paper, recovering shadow detail without flattening the image."
                ),
            )

        c3, c4 = st.columns(2)

        with c3:
            # 3. Chroma Noise Removal
            render_control_slider(
                "Chroma Smoothing",
                0.0,
                1.0,
                0.25,
                0.05,
                "c_noise_strength",
                format="%.2f",
                help_text=(
                    "Removes digital color artifacts ('confetti') often found in dense shadows. "
                    "Smooths the color channels while strictly preserving the natural "
                    "film grain structure in the luminance channel."
                ),
            )

        with c4:
            # 4. Output Sharpening
            render_control_slider(
                "Luma Sharpening",
                0.0,
                1.0,
                0.50,
                0.05,
                "sharpen",
                help_text=(
                    "Applies Unsharp Masking only to the Luminance channel. "
                    "Crispens details and grain structure without introducing "
                    "color halos or enhancing chromatic noise."
                ),
            )
