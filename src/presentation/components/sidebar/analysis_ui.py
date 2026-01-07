import streamlit as st
import numpy as np
from src.presentation.components.plots import plot_histogram, plot_photometric_curve
from src.presentation.state.view_models import ExposureViewModel


def render_analysis_section() -> None:
    """
    Renders the Analysis section (Histogram and Photometric Curve) of the sidebar.
    """
    exp_vm = ExposureViewModel()

    with st.expander(":material/analytics: Analysis", expanded=True):
        if "preview_raw" in st.session_state:
            if "last_pil_prev" in st.session_state:
                st.caption(
                    "Histogram",
                    help=(
                        "Visualizes the tonal distribution of the processed print. "
                        "The horizontal axis represents brightness from Shadows (left) to Highlights (right), "
                        "while the vertical axis shows the frequency of pixels at each level."
                    ),
                )

                st.pyplot(
                    plot_histogram(
                        np.array(st.session_state.last_pil_prev.convert("RGB")),
                        figsize=(3, 1.4),
                        dpi=150,
                    ),
                    width="stretch",
                )

            st.caption(
                "Photometric Curve",
                help=(
                    "This H&D Characteristic Curve represents the relationship between Subject Brightness and Print Density. "
                    "It visualizes how the engine simulates light-sensitive paper: 'Density' shifts the exposure, "
                    "'Grade' controls contrast slope, while 'Toe' and 'Shoulder' manage the roll-off in highlights "
                    "and shadows respectively."
                ),
            )

            st.pyplot(
                plot_photometric_curve(exp_vm.to_config(), figsize=(3, 1.4), dpi=150),
                width="stretch",
            )
