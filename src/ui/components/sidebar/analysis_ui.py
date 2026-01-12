import streamlit as st
import numpy as np
from src.ui.components.plots import plot_histogram, plot_photometric_curve
from src.ui.state.view_models import ExposureViewModel


def render_analysis_section() -> None:
    """
    Renders Histogram & Curve plots.
    """
    exp_vm = ExposureViewModel()

    with st.expander(":material/analytics: Analysis", expanded=True):
        if "preview_raw" in st.session_state:
            if "last_pil_prev" in st.session_state:
                st.caption(
                    "Histogram",
                    help="Tonal distribution (Shadows -> Highlights).",
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
                    help="H&D Characteristic Curve (Log-Exposure vs Density).",
                )
            st.pyplot(
                plot_photometric_curve(exp_vm.to_config(), figsize=(3, 1.4), dpi=150),
                width="stretch",
            )
