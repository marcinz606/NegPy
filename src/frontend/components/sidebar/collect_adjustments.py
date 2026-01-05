import streamlit as st
import numpy as np
from src.frontend.state import save_settings
from src.frontend.components.sidebar.local_adjustments_ui import (
    render_local_adjustments,
)
from src.frontend.components.sidebar.exposure_ui import render_exposure_section
from src.frontend.components.sidebar.paper_toning_ui import render_paper_section
from src.frontend.components.sidebar.lab_scanner_ui import render_lab_scanner_section
from src.frontend.components.sidebar.retouch_ui import render_retouch_section
from src.frontend.components.sidebar.export_ui import render_export_section
from src.frontend.components.sidebar.presets_ui import render_presets
from src.domain_objects import SidebarData
from src.frontend.components.navigation import render_navigation
from src.backend.image_logic.plots import plot_histogram, plot_photometric_curve


def reset_wb_settings() -> None:
    """
    Resets Cyan, Magenta, and Yellow sliders to 0.
    """
    st.session_state.wb_cyan = 0.0
    st.session_state.wb_magenta = 0.0
    st.session_state.wb_yellow = 0.0
    save_settings()


def render_adjustments() -> SidebarData:
    """
    Renders the various image adjustment expanders by delegating to sub-components.
    """
    # --- Top Controls ---
    st.selectbox(
        "Processing Mode",
        ["C41", "B&W"],
        key="process_mode",
        on_change=reset_wb_settings,
        help="Choose processing mode between Color Negative (C41) and B&W Negative",
    )

    # Replaced Auto-WB/Auto-D with Navigation
    export_btn_sidebar, process_all_btn = render_navigation()

    # --- Top Controls ---
    # Moved working copy slider to image view

    autocrop = st.checkbox(
        "Auto-Crop",
        key="autocrop",
        help="Automatically detect film borders and crop to desired aspect ratio.",
    )
    if autocrop:
        c_geo1, c_geo2, c_geo3 = st.columns(3)
        c_geo1.selectbox(
            "Ratio",
            ["3:2", "4:3", "5:4", "6:7", "1:1", "65:24"],
            key="autocrop_ratio",
            label_visibility="collapsed",
            help="Aspect ratio to crop to.",
        )
        c_geo2.slider(
            "Crop Offset",
            0,
            100,
            key="autocrop_offset",
            help="Buffer/offset (pixels) to crop beyond automatically detected border, might be useful when border is uneven.",
        )
        c_geo3.slider(
            "Fine Rotation (°)", -5.0, 5.0, step=0.05, key="fine_rotation"
        )

        render_presets()

        # 0. Analysis Plots

        with st.expander(":material/analytics: Analysis", expanded=True):
            if "preview_raw" in st.session_state:
                from src.frontend.main import get_processing_params

                current_params = get_processing_params(st.session_state)
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
                    plot_photometric_curve(current_params, figsize=(3, 1.4), dpi=150),
                    width="stretch",
                )

        # 1. Exposure & Tonality

        render_exposure_section()

    # 2. Lab Scanner Simulation
    render_lab_scanner_section()

    # 3. Color & Balance (includes Selective Color)
    render_paper_section()

    # 4. Dodge & Burn
    with st.expander(":material/pen_size_5: Dodge & Burn", expanded=False):
        render_local_adjustments()

    # 4. Retouch & Export
    render_retouch_section()
    export_data = render_export_section()
    export_data.export_btn = export_btn_sidebar
    export_data.process_btn = process_all_btn

    return export_data
