from src.presentation.state.view_models import GeometryViewModel, ExposureViewModel
from src.presentation.layouts.navigation import render_navigation
from src.presentation.components.plots import plot_histogram, plot_photometric_curve
import streamlit as st
import numpy as np
from src.presentation.state.state_manager import save_settings
from src.presentation.components.sidebar.local_adjustments_ui import (
    render_local_adjustments,
)
from src.presentation.components.sidebar.exposure_ui import render_exposure_section
from src.presentation.components.sidebar.paper_toning_ui import render_paper_section
from src.presentation.components.sidebar.lab_scanner_ui import (
    render_lab_scanner_section,
)
from src.presentation.components.sidebar.retouch_ui import render_retouch_section
from src.presentation.components.sidebar.export_ui import render_export_section
from src.presentation.components.sidebar.presets_ui import render_presets
from src.presentation.state.view_models import SidebarState
from src.presentation.components.sidebar.helpers import st_init, render_control_slider
from src.config import DEFAULT_WORKSPACE_CONFIG


def reset_wb_settings() -> None:
    """
    Resets Cyan, Magenta, and Yellow sliders to 0.
    """
    st.session_state.wb_cyan = 0.0
    st.session_state.wb_magenta = 0.0
    st.session_state.wb_yellow = 0.0
    save_settings()


def render_adjustments() -> SidebarState:
    """
    Renders the various image adjustment expanders by delegating to sub-components.
    """
    geo_vm = GeometryViewModel()
    exp_vm = ExposureViewModel()

    # --- Top Controls ---
    st_init("process_mode", DEFAULT_WORKSPACE_CONFIG.process_mode)
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

    st_init(geo_vm.get_key("autocrop"), DEFAULT_WORKSPACE_CONFIG.geometry.autocrop)
    autocrop = st.checkbox(
        "Auto-Crop",
        key=geo_vm.get_key("autocrop"),
        help="Automatically detect film borders and crop to desired aspect ratio.",
    )
    if autocrop:
        c_geo1, c_geo2, c_geo3 = st.columns(3)
        st_init(
            geo_vm.get_key("autocrop_ratio"),
            DEFAULT_WORKSPACE_CONFIG.geometry.autocrop_ratio,
        )
        c_geo1.selectbox(
            "Ratio",
            ["3:2", "4:3", "5:4", "6:7", "1:1", "65:24"],
            key=geo_vm.get_key("autocrop_ratio"),
            label_visibility="collapsed",
            help="Aspect ratio to crop to.",
        )
        
        with c_geo2:
            render_control_slider(
                label="Crop Offset",
                min_val=0.0,
                max_val=100.0,
                default_val=1.0,
                step=1.0,
                key=geo_vm.get_key("autocrop_offset"),
                format="%d",
                help_text="Buffer/offset (pixels) to crop beyond automatically detected border, might be useful when border is uneven.",
            )
            
        with c_geo3:
            render_control_slider(
                label="Fine Rotation (°)",
                min_val=-5.0,
                max_val=5.0,
                default_val=0.0,
                step=0.05,
                key=geo_vm.get_key("fine_rotation"),
            )

    render_presets()

    # 0. Analysis Plots
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
