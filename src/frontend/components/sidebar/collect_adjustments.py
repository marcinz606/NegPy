import streamlit as st
from src.frontend.state import save_settings, reset_file_settings
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
from src.backend.image_logic.exposure import (
    solve_photometric_exposure,
    prepare_exposure_analysis,
)
from src.backend.image_logic.geometry import apply_autocrop


def run_auto_wb() -> None:
    if "preview_raw" in st.session_state:
        img = st.session_state.preview_raw

        # Apply Auto-Crop if enabled to focus color analysis on subject area
        if st.session_state.get("autocrop"):
            img = apply_autocrop(
                img,
                offset_px=st.session_state.get("autocrop_offset", 0),
                scale_factor=1.0,
                ratio=st.session_state.get("autocrop_ratio", "3:2"),
            )

        # Delegate logic to the Backend Photometric Solver
        norm_log, bounds = prepare_exposure_analysis(img)
        c, m, y, density, grade = solve_photometric_exposure(norm_log, bounds)

        # ATOMIC UPDATE: Color Only
        st.session_state.wb_cyan = c
        st.session_state.wb_magenta = m
        st.session_state.wb_yellow = y

        st.session_state["auto_wb"] = False
        save_settings()


def run_auto_density() -> None:
    if "preview_raw" in st.session_state:
        img = st.session_state.preview_raw

        # Apply Auto-Crop if enabled to focus analysis on subject area only
        if st.session_state.get("autocrop"):
            img = apply_autocrop(
                img,
                offset_px=st.session_state.get("autocrop_offset", 0),
                scale_factor=1.0,
                ratio=st.session_state.get("autocrop_ratio", "3:2"),
            )

        # Delegate logic to the Backend Photometric Solver
        norm_log, bounds = prepare_exposure_analysis(img)
        c, m, y, density, grade = solve_photometric_exposure(norm_log, bounds)

        # ATOMIC UPDATE: Exposure and Contrast
        st.session_state.density = density
        st.session_state.grade = grade

        save_settings()


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
    mode = st.selectbox(
        "Processing Mode",
        ["C41", "B&W"],
        key="process_mode",
        on_change=reset_wb_settings,
        help="Choose processing mode between Color Negative (C41) and B&W Negative",
    )
    c1, c2, c3 = st.columns(3)
    is_bw = mode == "B&W"

    c1.button(
        ":material/wb_auto: Auto-WB",
        on_click=run_auto_wb,
        width="stretch",
        disabled=is_bw,
        help="Tries to neutralize color negative film mask by adjusting magenta and yellow filters.",
    )
    c2.button(
        ":material/exposure: Auto-D",
        on_click=run_auto_density,
        width="stretch",
        help="Automatically solves for the optimal Density and Shoulder settings based on negative dynamics.",
    )
    c3.button(
        ":material/reset_image: Reset",
        key="reset_s",
        on_click=reset_file_settings,
        width="stretch",
        type="secondary",
        help="Reset all settings for this negative to defaults.",
    )

    autocrop = st.checkbox(
        "Auto-Crop",
        key="autocrop",
        help="Automatically detect film borders and crop to desired aspect ratio.",
    )
    if autocrop:
        c_geo1, c_geo2 = st.columns(2)
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
    
    st.slider("Fine Rotation (°)", -5.0, 5.0, step=0.05, key="fine_rotation")

    render_presets()

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

    return export_data
