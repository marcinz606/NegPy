import streamlit as st
from src.frontend.state import save_settings, reset_file_settings
from src.frontend.components.local_ui import render_local_adjustments
from src.frontend.components.sidebar.exposure_ui import render_exposure_section
from src.frontend.components.sidebar.color_ui import render_color_section
from src.frontend.components.sidebar.retouch_ui import render_retouch_section
from src.frontend.components.sidebar.export import render_export_section
from src.frontend.components.sidebar.presets import render_presets
from src.domain_objects import SidebarData
from src.backend.image_logic.exposure import solve_photometric_exposure
from src.backend.image_logic.retouch import apply_autocrop


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
        results = solve_photometric_exposure(img)

        # ATOMIC UPDATE: Color Only
        st.session_state.wb_cyan = results["wb_cyan"]
        st.session_state.wb_magenta = results["wb_magenta"]
        st.session_state.wb_yellow = results["wb_yellow"]

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
        results = solve_photometric_exposure(img)

        # ATOMIC UPDATE: Exposure and Contrast
        st.session_state.density = results["density"]
        st.session_state.grade = results["grade"]

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
        c_geo3.slider("Rotate (°)", -5.0, 5.0, step=0.05, key="fine_rotation")

    render_presets()

    # 1. Exposure & Tonality
    render_exposure_section()

    # 2. Color & Balance (includes Selective Color)
    render_color_section()

    # 3. Dodge & Burn
    with st.expander(":material/pen_size_5: Dodge & Burn", expanded=False):
        render_local_adjustments()

    # 4. Retouch & Export
    render_retouch_section()
    export_data = render_export_section()

    return export_data
