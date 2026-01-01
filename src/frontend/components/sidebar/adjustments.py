import streamlit as st
import numpy as np
from src.frontend.state import save_settings, reset_file_settings
from src.frontend.components.local_ui import render_local_adjustments
from src.backend.processor import calculate_auto_mask_wb
from src.frontend.components.sidebar.helpers import apply_wb_gains_to_sliders
from src.frontend.components.sidebar.exposure_ui import render_exposure_section
from src.frontend.components.sidebar.color_ui import render_color_section
from src.frontend.components.sidebar.retouch_ui import render_retouch_section
from src.frontend.components.sidebar.export import render_export_section
from src.frontend.components.sidebar.presets import render_presets
from src.backend.image_logic.exposure import calculate_auto_exposure_params

def run_auto_wb(current_file_name: str):
    if 'preview_raw' in st.session_state:
        r, g, b = calculate_auto_mask_wb(st.session_state.preview_raw)
        slider_vals = apply_wb_gains_to_sliders(r, g, b)
        for k, v in slider_vals.items():
            st.session_state[k] = v
        st.session_state['auto_wb'] = False
        save_settings(current_file_name)


def run_auto_density(current_file_name: str):
    if 'preview_raw' in st.session_state:
        # Calculate current WB gains from sliders
        c_val = np.clip(st.session_state.get('wb_cyan', 0), 0, 170)
        m_val = np.clip(st.session_state.get('wb_magenta', 0), 0, 170)
        y_val = np.clip(st.session_state.get('wb_yellow', 0), 0, 170)
        
        r_gain = 10.0 ** (c_val / 100.0)
        g_gain = 10.0 ** (m_val / 100.0)
        b_gain = 10.0 ** (y_val / 100.0)

        # Delegate logic to the Backend Surgical Solver
        grade, s_toe, h_shoulder = calculate_auto_exposure_params(
            st.session_state.preview_raw,
            r_gain,
            g_gain,
            b_gain
        )
        
        st.session_state.grade = grade
        st.session_state.scan_gain_s_toe = s_toe
        st.session_state.scan_gain_h_shoulder = h_shoulder
        
        save_settings(current_file_name)

def reset_wb_settings(current_file_name: str):
    """
    Resets Cyan, Magenta, and Yellow sliders to 0.
    """
    st.session_state.wb_cyan = 0
    st.session_state.wb_magenta = 0
    st.session_state.wb_yellow = 0
    save_settings(current_file_name)

def render_adjustments(current_file_name: str):
    """
    Renders the various image adjustment expanders by delegating to sub-components.
    """
    # --- Top Controls ---
    mode = st.selectbox("Processing Mode", ["C41", "B&W"], key="process_mode", on_change=reset_wb_settings, args=(current_file_name,), help="Choose processing mode between Color Negative (C41) and B&W Negative")
    c1, c2, c3 = st.columns(3)
    is_bw = (mode == "B&W")
    
    c1.button(":material/wb_auto: Auto-WB", on_click=run_auto_wb, args=(current_file_name,), width='stretch', disabled=is_bw, help="Tries to neutralize color negative film mask by adjusting magenta and yellow filters.")
    c2.button(":material/exposure: Auto-D", on_click=run_auto_density, args=(current_file_name,), width='stretch', help="Automatically solves for the optimal Density and Shoulder settings based on negative dynamics.")
    c3.button(":material/reset_image: Reset", key="reset_s", on_click=reset_file_settings, args=(current_file_name,), width='stretch', type="secondary", help="Reset all settings for this negative to defaults.")


    autocrop = st.checkbox("Auto-Crop", key="autocrop", help="Automatically detect film borders and crop to desired aspect ratio.")
    if autocrop:
        c_geo1, c_geo2, c_geo3 = st.columns(3)
        c_geo1.selectbox("Ratio", ["3:2", "4:3", "5:4", "6:7", "1:1", "65:24"], key="autocrop_ratio", label_visibility="collapsed", help="Aspect ratio to crop to.")
        c_geo2.slider("Crop Offset", 0, 100, 1, 1, key="autocrop_offset", help="Buffer/offset (pixels) to crop beyond automatically detected border, might be useful when border is uneven.")
        c_geo3.slider("Rotate (°)", -5.0, 5.0, 0.0, 0.05, key="fine_rotation")

    render_presets(current_file_name)

    # 1. Exposure & Tonality
    render_exposure_section()

    # 2. Color & Balance (includes Selective Color)
    render_color_section(current_file_name)

    # 3. Dodge & Burn
    with st.expander(":material/pen_size_5: Dodge & Burn", expanded=False):
        render_local_adjustments()
    
    # 4. Retouch & Export
    retouch_data = render_retouch_section(current_file_name)
    export_data = render_export_section()
    
    return {**retouch_data, **export_data}
