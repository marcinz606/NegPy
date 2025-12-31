import streamlit as st
import numpy as np
from src.frontend.state import save_settings
from src.frontend.components.local_ui import render_local_adjustments
from src.backend.processor import calculate_auto_mask_wb
from src.frontend.components.sidebar.helpers import apply_wb_gains_to_sliders

from .exposure_ui import render_exposure_section
from .color_ui import render_color_section
from .retouch_ui import render_retouch_section

def run_auto_wb(current_file_name: str):
    if 'preview_raw' in st.session_state:
        r, g, b = calculate_auto_mask_wb(st.session_state.preview_raw)
        slider_vals = apply_wb_gains_to_sliders(r, g, b)
        for k, v in slider_vals.items():
            st.session_state[k] = v
        st.session_state['auto_wb'] = False
        save_settings(current_file_name)

from src.backend.image_logic.exposure import calculate_auto_exposure_params

def run_auto_density(current_file_name: str):
    if 'preview_raw' in st.session_state:
        # Delegate logic to the Backend Surgical Solver
        gain, s_toe, h_shoulder = calculate_auto_exposure_params(
            st.session_state.preview_raw,
            st.session_state.get('wb_manual_r', 1.0),
            st.session_state.get('wb_manual_g', 1.0),
            st.session_state.get('wb_manual_b', 1.0)
        )
        
        st.session_state.scan_gain = gain
        st.session_state.scan_gain_s_toe = s_toe
        st.session_state.scan_gain_h_shoulder = h_shoulder
        
        save_settings(current_file_name)

def render_adjustments(current_file_name: str):
    """
    Renders the various image adjustment expanders by delegating to sub-components.
    """
    # --- Top Controls ---
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    c1.checkbox("Auto-Crop", key="autocrop")
    c2.checkbox("Mono", key="monochrome")
    c3.button("Auto-WB", on_click=run_auto_wb, args=(current_file_name,), use_container_width=True)
    c4.button("Auto-D", on_click=run_auto_density, args=(current_file_name,), use_container_width=True, help="Automatically solves for the optimal Density and Shoulder settings based on negative dynamics.")
    
    if c5.button("Pick", use_container_width=True, type="secondary" if not st.session_state.pick_wb else "primary", help="Manual WB Picker"):
        st.session_state.pick_wb = not st.session_state.pick_wb
        st.rerun()
    
    c_geo1, c_geo2 = st.columns(2)
    if st.session_state.autocrop: 
        c_geo1.slider("Crop Offset", 0, 100, 1, 1, key="autocrop_offset")
        c_geo2.slider("Rotate (°)", -5.0, 5.0, 0.0, 0.05, key="fine_rotation")

    # 1. Exposure & Tonality
    render_exposure_section()

    # 2. Color & Balance (includes Selective Color)
    render_color_section(current_file_name)

    # 3. Dodge & Burn
    with st.expander("Dodge & Burn", expanded=False):
        render_local_adjustments()
    
    # 4. Retouch & Export
    return render_retouch_section(current_file_name)