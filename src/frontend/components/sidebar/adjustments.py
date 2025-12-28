import streamlit as st
import os
from src.backend.config import TONE_CURVES_PRESETS, APP_CONFIG
from src.frontend.state import save_settings, load_settings
from src.frontend.components.local_ui import render_local_adjustments

def render_adjustments(current_file_name: str):
    """
    Renders the various image adjustment expanders.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.checkbox("Auto-Crop", key="autocrop")
    c2.checkbox("Monochrome", key="monochrome")
    c3.checkbox("Auto-WB", key="auto_wb")
    if c4.button("Manual-WB", width="stretch", type="secondary" if not st.session_state.pick_wb else "primary"):
        st.session_state.pick_wb = not st.session_state.pick_wb
        st.rerun()
    
    c_geo1, c_geo2 = st.columns(2)
    if st.session_state.autocrop: 
        c_geo1.slider("Crop Offset", 0, 100, 1, 1, key="autocrop_offset")
        c_geo2.slider("Rotate (°)", -10.0, 10.0, 0.0, 0.1, key="fine_rotation")

    with st.expander("Color & Balance", expanded=True):
        st.slider("Color Separation", 0.0, 2.0, 1.0, 0.01, format="%.2f", key="saturation")
        st.caption("Global Tone")
        st.slider("Temperature", -0.25, 0.25, 0.0, 0.001, format="%.3f", key="temperature")
        c1, c2, c3 = st.columns(3)
        with c1: st.slider("Cyan - Red", 0.95, 1.05, 1.0, 0.001, format="%.3f", key="cr_balance")
        with c2: st.slider("Magenta - Green", 0.95, 1.05, 1.0, 0.001, format="%.3f", key="mg_balance")
        with c3: st.slider("Yellow - Blue", 0.95, 1.05, 1.0, 0.001, format="%.3f", key="yb_balance")

        st.caption("Shadows Tone")
        st.slider("S: Temperature", -0.25, 0.25, 0.0, 0.001, format="%.3f", key="shadow_temp")
        c1, c2, c3 = st.columns(3)
        c1.slider("S: Cyan - Red", 0.9, 1.1, 1.0, 0.001, format="%.3f", key="shadow_cr")
        c2.slider("S: Magenta - Green", 0.9, 1.1, 1.0, 0.001, format="%.3f", key="shadow_mg")
        c3.slider("S: Yellow - Blue", 0.9, 1.1, 1.0, 0.001, format="%.3f", key="shadow_yb")
        
        st.caption("Highlights Tone")
        st.slider("H: Temperature", -0.25, 0.25, 0.0, 0.001, format="%.3f", key="highlight_temp")
        c1, c2, c3 = st.columns(3)
        c1.slider("H: Cyan - Red", 0.9, 1.1, 1.0, 0.001, format="%.3f", key="highlight_cr")
        c2.slider("H: Magenta - Green", 0.9, 1.1, 1.0, 0.001, format="%.3f", key="highlight_mg")
        c3.slider("H: Yellow - Blue", 0.9, 1.1, 1.0, 0.001, format="%.3f", key="highlight_yb")

    with st.expander("Selective Color", expanded=False):
        color_opts = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']
        c1, c2 = st.columns([1, 1.5])
        selected_color_disp = c1.selectbox("Select Color", color_opts, key="sel_color_choice")
        selected_color = selected_color_disp.lower()
        c2.slider("Range", 0.1, 3.0, 1.0, 0.1, key=f"selective_{selected_color}_range")
        
        c1, c2, c3 = st.columns(3)
        c1.slider("Hue", -180.0, 180.0, 0.0, 1.0, key=f"selective_{selected_color}_hue")
        c2.slider("Saturation", -1.0, 1.0, 0.0, 0.01, key=f"selective_{selected_color}_sat")
        c3.slider("Luminance", -1.0, 1.0, 0.0, 0.01, key=f"selective_{selected_color}_lum")

    with st.expander("Exposure & Tonality", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: st.slider("Exposure", -2.0, 2.0, 0.0, 0.05, format="%.2f", key="exposure")
        with c2: st.slider("Contrast", 0.5, 1.5, 1.0, 0.05, format="%.2f", key="contrast")
        with c3: st.slider("Global Grade", 0.0, 5.0, 2.5, 0.05, format="%.2f", key="gamma")
        c1, c2 = st.columns(2)
        c1.slider("Shadows Grade", 0.0, 5.0, 2.5, 0.05, format="%.2f", key="grade_shadows")
        c2.slider("Higlights Grade", 0.0, 5.0, 2.5, 0.05, format="%.2f", key="grade_highlights")
        st.slider("Black & White Points", 0.0, 1.0, (0.0, 1.0), 0.01, key="bw_points")
        c1, c2 = st.columns([1, 1])
        c1.selectbox("Tone Curve", list(TONE_CURVES_PRESETS.keys()), key="curve_mode")
        c2.slider("Curve Strength", 0.0, 2.0, 1.0, 0.05, key="curve_strength")

    with st.expander("Dodge & Burn", expanded=False):
        render_local_adjustments()
    
    with st.expander("Retouch & Export", expanded=True):
        c1, c2, c3 = st.columns([0.8, 1.5, 1.5])
        c1.checkbox("Dust Removal", key="dust_remove")
        c2.slider("Dust Threshold", 0.01, 1.0, 0.55, 0.01, label_visibility="collapsed", disabled=not st.session_state.dust_remove, key="dust_threshold")
        c3.slider("Dust Size", 1, 20, 2, 1, label_visibility="collapsed", disabled=not st.session_state.dust_remove, key="dust_size")
        
        c1, c2 = st.columns([2, 1])
        c1.checkbox("Manual Dust Correction", key="pick_dust")
        if st.session_state.get('manual_dust_spots'):
            c2.caption(f"{len(st.session_state.manual_dust_spots)} spots")
        
        if st.session_state.pick_dust:
            st.slider("Manual Spot Size", 1, 50, 4, 1, key="manual_dust_size")
            st.checkbox("Scratch Mode (Click Start -> Click End)", key="dust_scratch_mode")
            st.checkbox("Show Patches", value=True, key="show_dust_patches")
            c1, c2 = st.columns(2)
            if c1.button("Undo Last", width="stretch"):
                if st.session_state.manual_dust_spots:
                    st.session_state.manual_dust_spots.pop()
                    save_settings(current_file_name)
                    st.rerun()
            if c2.button("Clear All", width="stretch"):
                st.session_state.manual_dust_spots = []
                save_settings(current_file_name)
                st.rerun()

        c1, c2 = st.columns([0.8, 3])
        c1.checkbox("Chroma Noise Removal", key="c_noise_remove")
        c2.slider("Chroma Noise Strength", 0, 100, 50, 1, label_visibility="collapsed", disabled=not st.session_state.c_noise_remove, key="c_noise_strength")
        st.divider()
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.5])
        out_fmt = c1.selectbox("Format", ["JPEG", "TIFF"])
        print_width = c2.number_input("Longer Side (cm)", 27.0)
        print_dpi = c3.number_input("DPI", 300)
        c4.slider("Output Sharpening", 0.0, 1.5, 0.75, 0.01, key="sharpen")
        c1, c2 = st.columns([3, 1])
        export_path = c1.text_input("Export Directory", APP_CONFIG['default_export_dir'])
        process_btn = st.button("Export All", type="primary", width="stretch")
    
    return {
        'out_fmt': out_fmt,
        'print_width': print_width,
        'print_dpi': print_dpi,
        'export_path': export_path,
        'process_btn': process_btn
    }
