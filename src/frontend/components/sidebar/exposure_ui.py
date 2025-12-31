import streamlit as st
from src.backend.config import TONE_CURVES_PRESETS

def render_exposure_section():
    """
    Renders the 'Exposure & Tonality' section of the sidebar.
    """
    # Safety Check: Clamp session state values to valid slider ranges
    # This prevents Streamlit errors when switching between different Auto-D logic versions
    if 'scan_gain' in st.session_state:
        st.session_state['scan_gain'] = max(1.0, min(st.session_state['scan_gain'], 5.0))
    if 'scan_gain_s_toe' in st.session_state:
        st.session_state['scan_gain_s_toe'] = max(0.0, min(st.session_state['scan_gain_s_toe'], 0.5))
    if 'scan_gain_h_shoulder' in st.session_state:
        st.session_state['scan_gain_h_shoulder'] = max(0.0, min(st.session_state['scan_gain_h_shoulder'], 0.5))

    with st.expander("Exposure & Tonality", expanded=True):
        st.caption("Film Base Neutralization (CMY Filtration)")
        # Darkroom Workflow: Cyan, Magenta, Yellow (0-170)
        c_wb1, c_wb2, c_wb3 = st.columns(3)
        c_wb1.slider("Cyan", 0, 170, 0, 1, key="wb_cyan", help="Adds Cyan filtration (removes Red).")
        c_wb2.slider("Magenta", 0, 170, 0, 1, key="wb_magenta", help="Adds Magenta filtration (removes Green).")
        c_wb3.slider("Yellow", 0, 170, 0, 1, key="wb_yellow", help="Adds Yellow filtration (removes Blue).")
        st.divider()

        # Primary Print Controls
        c_gain1, c_gain2, c_gain3 = st.columns(3)
        c_gain1.slider(
            "Scan Gain", 
            1.0, 5.0, 1.0, 0.01, 
            format="%.2f", 
            key="scan_gain", 
            help="Scanner-style gain. Darkens the print while smartly pulling in highlight detail."
        )
        c_gain2.slider(
            "Shadow Toe", 
            0.0, 0.5, 0.0, 0.001, 
            format="%.3f", 
            key="scan_gain_s_toe",
            help="Separates and lifts the deepest shadows to prevent a muddy look."
        )
        c_gain3.slider(
            "Highlight Shoulder", 
            0.0, 0.5, 0.0, 0.001, 
            format="%.3f", 
            key="scan_gain_h_shoulder",
            help="Controls the roll-off of the whites, recovering peak highlight detail."
        )
        
        st.slider(
            "Paper Grade", 
            0.2, 2.2, 1.0, 0.01, 
            format="%.2f", 
            key="gamma",
            help="Controls midtone contrast. Higher grades produce punchier, more contrasty prints, while lower grades are softer and flatter."
        )
        
        st.slider("Shadow Desaturation (Auto)", 0.0, 1.0, 1.0, 0.05, help="Prevents oversaturated shadows when lifting them.", key="shadow_desat_strength")
        st.divider()
        c_ex1, c_ex2 = st.columns([1.5, 1])
        c_ex1.slider("Shadows Exposure", -0.5, 0.5, 0.0, 0.005, format="%.3f", key="exposure_shadows")
        c_ex2.slider("Shadows Range", 0.1, 1.0, 1.0, 0.01, key="exposure_shadows_range")
        
        c_ex3, c_ex4 = st.columns([1.5, 1])
        c_ex3.slider("Highlights Exposure", -0.5, 0.5, 0.0, 0.005, format="%.3f", key="exposure_highlights")
        c_ex4.slider("Highlights Range", 0.1, 1.0, 1.0, 0.01, key="exposure_highlights_range")
        
        st.slider("Black & White Points", 0.0, 1.0, (0.0, 1.0), 0.01, key="bw_points")
        c1, c2 = st.columns([1, 1])
        c1.selectbox("Tone Curve", list(TONE_CURVES_PRESETS.keys()), key="curve_mode")
        c2.slider("Curve Strength", 0.0, 1.0, 1.0, 0.01, key="curve_strength")
