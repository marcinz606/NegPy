import streamlit as st

def render_exposure_section():
    """
    Renders the 'Exposure & Tonality' section of the sidebar.
    """
    # Safety Check: Clamp session state values to valid slider ranges
    if 'grade' in st.session_state:
        st.session_state['grade'] = max(0.0, min(st.session_state['grade'], 5.0))
    if 'scan_gain_s_toe' in st.session_state:
        st.session_state['scan_gain_s_toe'] = max(0.0, min(st.session_state['scan_gain_s_toe'], 0.5))
    if 'scan_gain_h_shoulder' in st.session_state:
        st.session_state['scan_gain_h_shoulder'] = max(0.0, min(st.session_state['scan_gain_h_shoulder'], 0.5))
    
    # Initialize filtration if missing
    for k in ['wb_cyan', 'wb_magenta', 'wb_yellow']:
        if k not in st.session_state:
            st.session_state[k] = 0

    with st.expander(":material/camera: Exposure & Tonality", expanded=True):
        st.caption("Film Base Neutralization (CMY Filtration)")
        # Full-width stacked sliders with color-coded labels
        st.slider(":blue-badge[Cyan]", 0.0, 170.0, 0.0, 0.5, key="wb_cyan", help="Adds Cyan filtration (removes Red cast).")
        st.slider(":violet-badge[Magenta]", 0.0, 170.0, 0.0, 0.5, key="wb_magenta", help="Adds Magenta filtration (removes Green cast).")
        st.slider(":orange-badge[Yellow]", 0.0, 170.0, 0.0, 0.5, key="wb_yellow", help="Adds Yellow filtration (removes Blue cast).")

        # Primary Print Controls
        st.slider(
            "Grade", 
            0.0, 5.0, 2.5, 0.01, 
            format="%.2f", 
            key="grade", 
            help="Unified Exposure and Contrast control. Mimics darkroom Variable Contrast (VC) paper grades. Higher values darken the print, increase contrast, and recover highlight detail."
        )
        c_gain1, c_gain2 = st.columns(2)
        c_gain1.slider(
            "Shadow Toe", 
            0.0, 0.5, 0.0, 0.005, 
            format="%.3f", 
            key="scan_gain_s_toe",
            help="Separates and lifts the deepest shadows to prevent a muddy look."
        )
        c_gain2.slider(
            "Highlight Shoulder", 
            0.0, 0.5, 0.0, 0.005, 
            format="%.3f", 
            key="scan_gain_h_shoulder",
            help="Controls the roll-off of the whites, recovering peak highlight detail."
        )
        
        st.slider("Shadow Desaturation (Auto)", 0.0, 1.0, 1.0, 0.05, help="Prevents oversaturated shadows when lifting them.", key="shadow_desat_strength")



