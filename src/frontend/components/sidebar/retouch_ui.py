import streamlit as st
from src.backend.config import APP_CONFIG
from src.frontend.state import save_settings

def render_retouch_section(current_file_name: str):
    """
    Renders the 'Retouch & Export' section of the sidebar.
    """
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
        c2.slider("Chroma Noise Strength", 0, 100, 25, 1, label_visibility="collapsed", disabled=not st.session_state.c_noise_remove, key="c_noise_strength")
        st.divider()
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.5])
        out_fmt = c1.selectbox("Format", ["JPEG", "TIFF"])
        print_width = c2.number_input("Longer Side (cm)", value=27.0, min_value=10.0)
        print_dpi = c3.number_input("DPI", 300)
        c4.slider("Output Sharpening", 0.0, 1.5, 0.75, 0.01, key="sharpen")
        c1, c2 = st.columns([3, 1])
        export_path = c1.text_input("Export Directory", APP_CONFIG['default_export_dir'])
        
        st.divider()
        c_exp1, c_exp2 = st.columns([1, 3])
        add_border = c_exp1.checkbox("Border", value=True)
        apply_icc = c_exp2.checkbox("Apply ICC Profile to Export", value=False, disabled=not st.session_state.get('icc_profile_path'))
        
        c_b1, c_b2 = st.columns([1.5, 1.5])
        border_size = c_b1.number_input("Size (cm)", value=0.25, min_value=0.1, step=0.05, disabled=not add_border)
        border_color = c_b2.color_picker("Color", value="#000000", disabled=not add_border)
        
        process_btn = st.button("Export All", type="primary", width="stretch")
    
    return {
        'out_fmt': out_fmt,
        'print_width': print_width,
        'print_dpi': print_dpi,
        'export_path': export_path,
        'add_border': add_border,
        'border_size': border_size,
        'border_color': border_color,
        'apply_icc': apply_icc,
        'process_btn': process_btn
    }
