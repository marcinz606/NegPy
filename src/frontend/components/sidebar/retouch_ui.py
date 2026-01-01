import streamlit as st
from src.backend.config import APP_CONFIG
from src.frontend.state import save_settings

def render_retouch_section(current_file_name: str):
    """
    Renders the 'Retouch' and 'Export' sections of the sidebar.
    """
    with st.expander(":material/brush: Retouch", expanded=True):
        st.checkbox("Automatic dust removal", key="dust_remove")
        c1, c2 = st.columns(2)
        c1.slider("Threshold", 0.01, 1.0, 0.55, 0.01, disabled=not st.session_state.dust_remove, key="dust_threshold", help="Sensitivity of automatic dust detection. Lower values detect more spots.")
        c2.slider("Size", 1, 20, 2, 1, disabled=not st.session_state.dust_remove, key="dust_size", help="Maximum size of spots to be automatically removed.")
        
        c1, c2 = st.columns([2, 1])
        c1.checkbox("Manual Dust Correction", key="pick_dust")
        if st.session_state.get('manual_dust_spots'):
            c2.caption(f"{len(st.session_state.manual_dust_spots)} spots")
        
        if st.session_state.pick_dust:
            st.slider("Manual Spot Size", 1, 50, 4, 1, key="manual_dust_size")
            st.checkbox("Scratch Mode (Click Start -> Click End)", key="dust_scratch_mode")
            st.checkbox("Show Patches", value=True, key="show_dust_patches")
            c1, c2 = st.columns(2)
            if c1.button("Undo Last", use_container_width=True):
                if st.session_state.manual_dust_spots:
                    st.session_state.manual_dust_spots.pop()
                    save_settings(current_file_name)
                    st.rerun()
            if c2.button("Clear All", use_container_width=True):
                st.session_state.manual_dust_spots = []
                save_settings(current_file_name)
                st.rerun()

        st.checkbox("Chroma Noise Removal", key="c_noise_remove")
        st.slider("Chroma Noise Strength", 0, 100, 25, 1, disabled=not st.session_state.c_noise_remove, key="c_noise_strength")

    with st.expander(":material/file_export: Export", expanded=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.5])
        out_fmt = c1.selectbox("Format", ["JPEG", "TIFF"])
        print_width = c2.number_input("Size (cm)", value=27.0, min_value=10.0)
        print_dpi = c3.number_input("DPI", 300)
        c4.slider("Sharpening", 0.0, 1.5, 0.75, 0.01, key="sharpen")
        c1, c2 = st.columns([3, 1])
        export_path = c1.text_input("Export Directory", APP_CONFIG['default_export_dir'])
        
        c_b1, c_b2, c_b3 = st.columns([1, 1.5, 1.5])
        add_border = c_b1.checkbox("Border", value=True)
        border_size = c_b2.number_input("Size (cm)", value=0.25, min_value=0.1, step=0.05, disabled=not add_border)
        border_color = c_b3.color_picker("Color", value="#000000", disabled=not add_border)
        
        c_exp1, c_exp2 = st.columns([1.5, 1.5])
        
        apply_icc = c_exp1.checkbox("Apply ICC Profile to Export", value=False, disabled=not st.session_state.get('icc_profile_path'))
        process_btn = c_exp2.button("Export All", type="primary", use_container_width=True)
    
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
