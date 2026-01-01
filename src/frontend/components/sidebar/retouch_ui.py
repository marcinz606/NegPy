import streamlit as st
from src.frontend.state import save_settings

def render_retouch_section(current_file_name: str):
    """
    Renders the 'Retouch' section of the sidebar.
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
            if c1.button("Undo Last", width='stretch'):
                if st.session_state.manual_dust_spots:
                    st.session_state.manual_dust_spots.pop()
                    save_settings(current_file_name)
                    st.rerun()
            if c2.button("Clear All", width='stretch'):
                st.session_state.manual_dust_spots = []
                save_settings(current_file_name)
                st.rerun()

        st.checkbox("Chroma Noise Removal", key="c_noise_remove")
        st.slider("Chroma Noise Strength", 0, 100, 25, 1, disabled=not st.session_state.c_noise_remove, key="c_noise_strength")

    return {}
