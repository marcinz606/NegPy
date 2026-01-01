import streamlit as st

def render_color_section(current_file_name: str):
    """
    Renders the 'Color & Toning' section of the sidebar.
    """
    is_bw = st.session_state.get('process_mode') == 'B&W'
    with st.expander(":material/colorize: Color & Toning", expanded=True):
        if not is_bw:
            c_sat1, c_sat2 = st.columns(2)
            c_sat1.slider(
                "Separation", 0.5, 1.5, 1.0, 0.005, format="%.3f", key="color_separation",
                help="Controls color depth and separation. Higher values mimic laboratory color separation techniques."
            )
            c_sat2.slider(
                "Saturation", 0.0, 1.5, 1.0, 0.005, format="%.3f", key="saturation",
                help="Overall color intensity of the print."
            )
        
        st.slider(
            "Paper Warmth", -0.2, 0.2, 0.0, 0.005, format="%.3f", key="temperature",
            help="Overall warmth or coolness of the paper. Mimics the base paper tint."
        )
        
        c_tone1, c_tone2 = st.columns(2)
        c_tone1.slider(
            "Shadow Tone", -0.2, 0.2, 0.0, 0.005, format="%.3f", key="shadow_temp",
            help="Amber (+) vs Blue (-) toning in the deep shadows. Mimics chemical toning baths."
        )
        c_tone2.slider(
            "Highlight Tone", -0.2, 0.2, 0.0, 0.005, format="%.3f", key="highlight_temp",
            help="Amber (+) vs Blue (-) toning in the paper highlights."
        )
