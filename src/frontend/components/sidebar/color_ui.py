import streamlit as st
from src.frontend.components.sidebar.helpers import render_control_slider


def render_color_section() -> None:
    """
    Renders the 'Color & Toning' section of the sidebar.
    """
    is_bw = st.session_state.get("process_mode") == "B&W"
    with st.expander(":material/colorize: Color & Toning", expanded=True):
        if not is_bw:
            c_sat1, c_sat2 = st.columns(2)
            with c_sat1:
                render_control_slider(
                    "Separation",
                    0.5,
                    1.5,
                    1.0,
                    0.005,
                    "color_separation",
                    format="%.3f",
                    help_text="Controls color depth and separation. Higher values mimic laboratory color separation techniques.",
                )
            with c_sat2:
                render_control_slider(
                    "Saturation",
                    0.0,
                    1.5,
                    1.0,
                    0.005,
                    "saturation",
                    format="%.3f",
                    help_text="Overall color intensity of the print.",
                )

        render_control_slider(
            "Paper Warmth",
            -0.2,
            0.2,
            0.0,
            0.005,
            "temperature",
            format="%.3f",
            help_text="Overall warmth or coolness of the paper. Mimics the base paper tint.",
        )

        c_tone1, c_tone2 = st.columns(2)
        with c_tone1:
            render_control_slider(
                "Shadow Tone",
                -0.2,
                0.2,
                0.0,
                0.005,
                "shadow_temp",
                format="%.3f",
                help_text="Amber (+) vs Blue (-) toning in the deep shadows. Mimics chemical toning baths.",
            )
        with c_tone2:
            render_control_slider(
                "Highlight Tone",
                -0.2,
                0.2,
                0.0,
                0.005,
                "highlight_temp",
                format="%.3f",
                help_text="Amber (+) vs Blue (-) toning in the paper highlights.",
            )
