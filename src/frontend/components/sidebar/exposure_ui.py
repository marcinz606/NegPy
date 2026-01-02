import streamlit as st
from src.frontend.components.sidebar.helpers import render_control_slider


def render_exposure_section() -> None:
    """
    Renders the 'Exposure & Tonality' section of the sidebar.
    """
    is_bw = st.session_state.get("process_mode") == "B&W"

    with st.expander(":material/camera: Exposure & Tonality", expanded=True):
        if not is_bw:
            render_control_slider(
                ":blue-badge[Cyan]",
                0.0,
                170.0,
                0.0,
                0.5,
                "wb_cyan",
                help_text=(
                    "Adds Cyan filtration (removes Red cast). Like in darkroom, "
                    "you SHOULD NOT be touching this unless you know what you are doing :)"
                ),
            )
            render_control_slider(
                ":violet-badge[Magenta]",
                0.0,
                170.0,
                0.0,
                0.5,
                "wb_magenta",
                help_text="Adds Magenta filtration (removes Green cast).",
            )
            render_control_slider(
                ":orange-badge[Yellow]",
                0.0,
                170.0,
                0.0,
                0.5,
                "wb_yellow",
                help_text="Adds Yellow filtration (removes Blue cast).",
            )

        # Primary Print Controls
        render_control_slider(
            "Grade",
            0.0,
            5.0,
            2.5,
            0.05,
            "grade",
            help_text=(
                "Unified Exposure and Contrast control. Mimics darkroom Variable Contrast (VC) paper grades. "
                "Higher values darken the print, increase contrast, and recover highlight detail."
            ),
        )

        c_gain1, c_gain2 = st.columns(2)
        with c_gain1:
            render_control_slider(
                "Shadow Toe",
                0.0,
                0.5,
                0.0,
                0.005,
                "scan_gain_s_toe",
                format="%.3f",
                help_text="Separates and lifts the deepest shadows to prevent a muddy look.",
            )
        with c_gain2:
            render_control_slider(
                "Highlight Shoulder",
                0.0,
                0.5,
                0.0,
                0.005,
                "scan_gain_h_shoulder",
                format="%.3f",
                help_text="Controls the roll-off of the whites, recovering peak highlight detail.",
            )
