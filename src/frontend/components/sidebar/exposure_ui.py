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
            "Density",
            -1.0,
            3.0,
            1.0,
            0.05,
            "density",
            help_text=(
                "Print Density. 1.0 is Normal. Higher = Darker."
            ),
        )

        render_control_slider(
            "Grade",
            0.0,
            5.0,
            2.0,
            0.05,
            "grade",
            help_text=(
                "Paper Contrast Grade. 2 is Normal. 0 is Soft. 5 is Hard."
            ),
        )

        render_control_slider(
            "Shoulder (Shadows)",
            0.0,
            1.0,
            0.0,
            0.05,
            "shoulder",
            help_text="Softens the Shadow roll-off (D-max approach). Higher = More shadow detail (flatter blacks).",
        )

        render_control_slider(
            "Toe (Highlights)",
            0.0,
            1.0,
            0.0,
            0.05,
            "toe",
            help_text="Softens the Highlight roll-off (D-min approach). Higher = Creamier highlights (less clipping).",
        )

        # Toe/Shoulder currently disabled in new Photometric Pipeline (Sigmoid has natural rolloff)
        # c_gain1, c_gain2 = st.columns(2)
        # with c_gain1:
        #     render_control_slider(
        #         "Shadow Toe",
        #         0.0,
        #         0.5,
        #         0.0,
        #         0.005,
        #         "scan_gain_s_toe",
        #         format="%.3f",
        #         help_text="Separates and lifts the deepest shadows to prevent a muddy look.",
        #     )
        # with c_gain2:
        #     render_control_slider(
        #         "Highlight Shoulder",
        #         0.0,
        #         0.5,
        #         0.0,
        #         0.005,
        #         "scan_gain_h_shoulder",
        #         format="%.3f",
        #         help_text="Controls the roll-off of the whites, recovering peak highlight detail.",
        #     )
