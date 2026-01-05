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
                -1.0,
                1.0,
                0.0,
                0.01,
                "wb_cyan",
                help_text=(
                    "Adds Cyan filtration (removes Red cast). Like in darkroom, "
                    "you SHOULD NOT be touching this unless you know what you are doing :)"
                ),
            )
            render_control_slider(
                ":violet-badge[Magenta]",
                -1.0,
                1.0,
                0.0,
                0.01,
                "wb_magenta",
                help_text="Adds Magenta filtration (removes Green cast).",
            )
            render_control_slider(
                ":orange-badge[Yellow]",
                -1.0,
                1.0,
                0.0,
                0.01,
                "wb_yellow",
                help_text="Adds Yellow filtration (removes Blue cast).",
            )

        # Primary Print Controls
        e1, e2 = st.columns(2)
        with e1:
            render_control_slider(
                "Density",
                -1.0,
                3.0,
                1.0,
                0.01,
                "density",
                help_text=("Print Density. 1.0 is Neutral. Higher = Darker."),
            )

        with e2:
            render_control_slider(
                "Grade",
                0.0,
                5.0,
                2.5,
                0.01,
                "grade",
                help_text=("Paper Contrast Grade. 2 is Neutral. 0 is Soft. 5 is Hard."),
            )

        c_toe1, c_toe2, c_toe3 = st.columns([1.5, 1, 1])
        c_sh1, c_sh2, c_sh3 = st.columns([1.5, 1, 1])

        with c_toe1:
            render_control_slider(
                "Toe (S roll-off)",
                -1.0,
                1.0,
                0.0,
                0.01,
                "toe",
                help_text="Controls Shadow roll-off. (+) Softens/lifts shadows, (-) Hardens/crushes shadows.",
            )
        with c_toe2:
            render_control_slider(
                "Reach",
                1.0,
                10.0,
                3.0,
                0.05,
                "toe_width",
                help_text="Controls how far into the midtones the shadow roll-off reaches.",
            )
        with c_toe3:
            render_control_slider(
                "Hardness",
                0.1,
                5.0,
                1.0,
                0.01,
                "toe_hardness",
                help_text="Transition shape. Higher = 'Snaps' to roll-off late (hard knee), Lower = Starts earlier and lazier (soft knee).",
            )

        with c_sh1:
            render_control_slider(
                "Shoulder (H roll-off)",
                -1.0,
                1.0,
                0.0,
                0.01,
                "shoulder",
                help_text="Controls Highlight roll-off. (+) Softens/creams highlights, (-) Hardens/punches highlights.",
            )
        with c_sh2:
            render_control_slider(
                "Reach",
                1.0,
                10.0,
                3.0,
                0.01,
                "shoulder_width",
                help_text="Controls how far into the midtones the highlight roll-off reaches.",
            )
        with c_sh3:
            render_control_slider(
                "Hardness",
                0.1,
                5.0,
                1.0,
                0.01,
                "shoulder_hardness",
                help_text="Transition shape. Higher = 'Snaps' to roll-off late (hard knee), Lower = Starts earlier and lazier (soft knee).",
            )
