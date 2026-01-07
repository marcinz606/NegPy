import streamlit as st
from src.presentation.state.view_models import ExposureViewModel
from src.presentation.components.sidebar.helpers import render_control_slider


def render_exposure_section() -> None:
    """
    Renders the 'Exposure & Tonality' section of the sidebar.
    """
    vm = ExposureViewModel()

    with st.expander(":material/camera: Exposure & Tonality", expanded=True):
        if not vm.is_bw:
            c1, c2, c3 = st.columns(3)
            with c1:
                render_control_slider(
                    label=":blue-badge[Cyan]",
                    min_val=-1.0,
                    max_val=1.0,
                    default_val=0.0,
                    step=0.01,
                    key=vm.get_key("wb_cyan"),
                    help_text="Cyan filtration (removes Red cast).",
                )
            with c2:
                render_control_slider(
                    label=":violet-badge[Magenta]",
                    min_val=-1.0,
                    max_val=1.0,
                    default_val=0.0,
                    step=0.01,
                    key=vm.get_key("wb_magenta"),
                    help_text="Magenta filtration (removes Green cast).",
                )
            with c3:
                render_control_slider(
                    label=":orange-badge[Yellow]",
                    min_val=-1.0,
                    max_val=1.0,
                    default_val=0.0,
                    step=0.01,
                    key=vm.get_key("wb_yellow"),
                    help_text="Yellow filtration (removes Blue cast).",
                )

        # Primary Print Controls
        e1, e2 = st.columns(2)
        with e1:
            render_control_slider(
                label="Density",
                min_val=-1.0,
                max_val=3.0,
                default_val=1.0,
                step=0.01,
                key=vm.get_key("density"),
                help_text="Print Density. 1.0 is Neutral. Higher = Darker.",
            )

        with e2:
            render_control_slider(
                label="Grade",
                min_val=0.0,
                max_val=5.0,
                default_val=2.5,
                step=0.01,
                key=vm.get_key("grade"),
                help_text="Paper Contrast Grade. 2 is Neutral. 0 is Soft. 5 is Hard.",
            )

        # Tone Curve Controls
        c_toe1, c_toe2, c_toe3 = st.columns([1.5, 1, 1])
        with c_toe1:
            render_control_slider(
                label="Toe (S roll-off)",
                min_val=-1.0,
                max_val=1.0,
                default_val=0.0,
                step=0.01,
                key=vm.get_key("toe"),
            )
        with c_toe2:
            render_control_slider(
                label="Reach",
                min_val=1.0,
                max_val=10.0,
                default_val=3.0,
                step=0.05,
                key=vm.get_key("toe_width"),
            )
        with c_toe3:
            render_control_slider(
                label="Hardness",
                min_val=0.1,
                max_val=5.0,
                default_val=1.0,
                step=0.01,
                key=vm.get_key("toe_hardness"),
            )

        c_sh1, c_sh2, c_sh3 = st.columns([1.5, 1, 1])
        with c_sh1:
            render_control_slider(
                label="Shoulder (H roll-off)",
                min_val=-1.0,
                max_val=1.0,
                default_val=0.0,
                step=0.01,
                key=vm.get_key("shoulder"),
            )
        with c_sh2:
            render_control_slider(
                label="Reach",
                min_val=1.0,
                max_val=10.0,
                default_val=3.0,
                step=0.01,
                key=vm.get_key("shoulder_width"),
            )
        with c_sh3:
            render_control_slider(
                label="Hardness",
                min_val=0.1,
                max_val=5.0,
                default_val=1.0,
                step=0.01,
                key=vm.get_key("shoulder_hardness"),
            )
