import streamlit as st
from src.presentation.state.view_models import ExposureViewModel


def render_control_slider(
    label: str,
    min_val: float,
    max_val: float,
    default: float,
    step: float,
    key: str,
    help_text: str = "",
) -> None:
    """
    Helper from original codebase, adapted slightly.
    """
    st.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        value=st.session_state.get(key, default),
        step=step,
        key=key,
        help=help_text,
    )


def render_exposure_view(vm: ExposureViewModel) -> None:
    """
    Renders the Exposure & Tonality section using the ViewModel.
    """
    with st.expander(":material/camera: Exposure & Tonality", expanded=True):
        if not vm.is_bw:
            c1, c2, c3 = st.columns(3)
            with c1:
                render_control_slider(
                    ":blue-badge[Cyan]",
                    -1.0,
                    1.0,
                    0.0,
                    0.01,
                    vm.get_key("wb_cyan"),
                    help_text="Cyan filtration (removes Red cast).",
                )
            with c2:
                render_control_slider(
                    ":violet-badge[Magenta]",
                    -1.0,
                    1.0,
                    0.0,
                    0.01,
                    vm.get_key("wb_magenta"),
                    help_text="Magenta filtration (removes Green cast).",
                )
            with c3:
                render_control_slider(
                    ":orange-badge[Yellow]",
                    -1.0,
                    1.0,
                    0.0,
                    0.01,
                    vm.get_key("wb_yellow"),
                    help_text="Yellow filtration (removes Blue cast).",
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
                vm.get_key("density"),
                help_text="Print Density. 1.0 is Neutral. Higher = Darker.",
            )

        with e2:
            render_control_slider(
                "Grade",
                0.0,
                5.0,
                2.5,
                0.01,
                vm.get_key("grade"),
                help_text="Paper Contrast Grade. 2 is Neutral. 0 is Soft. 5 is Hard.",
            )

        # Tone Curve Controls
        c_toe1, c_toe2, c_toe3 = st.columns([1.5, 1, 1])
        with c_toe1:
            render_control_slider(
                "Toe (S roll-off)", -1.0, 1.0, 0.0, 0.01, vm.get_key("toe")
            )
        with c_toe2:
            render_control_slider(
                "Reach", 1.0, 10.0, 3.0, 0.05, vm.get_key("toe_width")
            )
        with c_toe3:
            render_control_slider(
                "Hardness", 0.1, 5.0, 1.0, 0.01, vm.get_key("toe_hardness")
            )

        c_sh1, c_sh2, c_sh3 = st.columns([1.5, 1, 1])
        with c_sh1:
            render_control_slider(
                "Shoulder (H roll-off)", -1.0, 1.0, 0.0, 0.01, vm.get_key("shoulder")
            )
        with c_sh2:
            render_control_slider(
                "Reach", 1.0, 10.0, 3.0, 0.01, vm.get_key("shoulder_width")
            )
        with c_sh3:
            render_control_slider(
                "Hardness", 0.1, 5.0, 1.0, 0.01, vm.get_key("shoulder_hardness")
            )
