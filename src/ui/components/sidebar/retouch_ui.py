import streamlit as st
from src.ui.state.view_models import RetouchViewModel
from src.ui.state.state_manager import save_settings
from src.ui.components.sidebar.helpers import (
    render_control_slider,
    render_control_checkbox,
)


def render_retouch_section() -> None:
    vm = RetouchViewModel()

    with st.expander(":material/ink_eraser: Retouch", expanded=True):
        render_control_checkbox(
            "Automatic dust removal", default_val=False, key=vm.get_key("dust_remove")
        )

        if st.session_state.get(vm.get_key("dust_remove")):
            c1, c2 = st.columns(2)

            with c1:
                render_control_slider(
                    label="Threshold",
                    min_val=0.01,
                    max_val=1.0,
                    default_val=0.8,
                    step=0.01,
                    key=vm.get_key("dust_threshold"),
                    help_text="Dust detection sensitivity.",
                )

            with c2:
                render_control_slider(
                    label="Size",
                    min_val=2.0,
                    max_val=20.0,
                    default_val=3.0,
                    step=1.0,
                    key=vm.get_key("dust_size"),
                    format="%d",
                    help_text="Max spot diameter.",
                )

        c1, c2 = st.columns([2, 1])
        with c1:
            render_control_checkbox(
                "Manual Dust Correction", default_val=False, key=vm.get_key("pick_dust")
            )

        manual_spots_key = vm.get_key("manual_dust_spots")
        manual_spots = st.session_state.get(manual_spots_key)
        if manual_spots is not None and len(manual_spots) > 0:
            c2.caption(f"{len(manual_spots)} spots")

        if st.session_state.get(vm.get_key("pick_dust")):
            render_control_slider(
                label="Manual Spot Size",
                min_val=2.0,
                max_val=50.0,
                default_val=5.0,
                step=1.0,
                key=vm.get_key("manual_dust_size"),
                format="%d",
            )
            render_control_checkbox(
                "Scratch Mode (Click Start -> Click End)",
                default_val=False,
                key=vm.get_key("dust_scratch_mode"),
            )
            render_control_checkbox(
                "Show Patches", default_val=False, key=vm.get_key("show_dust_patches")
            )

            c1, c2 = st.columns(2)
            if c1.button("Undo Last", width="stretch"):
                if manual_spots:
                    manual_spots.pop()
                    save_settings()
                    st.rerun()
            if c2.button("Clear All", width="stretch"):
                st.session_state[manual_spots_key] = []
                save_settings()
                st.rerun()
