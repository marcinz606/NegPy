import streamlit as st
from src.presentation.state.view_models import RetouchViewModel
from src.presentation.state.state_manager import save_settings


def render_retouch_section() -> None:
    """
    Renders the 'Retouch' section of the sidebar.
    """
    vm = RetouchViewModel()

    with st.expander(":material/brush: Retouch", expanded=True):
        st.checkbox("Automatic dust removal", key=vm.get_key("dust_remove"))
        c1, c2 = st.columns(2)
        c1.slider(
            "Threshold",
            0.01,
            1.0,
            0.55,
            0.01,
            disabled=not st.session_state.get(vm.get_key("dust_remove")),
            key=vm.get_key("dust_threshold"),
            help="Sensitivity of automatic dust detection. Lower values detect more spots.",
        )
        c2.slider(
            "Size",
            1,
            20,
            2,
            1,
            disabled=not st.session_state.get(vm.get_key("dust_remove")),
            key=vm.get_key("dust_size"),
            help="Maximum size of spots to be automatically removed.",
        )

        c1, c2 = st.columns([2, 1])
        c1.checkbox("Manual Dust Correction", key="pick_dust")
        manual_spots_key = vm.get_key("manual_dust_spots")
        manual_spots = st.session_state.get(manual_spots_key)
        if manual_spots is not None and len(manual_spots) > 0:
            c2.caption(f"{len(manual_spots)} spots")

        if st.session_state.get("pick_dust"):
            st.slider(
                "Manual Spot Size", 1, 50, 4, 1, key=vm.get_key("manual_dust_size")
            )
            st.checkbox(
                "Scratch Mode (Click Start -> Click End)", key="dust_scratch_mode"
            )
            st.checkbox("Show Patches", value=True, key="show_dust_patches")
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
