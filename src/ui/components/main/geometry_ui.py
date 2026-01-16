import streamlit as st
from src.domain.constants import SUPPORTED_ASPECT_RATIOS
from src.features.geometry.models import CropMode
from src.ui.state.view_models import GeometryViewModel
from src.ui.components.sidebar.helpers import (
    render_control_slider,
    render_control_checkbox,
    render_control_selectbox,
)
from src.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from src.ui.state.state_manager import save_settings


def render_geometry_section() -> None:
    geo_vm = GeometryViewModel()
    geo_conf = geo_vm.to_config()

    with st.container(border=True):
        # Row 1: Crop Mode, Ratio/Pick, Assist/Reset, Keep Borders, Offset, Rotation
        c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.2, 1.2, 1.2, 2, 2])

        with c1:
            crop_modes = list(CropMode)
            current_mode = geo_vm.crop_mode

            selected_label = st.selectbox(
                "Crop Mode",
                options=[m.value for m in crop_modes],
                index=crop_modes.index(current_mode),
                key="geometry_crop_mode_select",
                help="Select cropping method.",
            )

            selected_mode = CropMode(selected_label)

            if selected_mode != current_mode:
                geo_vm.crop_mode = selected_mode
                save_settings()
                st.rerun()

        if geo_vm.crop_mode == CropMode.MANUAL:
            with c2:
                render_control_checkbox(
                    "Pick Crop",
                    default_val=False,
                    key=geo_vm.get_key("pick_manual_crop"),
                    is_toggle=True,
                    help_text="Click top-left and then bottom-right corner.",
                )
            with c3:
                if geo_conf.manual_crop_rect is not None:
                    st.write("##")  # Align with selectbox
                    if st.button(
                        "Reset Crop", key="reset_manual_crop_btn", width="stretch"
                    ):
                        st.session_state[geo_vm.get_key("manual_crop_rect")] = None
                        save_settings()
                        st.rerun()
        else:
            with c2:
                render_control_selectbox(
                    "Ratio",
                    SUPPORTED_ASPECT_RATIOS,
                    default_val=DEFAULT_WORKSPACE_CONFIG.geometry.autocrop_ratio,
                    key=geo_vm.get_key("autocrop_ratio"),
                    help_text="Aspect ratio to crop to.",
                )
            with c3:
                # Show Clear Assist if active, otherwise Pick Assist toggle
                if geo_conf.autocrop_assist_luma is not None:
                    st.write("##")  # Align with selectbox
                    if st.button(
                        "Clear Assist", key="clear_assist_btn", width="stretch"
                    ):
                        st.session_state[geo_vm.get_key("autocrop_assist_point")] = None
                        st.session_state[geo_vm.get_key("autocrop_assist_luma")] = None
                        save_settings()
                        st.rerun()
                else:
                    render_control_checkbox(
                        "Pick Assist",
                        default_val=False,
                        key=geo_vm.get_key("pick_assist"),
                        is_toggle=True,
                        help_text="Click on the film border to assist detection.",
                    )

        with c4:
            render_control_checkbox(
                "Keep Borders",
                default_val=DEFAULT_WORKSPACE_CONFIG.geometry.keep_full_frame,
                key=geo_vm.get_key("keep_full_frame"),
                help_text="Keep image and film borders.",
            )

        with c5:
            render_control_slider(
                label="Crop Offset",
                min_val=-20.0,
                max_val=100.0,
                default_val=4.0,
                step=1.0,
                key=geo_vm.get_key("autocrop_offset"),
                format="%d",
                help_text="Buffer/offset (pixels) to crop beyond detected border.",
            )
        with c6:
            render_control_slider(
                label="Fine Rotation (°)",
                min_val=-5.0,
                max_val=5.0,
                default_val=0.0,
                step=0.05,
                key=geo_vm.get_key("fine_rotation"),
            )
