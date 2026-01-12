import streamlit as st
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

    with st.expander(":material/crop: Geometry", expanded=True):
        c_main1, c_main2, c_main3 = st.columns([1, 1, 1.2])
        with c_main1:
            render_control_selectbox(
                "Ratio",
                ["3:2", "4:3", "5:4", "6:7", "1:1", "65:24"],
                default_val=DEFAULT_WORKSPACE_CONFIG.geometry.autocrop_ratio,
                key=geo_vm.get_key("autocrop_ratio"),
                label_visibility="collapsed",
                help_text="Aspect ratio to crop to.",
            )
        with c_main2:
            render_control_checkbox(
                "Auto-Crop",
                default_val=DEFAULT_WORKSPACE_CONFIG.geometry.autocrop,
                key=geo_vm.get_key("autocrop"),
                help_text="Automatically detect film borders and crop to desired aspect ratio.",
            )
        with c_main3:
            render_control_checkbox(
                "Keep Borders",
                default_val=DEFAULT_WORKSPACE_CONFIG.geometry.keep_full_frame,
                key=geo_vm.get_key("keep_full_frame"),
                help_text="Keep entire image and film borders in final export. "
                "Crop area is still used for internal analysis.",
            )

        c_a1, c_a2 = st.columns(2)
        with c_a1:
            render_control_checkbox(
                "Pick Assist",
                default_val=False,
                key=geo_vm.get_key("pick_assist"),
                is_toggle=True,
                help_text="Click on the film border (unexposed area) in the preview to assist crop detection.",
            )
        with c_a2:
            if geo_conf.autocrop_assist_luma is not None:
                if st.button("Clear Assist", use_container_width=True):
                    st.session_state[geo_vm.get_key("autocrop_assist_point")] = None
                    st.session_state[geo_vm.get_key("autocrop_assist_luma")] = None
                    save_settings()
                    st.rerun()

        c_geo1, c_geo2 = st.columns(2)
        with c_geo1:
            render_control_slider(
                label="Crop Offset",
                min_val=-20.0,
                max_val=100.0,
                default_val=4.0,
                step=1.0,
                key=geo_vm.get_key("autocrop_offset"),
                format="%d",
                help_text="Buffer/offset (pixels) to crop beyond automatically detected border. "
                "Positive values crop IN, negative values expand OUT.",
            )
        with c_geo2:
            render_control_slider(
                label="Fine Rotation (°)",
                min_val=-5.0,
                max_val=5.0,
                default_val=0.0,
                step=0.05,
                key=geo_vm.get_key("fine_rotation"),
            )
