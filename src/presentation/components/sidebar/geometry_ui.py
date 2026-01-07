import streamlit as st
from src.presentation.state.view_models import GeometryViewModel
from src.presentation.components.sidebar.helpers import (
    render_control_slider,
    render_control_checkbox,
    render_control_selectbox,
)
from src.config import DEFAULT_WORKSPACE_CONFIG


def render_geometry_section() -> None:
    """
    Renders the Geometry/Auto-Crop section of the sidebar.
    """
    geo_vm = GeometryViewModel()

    with st.expander(":material/crop: Geometry", expanded=True):
        autocrop = render_control_checkbox(
            "Auto-Crop",
            default_val=DEFAULT_WORKSPACE_CONFIG.geometry.autocrop,
            key=geo_vm.get_key("autocrop"),
            help_text="Automatically detect film borders and crop to desired aspect ratio.",
        )
    if autocrop:
        c_geo1, c_geo2, c_geo3 = st.columns(3)
        with c_geo1:
            render_control_selectbox(
                "Ratio",
                ["3:2", "4:3", "5:4", "6:7", "1:1", "65:24"],
                default_val=DEFAULT_WORKSPACE_CONFIG.geometry.autocrop_ratio,
                key=geo_vm.get_key("autocrop_ratio"),
                label_visibility="collapsed",
                help_text="Aspect ratio to crop to.",
            )

        with c_geo2:
            render_control_slider(
                label="Crop Offset",
                min_val=0.0,
                max_val=100.0,
                default_val=1.0,
                step=1.0,
                key=geo_vm.get_key("autocrop_offset"),
                format="%d",
                help_text="Buffer/offset (pixels) to crop beyond automatically detected border, might be useful when border is uneven.",
            )

        with c_geo3:
            render_control_slider(
                label="Fine Rotation (°)",
                min_val=-5.0,
                max_val=5.0,
                default_val=0.0,
                step=0.05,
                key=geo_vm.get_key("fine_rotation"),
            )
