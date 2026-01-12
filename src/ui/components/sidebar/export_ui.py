import streamlit as st
import os
from src.kernel.system.config import APP_CONFIG, DEFAULT_WORKSPACE_CONFIG
from src.ui.state.view_models import SidebarState
from src.ui.components.sidebar.helpers import (
    render_control_slider,
    render_control_selectbox,
    render_control_text_input,
    render_control_color_picker,
)
from src.infrastructure.loaders.native_picker import NativeFilePicker


def render_export_section() -> SidebarState:
    with st.expander("Export", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            render_control_selectbox(
                "Format",
                ["JPEG", "TIFF"],
                default_val=DEFAULT_WORKSPACE_CONFIG.export.export_fmt,
                key="export_fmt",
            )

        color_options = ["sRGB", "Adobe RGB", "Greyscale"]
        with c2:
            render_control_selectbox(
                "Color Space",
                color_options,
                default_val=DEFAULT_WORKSPACE_CONFIG.export.export_color_space,
                key="export_color_space",
                help_text="sRGB: screen, AdobeRGB: print, Greyscale: B&W.",
            )

        c1, c2 = st.columns(2)
        with c1:
            render_control_slider(
                label="Size (cm)",
                min_val=10.0,
                max_val=60.0,
                default_val=DEFAULT_WORKSPACE_CONFIG.export.export_print_size,
                step=0.5,
                key="export_print_size",
                help_text="Long dimension (cm).",
            )

        with c2:
            render_control_slider(
                label="DPI",
                min_val=100.0,
                max_val=1600.0,
                default_val=DEFAULT_WORKSPACE_CONFIG.export.export_dpi,
                step=100.0,
                key="export_dpi",
                format="%d",
                help_text="Print resolution (dots per inch).",
            )

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            render_control_slider(
                label="Border Size (cm)",
                min_val=0.0,
                max_val=2.5,
                default_val=DEFAULT_WORKSPACE_CONFIG.export.export_border_size,
                step=0.05,
                key="export_border_size",
                help_text="Total size is preserved (image scales down). 0 = off.",
            )

        with c_b2:
            render_control_color_picker(
                "Border Color",
                default_val=DEFAULT_WORKSPACE_CONFIG.export.export_border_color,
                key="export_border_color",
                help_text="Color (hex) of the added border.",
            )

        render_control_text_input(
            "Filename Pattern",
            default_val=DEFAULT_WORKSPACE_CONFIG.export.filename_pattern,
            key="filename_pattern",
            help_text="Jinja2 template. Available: {{ original_name }}, {{ date }}, {{ mode }}, {{ fmt }}, {{ colorspace }}, {{ border }}",
        )

        is_docker = os.path.exists("/.dockerenv")
        if not is_docker:
            c_path, c_btn = st.columns([0.8, 0.2])
            with c_path:
                render_control_text_input(
                    "Export Directory",
                    default_val=APP_CONFIG.default_export_dir,
                    key="export_path",
                )
            with c_btn:
                st.write("##")  # Spacer to align with text input
                if st.button(":material/folder_open:", help="Pick export folder"):
                    picker = NativeFilePicker()
                    new_path = picker.pick_export_folder(
                        initial_dir=st.session_state.get("export_path")
                    )
                    if new_path:
                        st.session_state.export_path = new_path
                        st.rerun()
        else:
            render_control_text_input(
                "Export Directory",
                default_val=APP_CONFIG.default_export_dir,
                key="export_path",
            )

    return SidebarState(
        out_fmt=st.session_state.export_fmt,
        color_space=st.session_state.export_color_space,
        print_width=float(st.session_state.export_print_size),
        print_dpi=int(st.session_state.export_dpi),
        export_path=st.session_state.export_path,
        add_border=float(st.session_state.export_border_size) > 0,
        border_size=float(st.session_state.export_border_size),
        border_color=st.session_state.export_border_color,
        filename_pattern=st.session_state.filename_pattern,
        apply_icc=bool(st.session_state.get("apply_icc", False)),
        process_btn=False,
    )
