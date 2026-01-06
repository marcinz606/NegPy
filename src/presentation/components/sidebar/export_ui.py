import streamlit as st
from src.config import APP_CONFIG
from src.presentation.state.view_models import SidebarState
from src.presentation.components.sidebar.helpers import st_init, render_control_slider


def render_export_section() -> SidebarState:
    """
    Renders the 'Export' section of the sidebar.
    """
    with st.expander(":material/file_export: Export", expanded=True):
        c1, c2 = st.columns(2)
        st_init("export_fmt", "JPEG")
        c1.selectbox("Format", ["JPEG", "TIFF"], key="export_fmt")

        st_init("export_color_space", "sRGB")
        color_options = ["sRGB", "Adobe RGB", "Greyscale"]
        c2.selectbox(
            "Color Space",
            color_options,
            key="export_color_space",
            help="Select color space of export file. sRGB is best for screen, AdobeRGB for print and Greyscale for B&W (not toned) prints.",
        )

        c1, c2 = st.columns(2)
        with c1:
            render_control_slider(
                label="Size (cm)",
                min_val=10.0,
                max_val=60.0,
                default_val=27.0,
                step=0.5,
                key="export_size",
                help_text="Longer dimension of the print.",
            )

        with c2:
            render_control_slider(
                label="DPI",
                min_val=100.0,
                max_val=1600.0,
                default_val=300.0,
                step=100.0,
                key="export_dpi",
                format="%d",
                help_text="Desired DPI (dots per inch) resolution for printing.",
            )

        st_init("export_path", APP_CONFIG.default_export_dir)
        st.text_input(
            "Export Directory",
            value=st.session_state.get("export_path", APP_CONFIG.default_export_dir),
            key="export_path",
        )

        c_b1, c_b2 = st.columns([2, 1])
        with c_b1:
            render_control_slider(
                label="Border Size (cm)",
                min_val=0.0,
                max_val=2.5,
                default_val=0.5,
                step=0.05,
                key="export_border_size",
                help_text=(
                    "Border width (cm). When border is added we retain our total print size, "
                    "actual image gets scaled down. Set to 0 to disable."
                ),
            )

        st_init("export_border_color", "#ffffff")
        c_b2.color_picker(
            "Border Color",
            key="export_border_color",
            help="Color (hex) of the added border.",
        )

    return SidebarState(
        out_fmt=st.session_state.export_fmt,
        color_space=st.session_state.export_color_space,
        print_width=float(st.session_state.export_size),
        print_dpi=int(st.session_state.export_dpi),
        export_path=st.session_state.export_path,
        add_border=float(st.session_state.export_border_size) > 0,
        border_size=float(st.session_state.export_border_size),
        border_color=st.session_state.export_border_color,
        apply_icc=bool(st.session_state.get("apply_icc", False)),
        process_btn=False,
    )
