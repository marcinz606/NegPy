import streamlit as st
from src.config import APP_CONFIG
from src.domain_objects import SidebarData


def render_export_section() -> SidebarData:
    """
    Renders the 'Export' section of the sidebar.
    Returns a dictionary of export parameters.
    """
    with st.expander(":material/file_export: Export", expanded=True):
        c1, c2 = st.columns(2)
        c1.selectbox("Format", ["JPEG", "TIFF"], key="export_fmt")

        color_options = ["sRGB", "Adobe RGB", "Greyscale"]
        current_cs = st.session_state.get("export_color_space", "sRGB")
        cs_idx = color_options.index(current_cs) if current_cs in color_options else 0

        c2.selectbox(
            "Color Space",
            color_options,
            index=cs_idx,
            key="export_color_space",
            help="Select color space of export file. sRGB is best for screen, AdobeRGB for print and Greyscale for B&W (not toned) prints.",
        )

        c1, c2 = st.columns(2)
        c1.number_input(
            "Size (cm)",
            value=st.session_state.get("export_size", 27.0),
            min_value=10.0,
            help="Longer dimension.",
            key="export_size",
        )
        c2.number_input(
            "DPI",
            value=st.session_state.get("export_dpi", 300),
            key="export_dpi",
            help="Desired DPI (dots per inch) resolution for printing.",
        )

        st.text_input(
            "Export Directory",
            st.session_state.get("export_path", APP_CONFIG.default_export_dir),
            key="export_path",
        )

        c_b1, c_b2 = st.columns([2, 1])
        c_b1.slider(
            "Border Size (cm)",
            min_value=0.0,
            max_value=2.5,
            value=st.session_state.get("export_border_size", 0.0),
            step=0.05,
            key="export_border_size",
            help="Border width (cm). When border is added we retain our total print size, actual image gets scaled down. Set to 0 to disable.",
        )
        c_b2.color_picker(
            "Border Color",
            value=st.session_state.get("export_border_color", "#ffffff"),
            key="export_border_color",
            help="Color (hex) of the added border.",
        )

    return SidebarData(
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
