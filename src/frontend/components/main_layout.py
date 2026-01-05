import streamlit as st
import numpy as np
from PIL import Image
from typing import Any, Tuple
from src.domain_objects import SidebarData
from src.backend.image_logic.plots import plot_histogram, plot_photometric_curve
from src.frontend.components.navigation import render_navigation
from src.frontend.components.contact_sheet import render_contact_sheet
from src.frontend.components.image_view import render_image_view


def render_layout_header() -> Tuple[Any, Any, Any]:
    """
    Initializes the main two-column layout and renders the title.
    Returns (main_col1, main_col2, status_area).
    """
    main_col1, main_col2 = st.columns([1, 5])
    with main_col1:
        st.title(":red[:material/camera_roll:] DarkroomPy")
        # reserve space in order to avoid shifting the layout when msg pops up.
        status_container = st.container(height=48, border=False)
        status_area = status_container.empty()
    return main_col1, main_col2, status_area


def render_main_layout(
    pil_prev: Image.Image,
    sidebar_data: SidebarData,
    main_col1: Any,
    main_col2: Any,
) -> bool:
    """
    Renders the remaining main content area into the provided columns.
    """
    from src.frontend.main import get_processing_params

    current_params = get_processing_params(st.session_state)

    with main_col1:
        export_btn_sidebar = render_navigation()

        st.caption(
            "Histogram",
            help=(
                "Visualizes the tonal distribution of the processed print. "
                "The horizontal axis represents brightness from Shadows (left) to Highlights (right), "
                "while the vertical axis shows the frequency of pixels at each level."
            ),
        )
        st.pyplot(
            plot_histogram(
                np.array(pil_prev.convert("RGB")), figsize=(3, 1.4), dpi=150
            ),
            width="stretch",
        )

        st.caption(
            "Photometric Curve",
            help=(
                "This H&D Characteristic Curve represents the relationship between Subject Brightness and Print Density. "
                "It visualizes how the engine simulates light-sensitive paper: 'Density' shifts the exposure, "
                "'Grade' controls contrast slope, while 'Toe' and 'Shoulder' manage the roll-off in highlights "
                "and shadows respectively."
            ),
        )
        st.pyplot(
            plot_photometric_curve(current_params, figsize=(3, 1.4), dpi=150),
            width="stretch",
        )

        render_contact_sheet()

    with main_col2:
        # 2. Main UI Render (Preview)
        render_image_view(pil_prev, border_config=sidebar_data)

    return export_btn_sidebar
