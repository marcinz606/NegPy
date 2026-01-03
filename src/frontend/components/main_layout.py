import streamlit as st
import numpy as np
from PIL import Image
from src.domain_objects import SidebarData
from src.backend.image_logic.plots import plot_histogram, plot_photometric_curve
from src.frontend.components.navigation import render_navigation
from src.frontend.components.contact_sheet import render_contact_sheet
from src.frontend.components.image_view import render_image_view


def render_main_layout(pil_prev: Image.Image, sidebar_data: SidebarData) -> bool:
    """
    Renders the main content area with a two-column layout:
    Left: Navigation, Plots, and Vertical Contact Sheet
    Right: Preview
    """
    from src.frontend.main import get_processing_params

    current_params = get_processing_params(st.session_state)

    main_col1, main_col2 = st.columns([1, 5])

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

        st.divider()
        render_contact_sheet()

    with main_col2:
        # 2. Main UI Render (Preview)
        render_image_view(pil_prev, border_config=sidebar_data)

    return export_btn_sidebar
