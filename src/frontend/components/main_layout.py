import streamlit as st
import numpy as np
from typing import List, Dict
from PIL import Image
from src.domain_objects import SidebarData
from src.backend.image_logic.histogram import plot_histogram
from src.frontend.components.navigation import render_navigation
from src.frontend.components.contact_sheet import render_contact_sheet
from src.frontend.components.image_view import render_image_view


def render_main_layout(
    uploaded_files: List[Dict[str, str]],
    pil_prev: Image.Image,
    sidebar_data: SidebarData,
) -> bool:
    """
    Renders the main content area with a two-column layout:
    Left: Navigation & Actions + Vertical Contact Sheet
    Right: Preview and Histogram
    """
    main_col1, main_col2 = st.columns([1, 5])

    with main_col1:
        export_btn_sidebar = render_navigation(uploaded_files)
        st.pyplot(
            plot_histogram(
                np.array(pil_prev.convert("RGB")), figsize=(3, 1.8), dpi=150
            ),
            width="stretch",
        )
        render_contact_sheet(uploaded_files)

    with main_col2:
        # Main UI Render (Preview)
        render_image_view(pil_prev, border_config=sidebar_data)

    return export_btn_sidebar
