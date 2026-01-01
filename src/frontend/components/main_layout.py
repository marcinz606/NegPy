import streamlit as st
import numpy as np
from typing import List, Any, Dict
from PIL import Image
from src.backend.image_logic.histogram import plot_histogram
from src.frontend.components.navigation import render_navigation
from src.frontend.components.contact_sheet import render_contact_sheet
from src.frontend.components.image_view import render_image_view

def render_main_layout(uploaded_files: List[Any], pil_prev: Image.Image, sidebar_data: Dict[str, Any]):
    """
    Renders the main content area with a two-column layout:
    Left: Navigation & Actions + Vertical Contact Sheet
    Right: Preview and Histogram
    """
    main_col1, main_col2 = st.columns([1, 5])

    with main_col1:
        export_btn_sidebar = render_navigation(uploaded_files)
        st.pyplot(plot_histogram(np.array(pil_prev.convert("RGB")), figsize=(3, 1.8), dpi=150), use_container_width=True)
        render_contact_sheet(uploaded_files)

    with main_col2:
        # Main UI Render (Preview)
        render_image_view(
            pil_prev,
            border_config={
                'add_border': sidebar_data.get('add_border', False),
                'size_cm': sidebar_data.get('border_size', 0.2),
                'color': sidebar_data.get('border_color', '#000000'),
                'print_width_cm': sidebar_data.get('print_width', 27.0)
            }
        )
    
    return export_btn_sidebar
