import streamlit as st
from PIL import Image
from typing import Any, Tuple
from src.presentation.state.session_context import SessionContext
from src.presentation.state.view_models import SidebarState
from src.presentation.layouts.contact_sheet import render_contact_sheet
from src.presentation.layouts.image_view import render_image_view


def render_layout_header(ctx: SessionContext) -> Tuple[Any, Any]:
    """
    Initializes the main layout and returns (main_area, status_area).
    """
    # 1. Determine Orientation & Sync Size
    rotation = st.session_state.get("rotation", 0)
    h_orig, w_orig = ctx.original_res

    # Apply rotation logic to dimensions
    if abs(rotation) % 180 != 0:
        h_orig, w_orig = w_orig, h_orig

    is_vertical = h_orig > w_orig
    target_key = (
        "working_copy_size_vertical" if is_vertical else "working_copy_size_horizontal"
    )

    # Ensure the slider reflects the correct stored value for this orientation
    if target_key in st.session_state:
        if st.session_state.working_copy_size != st.session_state[target_key]:
            st.session_state.working_copy_size = st.session_state[target_key]

    def update_orientation_size() -> None:
        """Callback to save the slider value to the orientation-specific key."""
        st.session_state[target_key] = st.session_state.working_copy_size

    main_area = st.container()
    with main_area:
        c_status, c_slider = st.columns([8, 1])
        with c_status:
            # reserve space in order to avoid shifting the layout when msg pops up.
            status_container = st.container(height=48, border=False)
            status_area = status_container.empty()

        with c_slider:
            st.slider(
                "Display Size",
                800,
                2800,
                step=100,
                key="working_copy_size",
                on_change=update_orientation_size,
                # label_visibility="collapsed",
                help="Scaling of the preview image in the browser. Does not affect internal processing resolution.",
            )

    return main_area, status_area


def render_main_layout(
    pil_prev: Image.Image,
    sidebar_data: SidebarState,
    main_area: Any,
) -> None:
    """
    Renders the image preview and a collapsible contact sheet fixed at the bottom.
    """
    with main_area:
        # 1. Main UI Render (Preview)
        # Wrap in a container that has padding-bottom to avoid being covered by the sticky footer
        preview_container = st.container()
        with preview_container:
            render_image_view(pil_prev, border_config=sidebar_data)

        # 2. Fixed Bottom Contact Sheet
        render_contact_sheet()
