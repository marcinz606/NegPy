import streamlit as st
from PIL import Image
from typing import Any, Tuple
from src.ui.state.session_context import SessionContext
from src.kernel.system.version import get_app_version
from src.ui.state.view_models import SidebarState
from src.ui.layouts.contact_sheet import render_contact_sheet
from src.ui.layouts.image_view import render_image_view


def render_layout_header(ctx: SessionContext) -> Tuple[Any, Any]:
    rotation = int(st.session_state.get("rotation", 0))
    h_orig, w_orig = ctx.original_res

    # rotation is k (number of 90deg CCW steps).
    # Swap dimensions only if rotation is 1 or 3 (90 or 270 degrees)
    if rotation % 2 != 0:
        h_orig, w_orig = w_orig, h_orig

    is_vertical = h_orig > w_orig
    target_key = (
        "working_copy_size_vertical" if is_vertical else "working_copy_size_horizontal"
    )

    if target_key in st.session_state:
        if st.session_state.working_copy_size != st.session_state[target_key]:
            st.session_state.working_copy_size = st.session_state[target_key]

    def update_orientation_size() -> None:
        """Callback to save the slider value to the orientation-specific key."""
        st.session_state[target_key] = st.session_state.working_copy_size

    main_area = st.container()
    with main_area:
        c_logo, c_status, c_empty, c_slider = st.columns([2, 3, 1, 1])
        with c_logo:
            version = get_app_version()
            st.title(
                f":red[:material/camera_roll:] NegPy :grey[{version}]",
                width="stretch",
            )
        with c_status:
            status_container = st.container(height=48, border=False, width="stretch")
            status_area = status_container.empty()
        with c_empty:
            pass
        with c_slider:
            st.slider(
                "Display Size",
                600,
                2400,
                step=100,
                key="working_copy_size",
                on_change=update_orientation_size,
                help="Scaling of the preview image in the browser. Does not affect internal processing resolution.",
            )

    return main_area, status_area


def render_main_layout(
    pil_prev: Image.Image,
    sidebar_data: SidebarState,
    main_area: Any,
) -> None:
    with main_area:
        preview_container = st.container()
        with preview_container:
            render_image_view(pil_prev, border_config=sidebar_data)
        render_contact_sheet()
