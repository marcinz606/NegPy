import streamlit as st
from src.ui.components.sidebar.navigation_ui import change_file
from src.domain.session import WorkspaceSession
from src.kernel.system.config import APP_CONFIG


def render_contact_sheet() -> None:
    """
    Renders a contact sheet of thumbnails in a scrollable container.
    """
    session: WorkspaceSession = st.session_state.session
    if not session.uploaded_files:
        return

    # Use a scrollable container with fixed height suitable for ~1 row when stuck to bottom
    with st.container(height=250):
        ts = APP_CONFIG.thumbnail_size
        # Dynamically calculate number of columns based on thumbnail size and container width
        # Assuming sidebar is ~300px and main is ~1200px.
        # But for simplicity, we can just use more columns or let streamlit handle it.
        # Let's use 10 columns for smaller thumbs.
        num_cols = 12
        uploaded_files = session.uploaded_files
        for i in range(0, len(uploaded_files), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i + j
                if idx < len(uploaded_files):
                    f_meta = uploaded_files[idx]
                    with cols[j]:
                        thumb = session.thumbnails.get(f_meta["name"])
                        is_selected = session.selected_file_idx == idx

                        display_name = (
                            f_meta["name"][:12] + "..."
                            if len(f_meta["name"]) > 15
                            else f_meta["name"]
                        )
                        st.caption(display_name)
                        if thumb:
                            st.image(thumb, width=ts)
                            st.button(
                                "Select",
                                key=f"sel_{idx}",
                                width=ts,
                                type="primary" if is_selected else "secondary",
                                on_click=change_file,
                                args=(idx,),
                            )
                        else:
                            st.write("Loading...")
