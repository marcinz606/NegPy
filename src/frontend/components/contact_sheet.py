import streamlit as st
from src.frontend.components.navigation import change_file
from src.backend.session import DarkroomSession


def render_contact_sheet() -> None:
    """
    Renders a vertical contact sheet of thumbnails in a scrollable container.
    """
    session: DarkroomSession = st.session_state.session
    if not session.uploaded_files:
        return

    # Use a scrollable container
    with st.container(height=600):
        # Create 2 columns for smaller thumbnails
        uploaded_files = session.uploaded_files
        for i in range(0, len(uploaded_files), 2):
            cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx < len(uploaded_files):
                    f_meta = uploaded_files[idx]
                    with cols[j]:
                        thumb = session.thumbnails.get(f_meta["name"])
                        is_selected = session.selected_file_idx == idx

                        display_name = (
                            f_meta["name"][:10] + "..."
                            if len(f_meta["name"]) > 12
                            else f_meta["name"]
                        )
                        st.caption(display_name)
                        if thumb:
                            st.image(thumb, width="stretch")
                            st.button(
                                "Select",
                                key=f"sel_{idx}",
                                width="stretch",
                                type="primary" if is_selected else "secondary",
                                on_click=change_file,
                                args=(idx,),
                            )
                        else:
                            st.write("Loading...")
