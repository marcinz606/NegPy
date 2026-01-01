import streamlit as st
from typing import List, Any
from src.frontend.components.navigation import change_file

def render_contact_sheet(uploaded_files: List[Any]) -> None:
    """
    Renders a vertical contact sheet of thumbnails in a scrollable container.
    """
    if not uploaded_files:
        return

    # Use a scrollable container
    with st.container(height=800):
        # Create 2 columns for smaller thumbnails
        for i in range(0, len(uploaded_files), 2):
            cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx < len(uploaded_files):
                    f = uploaded_files[idx]
                    with cols[j]:
                        thumb = st.session_state.thumbnails.get(f.name)
                        if thumb:
                            is_selected = (st.session_state.selected_file_idx == idx)
                            # Smaller thumbnails by using column width
                            display_name = f.name[:12] + "..." if len(f.name) > 15 else f.name
                            st.caption(display_name)
                            st.image(thumb, width='stretch')
                            st.button(
                                display_name, 
                                key=f"sel_{idx}", 
                                width='stretch', 
                                type="primary" if is_selected else "secondary",
                                on_click=change_file, 
                                args=(idx, uploaded_files)
                            )
                        else:
                            st.write("...")

