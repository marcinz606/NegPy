import streamlit as st
from typing import List, Any
from src.frontend.components.sidebar.navigation import change_file

def render_contact_sheet(uploaded_files: List[Any]) -> None:
    """
    Renders a collapsible contact sheet of thumbnails.
    """
    if not uploaded_files:
        return

    with st.expander("Contact Sheet", expanded=False):
        # Create a grid of thumbnails
        cols_per_row = 10
        for i in range(0, len(uploaded_files), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(uploaded_files):
                    f = uploaded_files[idx]
                    with cols[j]:
                        thumb = st.session_state.thumbnails.get(f.name)
                        if thumb:
                            is_selected = (st.session_state.selected_file_idx == idx)
                            st.image(thumb, width="stretch", caption=f.name if len(f.name) < 15 else f.name[:12]+"...")
                            st.button("Select", key=f"sel_{idx}", width="stretch", 
                                      type="primary" if is_selected else "secondary",
                                      on_click=change_file, args=(idx, uploaded_files))
                        else:
                            st.write("Loading...")
