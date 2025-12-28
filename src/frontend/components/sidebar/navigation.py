import streamlit as st
from typing import List, Any
from src.frontend.state import save_settings, load_settings, reset_file_settings

def change_file(new_idx: int, file_list: List[Any]) -> None:
    """
    Callback to switch the currently selected file.
    """
    if st.session_state.selected_file_idx < len(file_list):
        save_settings(file_list[st.session_state.selected_file_idx].name)
    st.session_state.selected_file_idx = new_idx
    load_settings(file_list[new_idx].name)
    st.session_state.dust_start_point = None
    st.session_state.last_dust_click = None

def unload_file(idx: int, file_list: List[Any]) -> None:
    """
    Removes a file from the uploaded list and clears its session cache.
    """
    file_to_remove = file_list[idx]
    filename = file_to_remove.name
    if filename in st.session_state.file_settings:
        del st.session_state.file_settings[filename]
    if filename in st.session_state.thumbnails:
        del st.session_state.thumbnails[filename]
    file_list.pop(idx)
    st.session_state.uploaded_files = file_list
    if st.session_state.selected_file_idx >= len(file_list):
        st.session_state.selected_file_idx = max(0, len(file_list) - 1)
    if 'last_file' in st.session_state and st.session_state.last_file == filename:
        if 'preview_raw' in st.session_state: del st.session_state.preview_raw
        if 'last_file' in st.session_state: del st.session_state.last_file

def rotate_file(direction: int, file_list: List[Any]) -> None:
    """
    Callback to rotate the image.
    1 for left (+90 deg), -1 for right (-90 deg).
    """
    st.session_state.rotation = (st.session_state.rotation + direction) % 4
    save_settings(file_list[st.session_state.selected_file_idx].name)

def render_navigation(uploaded_files: List[Any]):
    """
    Renders the navigation buttons, rotation, and file removal/reset actions.
    """
    st.subheader("Navigation & Actions")
    c1, c2 = st.columns(2)
    c1.button("Previous", key="prev_btn_s", width="stretch", 
            disabled=st.session_state.selected_file_idx == 0,
            on_click=change_file, args=(st.session_state.selected_file_idx - 1, uploaded_files))
    c2.button("Next", key="next_btn_s", width="stretch", 
            disabled=st.session_state.selected_file_idx == len(uploaded_files) - 1,
            on_click=change_file, args=(st.session_state.selected_file_idx + 1, uploaded_files))
    
    c1, c2 = st.columns(2)
    c1.button("Rotate Left", key="rot_l_s", width="stretch",
              on_click=rotate_file, args=(1, uploaded_files))
    c2.button("Rotate Right", key="rot_r_s", width="stretch",
              on_click=rotate_file, args=(-1, uploaded_files))

    f_idx = st.session_state.selected_file_idx
    f_current = uploaded_files[f_idx]
    
    c1, c2, c3 = st.columns(3)
    export_btn_sidebar = c1.button("Export", key="export_s", width="stretch", type="primary")
    c2.button("Remove", key="unload_s", width="stretch", type="secondary",
                on_click=unload_file, args=(f_idx, uploaded_files))
    c3.button("Reset", key="reset_s", width="stretch", type="secondary",
                on_click=reset_file_settings, args=(f_current.name,))
    
    return export_btn_sidebar
