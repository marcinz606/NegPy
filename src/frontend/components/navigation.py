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
    st.button("-> Next", key="next_btn_s", use_container_width=True, 
            disabled=st.session_state.selected_file_idx == len(uploaded_files) - 1,
            on_click=change_file, args=(st.session_state.selected_file_idx + 1, uploaded_files))
    st.button("<- Previous", key="prev_btn_s", use_container_width=True, 
            disabled=st.session_state.selected_file_idx == 0,
            on_click=change_file, args=(st.session_state.selected_file_idx - 1, uploaded_files))
    
    st.button(":material/rotate_left: Rotate Left", key="rot_l_s", use_container_width=True,
              on_click=rotate_file, args=(1, uploaded_files))
    st.button(":material/rotate_right: Rotate Right", key="rot_r_s", use_container_width=True,
              on_click=rotate_file, args=(-1, uploaded_files))

    f_idx = st.session_state.selected_file_idx
    f_current = uploaded_files[f_idx]
    
    export_btn_sidebar = st.button(":material/save: Export", key="export_s", use_container_width=True, type="primary")
    st.button(":material/delete: Remove", key="unload_s", use_container_width=True, type="secondary",
                on_click=unload_file, args=(f_idx, uploaded_files))
    
    return export_btn_sidebar
