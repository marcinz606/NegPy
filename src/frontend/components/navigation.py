import streamlit as st
from src.frontend.state import save_settings, load_settings
from src.backend.session import DarkroomSession


def change_file(new_idx: int) -> None:
    """
    Callback to switch the currently selected file.
    """
    session: DarkroomSession = st.session_state.session
    if session.selected_file_idx < len(session.uploaded_files):
        save_settings()

    session.selected_file_idx = new_idx
    load_settings()
    st.session_state.dust_start_point = None
    st.session_state.last_dust_click = None


def unload_file(idx: int) -> None:
    """
    Removes a file from the uploaded list and clears its session cache.
    """
    session: DarkroomSession = st.session_state.session
    file_list = session.uploaded_files
    file_to_remove = file_list[idx]
    filename = file_to_remove["name"]
    f_hash = file_to_remove["hash"]
    session.asset_manager.remove(file_to_remove["path"])

    if f_hash in session.file_settings:
        del session.file_settings[f_hash]
    if filename in session.thumbnails:
        del session.thumbnails[filename]

    file_list.pop(idx)
    session.uploaded_files = file_list

    if session.selected_file_idx >= len(file_list):
        session.selected_file_idx = max(0, len(file_list) - 1)

    if st.session_state.get("last_file") == filename:
        if "preview_raw" in st.session_state:
            del st.session_state.preview_raw
        if "last_file" in st.session_state:
            del st.session_state.last_file


def rotate_file(direction: int) -> None:
    """
    Callback to rotate the image.
    1 for left (+90 deg), -1 for right (-90 deg).
    """
    st.session_state.rotation = (st.session_state.get("rotation", 0) + direction) % 4
    save_settings()


def render_navigation() -> bool:
    """
    Renders the navigation buttons, rotation, and file removal/reset actions in 2 columns.
    """
    session: DarkroomSession = st.session_state.session
    c1, c2 = st.columns(2)

    with c1:
        st.button(
            ":material/arrow_back: Previous",
            key="prev_btn_s",
            width="stretch",
            disabled=session.selected_file_idx == 0,
            on_click=change_file,
            args=(session.selected_file_idx - 1,),
        )
        st.button(
            ":material/rotate_left: Left",
            key="rot_l_s",
            width="stretch",
            on_click=rotate_file,
            args=(1,),
        )
        st.button(
            ":material/delete: Remove",
            key="unload_s",
            width="stretch",
            type="secondary",
            on_click=unload_file,
            args=(session.selected_file_idx,),
        )

    with c2:
        st.button(
            "Next :material/arrow_forward:",
            key="next_btn_s",
            width="stretch",
            disabled=session.selected_file_idx == len(session.uploaded_files) - 1,
            on_click=change_file,
            args=(session.selected_file_idx + 1,),
        )
        st.button(
            ":material/rotate_right: Right",
            key="rot_r_s",
            width="stretch",
            on_click=rotate_file,
            args=(-1,),
        )
        export_btn_sidebar = st.button(
            ":material/save: Export",
            key="export_s",
            width="stretch",
            type="primary",
        )

    return export_btn_sidebar
