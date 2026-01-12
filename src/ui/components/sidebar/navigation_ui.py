from typing import Tuple
import streamlit as st
from src.ui.state.state_manager import (
    save_settings,
    load_settings,
    copy_settings,
    paste_settings,
    reset_file_settings,
)
from src.domain.session import WorkspaceSession


def change_file(new_idx: int) -> None:
    """
    Callback to switch the currently selected file.
    """
    session: WorkspaceSession = st.session_state.session

    save_settings(persist=True)

    session.selected_file_idx = new_idx

    load_settings()

    st.session_state.dust_start_point = None
    st.session_state.last_dust_click = None


def unload_file(idx: int) -> None:
    """
    Removes a file from the uploaded list and clears its session cache.
    """
    session: WorkspaceSession = st.session_state.session
    file_list = session.uploaded_files
    file_to_remove = file_list[idx]
    filename = file_to_remove["name"]
    f_hash = file_to_remove["hash"]
    session.asset_store.remove(file_to_remove["path"])

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
    save_settings(persist=True)


def render_navigation() -> Tuple[bool, bool]:
    session: WorkspaceSession = st.session_state.session

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.button(
            ":material/arrow_back:",
            key="prev_btn_s",
            width="stretch",
            disabled=session.selected_file_idx == 0,
            on_click=change_file,
            args=(session.selected_file_idx - 1,),
        )
    with c2:
        st.button(
            ":material/arrow_forward:",
            key="next_btn_s",
            width="stretch",
            disabled=session.selected_file_idx == len(session.uploaded_files) - 1,
            on_click=change_file,
            args=(session.selected_file_idx + 1,),
        )
    with c3:
        st.button(
            ":material/rotate_left:",
            key="rot_l_s",
            width="stretch",
            on_click=rotate_file,
            args=(1,),
        )
    with c4:
        st.button(
            ":material/rotate_right:",
            key="rot_r_s",
            width="stretch",
            on_click=rotate_file,
            args=(-1,),
        )

    with c5:
        st.button(
            ":red[:material/delete:]",
            key="unload_s",
            width="stretch",
            type="secondary",
            on_click=unload_file,
            args=(session.selected_file_idx,),
        )

    ca, cb, cc = st.columns(3)
    with ca:
        st.button(
            ":material/copy_all: Copy",
            on_click=copy_settings,
            width="stretch",
            help="Copy current settings to clipboard.",
        )
    with cb:
        st.button(
            ":material/content_copy: Paste",
            on_click=paste_settings,
            disabled=session.clipboard is None,
            width="stretch",
            help="Paste settings from clipboard.",
        )
    with cc:
        st.button(
            ":material/reset_image: Reset",
            key="reset_s",
            on_click=reset_file_settings,
            width="stretch",
            type="secondary",
            help="Reset all settings for this negative to defaults.",
        )

    ea, eb = st.columns(2)
    with ea:
        export_btn_sidebar = st.button(
            ":material/save: Export",
            key="export_s",
            width="stretch",
            type="primary",
        )
    with eb:
        process_all_btn = st.button(
            ":material/batch_prediction: Export All",
            key="export_all_s",
            type="primary",
            width="stretch",
            help="Process and export all loaded files using their individual settings.",
        )

    return export_btn_sidebar, process_all_btn
