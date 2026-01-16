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
    1 for left (+90 deg CCW), -1 for right (-90 deg CW).
    """
    st.session_state.rotation = (st.session_state.get("rotation", 0) + direction) % 4

    # Transform manual crop if exists to preserve it across rotations
    manual_crop = st.session_state.get("manual_crop_rect")
    if manual_crop:
        x1, y1, x2, y2 = manual_crop
        if direction == 1:  # Left (90 CCW)
            # Point (x, y) -> (y, 1-x)
            nx1, ny1 = y1, 1.0 - x1
            nx2, ny2 = y2, 1.0 - x2
        else:  # Right (90 CW)
            # Point (x, y) -> (1-y, x)
            nx1, ny1 = 1.0 - y1, x1
            nx2, ny2 = 1.0 - y2, x2

        # Re-sort to maintain (xmin, ymin, xmax, ymax)
        final_x1, final_x2 = sorted([nx1, nx2])
        final_y1, final_y2 = sorted([ny1, ny2])
        st.session_state.manual_crop_rect = (final_x1, final_y1, final_x2, final_y2)

    save_settings(persist=True)


def flip_file(axis: str) -> None:
    """
    Toggles horizontal or vertical flip.
    """
    key = f"flip_{axis}"
    st.session_state[key] = not st.session_state.get(key, False)
    save_settings(persist=True)


def render_actions_menu() -> bool:
    """
    Renders 2-row horizontal toolbar for file actions and navigation.
    """
    session: WorkspaceSession = st.session_state.session

    # Row 1: Navigation & Transform (6 columns)
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])
    with c1:
        st.button(
            ":material/arrow_back: Prev",
            key="prev_btn_m",
            width="stretch",
            disabled=session.selected_file_idx == 0,
            on_click=change_file,
            args=(session.selected_file_idx - 1,),
        )
    with c2:
        st.button(
            ":material/arrow_forward: Next",
            key="next_btn_m",
            width="stretch",
            disabled=session.selected_file_idx == len(session.uploaded_files) - 1,
            on_click=change_file,
            args=(session.selected_file_idx + 1,),
        )
    with c3:
        st.button(
            ":material/rotate_left: Rot L",
            key="rot_l_m",
            width="stretch",
            on_click=rotate_file,
            args=(1,),
        )
    with c4:
        st.button(
            ":material/rotate_right: Rot R",
            key="rot_r_m",
            width="stretch",
            on_click=rotate_file,
            args=(-1,),
        )
    with c5:
        st.button(
            ":material/flip: Flip H",
            key="flip_h_m",
            width="stretch",
            on_click=flip_file,
            args=("horizontal",),
        )
    with c6:
        st.button(
            ":material/flip: Flip V",
            key="flip_v_m",
            width="stretch",
            on_click=flip_file,
            args=("vertical",),
        )

    # Row 2: Clipboard & Primary Action
    ca, cb, cc, cd, ce = st.columns([1, 1, 2, 1, 1])
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
        export_btn = st.button(
            ":material/save: Export Image",
            key="export_m",
            width="stretch",
            type="primary",
        )
    with cd:
        st.button(
            ":material/reset_image: Reset",
            key="reset_m",
            on_click=reset_file_settings,
            width="stretch",
            type="secondary",
            help="Reset all settings for this negative to defaults.",
        )
    with ce:
        st.button(
            ":red[:material/delete:] Unload",
            key="unload_m",
            width="stretch",
            type="secondary",
            on_click=unload_file,
            args=(session.selected_file_idx,),
        )

    return export_btn
