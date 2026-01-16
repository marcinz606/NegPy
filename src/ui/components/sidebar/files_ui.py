import streamlit as st
import os
from src.domain.session import WorkspaceSession
from src.infrastructure.loaders.native_picker import NativeFilePicker
from src.kernel.system.config import APP_CONFIG
from src.ui.state.state_manager import save_settings
from src.ui.components.sidebar.helpers import render_control_checkbox
from src.infrastructure.loaders.helpers import get_supported_raw_dotless


def unload_all_files() -> None:
    """
    Clears session and specific streamlit state keys.
    """
    session: WorkspaceSession = st.session_state.session
    session.clear_all_files()

    keys_to_clear = [
        "preview_raw",
        "last_file",
        "last_metrics",
        "base_positive",
        "dust_start_point",
        "last_dust_click",
        "manual_crop_start_point",
        "last_manual_crop_click",
        "last_wb_click",
        "last_assist_click",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


def render_file_manager() -> None:
    """
    File picker & session sync.
    """
    session: WorkspaceSession = st.session_state.session
    is_docker = os.path.exists("/.dockerenv")

    with st.sidebar:
        # Native Picker (Desktop)
        if not is_docker:
            picker = NativeFilePicker()

            # Resume last dir
            last_dir = st.session_state.get("last_picker_dir")
            if not last_dir or not os.path.exists(last_dir):
                last_dir = os.path.dirname(APP_CONFIG.edits_db_path)

            c1, c2 = st.columns(2)
            with c1:
                if st.button(":material/file_open: Pick Files", width="stretch"):
                    save_settings(persist=True)
                    paths = picker.pick_files(initial_dir=last_dir)
                    if paths:
                        session.add_local_assets(paths)
                        st.session_state.last_picker_dir = os.path.dirname(paths[0])
                        # Update hot folder
                        if st.session_state.get("hot_folder_mode"):
                            session.watched_folders.add(os.path.dirname(paths[0]))
                        st.rerun()
            with c2:
                if st.button(":material/folder_open: Pick Folder", width="stretch"):
                    save_settings(persist=True)
                    root_path, paths = picker.pick_folder(initial_dir=last_dir)
                    if paths:
                        session.add_local_assets(paths)
                        # Update last used directory
                        st.session_state.last_picker_dir = root_path
                        # Update hot folder
                        if st.session_state.get("hot_folder_mode") and root_path:
                            session.watched_folders.add(root_path)
                        st.rerun()

            if session.uploaded_files:
                if st.button(
                    ":red[:material/delete_sweep:] Unload All",
                    width="stretch",
                    on_click=unload_all_files,
                    help="Remove all files from the workspace and clear cached data.",
                ):
                    pass

            render_control_checkbox(
                "Hot Folder Mode",
                default_val=False,
                key="hot_folder_mode",
                help_text="Automatically discover new files in picked folders.",
            )
        else:
            # Streamlit uploader (Docker)
            raw_uploaded_files = st.file_uploader(
                "Load RAW files",
                type=get_supported_raw_dotless(),
                accept_multiple_files=True,
            )
            current_uploaded_names = (
                {f.name for f in raw_uploaded_files} if raw_uploaded_files else set()
            )

            session.sync_files(
                current_uploaded_names, raw_uploaded_files if raw_uploaded_files else []
            )
