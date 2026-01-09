import streamlit as st
import os
from src.core.session.manager import WorkspaceSession
from src.infrastructure.loaders.native_picker import NativeFilePicker
from src.config import APP_CONFIG
from src.presentation.state.state_manager import save_settings
from src.presentation.components.sidebar.helpers import render_control_checkbox


def render_file_manager() -> None:
    """
    Handles file uploading and session synchronization.
    """
    session: WorkspaceSession = st.session_state.session
    is_docker = os.path.exists("/.dockerenv")

    with st.sidebar:

        # 1. Native Picker - Hidden if in Docker
        if not is_docker:
            picker = NativeFilePicker()

            # Use persisted last dir or default to project user root
            last_dir = st.session_state.get("last_picker_dir")
            if not last_dir or not os.path.exists(last_dir):
                last_dir = os.path.dirname(APP_CONFIG.edits_db_path)

            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    ":material/file_open: Pick Files", use_container_width=True
                ):
                    save_settings(persist=True)
                    paths = picker.pick_files(initial_dir=last_dir)
                    if paths:
                        session.add_local_assets(paths)
                        st.session_state.last_picker_dir = os.path.dirname(paths[0])
                        st.rerun()
            with c2:
                if st.button(
                    ":material/folder_open: Pick Folder", use_container_width=True
                ):
                    save_settings(persist=True)
                    root_path, paths = picker.pick_folder(initial_dir=last_dir)
                    if paths:
                        session.add_local_assets(paths)
                        # Update last used directory
                        st.session_state.last_picker_dir = root_path
                        # If hot folder mode is enabled, start watching this root folder
                        if st.session_state.get("hot_folder_mode") and root_path:
                            session.watched_folders.add(root_path)
                        st.rerun()

            render_control_checkbox(
                "Hot Folder Mode",
                default_val=False,
                key="hot_folder_mode",
                help_text="Automatically discover new files in picked folders.",
            )
        else:
            # 2. Standard Web Uploader (Docker Mode)
            raw_uploaded_files = st.file_uploader(
                "Load RAW files",
                type=["dng", "tiff", "nef", "arw", "raw", "raf"],
                accept_multiple_files=True,
            )
            current_uploaded_names = (
                {f.name for f in raw_uploaded_files} if raw_uploaded_files else set()
            )

            # Sync files via Session object
            session.sync_files(
                current_uploaded_names, raw_uploaded_files if raw_uploaded_files else []
            )