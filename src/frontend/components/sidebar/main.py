import streamlit as st
import os
from src.frontend.state import copy_settings, paste_settings
from src.backend.session import DarkroomSession
from src.domain_objects import SidebarData
from src.config import APP_CONFIG
from .adjustments import render_adjustments


def render_file_manager() -> None:
    """
    Handles file uploading and session synchronization.
    """
    session: DarkroomSession = st.session_state.session
    with st.sidebar:
        st.title(":material/camera_roll: DarkroomPy")

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


def render_sidebar_content() -> SidebarData:
    """
    Renders the main sidebar content.
    """
    session: DarkroomSession = st.session_state.session
    with st.sidebar:
        # 0. Global Actions
        process_btn = st.button(
            ":material/batch_prediction: Export All",
            type="primary",
            width="stretch",
            help="Process and export all loaded files using their individual settings.",
        )

        c1, c2 = st.columns(2)

        current_file = session.current_file
        if not current_file:
            return SidebarData()

        # Ensure settings are loaded
        if current_file["hash"] not in session.file_settings:
            session.load_active_settings()

        c1.button(
            ":material/copy_all: Copy Settings",
            on_click=copy_settings,
            width="stretch",
        )
        c2.button(
            ":material/content_copy: Paste Settings",
            on_click=paste_settings,
            disabled=session.clipboard is None,
            width="stretch",
        )

        # 4. Main Adjustments
        adjustments_data = render_adjustments()

        st.divider()

        # 5. Color Management (ICC Profiles)
        st.subheader("Soft Proofing")

        # 5a. List all profiles
        built_in_icc = [
            os.path.join("icc", f)
            for f in os.listdir("icc")
            if f.lower().endswith((".icc", ".icm"))
        ]
        user_icc = []
        if os.path.exists(APP_CONFIG.user_icc_dir):
            user_icc = [
                os.path.join(APP_CONFIG.user_icc_dir, f)
                for f in os.listdir(APP_CONFIG.user_icc_dir)
                if f.lower().endswith((".icc", ".icm"))
            ]

        all_icc_paths = built_in_icc + user_icc

        selected_idx = 0
        if session.icc_profile_path in all_icc_paths:
            selected_idx = all_icc_paths.index(session.icc_profile_path) + 1

        selected_path = st.selectbox(
            "ICC Profile",
            ["None"] + all_icc_paths,
            index=selected_idx,
            format_func=lambda x: os.path.basename(x) if x != "None" else "None",
        )

        if selected_path == "None":
            session.icc_profile_path = None
        else:
            session.icc_profile_path = str(selected_path)

        uploaded_icc = st.file_uploader(
            "Upload ICC Profile", type=["icc", "icm"], label_visibility="collapsed"
        )
        if uploaded_icc:
            os.makedirs(APP_CONFIG.user_icc_dir, exist_ok=True)
            upload_path = os.path.join(APP_CONFIG.user_icc_dir, uploaded_icc.name)
            with open(upload_path, "wb") as f:
                f.write(uploaded_icc.getbuffer())
            session.icc_profile_path = upload_path
            st.rerun()

        st.divider()

        # Consolidate data for the main app
        adjustments_data.process_btn = process_btn
        return adjustments_data
