import streamlit as st
from src.core.session.manager import WorkspaceSession
from src.domain_objects import SidebarData
from src.presentation.components.sidebar.collect_adjustments import render_adjustments
from src.presentation.components.sidebar.soft_proofing_ui import render_soft_proofing


def render_file_manager() -> None:
    """
    Handles file uploading and session synchronization.
    """
    session: WorkspaceSession = st.session_state.session
    with st.sidebar:
        st.title(":red[:material/camera_roll:] DarkroomPy")
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
    session: WorkspaceSession = st.session_state.session
    with st.sidebar:
        # 0. Global Actions
        current_file = session.current_file
        if not current_file:
            return SidebarData()

        # 4. Main Adjustments
        adjustments_data = render_adjustments()

        st.divider()

        # 5. Soft Proofing
        render_soft_proofing()

        st.divider()

        # Consolidate data for the main app
        return adjustments_data
