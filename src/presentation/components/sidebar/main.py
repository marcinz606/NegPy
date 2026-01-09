import streamlit as st
from src.core.session.manager import WorkspaceSession
from src.presentation.state.view_models import SidebarState
from src.presentation.components.sidebar.collect_adjustments import render_adjustments


def render_sidebar_content() -> SidebarState:
    """
    Renders the main sidebar content.
    """
    session: WorkspaceSession = st.session_state.session
    with st.sidebar:
        st.title(":red[:material/camera_roll:] DarkroomPy")

        current_file = session.current_file
        if not current_file:
            return SidebarState()

        adjustments_data = render_adjustments()

        return adjustments_data
