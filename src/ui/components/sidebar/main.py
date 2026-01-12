import streamlit as st
from src.domain.session import WorkspaceSession
from src.ui.state.view_models import SidebarState
from src.ui.components.sidebar.collect_adjustments import render_adjustments


def render_sidebar_content() -> SidebarState:
    session: WorkspaceSession = st.session_state.session
    with st.sidebar:
        current_file = session.current_file
        if not current_file:
            return SidebarState()

        adjustments_data = render_adjustments()

        return adjustments_data
