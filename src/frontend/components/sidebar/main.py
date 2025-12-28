import streamlit as st
from typing import Optional, Dict, Any
from src.frontend.state import load_settings, copy_settings, paste_settings
from .navigation import render_navigation
from .presets import render_presets
from .adjustments import render_adjustments
from .ai import render_ai_tab

def render_sidebar() -> Optional[Dict[str, Any]]:
    """
    Main orchestrator for the sidebar UI.
    """
    with st.sidebar:
        st.title("🎞️ DarkroomPy")
        
        raw_uploaded_files = st.file_uploader("Upload RAW files", type=['dng', 'cr2', 'nef', 'arw'], accept_multiple_files=True)
        current_uploaded_names = {f.name for f in raw_uploaded_files} if raw_uploaded_files else set()
        
        # Sync uploaded files with session state
        new_names = current_uploaded_names - st.session_state.last_uploaded_names
        if new_names:
            for f in raw_uploaded_files:
                if f.name in new_names and f.name not in {x.name for x in st.session_state.uploaded_files}:
                    st.session_state.uploaded_files.append(f)
        
        removed_from_widget = st.session_state.last_uploaded_names - current_uploaded_names
        if removed_from_widget:
            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f.name not in removed_from_widget]
        
        st.session_state.last_uploaded_names = current_uploaded_names
        uploaded_files = st.session_state.uploaded_files

        if not uploaded_files:
            return None

        # 1. Navigation & Actions
        export_btn_sidebar = render_navigation(uploaded_files)

        st.divider()
        
        # 2. Settings Clipboard
        c1, c2 = st.columns(2)
        c1.button("Copy Settings", on_click=copy_settings, width="stretch")
        c2.button("Paste Settings", on_click=paste_settings, disabled=st.session_state.clipboard is None, width="stretch")
        
        # 3. Presets
        current_file_name = uploaded_files[st.session_state.selected_file_idx].name
        
        # Automatically load from DB if this file is encountered for the first time in this session
        # This MUST happen before render_adjustments to avoid modifying session_state after widget instantiation
        if current_file_name not in st.session_state.file_settings:
            load_settings(current_file_name)
            
        render_presets(current_file_name)

        st.divider()

        # 4. Main Tabs (Adjustments vs AI)
        tab_adj, tab_ai = st.tabs(["Adjustments", "AI"])
        
        with tab_adj:
            adjustments_data = render_adjustments(current_file_name)
            
        with tab_ai:
            render_ai_tab(current_file_name)
        
        st.divider()
        
        # Consolidate data for the main app
        return {
            'export_btn_sidebar': export_btn_sidebar,
            **adjustments_data
        }
