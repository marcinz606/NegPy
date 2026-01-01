import streamlit as st
import os
from typing import Dict, Any, List
from src.frontend.state import load_settings, copy_settings, paste_settings
from .adjustments import render_adjustments

def render_file_manager() -> List[Any]:
    """
    Handles file uploading and session state synchronization.
    Returns the list of uploaded files.
    """
    with st.sidebar:
        st.title(":material/camera_roll: DarkroomPy")
        
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
        return st.session_state.uploaded_files

def render_sidebar_content(uploaded_files: List[Any]) -> Dict[str, Any]:

    """

    Renders the main sidebar content (Nav, Settings, Adjustments).

    Should be called AFTER any auto-adjustments have been applied to session state.

    """

    with st.sidebar:

        # 1. Settings Clipboard

        c1, c2 = st.columns(2)


        if not uploaded_files:
            return {}
        
        current_file_name = uploaded_files[st.session_state.selected_file_idx].name
        
        # NOTE: We assume 'load_settings' or auto-logic has already run in main.py if needed.
        # But we still need to load settings if switching manually.
        # Check if settings are missing for this file
        if current_file_name not in st.session_state.file_settings:
            load_settings(current_file_name)
            


        c1.button(":material/copy_all: Copy Settings", on_click=copy_settings, width="stretch")
        c2.button(":material/content_copy: Paste Settings", on_click=paste_settings, disabled=st.session_state.clipboard is None, width="stretch")

        # 4. Main Adjustments
        adjustments_data = render_adjustments(current_file_name)
        
        st.divider()

        # 5. Color Management (ICC Profiles)
        st.subheader("Soft Proofing")
        icc_files = [f for f in os.listdir("icc") if f.lower().endswith(('.icc', '.icm'))]
        
        selected_icc = st.selectbox("ICC Profile", ["None"] + icc_files, 
                                    index=0 if st.session_state.icc_profile_path is None 
                                    else (icc_files.index(os.path.basename(st.session_state.icc_profile_path)) + 1 if os.path.basename(st.session_state.icc_profile_path) in icc_files else 0))
        
        if selected_icc == "None":
            st.session_state.icc_profile_path = None
        else:
            st.session_state.icc_profile_path = os.path.join("profiles", selected_icc)

        uploaded_icc = st.file_uploader("Upload ICC Profile", type=['icc', 'icm'], label_visibility="collapsed")
        if uploaded_icc:
            with open(os.path.join("profiles", uploaded_icc.name), "wb") as f:
                f.write(uploaded_icc.getbuffer())
            st.session_state.icc_profile_path = os.path.join("profiles", uploaded_icc.name)
            st.rerun()
        
        st.divider()
        
        # Consolidate data for the main app
        return {
            **adjustments_data
        }
