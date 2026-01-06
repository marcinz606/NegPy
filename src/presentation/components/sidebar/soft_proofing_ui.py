import streamlit as st
import os
from src.config import APP_CONFIG
from src.core.session.manager import WorkspaceSession


def render_soft_proofing() -> None:
    """
    Renders the Soft Proofing section of the sidebar.
    """
    session: WorkspaceSession = st.session_state.session

    with st.expander(":material/imagesearch_roller: Soft Proofing", expanded=False):
        # List all profiles
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
