import streamlit as st
import os
from src.kernel.system.config import APP_CONFIG
from src.domain.session import WorkspaceSession
from src.ui.components.sidebar.helpers import render_control_selectbox


def render_soft_proofing() -> None:
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

        if "soft_proof_icc" not in st.session_state:
            st.session_state.soft_proof_icc = session.icc_profile_path or "None"

        selected_path = render_control_selectbox(
            "ICC Profile",
            ["None"] + all_icc_paths,
            default_val=session.icc_profile_path or "None",
            key="soft_proof_icc",
            format_func=lambda x: os.path.basename(x) if x != "None" else "None",
            help_text="Select ICC for screen simulation.",
        )

        if selected_path == "None":
            session.icc_profile_path = None
        else:
            session.icc_profile_path = str(selected_path)
