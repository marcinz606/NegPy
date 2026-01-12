import streamlit as st


def apply_custom_css() -> None:
    """
    Injects darkroom CSS.
    """
    st.markdown(
        """
        <style>
        /* Don't round the borders in preview
        looks stupid when we apply the border */
        img { border-radius: 0px !important; }

        /* Hide default streamlit 'deploy' button */
        .stDeployButton {
            visibility: hidden;
        }

        /* Adjust the padding of main view */
        [data-testid="stMainBlockContainer"], .block-container {
            max-width: none !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
