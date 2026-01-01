import streamlit as st

def apply_custom_css():
    """
    Applies custom CSS to the Streamlit app for professional darkroom styling.
    """
    st.markdown("""
        <style>
        /* Don't round the borders in preview */
        img { border-radius: 0px !important; }

        /* Hide default streamlit 'deploy' button */
        .stDeployButton {
            visibility: hidden;
        }
        /* Target scrollable containers to fit viewport height */
        [data-testid="stVerticalBlockBorderWrapper"] > div:has(> [data-testid="stVerticalBlock"]) {
             /* This is a bit generic, but we can target the height-capping containers */
        }
        
        /* Capping the contact sheet height specifically if possible, 
           otherwise we rely on the height parameter in st.container which 
           is safer in Streamlit. Let's try to use a slightly more responsive approach. */
        div[data-testid="stVerticalBlock"] > div[style*="height:"] {
            max-height: 85vh !important;
            height: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
