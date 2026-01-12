import streamlit as st
from typing import Protocol
from src.core.session.manager import WorkspaceSession
from src.presentation.components.sidebar.navigation_ui import change_file


class IFilmStrip(Protocol):
    def render(self) -> None: ...


class FilmStrip:
    """
    A horizontal navigation bar (filmstrip) anchored at the bottom of the main view.
    Uses native Streamlit widgets for interactivity (avoiding page reloads)
    and advanced CSS (:has selector) for layout.
    """

    def __init__(self, session: WorkspaceSession):
        self.session = session

    def _inject_custom_css(self) -> str:
        """
        Returns the CSS to force the specific Streamlit container to behave like a filmstrip.
        Uses :has() to target the container that holds our unique marker ID.
        """
        return """
        <style>
            /* 
               Target the specific vertical block that contains our filmstrip marker.
               Make it sticky at the bottom.
            */
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) {
                position: sticky;
                bottom: 0;
                left: 0;
                width: 100%;
                z-index: 999;
                background-color: #1e1e1e;
                border-top: 1px solid #333;
                padding: 10px 0;
                margin-bottom: 0 !important;
            }

            /* 
               Target the internal horizontal block (created by st.columns) inside our container.
               Force it to scroll horizontally and not wrap.
            */
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) [data-testid="stHorizontalBlock"] {
                flex-wrap: nowrap !important;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                gap: 0.5rem !important;
                padding-bottom: 5px; /* Space for scrollbar */
            }

            /* Hide Scrollbar for cleaner look (optional, but requested) */
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) [data-testid="stHorizontalBlock"]::-webkit-scrollbar {
                height: 6px;
            }
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
                background: #444;
                border-radius: 3px;
            }

            /* 
               Target the individual columns (items).
               Force fixed width and prevent shrinking.
            */
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) [data-testid="column"] {
                flex: 0 0 auto !important;
                width: 140px !important;
                min-width: 140px !important;
                display: flex;
                flex-direction: column;
                align-items: center;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
                padding: 5px;
                border: 1px solid transparent;
                transition: background 0.2s;
            }

            /* Highlight selected item (we can't easily target parent based on child button type via CSS only) 
               so we rely on the button type visualization mostly. */
            
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) [data-testid="column"]:hover {
                background: rgba(255, 255, 255, 0.1);
            }

            /* Adjust Image styling */
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) img {
                object-fit: contain !important;
                max-height: 80px !important;
            }
            
            /* Compact buttons */
            div[data-testid="stVerticalBlock"]:has(#filmstrip-unique-marker) button {
                padding: 0px 8px !important;
                min-height: 0px !important;
                height: 28px !important;
                line-height: 1 !important;
                margin-top: 5px !important;
                font-size: 12px !important;
            }
        </style>
        """

    def render(self) -> None:
        """
        Renders the filmstrip component using native Streamlit widgets.
        """
        if not self.session.uploaded_files:
            return

        # 1. Inject CSS
        st.markdown(self._inject_custom_css(), unsafe_allow_html=True)

        # 2. Container with Marker
        # We use a container to group the layout, and the CSS targets this container via the marker.
        with st.container():
            # The Marker
            st.markdown(
                '<div id="filmstrip-unique-marker"></div>', unsafe_allow_html=True
            )

            # 3. Native Columns (Horizontal Layout via CSS)
            files = self.session.uploaded_files

            # Create N columns.
            # Note: Streamlit handles large numbers of columns reasonably well visually if they are empty,
            # but here we populate them. Performance for >50 images might degrade, but it's the standard way.
            cols = st.columns(len(files))

            for idx, col in enumerate(cols):
                f_meta = files[idx]
                thumb = self.session.thumbnails.get(f_meta["name"])
                is_selected = self.session.selected_file_idx == idx

                with col:
                    # Thumbnail
                    if thumb:
                        st.image(thumb, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='height:80px;display:flex;align-items:center;justify-content:center;color:#666;font-size:10px;'>No Thumb</div>",
                            unsafe_allow_html=True,
                        )

                    # Filename (Truncated)
                    display_name = f_meta["name"]
                    if len(display_name) > 12:
                        display_name = display_name[:10] + ".."

                    # Selection Button
                    # This triggers the callback directly without reload
                    st.button(
                        display_name,
                        key=f"fs_btn_{idx}",
                        type="primary" if is_selected else "secondary",
                        on_click=change_file,
                        args=(idx,),
                        use_container_width=True,
                        help=f_meta["name"],
                    )


def render_film_strip() -> None:
    """
    Functional wrapper for the FilmStrip component.
    """
    if "session" in st.session_state:
        renderer = FilmStrip(st.session_state.session)
        renderer.render()
