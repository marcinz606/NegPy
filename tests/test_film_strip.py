from unittest.mock import MagicMock, patch
import pytest
from src.presentation.layouts.film_strip import FilmStrip, render_film_strip
from src.core.session.manager import WorkspaceSession
from PIL import Image


@pytest.fixture
def mock_session():
    session = MagicMock(spec=WorkspaceSession)
    session.uploaded_files = [
        {"name": "test1.jpg", "path": "/tmp/test1.jpg", "hash": "h1"},
        {"name": "test2.jpg", "path": "/tmp/test2.jpg", "hash": "h2"},
    ]
    img = Image.new("RGB", (10, 10), color="red")
    session.thumbnails = {"test1.jpg": img, "test2.jpg": img}
    session.selected_file_idx = 0
    return session


@patch("src.presentation.layouts.film_strip.st")
def test_film_strip_render_native(mock_st, mock_session):
    # Setup mocks for columns context manager
    mock_col = MagicMock()
    mock_st.columns.return_value = [mock_col, mock_col]
    mock_col.__enter__.return_value = mock_col

    fs = FilmStrip(mock_session)
    fs.render()

    # Check CSS injection
    assert mock_st.markdown.call_count >= 2  # CSS + Marker

    # Check columns creation
    mock_st.columns.assert_called_once_with(2)

    # Check widget calls inside columns
    # We expect 2 images and 2 buttons
    assert mock_st.image.call_count == 2
    assert mock_st.button.call_count == 2

    # Verify button arguments
    args, kwargs = mock_st.button.call_args_list[0]
    assert "test1" in args[0] or "test1" in kwargs.get("label", "")
    assert kwargs["key"] == "fs_btn_0"
    assert kwargs["type"] == "primary"  # Selected

    args, kwargs = mock_st.button.call_args_list[1]
    assert kwargs["key"] == "fs_btn_1"
    assert kwargs["type"] == "secondary"  # Not selected


@patch("src.presentation.layouts.film_strip.st")
def test_render_film_strip_wrapper(mock_st):
    mock_session = MagicMock()
    mock_st.session_state = MagicMock()
    mock_st.session_state.session = mock_session
    mock_st.session_state.__contains__.return_value = True

    with patch("src.presentation.layouts.film_strip.FilmStrip") as MockFS:
        render_film_strip()
        MockFS.assert_called_once_with(mock_session)
        MockFS.return_value.render.assert_called_once()
