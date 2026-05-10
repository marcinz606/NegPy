from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def session(qapp):
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.get_max_history_index.return_value = 0
    mgr = DesktopSessionManager(repo)
    mgr.state.uploaded_files = [
        {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
        {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
        {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        {"name": "note.txt", "path": "/tmp/note.txt", "hash": "h4"},
    ]
    mgr.asset_model.refresh()
    return mgr


@pytest.fixture
def browser(session):
    controller = MagicMock()
    controller.session = session
    return FileBrowser(controller)


def test_search_input_is_present(browser):
    assert browser.search_input is not None
    assert browser.search_input.placeholderText() == "Filter by filename..."
    assert browser.regex_btn.isCheckable()


def test_apply_filter_narrows_visible_files(browser, session):
    browser.search_input.setText("IMG")
    browser._apply_filter()
    visible = session.asset_model.visible_actual_indices_ordered()
    visible_names = {session.state.uploaded_files[i]["name"] for i in visible}
    assert visible_names == {"IMG_0001.cr2", "IMG_0002.cr2"}


def test_regex_toggle_compiles_pattern(browser, session):
    browser.regex_btn.setChecked(True)
    browser.search_input.setText(r"^IMG_\d{4}")
    browser._apply_filter()
    assert session.asset_model._filter_pattern is not None
    visible = {session.state.uploaded_files[i]["name"] for i in session.asset_model._sorted_indices}
    assert visible == {"IMG_0001.cr2", "IMG_0002.cr2"}


def test_invalid_regex_sets_error_stylesheet(browser):
    browser.regex_btn.setChecked(True)
    browser.search_input.setText("[")
    browser._apply_filter()
    assert THEME.accent_primary in browser.search_input.styleSheet()


def test_invalid_regex_does_not_change_visible(browser, session):
    browser.search_input.setText("IMG")
    browser._apply_filter()
    before = list(session.asset_model._sorted_indices)
    browser.regex_btn.setChecked(True)
    browser.search_input.setText("[")
    browser._apply_filter()
    assert session.asset_model._sorted_indices == before


def test_selection_pruned_to_visible(browser, session):
    session.state.selected_indices = [0, 1, 2, 3]
    session.state.selected_file_idx = 0
    browser.search_input.setText("IMG")
    browser._apply_filter()
    assert set(session.state.selected_indices) == {0, 1}
    assert session.state.selected_file_idx in {0, 1}


def test_selection_cleared_when_no_visible_match(browser, session):
    session.state.selected_indices = [0, 1, 2, 3]
    session.state.selected_file_idx = 0
    browser.search_input.setText("zzzzz")
    browser._apply_filter()
    assert session.state.selected_indices == []
    assert session.state.selected_file_idx == -1


def test_active_file_preserved_when_still_visible(browser, session):
    session.state.selected_indices = [0, 1, 2]
    session.state.selected_file_idx = 1  # IMG_0002.cr2
    browser.search_input.setText("IMG")
    browser._apply_filter()
    assert session.state.selected_file_idx == 1
    assert set(session.state.selected_indices) == {0, 1}


def test_clearing_filter_clears_error_stylesheet(browser):
    browser.regex_btn.setChecked(True)
    browser.search_input.setText("[")
    browser._apply_filter()
    assert browser.search_input.styleSheet() != ""

    browser.regex_btn.setChecked(False)
    browser.search_input.setText("")
    browser._apply_filter()
    assert browser.search_input.styleSheet() == ""
