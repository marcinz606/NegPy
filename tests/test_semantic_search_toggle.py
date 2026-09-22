"""The Files sidebar's search-by-meaning toggle: visible only when the feature is on,
mutually exclusive with regex, and routes the search box through set_semantic_query
instead of the structured filter."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def session(qapp):
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings_many.return_value = {}
    mgr = DesktopSessionManager(repo)
    mgr.state.uploaded_files = [{"name": "a.nef", "path": "/tmp/a.nef", "hash": "h1"}]
    mgr.asset_model.refresh()
    return mgr


@pytest.fixture
def browser(session):
    controller = MagicMock()
    controller.session = session
    controller.half_frame_mode_for_roll.return_value = False
    return FileBrowser(controller)


def test_hidden_by_default(browser):
    assert browser.semantic_btn.isHidden()


def test_sync_ui_shows_it_once_the_preference_is_on(browser, session):
    session.state.semantic_search_enabled = True
    browser.sync_ui()
    assert not browser.semantic_btn.isHidden()


def test_sync_ui_unchecks_and_hides_it_when_the_preference_turns_off(browser, session):
    session.state.semantic_search_enabled = True
    browser.sync_ui()
    browser.semantic_btn.setChecked(True)

    session.state.semantic_search_enabled = False
    browser.sync_ui()

    assert browser.semantic_btn.isHidden()
    assert not browser.semantic_btn.isChecked()


def test_checking_it_disables_regex(browser):
    browser.semantic_btn.setChecked(True)
    assert not browser.regex_btn.isEnabled()
    browser.semantic_btn.setChecked(False)
    assert browser.regex_btn.isEnabled()


def test_apply_filter_routes_through_the_embedding_when_checked(browser, session):
    vector = np.array([1.0, 0.0], dtype=np.float32)
    browser.controller.embed_search_query.return_value = vector
    browser.semantic_btn.setChecked(True)
    browser.search_input.setText("a photo of a cat")

    browser._apply_filter()

    browser.controller.embed_search_query.assert_called_once_with("a photo of a cat")
    assert session.asset_model.semantic_query_active is True


def test_apply_filter_flags_an_error_when_the_model_is_not_ready(browser):
    browser.controller.embed_search_query.return_value = None
    browser.semantic_btn.setChecked(True)
    browser.search_input.setText("a photo of a cat")

    browser._apply_filter()

    assert "border" in browser.search_input.styleSheet()


def test_unchecking_reverts_to_the_structured_filter(browser, session):
    browser.controller.embed_search_query.return_value = np.zeros(2, dtype=np.float32)
    browser.semantic_btn.setChecked(True)
    browser.search_input.setText("cats")
    browser._apply_filter()
    assert session.asset_model.semantic_query_active is True

    browser.semantic_btn.setChecked(False)
    browser.search_input.setText("a")
    browser._apply_filter()

    assert session.asset_model.semantic_query_active is False


def test_search_library_runs_the_semantic_variant_when_checked(browser):
    browser.semantic_btn.setChecked(True)
    browser.search_input.setText("a sunset")

    browser.search_library()

    browser.controller.request_library_semantic_search.assert_called_once_with("a sunset")
    browser.controller.request_library_search.assert_not_called()


def test_search_library_runs_the_keyword_variant_when_unchecked(browser):
    browser.search_input.setText("film:portra")

    browser.search_library()

    browser.controller.request_library_search.assert_called_once_with("film:portra")
    browser.controller.request_library_semantic_search.assert_not_called()


def test_search_library_stops_the_pending_live_filter_debounce(browser):
    """A keystroke just before Enter/click leaves filter_timer running. Left alone, it
    would fire _apply_filter after the hand-off replaces the file list -- reapplying the
    in-session query the hand-off itself just cleared, right back onto the new results."""
    browser.semantic_btn.setChecked(True)
    browser.search_input.setText("a sunset")  # setText's own textChanged starts the timer
    assert browser.filter_timer.isActive()

    browser.search_library()

    assert not browser.filter_timer.isActive()
