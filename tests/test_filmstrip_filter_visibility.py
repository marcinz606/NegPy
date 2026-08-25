"""The Film Strip must say when a filter, not a load failure, emptied it.

The tally counts the session and the strip shows the model, so a filter that hides
every frame reads as a blank panel under a full count unless both are told about it.
"""

import pytest

from negpy.desktop.session import AssetListModel
from negpy.desktop.view.sidebar.session_panel import SessionPanel

from conftest import FakeController as _Controller, FakeRepo as _Repo


def _files(n, keepers=(), rejected=()):
    return [
        {
            "name": f"_DSC{1000 + i}.NEF",
            "path": f"/tmp/_DSC{1000 + i}.NEF",
            "hash": f"hash{i}",
            "keeper": i in keepers,
            "excluded": i in rejected,
        }
        for i in range(n)
    ]


def _browser(qapp, repo=None, files=None):
    controller = _Controller(repo if repo is not None else _Repo())
    controller.state.uploaded_files = files if files is not None else _files(36)
    controller.session.asset_model = AssetListModel(controller.state)
    panel = SessionPanel(controller)
    panel.resize(300, 700)
    panel.show()
    qapp.processEvents()
    return panel.file_browser


@pytest.fixture
def browser(qapp):
    return _browser(qapp)


def test_the_tally_counts_every_frame_when_nothing_is_filtered(browser):
    assert browser.tally_label.text() == "36 frames"
    assert browser.list_view.isVisible()
    assert not browser.empty_label.isVisible()


def test_the_tally_names_the_filter_that_emptied_the_strip(qapp):
    """The reported bug: a persisted Keepers filter over a roll with no keepers
    left a blank strip under a full count, with the funnel tint the only clue."""
    browser = _browser(qapp, repo=_Repo(sheet_filter="keepers"))
    assert browser.session.asset_model.rowCount() == 0
    assert browser.tally_label.text() == "0 of 36 frames · Keepers filter"


def test_the_strip_gives_way_to_a_message_when_a_filter_hides_everything(qapp):
    browser = _browser(qapp, repo=_Repo(sheet_filter="keepers"))
    assert browser.empty_label.isVisible()
    assert not browser.list_view.isVisible()
    assert "No frames match the Keepers filter" in browser.empty_label.text()


def test_clearing_from_the_empty_state_brings_every_frame_back(qapp):
    browser = _browser(qapp, repo=_Repo(sheet_filter="keepers"))
    browser._clear_frame_filters()
    qapp.processEvents()
    assert browser.session.asset_model.rowCount() == 36
    assert browser.tally_label.text() == "36 frames"
    assert browser.list_view.isVisible()
    assert not browser.empty_label.isVisible()


def test_a_filter_that_still_shows_frames_reports_the_shortfall(qapp):
    browser = _browser(qapp, repo=_Repo(sheet_filter="keepers"), files=_files(36, keepers=(0, 1, 2)))
    assert browser.session.asset_model.rowCount() == 3
    assert browser.tally_label.text() == "3 of 36 frames · Keepers filter · 3 keepers"
    assert browser.list_view.isVisible()
    assert not browser.empty_label.isVisible()


def test_the_search_filter_is_named_too(browser):
    browser.search_input.setText("nothing-matches-this")
    browser._apply_filter()
    assert browser.session.asset_model.rowCount() == 0
    assert browser.tally_label.text() == "0 of 36 frames · search filter"
    assert "No frames match the search filter" in browser.empty_label.text()


def test_both_filters_are_named_together(qapp):
    browser = _browser(qapp, repo=_Repo(sheet_filter="keepers"))
    browser.search_input.setText("nothing-matches-this")
    browser._apply_filter()
    assert browser.tally_label.text() == "0 of 36 frames · search filter · Keepers filter"
    assert "search and Keepers filter" in browser.empty_label.text()


def test_an_empty_session_shows_neither_tally_nor_message(qapp):
    """No frames loaded is not a filtered strip: the panel stays quiet."""
    browser = _browser(qapp, files=[])
    assert not browser.tally_label.isVisible()
    assert not browser.empty_label.isVisible()
    assert browser.list_view.isVisible()
