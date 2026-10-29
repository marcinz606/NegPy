"""The Film Strip must say when a filter, not a load failure, emptied it.

The tally counts the session and the strip shows the model, so a filter that hides
every frame reads as a blank panel under a full count unless both are told about it.
"""

import pytest

from negpy.desktop.session import AssetListModel
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.services.assets.rolls import create_virtual_roll

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
    panel = FileBrowser(controller)
    panel.resize(300, 700)
    panel.show()
    qapp.processEvents()
    return panel


@pytest.fixture
def browser(qapp):
    return _browser(qapp)


def test_the_tally_counts_every_frame_when_nothing_is_filtered(browser):
    assert browser.tally_label.text() == "36 frames"
    assert browser.list_view.isVisible()
    assert not browser.empty_label.isVisible()


def test_the_tally_counts_what_a_filter_leaves_and_the_button_shows_it_is_on(qapp):
    """A persisted Keepers filter over a roll with no keepers must not read as a full count over a blank strip."""
    browser = _browser(qapp, repo=_Repo(sheet_filter="keepers"))
    assert browser.session.asset_model.rowCount() == 0
    assert browser.tally_label.text() == "0 of 36 frames"
    assert browser.sheet_btn.property("filtering") is True


def test_an_unknown_saved_filter_and_sort_fall_back_to_all_and_name(qapp):
    browser = _browser(qapp, repo=_Repo(sheet_filter="rejected", file_sort_order="size"))
    assert browser.session.asset_model.sheet_filter == "all"
    assert browser.act_sheet_all.isChecked()
    assert browser.session.asset_model.effective_sort_order == "name"
    assert browser.act_sort_name.isChecked()


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
    assert browser.tally_label.text() == "3 of 36 frames · 3 keepers"
    assert browser.list_view.isVisible()
    assert not browser.empty_label.isVisible()


def test_the_empty_strip_names_the_search_filter_too(browser):
    browser.search_input.setText("nothing-matches-this")
    browser._apply_filter()
    assert browser.session.asset_model.rowCount() == 0
    assert browser.tally_label.text() == "0 of 36 frames"
    assert "No frames match the search filter" in browser.empty_label.text()


def test_the_empty_strip_names_both_filters_together(qapp):
    browser = _browser(qapp, repo=_Repo(sheet_filter="keepers"))
    browser.search_input.setText("nothing-matches-this")
    browser._apply_filter()
    assert browser.tally_label.text() == "0 of 36 frames"
    assert "search and Keepers filter" in browser.empty_label.text()


def test_the_row_leads_with_the_active_rolls_name(qapp):
    repo = _Repo()
    roll_id = create_virtual_roll(repo, "Portra 400", [])
    browser = _browser(qapp, repo=repo)
    browser.controller.state.active_roll_id = roll_id

    browser._update_tally()

    assert browser.roll_name_label.text() == "Portra 400"
    assert browser.tally_label.text() == "36 frames"
    assert browser.save_roll_btn.isHidden()


def test_frames_that_are_not_one_roll_lead_with_collection_and_say_what_that_costs(qapp):
    """The name slot holds the roll's name, so a batch that is not one roll would show
    nothing there and look exactly like a roll -- while an edit in it reaches the roll each
    frame came from. Named, plus a tooltip for the consequence and the save button beside it."""
    browser = _browser(qapp)
    browser.controller.state.active_roll_id = None

    browser._update_tally()

    assert browser.roll_name_label.text() == "No roll"
    assert "also shows in the roll the frame came from" in browser.roll_title.toolTip()
    assert not browser.save_roll_btn.isHidden()


def test_the_active_rolls_name_leaves_no_collection_tooltip(qapp):
    repo = _Repo()
    roll_id = create_virtual_roll(repo, "Portra 400", [])
    browser = _browser(qapp, repo=repo)
    browser.controller.state.active_roll_id = roll_id

    browser._update_tally()

    assert browser.roll_title.toolTip() == ""


def test_an_empty_session_shows_neither_tally_nor_message(qapp):
    """No frames loaded is not a filtered strip: the panel stays quiet."""
    browser = _browser(qapp, files=[])
    assert not browser.roll_title.isVisible()
    assert not browser.empty_label.isVisible()
    assert browser.list_view.isVisible()


def test_a_long_tally_does_not_hold_the_panel_wide(qapp):
    browser = _browser(qapp, files=_files(36, keepers=(1, 2), rejected=(3,)))
    label = browser.tally_label
    assert label.minimumSizeHint().width() == 0
    label.setFixedWidth(40)
    assert label.is_elided()
    assert label.text() == "36 frames · 2 keepers · 1 rejected"


def test_a_thumbnail_refresh_leads_the_tally_and_leaves_with_it(browser):
    browser._on_thumbnail_refresh_progress("Thumbnails 3/36 · ~1 min left")
    assert browser.tally_label.text() == "Thumbnails 3/36 · ~1 min left · 36 frames"
    browser._on_thumbnail_refresh_progress("")
    assert browser.tally_label.text() == "36 frames"
