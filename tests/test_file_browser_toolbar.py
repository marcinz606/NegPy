from types import SimpleNamespace

import pytest

from PyQt6.QtCore import Qt

from negpy.desktop.view.styles.templates import BAR_BUTTON_HEIGHT, TOOLBAR_BUTTON_HEIGHT, BAR_TRACK_BUTTON_HEIGHT
from negpy.desktop.session import AssetListModel
from negpy.desktop.view.sidebar.files import FileBrowser

from conftest import FakeController as _Controller, FakeRepo as _Repo


@pytest.fixture
def panel(qapp):
    controller = _Controller(_Repo())
    controller.session.asset_model = AssetListModel(controller.state)
    browser = FileBrowser(controller)
    browser.resize(300, 700)
    browser.show()
    browser.library_tree.resize(300, 400)
    browser.library_tree.show()
    qapp.processEvents()
    return SimpleNamespace(file_browser=browser, library_tree=browser.library_tree)


def test_each_section_toolbar_has_its_own_sort(panel):
    """The Film Strip's Sort orders the frames, the Library's the roll list. The strip's sits
    in the canvas bar, inside a track as tall as the bar's other boxes."""
    browser = panel.file_browser
    tree = panel.library_tree

    assert browser.roll_row.isAncestorOf(browser.sort_btn)
    assert tree.isAncestorOf(tree.sort_btn)
    assert browser.sort_btn is not tree.sort_btn
    assert (browser.sort_btn.height(), tree.sort_btn.height()) == (BAR_TRACK_BUTTON_HEIGHT, TOOLBAR_BUTTON_HEIGHT)
    assert browser.search_input.height() == BAR_BUTTON_HEIGHT


def test_the_roll_row_buttons_are_icons_and_a_filter_on_takes_the_chip(panel):
    browser = panel.file_browser
    buttons = (browser.sort_btn, browser.sheet_btn, browser.roll_settings_btn, browser.apply_btn)
    assert all(btn.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly and not btn.text() for btn in buttons)

    browser._apply_sheet_filter("keepers", save=False)
    assert browser.sheet_btn.property("filtering") is True
    browser._apply_sheet_filter("all", save=False)
    assert browser.sheet_btn.property("filtering") is False


def test_roll_settings_sits_before_apply_and_opens_its_dialog(panel, monkeypatch):
    browser = panel.file_browser
    layout = browser.roll_row.layout()
    opened = []
    monkeypatch.setattr(browser, "_build_roll_settings_dialog", lambda: opened.append(True) or None)

    browser.roll_settings_btn.click()

    assert layout.indexOf(browser.roll_settings_btn) + 1 == layout.indexOf(browser.apply_btn)
    assert opened == [True]


def test_the_roll_menu_holds_the_actions_out_of_view(panel):
    browser = panel.file_browser
    labels = [action.text() for action in browser.roll_menu_btn.menu().actions() if not action.isSeparator()]
    assert [label.split("\t")[0] for label in labels] == [
        "Add Files…",
        "Add Folder…",
        "Save to Folder Roll",
        "Save as Virtual Roll…",
        "Update Thumbnails",
        "Hot Folder",
        "Unload…",
        "Unload All…",
        "Reset Roll to Defaults…",
        "Restore Removed Frames…",
        "Film Strip Guide…",
    ]


def test_the_hot_folder_button_names_the_folder_it_watches(panel):
    browser = panel.file_browser
    files = browser.session.state.uploaded_files
    files[:] = [{"name": "a.tif", "path": "/scans/500T/a.tif", "hash": "h1"}, {"name": "b.tif", "path": "/scans/HP5/b.tif", "hash": "h2"}]
    browser.hot_folder_action.setChecked(True)
    assert "/scans/HP5" in browser.hot_folder_btn.toolTip(), "the newest frame's folder, the one the poll walks"
    files.clear()
    browser._update_tally()
    assert "Hot Folder is on and waits for a frame to load." in browser.hot_folder_btn.toolTip()
    assert browser.hot_folder_action.isEnabled(), "a Hot Folder that is on can always turn off"
    browser.hot_folder_action.setChecked(False)
    assert not browser.hot_folder_action.isEnabled()
    assert browser.hot_folder_action.toolTip().startswith("Load a frame first")
    files.append({"name": "a.tif", "path": "/scans/500T/a.tif", "hash": "h1"})
    browser._update_tally()
    assert browser.hot_folder_action.isEnabled()


def test_the_hot_folder_shows_in_the_row_while_it_is_on(panel):
    browser = panel.file_browser
    assert browser.hot_folder_btn.isHidden()

    browser.hot_folder_action.setChecked(True)
    assert not browser.hot_folder_btn.isHidden() and browser.scan_timer.isActive()

    browser.hot_folder_btn.click()
    assert not browser.hot_folder_action.isChecked()
    assert browser.hot_folder_btn.isHidden() and not browser.scan_timer.isActive()


@pytest.mark.parametrize("restored", [True, False])
def test_the_hot_folder_comes_back_only_with_a_restored_session(qapp, restored):
    """Hot Folder belongs to the saved session: a restore turns it back on, a fresh start forgets it."""
    repo = _Repo()
    controller = _Controller(repo)
    controller.session.asset_model = AssetListModel(controller.state)
    FileBrowser(controller).hot_folder_action.setChecked(True)

    controller = _Controller(repo)
    controller.session.asset_model = AssetListModel(controller.state)
    browser = FileBrowser(controller)
    browser.restore_hot_folder(restored)

    assert browser.hot_folder_action.isChecked() is restored
    assert browser.scan_timer.isActive() is restored
    assert repo.get_global_setting("session_hot_folder") is restored


def test_update_thumbnails_refreshes_the_whole_roll(panel):
    panel.file_browser.update_thumbnails_action.trigger()
    panel.file_browser.controller.request_thumbnail_refresh.assert_called_once_with("roll")


def test_update_thumbnails_cancels_instead_while_running(panel):
    panel.file_browser.controller.thumbnail_refresh_running = True

    panel.file_browser.update_thumbnails_action.trigger()

    panel.file_browser.controller.cancel_thumbnail_refresh.assert_called_once_with()
    panel.file_browser.controller.request_thumbnail_refresh.assert_not_called()


def test_update_thumbnails_reflects_the_running_state(panel):
    browser = panel.file_browser
    idle = browser.update_thumbnails_action.text()

    browser._on_thumbnail_refresh_state_changed(True)
    assert browser.update_thumbnails_action.text() == "Cancel Thumbnail Update"

    browser._on_thumbnail_refresh_state_changed(False)
    assert browser.update_thumbnails_action.text() == idle


def test_a_narrow_row_moves_save_as_roll_to_the_menu(panel, qapp):
    browser = panel.file_browser
    row = browser.roll_row
    browser._show_save_roll(True)
    row.resize(1400, 40)
    row.show()
    qapp.processEvents()
    assert not browser.save_roll_btn.isHidden()

    row.resize(400, 40)
    qapp.processEvents()
    assert browser.save_roll_btn.isHidden()

    row.resize(1400, 40)
    qapp.processEvents()
    assert not browser.save_roll_btn.isHidden()
