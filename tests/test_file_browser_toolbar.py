import pytest

from negpy.desktop.view.styles.templates import TOOLBAR_BUTTON_HEIGHT
from negpy.desktop.session import AssetListModel
from negpy.desktop.view.sidebar.session_panel import SessionPanel

from conftest import FakeController as _Controller, FakeRepo as _Repo


@pytest.fixture
def panel(qapp):
    controller = _Controller(_Repo())
    controller.session.asset_model = AssetListModel(controller.state)
    panel = SessionPanel(controller)
    panel.resize(300, 700)
    panel.show()
    qapp.processEvents()
    return panel


def test_sort_joins_the_library_toolbar(panel):
    """No top-level toolbar: Sort sits in LibraryTree's own row, sized like every other
    section-toolbar button rather than as a one-off."""
    browser = panel.file_browser
    tree = panel.library_tree

    assert tree.isAncestorOf(browser.sort_btn)
    assert browser.sort_btn.height() == TOOLBAR_BUTTON_HEIGHT
    assert browser.sort_btn in tree.toolbar.buttons


def test_both_section_toolbars_size_their_buttons_the_same(panel):
    browser = panel.file_browser
    tree = panel.library_tree

    heights = {b.height() for b in tree.toolbar.buttons + browser.film_strip_toolbar.buttons}

    assert heights == {TOOLBAR_BUTTON_HEIGHT}


def test_film_strip_toolbar_holds_roll_scoped_actions(panel):
    browser = panel.file_browser
    expected = [
        browser.save_roll_btn,
        browser.add_btn,
        browser.hot_folder_btn,
        browser.apply_btn,
        browser.roll_settings_btn,
        browser.update_thumbnails_btn,
        browser.unload_btn,
        browser.scenes_btn,
        browser.sheet_btn,
    ]
    assert browser.film_strip_toolbar.buttons == expected


def test_new_roll_lives_on_the_film_strip_section_header(panel):
    """Not a toolbar button (it wrapped the row to a second line): the section header's
    actions menu instead, next to the info and chevron icons."""
    browser = panel.file_browser
    assert not hasattr(browser, "new_roll_btn")
    actions_btn = browser.frames_section.actions_btn
    assert actions_btn is not None
    labels = [action.text() for action in actions_btn.menu().actions()]
    assert labels == ["New Roll…", "Reset Roll to Defaults…"]


def test_update_thumbnails_button_refreshes_the_whole_roll(panel):
    panel.file_browser.update_thumbnails_btn.click()
    panel.file_browser.controller.request_thumbnail_refresh.assert_called_once_with("roll")


def test_update_thumbnails_button_cancels_instead_while_running(panel):
    panel.file_browser.controller.thumbnail_refresh_running = True

    panel.file_browser.update_thumbnails_btn.click()

    panel.file_browser.controller.cancel_thumbnail_refresh.assert_called_once_with()
    panel.file_browser.controller.request_thumbnail_refresh.assert_not_called()


def test_update_thumbnails_button_reflects_the_running_state(panel):
    browser = panel.file_browser
    idle_tip = browser.update_thumbnails_btn.toolTip()

    browser._on_thumbnail_refresh_state_changed(True)
    running_tip = browser.update_thumbnails_btn.toolTip()
    assert running_tip != idle_tip
    assert "Cancel" in running_tip

    browser._on_thumbnail_refresh_state_changed(False)
    assert browser.update_thumbnails_btn.toolTip() == idle_tip


def test_narrowing_the_panel_raises_a_populated_overflow_menu(panel, qapp):
    """QToolBar's native extension menu was tried first and came up empty: widgets added with
    addWidget() become QWidgetActions its popup cannot host."""
    toolbar = panel.file_browser.film_strip_toolbar
    panel.resize(420, 700)
    qapp.processEvents()
    assert not toolbar.overflow_btn.isVisible()

    panel.resize(200, 700)
    qapp.processEvents()
    assert toolbar.overflow_btn.isVisible()

    labels = [action.text() for action in toolbar.build_overflow_menu().actions()]
    assert labels, "overflow button with an empty menu"
    assert "Sheet filter" in labels


def test_film_strip_toolbar_minimum_is_not_the_sum_of_its_buttons(panel):
    toolbar = panel.file_browser.film_strip_toolbar
    assert toolbar.minimumSizeHint().width() < toolbar.sizeHint().width()


def test_session_panel_shrinks_below_the_old_button_row_floor(panel):
    assert panel.minimumSizeHint().width() < 268


def test_a_switched_off_toolbar_button_stays_off_across_a_resize(panel):
    """_relayout shows whatever fits, so an opt-in button hidden with a bare
    setVisible would come back on the next resize."""
    tree = panel.library_tree

    assert not tree.index_btn.isVisible()

    tree.toolbar.resize(600, tree.toolbar.height())

    assert not tree.index_btn.isVisible()
    assert [a.text() for a in tree.toolbar.build_overflow_menu().actions()] == []


def test_turning_an_opt_in_button_on_places_it(panel):
    tree = panel.library_tree

    tree.toolbar.set_button_visible(tree.index_btn, True)
    tree.toolbar.resize(600, tree.toolbar.height())

    assert tree.index_btn.isVisible()
