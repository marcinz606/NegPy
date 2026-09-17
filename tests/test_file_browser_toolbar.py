import pytest

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


def test_sort_joins_the_library_tree_corner_row(panel):
    """No top-level toolbar: Sort sits with LibraryTree's own +/refresh, sized to match
    (20x20), not the taller film-strip-toolbar convention."""
    browser = panel.file_browser
    tree = panel.library_tree

    assert tree.isAncestorOf(browser.sort_btn)
    assert (browser.sort_btn.width(), browser.sort_btn.height()) == (20, 20)


def test_film_strip_toolbar_holds_roll_scoped_actions(panel):
    browser = panel.file_browser
    expected = [
        browser.save_roll_btn,
        browser.add_btn,
        browser.hot_folder_btn,
        browser.rgb_scan_btn,
        browser.half_frame_btn,
        browser.half_frame_menu_btn,
        browser.apply_btn,
        browser.roll_settings_btn,
        browser.unload_btn,
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
