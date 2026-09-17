import pytest
from negpy.desktop.view.widgets.overflow_bar import OverflowBar

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


def _toolbar(panel) -> OverflowBar:
    return panel.file_browser.findChild(OverflowBar)


def test_toolbar_minimum_is_not_the_sum_of_its_buttons(panel):
    """The regression this guards: a plain QHBoxLayout made the session panel unshrinkable
    below every button laid end to end, so each new tool widened the panel for good."""
    toolbar = _toolbar(panel)
    assert toolbar.minimumSizeHint().width() < toolbar.sizeHint().width() / 2


def test_toolbar_keeps_every_action(panel):
    browser = panel.file_browser
    toolbar = _toolbar(panel)
    expected = [
        browser.library_btn,
        browser.add_files_btn,
        browser.add_folder_btn,
        browser.unload_btn,
        browser.hot_folder_btn,
        browser.rgb_scan_btn,
        browser.half_frame_btn,
        browser.half_frame_menu_btn,
        browser.apply_btn,
        browser.update_thumbnails_btn,
        browser.sheet_btn,
        browser.sort_btn,
    ]
    assert toolbar.buttons == expected


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
    toolbar = _toolbar(panel)
    panel.resize(420, 700)
    qapp.processEvents()
    assert not toolbar.overflow_btn.isVisible()

    panel.resize(200, 700)
    qapp.processEvents()
    assert toolbar.overflow_btn.isVisible()

    labels = [action.text() for action in toolbar.build_overflow_menu().actions()]
    assert labels, "overflow button with an empty menu"
    assert "Sort" in labels


def test_session_panel_shrinks_below_the_old_button_row_floor(panel):
    assert panel.minimumSizeHint().width() < 268
