from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QListView, QStackedWidget, QToolButton, QWidget

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.keyboard_shortcuts import _context_cancel
from negpy.desktop.view.main_window import MainWindow
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.infrastructure.storage.repository import StorageRepository


class _Dock:
    def __init__(self, visible=True, floating=False):
        self.visible = visible
        self.floating = floating

    def isVisible(self):
        return self.visible

    def isFloating(self):
        return self.floating

    def setVisible(self, visible):
        self.visible = visible


def _window():
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    grid = QListView()
    stack.addWidget(grid)
    button = QToolButton()
    button.setCheckable(True)
    win = SimpleNamespace(
        central_stack=stack,
        drawer=_Dock(),
        session_dock=_Dock(),
        session_panel=SimpleNamespace(file_browser=SimpleNamespace(light_table_btn=button, light_table_view=grid)),
        controller=MagicMock(),
        state=SimpleNamespace(selected_file_idx=0),
    )
    win.controller.session.asset_model.actual_to_display.return_value = -1
    win.light_table_active = lambda: MainWindow.light_table_active(win)
    return win


def test_the_light_table_takes_the_canvas_place_and_gives_it_back(qapp):
    win = _window()

    MainWindow.set_light_table(win, True)
    assert win.central_stack.currentIndex() == 1
    assert win.session_panel.file_browser.light_table_btn.isChecked()
    assert not win.drawer.visible

    MainWindow.set_light_table(win, False)
    assert win.central_stack.currentIndex() == 0
    assert not win.session_panel.file_browser.light_table_btn.isChecked()
    assert win.drawer.visible


def test_leaving_the_light_table_keeps_a_hidden_controls_panel_hidden(qapp):
    win = _window()
    win.drawer.visible = False

    MainWindow.set_light_table(win, True)
    MainWindow.set_light_table(win, False)

    assert not win.drawer.visible


def test_escape_closes_the_light_table_before_anything_else(qapp):
    win = _window()
    MainWindow.set_light_table(win, True)
    win.set_light_table = lambda on: MainWindow.set_light_table(win, on)
    controller = MagicMock()

    _context_cancel(controller, win)

    assert not win.light_table_active()
    controller.cancel_active_tool.assert_not_called()


def test_one_key_hides_both_panels_then_brings_both_back(qapp):
    win = _window()

    MainWindow.toggle_side_panels(win)
    assert not win.drawer.visible and not win.session_dock.visible
    MainWindow.toggle_side_panels(win)
    assert win.drawer.visible and win.session_dock.visible
    saved = win.controller.session.repo.save_global_setting.call_args_list
    assert saved[-2:] == [(("panel_left_visible", True),), (("panel_right_visible", True),)]


@pytest.fixture
def browser(qapp):
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.load_file_settings_many.return_value = {}
    repo.get_max_history_index.return_value = 0
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": f"{i}.tif", "path": f"/r/{i}.tif", "hash": f"h{i}"} for i in range(3)]
    session.asset_model.refresh()
    controller = MagicMock()
    controller.session = session
    controller.thumbnail_refresh_running = False
    return FileBrowser(controller)


def test_the_light_table_shares_the_strip_selection(browser):
    assert browser.light_table_view.model() is browser.list_view.model()
    assert browser.light_table_view.selectionModel() is browser.list_view.selectionModel()


def test_opening_a_frame_from_the_light_table_asks_for_the_canvas(browser, monkeypatch):
    opened, left = [], []
    monkeypatch.setattr(browser, "_activate_file", opened.append)
    browser.light_table_opened.connect(lambda: left.append(True))
    index = browser.light_table_view.model().index(1, 0)

    browser._open_from_light_table(index)

    assert opened == [index] and left == [True]
