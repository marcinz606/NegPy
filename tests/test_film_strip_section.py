from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from negpy.desktop.view.main_window import MainWindow


@pytest.fixture
def win(qapp):
    splitter = QSplitter(Qt.Orientation.Vertical)
    pane = QWidget()
    strip = QWidget()
    QVBoxLayout(pane).addWidget(strip)
    toolbar = MagicMock()
    toolbar.sizeHint.return_value = QSize(400, 30)
    splitter.addWidget(QWidget())
    splitter.addWidget(pane)
    splitter.resize(400, 800)
    splitter.show()
    splitter.setSizes([500, 300])
    qapp.processEvents()
    win = SimpleNamespace(
        file_browser=strip,
        strip_pane=pane,
        strip_splitter=splitter,
        toolbar=toolbar,
        _strip_open_height=300,
        _strip_handle_width=splitter.handleWidth(),
        controller=MagicMock(),
        state=SimpleNamespace(canvas_bg_index=0),
    )
    for name in ("film_strip_expanded", "set_film_strip_expanded", "_apply_strip_constraints"):
        setattr(win, name, getattr(MainWindow, name).__get__(win))
    yield win
    splitter.hide()


def test_a_collapsed_strip_is_its_toolbar_and_the_canvas_takes_the_rest(win, qapp):
    before = win.strip_splitter.sizes()
    handle = win.strip_splitter.handleWidth()

    MainWindow.toggle_film_strip(win)
    qapp.processEvents()
    assert win.strip_splitter.handleWidth() == 0, "the canvas sits on the toolbar"
    assert win.strip_splitter.sizes()[1] == 30
    assert win.file_browser.isHidden()
    win.toolbar.set_strip_expanded.assert_called_with(False)
    win.controller.session.repo.save_global_setting.assert_called_with("section_expanded_frames", False)

    MainWindow.toggle_film_strip(win)
    qapp.processEvents()
    assert win.strip_splitter.sizes()[1] == before[1]
    assert win.strip_splitter.handleWidth() == handle
    assert not win.file_browser.isHidden()


def test_the_handle_above_the_strip_takes_the_table_color(win):
    from negpy.desktop.view.canvas.toolbar import CANVAS_COLORS

    MainWindow._sync_strip_handle(win)

    assert f"background: {CANVAS_COLORS[0][0]}" in win.strip_splitter.styleSheet()


def test_only_an_open_strip_saves_its_height(win):
    win.set_film_strip_expanded(False)
    save = win.controller.session.repo.save_global_setting
    save.reset_mock()
    MainWindow._on_strip_splitter_moved(win)
    save.assert_not_called()

    win.set_film_strip_expanded(True)
    save.reset_mock()
    MainWindow._on_strip_splitter_moved(win)
    save.assert_called_once_with("film_strip_splitter_sizes", win.strip_splitter.sizes())


def test_the_filter_key_opens_the_roll_row_and_the_strip(win):
    win.file_browser.focus_search = MagicMock()
    win.set_film_strip_expanded(False)

    MainWindow.focus_search(win)

    win.toolbar.set_mode.assert_called_once_with("roll")
    assert win.film_strip_expanded()
    win.file_browser.focus_search.assert_called_once()
