from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.view.main_window import MainWindow


def _window(*, has_rolls: bool, drawer_visible: bool) -> SimpleNamespace:
    win = SimpleNamespace(
        controller=MagicMock(),
        right_panel=MagicMock(),
        drawer=SimpleNamespace(isVisible=lambda: drawer_visible),
        toggle_controls_dock=MagicMock(),
    )
    win.controller.has_rolls.return_value = has_rolls
    return win


def test_show_library_opens_the_tab_and_shows_a_hidden_panel():
    win = _window(has_rolls=True, drawer_visible=False)

    MainWindow.show_library(win)

    win.toggle_controls_dock.assert_called_once_with()
    win.right_panel.show_tab_by_key.assert_called_once_with("library")
    win.right_panel.library_tree.prompt_import_folder.assert_not_called()


def test_an_empty_library_offers_an_import_unless_asked_not_to():
    win = _window(has_rolls=False, drawer_visible=True)

    MainWindow.show_library(win)
    MainWindow.show_library(win, ask_if_unset=False)

    win.right_panel.library_tree.prompt_import_folder.assert_called_once_with()
    win.toggle_controls_dock.assert_not_called()
