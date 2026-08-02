from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.sidebar.session_panel import SessionPanel
from negpy.infrastructure.storage.repository import StorageRepository


def _controller(tmp_path, roots: list[str]) -> MagicMock:
    """A mock controller around a real session — the film strip needs a real model."""
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    repo.save_global_setting("library_roots", roots)

    controller = MagicMock()
    controller.session = DesktopSessionManager(repo)
    controller.library_roots.return_value = roots
    return controller


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    # The update check would hit the network on construction.
    monkeypatch.setattr("negpy.desktop.view.sidebar.session_panel.check_for_updates", lambda: None)
    root = tmp_path / "library"
    root.mkdir()
    return SessionPanel(_controller(tmp_path, [str(root)]))


def test_tree_sits_above_the_film_strip(panel):
    assert panel.splitter.widget(0) is panel.library_tree
    assert panel.splitter.widget(1) is panel.file_browser


def test_tree_is_shown_when_the_library_has_roots(panel):
    assert panel.library_tree.isVisibleTo(panel)


def test_tree_hidden_when_no_roots(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.desktop.view.sidebar.session_panel.check_for_updates", lambda: None)

    panel = SessionPanel(_controller(tmp_path, []))

    assert not panel.library_tree.isVisibleTo(panel)


def test_splitter_height_is_restored_only_once_the_panel_has_one(panel):
    panel.controller.session.repo.save_global_setting("library_tree_height", 120)
    panel.resize(300, 900)
    panel.show()

    assert panel._splitter_restored
    assert panel.splitter.sizes()[0] == 120
    panel.hide()


def test_tree_browses_through_the_film_strip(panel, tmp_path):
    """A folder entered from the tree takes the same route as one entered from the
    sheet — browse first, prompt before loading."""
    (tmp_path / "library" / "roll_a").mkdir()

    panel.library_tree.folder_opened.emit(str(tmp_path / "library"), False)

    # An image-less folder just lists itself; nothing is hashed and nothing is loaded.
    assert [f.name for f in panel.file_browser.session.asset_model._folders] == ["roll_a"]
    panel.controller.open_library_folder.assert_not_called()


def test_changing_roots_drops_the_cached_walk(panel):
    panel.library_tree.roots_changed.emit()

    panel.controller.invalidate_library_walk.assert_called_once_with()


def test_toggle_hides_and_shows_the_tree(panel):
    panel.toggle_library_tree()
    assert not panel.library_tree.isVisibleTo(panel)

    panel.toggle_library_tree()
    assert panel.library_tree.isVisibleTo(panel)
