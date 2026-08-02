from unittest.mock import MagicMock

import pytest

from negpy.desktop.view.sidebar.library_tree import LibraryTree
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def tree_dirs(tmp_path):
    (tmp_path / "roll_a" / "2024" / "march").mkdir(parents=True)
    (tmp_path / "roll_a" / "2023").mkdir()
    (tmp_path / "roll_b").mkdir()
    (tmp_path / "roll_a" / ".hidden").mkdir()
    (tmp_path / "roll_a" / "IMG_0001.NEF").write_bytes(b"a")
    return tmp_path


@pytest.fixture
def widget(qapp, tmp_path, tree_dirs):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    repo.save_global_setting("library_roots", [str(tree_dirs / "roll_a"), str(tree_dirs / "roll_b")])
    controller = MagicMock()
    controller.session.repo = repo
    return LibraryTree(controller)


def _labels(item) -> list[str]:
    return [item.child(i).text(0) for i in range(item.childCount())]


def test_roots_appear_as_top_level_items(widget):
    assert [widget.tree.topLevelItem(i).text(0) for i in range(widget.tree.topLevelItemCount())] == ["roll_a", "roll_b"]


def test_children_are_read_only_on_expand(widget):
    root = widget.tree.topLevelItem(0)
    assert _labels(root) == ["__unpopulated__"]

    root.setExpanded(True)
    assert _labels(root) == ["2023", "2024"]  # sorted, files and dot-dirs excluded


def test_leaf_folder_has_no_expander(widget):
    roll_b = widget.tree.topLevelItem(1)
    assert roll_b.childCount() == 0


def test_expanding_twice_does_not_duplicate_children(widget):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    root.setExpanded(False)
    root.setExpanded(True)
    assert _labels(root) == ["2023", "2024"]


def test_click_asks_to_open_that_folder(widget, tree_dirs):
    opened = []
    widget.folder_opened.connect(lambda path, add: opened.append((path, add)))

    widget._on_clicked(widget.tree.topLevelItem(1), 0)

    assert opened == [(str(tree_dirs / "roll_b"), False)]


def test_remove_root_persists_and_reloads(widget, tree_dirs):
    widget.remove_root(str(tree_dirs / "roll_a"))

    assert widget.roots() == [str(tree_dirs / "roll_b")]
    assert widget.tree.topLevelItemCount() == 1


def test_expanded_folders_survive_a_reload(widget):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)

    widget.reload()

    reloaded = widget.tree.topLevelItem(0)
    assert reloaded.isExpanded()
    assert _labels(reloaded) == ["2023", "2024"]


def test_empty_library_shows_the_hint(qapp, tmp_path):
    repo = StorageRepository(str(tmp_path / "e.db"), str(tmp_path / "s.db"))
    repo.initialize()
    controller = MagicMock()
    controller.session.repo = repo

    widget = LibraryTree(controller)

    assert widget.tree.topLevelItemCount() == 0
    assert widget.empty_label.isVisible() or not widget.isVisible()
