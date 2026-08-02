from unittest.mock import MagicMock

import pytest

from negpy.desktop.view.sidebar.library_tree import LibraryTree
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def tree_dirs(tmp_path):
    """scans/ with a container folder, two rolls, and a roll holding a subfolder."""
    (tmp_path / "scans" / "roll_a").mkdir(parents=True)
    (tmp_path / "scans" / "roll_b" / "rescans").mkdir(parents=True)
    (tmp_path / "scans" / "empty_box" / "nothing").mkdir(parents=True)
    (tmp_path / "scans" / ".hidden").mkdir()
    (tmp_path / "scans" / "roll_a" / "a1.NEF").write_bytes(b"1")
    (tmp_path / "scans" / "roll_a" / "a2.NEF").write_bytes(b"2")
    (tmp_path / "scans" / "roll_b" / "b1.NEF").write_bytes(b"3")
    (tmp_path / "scans" / "roll_b" / "rescans" / "b1_v2.NEF").write_bytes(b"4")
    return tmp_path / "scans"


def _make(tmp_path, roots) -> LibraryTree:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    repo.save_global_setting("library_roots", [str(r) for r in roots])
    controller = MagicMock()
    controller.session.repo = repo
    return LibraryTree(controller)


@pytest.fixture
def widget(qapp, tmp_path, tree_dirs):
    return _make(tmp_path, [tree_dirs])


def _labels(item) -> list[str]:
    return [item.child(i).text(0) for i in range(item.childCount())]


def _counts(item) -> list[str]:
    return [item.child(i).text(1) for i in range(item.childCount())]


def test_roots_appear_as_top_level_items(widget, tree_dirs):
    assert widget.tree.topLevelItem(0).text(0) == "scans"


def test_children_are_read_only_on_expand(widget):
    root = widget.tree.topLevelItem(0)
    assert _labels(root) == ["__unpopulated__"]

    root.setExpanded(True)
    assert _labels(root) == ["empty_box", "roll_a", "roll_b"]  # dot-dirs excluded


def test_each_folder_reports_what_is_inside(widget):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)

    assert _counts(root) == ["1 folder", "2 photos", "1 photo · 1 folder"]


def test_a_leaf_folder_has_no_expander(widget):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    roll_a = root.child(1)

    assert roll_a.childCount() == 0  # roll_a holds photos but no subfolders


def test_expanding_twice_does_not_duplicate_children(widget):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    root.setExpanded(False)
    root.setExpanded(True)

    assert _labels(root) == ["empty_box", "roll_a", "roll_b"]


def test_expanded_and_selected_folders_survive_a_reload(widget):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    root.child(1).setSelected(True)

    widget.reload()

    reloaded = widget.tree.topLevelItem(0)
    assert reloaded.isExpanded()
    assert [i.text(0) for i in widget.tree.selectedItems()] == ["roll_a"]


# --- opening ------------------------------------------------------------------


def test_a_single_click_does_not_open_anything(widget):
    opened = []
    widget.folders_activated.connect(opened.append)
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)

    widget.tree.itemClicked.emit(root.child(1), 0)

    assert opened == []


def test_double_click_opens_that_folder(widget, tree_dirs):
    opened = []
    widget.folders_activated.connect(opened.append)
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)

    widget._on_double_clicked(root.child(1), 0)

    assert opened == [[str(tree_dirs / "roll_a")]]


def test_opening_a_row_inside_a_selection_opens_the_whole_selection(widget, tree_dirs):
    opened = []
    widget.folders_activated.connect(opened.append)
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    root.child(1).setSelected(True)
    root.child(2).setSelected(True)

    widget._on_double_clicked(root.child(1), 0)

    assert opened == [[str(tree_dirs / "roll_a"), str(tree_dirs / "roll_b")]]


def test_opening_a_row_outside_the_selection_opens_only_that_row(widget, tree_dirs):
    opened = []
    widget.folders_activated.connect(opened.append)
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    root.child(1).setSelected(True)

    widget._on_double_clicked(root.child(2), 0)

    assert opened == [[str(tree_dirs / "roll_b")]]


def test_select_parent_moves_up_one_level(widget, tree_dirs):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    widget.tree.setCurrentItem(root.child(1))

    widget.select_parent()

    assert widget.tree.currentItem().text(0) == "scans"


def test_select_parent_at_a_root_does_nothing(widget):
    widget.tree.setCurrentItem(widget.tree.topLevelItem(0))

    widget.select_parent()

    assert widget.tree.currentItem().text(0) == "scans"


def test_reveal_selects_and_expands_a_folder(widget, tree_dirs):
    widget.reveal(str(tree_dirs / "roll_b"))

    current = widget.tree.currentItem()
    assert current.text(0) == "roll_b"
    assert current.isSelected() and current.isExpanded()


# --- sorting ------------------------------------------------------------------


def test_folders_follow_the_sheet_sort(widget):
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    assert _labels(root) == ["empty_box", "roll_a", "roll_b"]

    widget.set_sort("name", True)
    assert _labels(widget.tree.topLevelItem(0)) == ["roll_b", "roll_a", "empty_box"]


def test_sorting_by_date_uses_folder_mtime(widget, tree_dirs):
    import os
    import time

    now = time.time()
    os.utime(tree_dirs / "roll_a", (now - 10_000, now - 10_000))
    os.utime(tree_dirs / "roll_b", (now - 5_000, now - 5_000))
    os.utime(tree_dirs / "empty_box", (now, now))

    widget.set_sort("date", False)
    widget.tree.topLevelItem(0).setExpanded(True)

    assert _labels(widget.tree.topLevelItem(0)) == ["roll_a", "roll_b", "empty_box"]


# --- roots --------------------------------------------------------------------


def test_remove_root_persists_and_reloads(widget, tree_dirs):
    widget.remove_root(str(tree_dirs))

    assert widget.roots() == []
    assert widget.tree.topLevelItemCount() == 0
    assert widget.empty_label.isVisibleTo(widget)


def test_add_root_puts_the_new_folder_first(qapp, tmp_path, tree_dirs):
    widget = _make(tmp_path, [tree_dirs / "roll_a"])

    widget.add_root(str(tree_dirs))

    assert widget.roots() == [str(tree_dirs), str(tree_dirs / "roll_a")]
    assert widget.primary_root() == str(tree_dirs)


def test_primary_root_skips_a_folder_that_is_gone(qapp, tmp_path, tree_dirs):
    widget = _make(tmp_path, [tmp_path / "unplugged", tree_dirs])

    assert widget.primary_root() == str(tree_dirs)


def test_selection_uses_the_accent_colour(widget):
    assert THEME.accent_primary in widget.tree.styleSheet()


def test_empty_library_shows_the_hint(qapp, tmp_path):
    widget = _make(tmp_path, [])

    assert widget.tree.topLevelItemCount() == 0
    assert widget.empty_label.isVisibleTo(widget)


def test_enter_opens_the_selection(widget, tree_dirs):
    opened = []
    widget.folders_activated.connect(opened.append)
    root = widget.tree.topLevelItem(0)
    root.setExpanded(True)
    root.child(2).setSelected(True)

    widget.open_selection()

    assert opened == [[str(tree_dirs / "roll_b")]]


def test_enter_with_nothing_selected_opens_nothing(widget):
    opened = []
    widget.folders_activated.connect(opened.append)

    widget.open_selection()

    assert opened == []
