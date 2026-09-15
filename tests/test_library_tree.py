from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from negpy.desktop.view.sidebar.library_tree import LibraryTree
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.rolls import create_virtual_roll, recognize_folder, roll_for_id


@pytest.fixture
def tree_dirs(tmp_path):
    """scans/ with two roll folders, one still un-imported."""
    (tmp_path / "scans" / "roll_a").mkdir(parents=True)
    (tmp_path / "scans" / "roll_b").mkdir(parents=True)
    (tmp_path / "scans" / "roll_a" / "a1.NEF").write_bytes(b"1")
    (tmp_path / "scans" / "roll_a" / "a2.NEF").write_bytes(b"2")
    (tmp_path / "scans" / "roll_b" / "b1.NEF").write_bytes(b"3")
    return tmp_path / "scans"


def _make(tmp_path) -> LibraryTree:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    controller = MagicMock()
    controller.session.repo = repo
    return LibraryTree(controller)


@pytest.fixture
def widget(qapp, tmp_path):
    return _make(tmp_path)


def _names(widget) -> list[str]:
    return [widget.tree.topLevelItem(i).text(0) for i in range(widget.tree.topLevelItemCount())]


# --- listing --------------------------------------------------------------


def test_empty_library_shows_the_hint(widget):
    assert widget.tree.topLevelItemCount() == 0
    assert widget.empty_label.isVisibleTo(widget)


def test_folder_and_virtual_rolls_appear_together_sorted_by_name(widget, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_b"), name="Zebra")
    create_virtual_roll(widget.repo, "apple", [])
    widget.reload()

    assert _names(widget) == ["apple", "Zebra"]
    assert not widget.empty_label.isVisibleTo(widget)


def test_a_folder_roll_shows_an_amber_icon_and_its_live_count(widget, tree_dirs, monkeypatch):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    colors = []
    monkeypatch.setattr(
        "negpy.desktop.view.sidebar.library_tree.qta.icon",
        lambda name, color=None: colors.append(color) or QIcon(),
    )

    widget.reload()

    assert colors == [THEME.mode_c41]
    assert widget.tree.topLevelItem(0).text(1) == "2 photos"


def test_a_virtual_roll_shows_a_red_icon_and_its_member_count(widget, monkeypatch):
    create_virtual_roll(widget.repo, "Portra", ["/a.nef", "/b.nef"])
    colors = []
    monkeypatch.setattr(
        "negpy.desktop.view.sidebar.library_tree.qta.icon",
        lambda name, color=None: colors.append(color) or QIcon(),
    )

    widget.reload()

    assert colors == [THEME.roll_virtual]
    assert widget.tree.topLevelItem(0).text(1) == "2 photos"


def test_selection_survives_a_reload(widget, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    recognize_folder(widget.repo, str(tree_dirs / "roll_b"))
    widget.reload()
    widget.tree.setCurrentItem(widget.tree.topLevelItem(1))

    widget.reload()

    assert widget.tree.currentItem().text(0) == "roll_b"


def test_selection_uses_the_accent_color(widget):
    assert THEME.accent_primary in widget.tree.styleSheet()


# --- sorting ----------------------------------------------------------------


def test_rolls_follow_name_sort(widget):
    create_virtual_roll(widget.repo, "apple", [])
    create_virtual_roll(widget.repo, "Zebra", [])
    widget.reload()
    assert _names(widget) == ["apple", "Zebra"]

    widget.set_sort("name", True)

    assert _names(widget) == ["Zebra", "apple"]


def test_rolls_follow_date_sort_by_created_at(widget, monkeypatch):
    import negpy.services.assets.rolls as rolls_module

    times = iter([100.0, 200.0])
    monkeypatch.setattr(rolls_module.time, "time", lambda: next(times))
    create_virtual_roll(widget.repo, "older", [])
    create_virtual_roll(widget.repo, "newer", [])

    widget.set_sort("date", False)

    assert _names(widget) == ["older", "newer"]


# --- opening ------------------------------------------------------------------


def test_double_click_opens_the_roll(widget):
    roll_id = create_virtual_roll(widget.repo, "Portra", ["/a.nef"])
    widget.reload()

    widget._on_double_clicked(widget.tree.topLevelItem(0), 0)

    widget.controller.open_roll.assert_called_once_with(roll_id)


def test_enter_opens_the_selected_roll(widget):
    roll_id = create_virtual_roll(widget.repo, "Portra", ["/a.nef"])
    widget.reload()
    widget.tree.setCurrentItem(widget.tree.topLevelItem(0))

    widget.open_selection()

    widget.controller.open_roll.assert_called_once_with(roll_id)


def test_enter_with_nothing_selected_opens_nothing(widget):
    widget.open_selection()

    widget.controller.open_roll.assert_not_called()


# --- rename / delete ------------------------------------------------------------


def test_renaming_a_roll(widget, monkeypatch):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Portra 400", True)))

    widget._rename_roll(roll_id, "Portra")

    assert roll_for_id(widget.repo, roll_id)["name"] == "Portra 400"
    assert widget.tree.topLevelItem(0).text(0) == "Portra 400"


def test_renaming_to_an_invalid_name_is_rejected(widget, monkeypatch):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("bad/name", True)))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    widget._rename_roll(roll_id, "Portra")

    assert roll_for_id(widget.repo, roll_id)["name"] == "Portra"


def test_deleting_a_roll(widget, monkeypatch):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_named", lambda *a, **k: True)

    widget._delete_roll(roll_id, "Portra")

    assert roll_for_id(widget.repo, roll_id) is None
    assert widget.tree.topLevelItemCount() == 0


def test_right_clicking_a_multi_selection_offers_a_bulk_delete(widget, monkeypatch):
    create_virtual_roll(widget.repo, "apple", [])
    create_virtual_roll(widget.repo, "banana", [])
    widget.reload()
    item = widget.tree.topLevelItem(0)
    widget.tree.topLevelItem(0).setSelected(True)
    widget.tree.topLevelItem(1).setSelected(True)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: item)
    menu = MagicMock()
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.QMenu", lambda *a, **k: menu)

    widget._show_context_menu(QPoint(0, 0))

    labels = [call.args[0] for call in menu.addAction.call_args_list]
    assert any("Delete 2 Rolls" in label for label in labels)
    assert "Open" not in labels
    assert "Rename…" not in labels


def test_right_clicking_outside_a_multi_selection_still_targets_just_that_row(widget, monkeypatch):
    create_virtual_roll(widget.repo, "apple", [])
    create_virtual_roll(widget.repo, "banana", [])
    widget.reload()
    widget.tree.topLevelItem(0).setSelected(True)
    other = widget.tree.topLevelItem(1)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: other)
    menu = MagicMock()
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.QMenu", lambda *a, **k: menu)

    widget._show_context_menu(QPoint(0, 0))

    labels = [call.args[0] for call in menu.addAction.call_args_list]
    assert "Open" in labels
    assert "Delete…" in labels


def test_deleting_a_multi_selection_removes_every_selected_roll(widget, monkeypatch):
    id_a = create_virtual_roll(widget.repo, "apple", [])
    id_b = create_virtual_roll(widget.repo, "banana", [])
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_several", lambda *a, **k: True)

    widget._delete_rolls([(id_a, "apple"), (id_b, "banana")])

    assert roll_for_id(widget.repo, id_a) is None
    assert roll_for_id(widget.repo, id_b) is None
    assert widget.tree.topLevelItemCount() == 0


def test_deleting_a_folder_roll_only_forgets_the_record(widget, tree_dirs, monkeypatch):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_named", lambda *a, **k: True)

    widget._delete_roll(roll_id, "roll_a")

    assert roll_for_id(widget.repo, roll_id) is None
    assert (tree_dirs / "roll_a" / "a1.NEF").exists()


# --- importing ------------------------------------------------------------------


def test_import_folder_recognizes_and_opens_it(widget, tree_dirs, monkeypatch):
    path = str(tree_dirs / "roll_a")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: path))
    widget.repo.save_global_setting("library_autoload_folders", True)  # skip the confirm prompt

    imported = widget.prompt_import_folder()

    assert imported is True
    widget.controller.open_library_folder.assert_called_once_with(path)


def test_import_folder_reports_a_newly_recognized_folder(widget, tree_dirs, monkeypatch):
    path = str(tree_dirs / "roll_a")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: path))
    widget.repo.save_global_setting("library_autoload_folders", True)
    created = []
    widget.folder_roll_created.connect(created.append)

    widget.prompt_import_folder()

    assert created == [path]


def test_import_folder_says_nothing_for_an_already_recognized_folder(widget, tree_dirs, monkeypatch):
    path = str(tree_dirs / "roll_a")
    recognize_folder(widget.repo, path)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: path))
    widget.repo.save_global_setting("library_autoload_folders", True)
    created = []
    widget.folder_roll_created.connect(created.append)

    widget.prompt_import_folder()

    assert created == []


def test_import_folder_with_no_images_reports_status_without_opening(widget, tree_dirs, monkeypatch):
    empty = tree_dirs / "empty"
    empty.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(empty)))

    imported = widget.prompt_import_folder()

    assert imported is False
    widget.controller.open_library_folder.assert_not_called()
    widget.controller.set_status.assert_called_once()


def test_cancelling_the_folder_picker_imports_nothing(widget, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))

    assert widget.prompt_import_folder() is False
    widget.controller.open_library_folder.assert_not_called()


def test_import_subfolders_delegates_to_the_controller(widget, tree_dirs, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tree_dirs)))
    widget.controller.import_subfolders_as_rolls.return_value = ["id1", "id2"]

    imported = widget.prompt_import_subfolders()

    assert imported is True
    widget.controller.import_subfolders_as_rolls.assert_called_once_with(str(tree_dirs))
    widget.controller.set_status.assert_called_once()


def test_import_subfolders_with_none_found_reports_status(widget, tree_dirs, monkeypatch):
    empty = tree_dirs / "roll_a"  # holds only files, no subfolders
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(empty)))
    widget.controller.import_subfolders_as_rolls.return_value = []

    imported = widget.prompt_import_subfolders()

    assert imported is False
    widget.controller.set_status.assert_called_once()
