from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QMessageBox

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


def _icon_names(widget, monkeypatch) -> list:
    names = []
    monkeypatch.setattr(
        "negpy.desktop.view.sidebar.library_tree.qta.icon",
        lambda name, color=None: names.append(name) or QIcon(),
    )
    widget.reload()
    return names


def test_a_folder_roll_shows_a_folder_icon_and_its_live_count(widget, tree_dirs, monkeypatch):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"))

    assert _icon_names(widget, monkeypatch) == ["fa5s.folder"]
    assert widget.tree.topLevelItem(0).text(1) == "2 photos"


def test_a_virtual_roll_shows_a_search_icon_and_its_member_count(widget, monkeypatch):
    """Roll kind reads off the icon's shape: colour carries other meanings already."""
    create_virtual_roll(widget.repo, "Portra", ["/a.nef", "/b.nef"])

    assert _icon_names(widget, monkeypatch) == ["fa5s.search"]
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


def test_its_own_sort_button_orders_and_saves_the_roll_list(widget):
    create_virtual_roll(widget.repo, "apple", [])
    create_virtual_roll(widget.repo, "Zebra", [])
    widget.reload()

    widget.sort_btn.descending_action.trigger()

    assert _names(widget) == ["Zebra", "apple"]
    assert widget.repo.get_global_setting("library_sort_descending") is True
    assert widget.sort_btn.order_action("name").isChecked()
    assert widget.sort_btn in widget.toolbar.buttons


def test_the_roll_list_starts_from_the_film_strips_sort_until_it_has_its_own(qapp, tmp_path):
    """It used to follow the Film Strip; Scene reads as Name, since rolls have no scenes."""
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    repo.save_global_setting("file_sort_order", "scene")
    repo.save_global_setting("file_sort_descending", True)
    controller = MagicMock()
    controller.session.repo = repo

    tree = LibraryTree(controller)

    assert (tree._sort_order, tree._sort_descending) == ("name", True)
    repo.save_global_setting("library_sort_order", "date")
    repo.save_global_setting("library_sort_descending", False)
    assert LibraryTree(controller)._saved_sort() == ("date", False)


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
    _FakeRenameDialog._outcome = ("Portra 400", False)
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _FakeRenameDialog)

    widget._rename_roll(roll_id, "Portra")

    assert roll_for_id(widget.repo, roll_id)["name"] == "Portra 400"
    assert widget.tree.topLevelItem(0).text(0) == "Portra 400"


def test_renaming_to_an_invalid_name_is_rejected(widget, monkeypatch):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    _FakeRenameDialog._outcome = ("bad/name", False)
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _FakeRenameDialog)
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


def _menu_with_distinct_actions(monkeypatch):
    """A QMenu stub whose addAction(label) returns its own mock per label, so each
    action's setEnabled/triggered calls can be checked independently."""
    menu = MagicMock()
    actions: dict[str, MagicMock] = {}

    def add_action(label, *a, **k):
        actions.setdefault(label, MagicMock())
        return actions[label]

    menu.addAction.side_effect = add_action
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.QMenu", lambda *a, **k: menu)
    return actions


def test_right_click_on_the_loaded_roll_offers_an_enabled_analyze_action(widget, monkeypatch):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    widget.controller.state.active_roll_id = roll_id
    widget.reload()
    item = widget.tree.topLevelItem(0)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: item)
    actions = _menu_with_distinct_actions(monkeypatch)

    widget._show_context_menu(QPoint(0, 0))

    actions["Roll Analysis"].setEnabled.assert_called_once_with(True)


def test_right_click_on_a_different_roll_offers_a_disabled_analyze_action(widget, monkeypatch):
    create_virtual_roll(widget.repo, "Portra", [])
    widget.controller.state.active_roll_id = "some-other-roll"
    widget.reload()
    item = widget.tree.topLevelItem(0)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: item)
    actions = _menu_with_distinct_actions(monkeypatch)

    widget._show_context_menu(QPoint(0, 0))

    actions["Roll Analysis"].setEnabled.assert_called_once_with(False)


def test_analyze_action_reaches_the_controller(widget, monkeypatch):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    widget.controller.state.active_roll_id = roll_id
    widget.reload()
    item = widget.tree.topLevelItem(0)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: item)
    actions = _menu_with_distinct_actions(monkeypatch)

    widget._show_context_menu(QPoint(0, 0))

    actions["Roll Analysis"].triggered.connect.assert_called_once_with(widget.controller.request_batch_normalization)


def test_rename_roll_dialog_checkbox_defaults_off(qapp):
    from negpy.desktop.view.widgets.rename_roll_dialog import RenameRollDialog

    dlg = RenameRollDialog("roll_a")
    assert dlg.rename_folder() is False
    assert dlg.name() == "roll_a"


def test_rename_roll_dialog_name_is_trimmed(qapp):
    from negpy.desktop.view.widgets.rename_roll_dialog import RenameRollDialog

    dlg = RenameRollDialog("roll_a")
    dlg.name_edit.setText("  new name  ")
    assert dlg.name() == "new name"


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


class _FakeRenameDialog:
    """Stands in for RenameRollDialog: exec() reports the outcome an earlier call to
    accept_as()/cancelled() set up, without opening a real modal dialog."""

    _outcome = None  # ("name", rename_folder) or None for rejected, set per test

    def __init__(self, *_a, **_k):
        pass

    def exec(self):
        from PyQt6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted if self._outcome is not None else QDialog.DialogCode.Rejected

    def name(self):
        return self._outcome[0]

    def rename_folder(self):
        return self._outcome[1]


def test_renaming_a_folder_roll_shows_the_rename_dialog_not_the_plain_one(widget, tree_dirs, monkeypatch):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    _FakeRenameDialog._outcome = ("roll_a_renamed", True)
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _FakeRenameDialog)
    widget.controller.request_rename_roll.return_value = True

    widget._rename_roll(roll_id, "roll_a")

    widget.controller.request_rename_roll.assert_called_once_with(roll_id, "roll_a_renamed", True)


def test_renaming_a_folder_roll_without_the_checkbox_never_touches_disk(widget, tree_dirs, monkeypatch):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    _FakeRenameDialog._outcome = ("roll_a_renamed", False)
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _FakeRenameDialog)

    widget._rename_roll(roll_id, "roll_a")

    widget.controller.request_rename_roll.assert_not_called()
    assert roll_for_id(widget.repo, roll_id)["name"] == "roll_a_renamed"
    assert roll_for_id(widget.repo, roll_id)["folder_path"] == str(tree_dirs / "roll_a")
    assert (tree_dirs / "roll_a").exists()  # nothing on disk moved


def test_renaming_a_folder_roll_cancelled_calls_nothing(widget, tree_dirs, monkeypatch):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    _FakeRenameDialog._outcome = None
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _FakeRenameDialog)

    widget._rename_roll(roll_id, "roll_a")

    widget.controller.request_rename_roll.assert_not_called()


def test_renaming_a_folder_roll_disk_failure_warns_and_does_not_reload(widget, tree_dirs, monkeypatch):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    _FakeRenameDialog._outcome = ("roll_b", True)  # already taken, per tree_dirs
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _FakeRenameDialog)
    widget.controller.request_rename_roll.return_value = False
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    reloaded = []
    monkeypatch.setattr(widget, "reload", lambda: reloaded.append(True))

    widget._rename_roll(roll_id, "roll_a")

    assert len(warned) == 1
    assert reloaded == []


def test_renaming_a_virtual_roll_uses_the_same_dialog_without_the_folder_row(widget, monkeypatch):
    """One action, one dialog: only the disk-rename row differs by roll kind."""
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    _FakeRenameDialog._outcome = ("Portra 400", False)
    built = []

    class _Recording(_FakeRenameDialog):
        def __init__(self, *_a, **kwargs):
            built.append(kwargs.get("folder_backed"))

    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _Recording)

    widget._rename_roll(roll_id, "Portra")

    assert built == [False]
    assert roll_for_id(widget.repo, roll_id)["name"] == "Portra 400"


def test_renaming_a_virtual_roll_never_renames_a_folder(widget, monkeypatch):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    _FakeRenameDialog._outcome = ("Portra 400", False)
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _FakeRenameDialog)

    widget._rename_roll(roll_id, "Portra")

    widget.controller.request_rename_roll.assert_not_called()


def test_the_rename_dialog_hides_the_disk_row_for_a_virtual_roll(qapp):
    from negpy.desktop.view.widgets.rename_roll_dialog import RenameRollDialog

    dlg = RenameRollDialog("Portra", folder_backed=False)

    assert dlg.rename_folder_check.isHidden()
    dlg.rename_folder_check.setChecked(True)
    assert dlg.rename_folder() is False


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


# --- Index Library ----------------------------------------------------------


def test_index_button_hidden_by_default(widget):
    widget.controller.state.semantic_search_enabled = False
    widget.sync_ui()
    assert widget.index_btn.isHidden()


def test_sync_ui_shows_the_button_once_the_model_is_ready(widget, monkeypatch):
    widget.controller.state.semantic_search_enabled = True
    monkeypatch.setattr("negpy.services.assets.semantic_model.clip_model_ready", lambda: True)

    widget.sync_ui()

    assert not widget.index_btn.isHidden()
    assert widget.index_btn.isEnabled()


def test_sync_ui_disables_the_button_before_the_model_is_downloaded(widget, monkeypatch):
    widget.controller.state.semantic_search_enabled = True
    monkeypatch.setattr("negpy.services.assets.semantic_model.clip_model_ready", lambda: False)

    widget.sync_ui()

    assert not widget.index_btn.isHidden()
    assert not widget.index_btn.isEnabled()


def test_clicking_index_calls_the_controller(widget):
    # A MagicMock slot can't be introspected for arity the way a real bound method
    # can, so clicked's bool argument passes through here where it wouldn't in
    # production (confirmed separately against a real bound method) -- only whether
    # it fired is the point of this test.
    widget.index_btn.click()
    widget.controller.index_library.assert_called_once()
