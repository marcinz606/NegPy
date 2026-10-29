from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from negpy.desktop.view.sidebar.library_tree import _DETAIL_ROLE, _FOLDER_ROLE, _MISSING_ROLE, LibraryTree
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.rolls import create_virtual_roll, delete_roll, recognize_folder, roll_for_id, saved_rolls


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
    controller.removed_frame_count.return_value = 0
    controller.save_choice.return_value = ("hidden", "")
    return LibraryTree(controller)


@pytest.fixture
def widget(qapp, tmp_path):
    return _make(tmp_path)


def _names(widget) -> list[str]:
    return [widget.tree.topLevelItem(i).text(0) for i in range(widget.tree.topLevelItemCount())]


# --- listing --------------------------------------------------------------


def _count(item) -> str:
    """The photo count that leads a roll row's line."""
    return item.data(0, _DETAIL_ROLE).split(" · ")[0]


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

    assert _icon_names(widget, monkeypatch) == ["mdi6.folder-outline"]
    assert _count(widget.tree.topLevelItem(0)) == "2 photos"


def _triplet_roll(widget, tree_dirs) -> str:
    """roll_c holds two captured triplets; the first is remembered as assembled."""
    folder = tree_dirs / "roll_c"
    folder.mkdir()
    for frame in ("f1", "f2"):
        for channel in "RGB":
            (folder / f"{frame}_{channel}.NEF").write_bytes(b"x")
    red, green, blue = (str(folder / f"f1_{c}.NEF") for c in "RGB")
    widget.repo.save_global_setting("triplets_by_path", {red: [green, blue, True, ["", "", ""]]})
    return recognize_folder(widget.repo, str(folder), name="roll_c")


def test_an_assembled_triplet_counts_as_one_photo(widget, tree_dirs):
    _triplet_roll(widget, tree_dirs)
    widget.controller.rgb_scan_mode_for_roll.return_value = True
    widget.reload()

    assert _count(widget.tree.topLevelItem(0)) == "4 photos"


def test_a_roll_out_of_trichrome_mode_counts_every_file(widget, tree_dirs):
    _triplet_roll(widget, tree_dirs)
    widget.controller.rgb_scan_mode_for_roll.return_value = False
    widget.reload()

    assert _count(widget.tree.topLevelItem(0)) == "6 photos"


def test_a_stitch_counts_as_one_photo(widget, tree_dirs):
    folder = tree_dirs / "roll_a"
    widget.repo.save_global_setting("composites_by_path", {str(folder / "a1.NEF"): {"kind": "stitch", "paths": [str(folder / "a2.NEF")]}})
    widget.controller.rgb_scan_mode_for_roll.return_value = False
    recognize_folder(widget.repo, str(folder), name="roll_a")
    widget.reload()

    assert _count(widget.tree.topLevelItem(0)) == "1 photo"


def test_the_open_roll_recounts_when_its_files_change(widget, tree_dirs):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="roll_a")
    widget.controller.rgb_scan_mode_for_roll.return_value = False
    widget.controller.state.active_roll_id = roll_id
    widget.reload()
    (tree_dirs / "roll_a" / "a3.NEF").write_bytes(b"3")

    widget.refresh_active_count()

    assert _count(widget.tree.topLevelItem(0)) == "3 photos"


def test_a_virtual_roll_shows_a_search_icon_and_its_member_count(widget, monkeypatch):
    """Roll kind reads off the icon's shape: colour carries other meanings already."""
    create_virtual_roll(widget.repo, "Portra", ["/a.nef", "/b.nef"])

    assert _icon_names(widget, monkeypatch) == ["mdi6.text-search"]
    assert _count(widget.tree.topLevelItem(0)) == "2 photos"


def test_selection_survives_a_reload(widget, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"))
    recognize_folder(widget.repo, str(tree_dirs / "roll_b"))
    widget.reload()
    widget.tree.setCurrentItem(widget.tree.topLevelItem(1))

    widget.reload()

    assert widget.tree.currentItem().text(0) == "roll_b"


def test_a_collapsed_folder_stays_collapsed_in_the_next_session(qapp, tmp_path, tree_dirs):
    widget = _make(tmp_path)
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="scans/roll_a")
    recognize_folder(widget.repo, str(tree_dirs / "roll_b"), name="scans/roll_b")
    widget.reload()
    folder = widget.tree.topLevelItem(0)
    assert folder.data(0, _FOLDER_ROLE) == [str(tree_dirs)]
    assert folder.isExpanded()

    folder.setExpanded(False)
    folder = _make(tmp_path).tree.topLevelItem(0)
    assert not folder.isExpanded()

    folder.setExpanded(True)
    assert _make(tmp_path).tree.topLevelItem(0).isExpanded()


def test_a_reload_opens_the_folder_of_the_selected_roll_without_saving_it(widget, tmp_path, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="scans/roll_a")
    widget.show()
    widget.reload()
    folder = widget.tree.topLevelItem(0)
    widget.tree.setCurrentItem(folder.child(0))
    folder.setExpanded(False)

    widget.reload()

    assert widget.tree.topLevelItem(0).isExpanded()
    assert widget.tree.currentItem().text(0) == "roll_a"
    assert widget.repo.get_global_setting("library_collapsed_folders") == ["scans"]
    assert not _make(tmp_path).tree.topLevelItem(0).isExpanded()


def test_the_open_roll_shows_inside_collapsed_folders_without_saving_them(widget, tmp_path):
    day = tmp_path / "2026" / "2026-10" / "2026-10-18"
    (day / "Roll01").mkdir(parents=True)
    (day / "Roll02").mkdir()
    roll_id = recognize_folder(widget.repo, str(day / "Roll01"), name="2026/2026-10/2026-10-18/Roll01")
    recognize_folder(widget.repo, str(day / "Roll02"), name="2026/2026-10/2026-10-18/Roll02")
    widget.repo.save_global_setting("library_collapsed_folders", ["2026/2026-10"])
    widget = _make(tmp_path)
    widget.show()
    month = widget.tree.topLevelItem(0).child(0)
    assert not month.isExpanded()

    widget.controller.state.active_roll_id = roll_id
    widget.sync_ui()

    assert month.isExpanded() and month.child(0).isExpanded()
    assert widget.repo.get_global_setting("library_collapsed_folders") == ["2026/2026-10"]
    assert not _make(tmp_path).tree.topLevelItem(0).child(0).isExpanded()


def test_a_folder_shared_by_two_disk_trees_stays_collapsed_after_a_sort_change(widget, tmp_path):
    for parent in ("a", "b"):
        (tmp_path / parent / "scans" / f"roll_{parent}").mkdir(parents=True)
    recognize_folder(widget.repo, str(tmp_path / "a" / "scans" / "roll_a"), name="scans/roll_a")
    recognize_folder(widget.repo, str(tmp_path / "b" / "scans" / "roll_b"), name="scans/roll_b")
    widget.reload()
    widget.tree.topLevelItem(0).setExpanded(False)

    widget.set_sort("name", descending=True)

    assert not widget.tree.topLevelItem(0).isExpanded()


def test_a_deleted_folder_is_forgotten(widget, tree_dirs):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="scans/roll_a")
    widget.reload()
    widget.tree.topLevelItem(0).setExpanded(False)

    delete_roll(widget.repo, roll_id)
    widget.reload()

    assert widget.repo.get_global_setting("library_collapsed_folders") == []


@pytest.mark.parametrize("saved", ["scans", 5, None, ["scans", 5]])
def test_a_bad_saved_value_still_builds_the_tree(qapp, tmp_path, tree_dirs, saved):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    repo.save_global_setting("library_collapsed_folders", saved)
    recognize_folder(repo, str(tree_dirs / "roll_a"), name="scans/roll_a")

    folder = _make(tmp_path).tree.topLevelItem(0)

    assert folder.isExpanded() == (not isinstance(saved, list))


def test_a_roll_row_says_what_the_roll_is(widget):
    from negpy.desktop.view.sidebar.library_tree import _DOT_ROLE
    from negpy.services.assets import rolls

    roll_id = create_virtual_roll(widget.repo, "portra", [])
    rolls.set_roll_defaults(widget.repo, roll_id, process_mode="B&W Negative")
    widget.reload()

    item = widget.tree.topLevelItem(0)
    parts = item.data(0, _DETAIL_ROLE).split(" · ")
    assert parts[:2] == ["0 photos", "B&W"] and len(parts) == 3, "count, film mode, import date"
    assert item.data(0, _DOT_ROLE) == THEME.mode_bw


def test_the_filter_box_hides_the_rolls_it_rules_out(widget, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="20260901/kentmere_400_1")
    create_virtual_roll(widget.repo, "portra", [])
    widget.reload()

    widget.filter_input.setText("PORTRA")

    folder, portra = widget.tree.topLevelItem(0), widget.tree.topLevelItem(1)
    assert folder.isHidden() and not portra.isHidden()
    assert widget.tally_label.text() == "1 of 2 rolls"
    widget.filter_input.clear()
    assert not folder.isHidden() and widget.tally_label.text().startswith("2 rolls")


def test_a_folder_roll_row_counts_its_removed_frames_and_the_filter_finds_them(widget, tree_dirs):
    a = recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="A")
    recognize_folder(widget.repo, str(tree_dirs / "roll_b"), name="B")
    widget.controller.removed_frame_count.side_effect = lambda roll_id: 3 if roll_id == a else 0
    widget.reload()

    rows = {widget.tree.topLevelItem(i).text(0): widget.tree.topLevelItem(i) for i in range(2)}
    assert rows["A"].data(0, _DETAIL_ROLE).split(" · ")[1] == "3 removed"
    assert "removed" not in rows["B"].data(0, _DETAIL_ROLE)
    widget.filter_input.setText("removed")
    assert not rows["A"].isHidden() and rows["B"].isHidden()


@pytest.mark.parametrize(("kind", "removed"), [("folder", 2), ("folder", 0), ("virtual", 0)])
def test_a_folder_roll_offers_restore_removed_frames(widget, tree_dirs, monkeypatch, kind, removed):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a")) if kind == "folder" else create_virtual_roll(widget.repo, "V", [])
    widget.controller.removed_frame_count.return_value = removed
    widget.reload()
    item = widget.tree.topLevelItem(0)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: item)
    actions = _menu_with_distinct_actions(monkeypatch)

    widget._show_context_menu(QPoint(0, 0))

    restore = [label for label in actions if label.startswith("Restore Removed Frames…")]
    if kind == "virtual":
        assert restore == []
        return
    actions[restore[0]].setEnabled.assert_called_once_with(bool(removed))
    with monkeypatch.context() as m:
        m.setattr("negpy.desktop.view.sidebar.library_tree.restore_removed_frames", lambda w, c, r: r == roll_id)
        reload = MagicMock()
        m.setattr(widget, "reload", reload)
        actions[restore[0]].triggered.connect.call_args.args[0]()
    reload.assert_called_once()


@pytest.mark.parametrize(
    "choice", [("ready", ""), ("blocked", "“500T” is already a folder roll, so open it from the Library."), ("hidden", "")]
)
def test_a_virtual_roll_offers_save_to_folder_roll(widget, monkeypatch, choice):
    roll_id = create_virtual_roll(widget.repo, "Picks", [])
    widget.controller.save_choice.return_value = choice
    widget.reload()
    item = widget.tree.topLevelItem(0)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: item)
    actions = _menu_with_distinct_actions(monkeypatch)

    widget._show_context_menu(QPoint(0, 0))

    found = [label for label in actions if label.startswith("Save to Folder Roll")]
    if choice[0] == "hidden":
        assert found == []
        return
    assert found == ["Save to Folder Roll"], "no shortcut label: the shortcut acts on the open roll"
    action = actions[found[0]]
    action.setEnabled.assert_called_once_with(choice[0] == "ready")
    if choice[1]:
        action.setToolTip.assert_called_once_with(choice[1])
    widget.controller.save_to_folder_roll.return_value = roll_id
    reload = MagicMock()
    monkeypatch.setattr(widget, "reload", reload)
    action.triggered.connect.call_args.args[0]()
    widget.controller.save_to_folder_roll.assert_called_once_with(roll_id)
    reload.assert_called_once()


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
    assert widget.sort_btn.text() == "Name"


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


@pytest.mark.parametrize("open_roll", [True, False])
def test_only_the_open_roll_offers_unload_all(widget, monkeypatch, open_roll):
    roll_id = create_virtual_roll(widget.repo, "Portra", [])
    widget.controller.state.active_roll_id = roll_id if open_roll else "some-other-roll"
    widget.reload()
    item = widget.tree.topLevelItem(0)
    monkeypatch.setattr(widget.tree, "itemAt", lambda pos: item)
    actions = _menu_with_distinct_actions(monkeypatch)

    widget._show_context_menu(QPoint(0, 0))

    assert [label.split("\t")[0] for label in actions if label.startswith(("Unload All", "Close Roll"))] == (
        ["Unload All…"] if open_roll else []
    )


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
    widget.controller.import_subfolders_as_rolls.return_value = []

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
    empty = tree_dirs / "empty"
    empty.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(empty)))
    widget.controller.import_subfolders_as_rolls.return_value = []

    imported = widget.prompt_import_subfolders()

    assert imported is False
    widget.controller.set_status.assert_called_once()


def test_refresh_discovers_new_rolls_then_rereads_counts(widget, tree_dirs):
    widget.controller.rediscover_rolls.side_effect = lambda: (recognize_folder(widget.repo, str(tree_dirs / "roll_a")) and 1, 0)
    changed = []
    widget.rolls_changed.connect(lambda: changed.append(True))

    widget.refresh()

    assert _names(widget) == ["roll_a"]
    assert changed == [True]
    widget.controller.set_status.assert_called_once()


def test_refresh_without_new_rolls_stays_quiet(widget):
    widget.controller.rediscover_rolls.return_value = (0, 0)

    widget.refresh()

    widget.controller.set_status.assert_not_called()


def test_a_roll_whose_folder_is_gone_is_marked_missing(widget, tmp_path):
    recognize_folder(widget.repo, str(tmp_path / "gone"))
    widget.reload()

    item = widget.tree.topLevelItem(0)
    assert item.data(0, _DETAIL_ROLE) == "folder missing" and item.data(0, _MISSING_ROLE)


def test_importing_a_folder_without_images_imports_its_subfolders(widget, tree_dirs, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tree_dirs)))
    widget.controller.import_subfolders_as_rolls.return_value = ["id1", "id2"]

    imported = widget.prompt_import_folder()

    assert imported is True
    widget.controller.import_subfolders_as_rolls.assert_called_once_with(str(tree_dirs))
    widget.controller.open_library_folder.assert_not_called()


# --- Index Library ----------------------------------------------------------


def test_index_button_hidden_by_default(widget):
    widget.controller.state.semantic_search_enabled = False
    widget.sync_ui()
    assert not widget.index_action.isVisible()


def test_sync_ui_shows_the_button_once_the_model_is_ready(widget, monkeypatch):
    widget.controller.state.semantic_search_enabled = True
    monkeypatch.setattr("negpy.services.assets.semantic_model.clip_model_ready", lambda: True)

    widget.sync_ui()

    assert widget.index_action.isVisible()
    assert widget.index_action.isEnabled()


def test_sync_ui_disables_the_button_before_the_model_is_downloaded(widget, monkeypatch):
    widget.controller.state.semantic_search_enabled = True
    monkeypatch.setattr("negpy.services.assets.semantic_model.clip_model_ready", lambda: False)

    widget.sync_ui()

    assert widget.index_action.isVisible()
    assert not widget.index_action.isEnabled()


def test_clicking_index_calls_the_controller(widget):
    widget.controller.state.semantic_search_enabled = True
    widget.index_action.setEnabled(True)
    widget.index_action.trigger()
    widget.controller.index_library.assert_called_once()


def test_nested_rolls_sit_under_their_folder_row(widget, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="20260901/kentmere_400_1")
    recognize_folder(widget.repo, str(tree_dirs / "roll_b"), name="20260901/kentmere_400_2")
    create_virtual_roll(widget.repo, "portra", [])
    widget.reload()

    assert _names(widget) == ["20260901", "portra"]
    folder = widget.tree.topLevelItem(0)
    assert [folder.child(i).text(0) for i in range(folder.childCount())] == ["kentmere_400_1", "kentmere_400_2"]
    assert folder.data(0, _DETAIL_ROLE) == "2 rolls"
    assert folder.isExpanded()


def test_a_folder_row_opens_nothing(widget, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="20260901/kentmere_400_1")
    widget.reload()

    widget._on_double_clicked(widget.tree.topLevelItem(0), 0)

    widget.controller.open_roll.assert_not_called()


def test_a_collapsed_folder_row_stays_collapsed_across_reload(widget, tree_dirs):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="scans/roll_a")
    widget.reload()
    widget.tree.topLevelItem(0).setExpanded(False)

    widget.reload()

    assert not widget.tree.topLevelItem(0).isExpanded()


def test_a_nested_roll_can_be_selected_after_reload(widget, tree_dirs):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="20260901/kentmere_400_1")
    widget.reload()
    widget._select_roll(roll_id)

    widget.reload()

    assert widget._selected_roll_id() == roll_id


def test_rename_starts_from_the_folder_name_not_the_path(widget, tree_dirs, monkeypatch):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="20260901/kentmere_400_1")
    seen = []

    class _Dialog:
        def __init__(self, current, *_a, **_k):
            seen.append(current)

        def exec(self):
            return 0

    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _Dialog)
    widget._rename_roll(roll_id, "20260901/kentmere_400_1")

    assert seen == ["kentmere_400_1"]


def test_rename_keeps_the_roll_under_its_folder_row(widget, tree_dirs, monkeypatch):
    roll_id = recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="20260901/kentmere_400_1")

    class _Dialog:
        def __init__(self, *_a, **_k):
            pass

        def exec(self):
            return 1

        def name(self):
            return "kentmere_best"

        def rename_folder(self):
            return False

    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.RenameRollDialog", _Dialog)
    widget._rename_roll(roll_id, "20260901/kentmere_400_1")

    assert roll_for_id(widget.repo, roll_id)["name"] == "20260901/kentmere_best"


def test_discovery_filters_popup_saves_one_filter_per_line(widget, monkeypatch):
    seen = []

    def _ask(_parent, _title, _label, text):
        seen.append(text)
        return "export\nraw_*\n", True

    monkeypatch.setattr(QInputDialog, "getMultiLineText", staticmethod(_ask))
    widget.controller.rediscover_rolls.return_value = (0, 2)

    assert widget.edit_discovery_filters() is True
    assert seen == ["export"]
    assert widget.repo.get_global_setting("roll_discovery_filters") == ["export", "raw_*"]
    widget.controller.rediscover_rolls.assert_called_once()


def test_cancelling_the_discovery_filters_popup_keeps_them(widget, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getMultiLineText", staticmethod(lambda *a: ("", False)))

    assert widget.edit_discovery_filters() is False
    assert widget.repo.get_global_setting("roll_discovery_filters") is None


def test_deleting_a_folder_row_forgets_every_roll_under_it(widget, tree_dirs, monkeypatch):
    recognize_folder(widget.repo, str(tree_dirs / "roll_a"), name="scans/roll_a")
    recognize_folder(widget.repo, str(tree_dirs / "roll_b"), name="scans/roll_b")
    create_virtual_roll(widget.repo, "portra", [])
    widget.reload()
    folder = widget.tree.topLevelItem(_names(widget).index("scans"))
    assert folder.data(0, _FOLDER_ROLE) == [str(tree_dirs)]
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_named", lambda *a, **k: True)

    widget._delete_folder(folder)

    assert _names(widget) == ["portra"]


def test_a_folder_row_shared_by_two_folders_deletes_the_rolls_of_both(widget, tmp_path, monkeypatch):
    for parent in ("a", "b"):
        (tmp_path / parent / "scans" / f"roll_{parent}").mkdir(parents=True)
    recognize_folder(widget.repo, str(tmp_path / "a" / "scans" / "roll_a"), name="scans/roll_a")
    recognize_folder(widget.repo, str(tmp_path / "b" / "scans" / "roll_b"), name="scans/roll_b")
    create_virtual_roll(widget.repo, "portra", [])
    widget.reload()
    folder = widget.tree.topLevelItem(_names(widget).index("scans"))
    paths = sorted(folder.data(0, _FOLDER_ROLE))
    assert paths == [str(tmp_path / "a" / "scans"), str(tmp_path / "b" / "scans")]
    assert folder.toolTip(0).split("\n") == folder.data(0, _FOLDER_ROLE)
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_named", lambda *a, **k: True)

    widget._delete_folder(folder)

    assert _names(widget) == ["portra"]


def test_deleting_a_folder_row_keeps_the_rolls_the_filter_hides(widget, tmp_path, monkeypatch):
    for name in ("2023-01", "2024-01"):
        (tmp_path / "scans" / name).mkdir(parents=True)
        recognize_folder(widget.repo, str(tmp_path / "scans" / name), name=f"scans/{name}")
    widget.repo.save_global_setting("roll_import_sources", [str(tmp_path / "scans")])
    widget.reload()
    widget.filter_input.setText("2024")
    widget._apply_filter()
    folder = widget.tree.topLevelItem(0)
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_named", lambda *a, **k: True)

    widget._delete_folder(folder)

    assert [entry["name"] for entry in saved_rolls(widget.repo).values()] == ["scans/2023-01"]
    assert widget.repo.get_global_setting("roll_import_sources") == [str(tmp_path / "scans")]


def test_deleting_a_folder_row_keeps_a_roll_shown_outside_it(widget, tmp_path, monkeypatch):
    (tmp_path / "y" / "a" / "roll").mkdir(parents=True)
    recognize_folder(widget.repo, str(tmp_path / "y" / "a" / "roll"), name="a/roll")
    recognize_folder(widget.repo, str(tmp_path / "y" / "a"))
    widget.reload()
    assert _names(widget) == ["a", "a"]
    folder = next(item for item in (widget.tree.topLevelItem(i) for i in range(2)) if item.childCount())
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_named", lambda *a, **k: True)

    widget._delete_folder(folder)

    assert _names(widget) == ["a"]
    assert widget.tree.topLevelItem(0).childCount() == 0


def test_delete_says_what_the_roll_loses_by_kind(widget, tmp_path, monkeypatch):
    (tmp_path / "scans" / "roll_a").mkdir(parents=True)
    folder_id = recognize_folder(widget.repo, str(tmp_path / "scans" / "roll_a"), name="scans/roll_a")
    set_id = create_virtual_roll(widget.repo, "Picks", [])
    said: list = []
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_named", lambda *a, **k: said.append(k["informative"]))
    monkeypatch.setattr("negpy.desktop.view.sidebar.library_tree.confirm_delete_several", lambda *a, **k: said.append(k["informative"]))
    widget.reload()

    widget._delete_roll(folder_id, "roll_a")
    widget._delete_roll(set_id, "Picks")
    widget._delete_folder(
        next(widget.tree.topLevelItem(i) for i in range(widget.tree.topLevelItemCount()) if widget.tree.topLevelItem(i).childCount())
    )
    widget._delete_rolls([(folder_id, "roll_a")])
    widget._delete_rolls([(folder_id, "roll_a"), (set_id, "Picks")])

    shared = "The files and their shared edits stay."
    assert all(text.startswith(shared) and "film base" in text and "edits made only in" in text for text in said)
    assert "Refresh will not add it again." in said[0] and "list of frames" in said[1] and "Refresh" not in said[1]
    assert "Refresh will not add these rolls again." in said[2]
    assert "The folder roll's settings" in said[0] and "The virtual roll's list of frames" in said[1]
    assert "virtual roll loses" not in said[3] and said[4].endswith("A virtual roll loses its list of frames too.")
