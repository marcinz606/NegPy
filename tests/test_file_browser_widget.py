from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QModelIndex, QPoint, QPointF, QPropertyAnimation, QRect, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QDialog, QStyleOptionViewItem

from negpy.desktop.session import DesktopSessionManager, composite_kind, composite_summary
from negpy.desktop.view.sidebar.files import (
    THUMB_CELL_MAX,
    THUMB_CELL_MIN,
    FileBrowser,
    ThumbnailGridView,
    _ThumbnailDelegate,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.thumbnails import asset_thumbnail_key


def _edited_cfg() -> WorkspaceConfig:
    """A config with a couple of non-default settings so the picker renders rows."""
    c = WorkspaceConfig()
    return replace(
        c,
        exposure=replace(c.exposure, density=1.5),
        geometry=replace(c.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9)),
    )


@pytest.fixture
def session(qapp):
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.load_file_settings_many.return_value = {}
    repo.get_max_history_index.return_value = 0
    mgr = DesktopSessionManager(repo)
    mgr.state.uploaded_files = [
        {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
        {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
        {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        {"name": "note.txt", "path": "/tmp/note.txt", "hash": "h4"},
    ]
    mgr.asset_model.refresh()
    return mgr


@pytest.fixture
def browser(session):
    controller = MagicMock()
    controller.session = session
    controller.thumbnail_refresh_running = False
    return FileBrowser(controller)


def test_search_input_is_present(browser):
    assert browser.search_input is not None
    assert "film:portra" in browser.search_input.placeholderText()
    assert browser.regex_btn.isCheckable()


def test_apply_filter_narrows_visible_files(browser, session):
    browser.search_input.setText("IMG")
    browser._apply_filter()
    visible = session.asset_model.visible_actual_indices_ordered()
    visible_names = {session.state.uploaded_files[i]["name"] for i in visible}
    assert visible_names == {"IMG_0001.cr2", "IMG_0002.cr2"}


def test_regex_toggle_compiles_pattern(browser, session):
    browser.regex_btn.setChecked(True)
    browser.search_input.setText(r"^IMG_\d{4}")
    browser._apply_filter()
    assert session.asset_model._filter_pattern is not None
    visible = {session.state.uploaded_files[i]["name"] for i in session.asset_model._sorted_indices}
    assert visible == {"IMG_0001.cr2", "IMG_0002.cr2"}


def test_invalid_regex_sets_error_stylesheet(browser):
    browser.regex_btn.setChecked(True)
    browser.search_input.setText("[")
    browser._apply_filter()
    assert THEME.accent_primary in browser.search_input.styleSheet()


def test_invalid_regex_does_not_change_visible(browser, session):
    browser.search_input.setText("IMG")
    browser._apply_filter()
    before = list(session.asset_model._sorted_indices)
    browser.regex_btn.setChecked(True)
    browser.search_input.setText("[")
    browser._apply_filter()
    assert session.asset_model._sorted_indices == before


def test_selection_pruned_to_visible(browser, session):
    session.state.selected_indices = [0, 1, 2, 3]
    session.state.selected_file_idx = 0
    browser.search_input.setText("IMG")
    browser._apply_filter()
    assert set(session.state.selected_indices) == {0, 1}
    assert session.state.selected_file_idx in {0, 1}


def test_selection_cleared_when_no_visible_match(browser, session):
    session.state.selected_indices = [0, 1, 2, 3]
    session.state.selected_file_idx = 0
    browser.search_input.setText("zzzzz")
    browser._apply_filter()
    assert session.state.selected_indices == []
    assert session.state.selected_file_idx == -1


def test_active_file_preserved_when_still_visible(browser, session):
    session.state.selected_indices = [0, 1, 2]
    session.state.selected_file_idx = 1  # IMG_0002.cr2
    browser.search_input.setText("IMG")
    browser._apply_filter()
    assert session.state.selected_file_idx == 1
    assert set(session.state.selected_indices) == {0, 1}


def _action_labels(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def test_context_menu_single_selection_items(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    labels = _action_labels(browser._build_context_menu())
    assert "Export Current Frame" in labels
    assert "Export Selected Frames" not in labels
    assert "Reset Settings" in labels
    assert "Unload…" in labels
    assert "Apply Settings…" in labels
    assert "Update Thumbnail" in labels
    assert "Update Thumbnails" not in labels


def test_context_menu_update_thumbnail_requests_the_selection_scope(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    menu = browser._build_context_menu()
    action = next(a for a in menu.actions() if a.text() == "Update Thumbnail")
    action.trigger()
    browser.controller.request_thumbnail_refresh.assert_called_once_with("selection")


def test_context_menu_offers_cancel_while_a_refresh_is_running(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    browser.controller.thumbnail_refresh_running = True

    labels = _action_labels(browser._build_context_menu())

    assert "Cancel Thumbnail Update" in labels
    assert "Update Thumbnail" not in labels

    menu = browser._build_context_menu()
    action = next(a for a in menu.actions() if a.text() == "Cancel Thumbnail Update")
    action.trigger()
    browser.controller.cancel_thumbnail_refresh.assert_called_once_with()


def test_context_menu_offers_unsplit_only_for_a_diptych(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    assert "Unsplit Diptych" not in _action_labels(browser._build_context_menu())

    session.state.uploaded_files[0]["diptych"] = True
    assert "Unsplit Diptych" in _action_labels(browser._build_context_menu())


def test_context_menu_offers_per_frame_split_only_for_a_half(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    browser.controller.half_frame_override.return_value = None
    assert "Adjust Split for This Frame…" not in _action_labels(browser._build_context_menu())

    session.state.uploaded_files[0]["half"] = 1
    session.state.uploaded_files[0]["hash"] = "h1#1"
    assert "Adjust Split for This Frame…" in _action_labels(browser._build_context_menu())
    # No override saved yet, so nothing to reset.
    assert "Reset Split to Roll Default" not in _action_labels(browser._build_context_menu())


def test_context_menu_offers_reset_only_with_a_saved_override(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    session.state.uploaded_files[0]["half"] = 1
    session.state.uploaded_files[0]["hash"] = "h1#1"
    browser.controller.half_frame_override.return_value = {"split_x": 0.4}
    assert "Reset Split to Roll Default" in _action_labels(browser._build_context_menu())


def _set_rolls_store(session, rolls_store):
    session.repo.get_global_setting.side_effect = lambda key, default=None: rolls_store if key == "rolls_by_id" else default


def test_context_menu_offers_fork_only_when_the_file_is_in_two_rolls(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    session.state.active_roll_id = "r1"
    rolls_store = {"r1": {"kind": "virtual", "name": "A", "member_paths": ["/tmp/IMG_0001.cr2"]}}
    _set_rolls_store(session, rolls_store)
    assert "Edit Independently in This Roll" not in _action_labels(browser._build_context_menu())

    rolls_store["r2"] = {"kind": "virtual", "name": "B", "member_paths": ["/tmp/IMG_0001.cr2"]}
    assert "Edit Independently in This Roll" in _action_labels(browser._build_context_menu())


def test_context_menu_needs_an_active_roll_to_offer_fork(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    session.state.active_roll_id = None
    rolls_store = {
        "r1": {"kind": "virtual", "name": "A", "member_paths": ["/tmp/IMG_0001.cr2"]},
        "r2": {"kind": "virtual", "name": "B", "member_paths": ["/tmp/IMG_0001.cr2"]},
    }
    _set_rolls_store(session, rolls_store)
    assert "Edit Independently in This Roll" not in _action_labels(browser._build_context_menu())


def test_context_menu_offers_unfork_once_forked(browser, session):
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    session.state.active_roll_id = "r1"
    rolls_store = {
        "r1": {"kind": "virtual", "name": "A", "member_paths": ["/tmp/IMG_0001.cr2"], "forked_hashes": ["h1"]},
        "r2": {"kind": "virtual", "name": "B", "member_paths": ["/tmp/IMG_0001.cr2"]},
    }
    _set_rolls_store(session, rolls_store)
    labels = _action_labels(browser._build_context_menu())
    assert "Use the Shared Edit Again…" in labels
    assert "Edit Independently in This Roll" not in labels


def test_hot_folder_stops_re_offering_a_duplicate_it_already_turned_away(browser, session):
    """The poll decided what was new by path while add_files turns files away by content
    hash, so a byte-identical copy under another name was never in the file list to
    compare against: hashed, rejected and offered again every 2s, forever."""
    session.state.duplicate_paths.add("/tmp/IMG_0001 copy.cr2")

    with patch("negpy.desktop.view.sidebar.files.FolderWatchService.scan_for_new_files", return_value=[]) as scan:
        browser._scan_folder()

    assert "/tmp/IMG_0001 copy.cr2" in scan.call_args[0][1]
    browser.controller.request_asset_discovery.assert_not_called()


def test_adjust_half_frame_split_reloads_only_on_apply(browser, session):
    browser.controller.open_half_frame_dialog.return_value = None
    browser._on_adjust_half_frame_split("/tmp/scan.tif", "h1")
    browser.controller.reload_after_half_frame_change.assert_not_called()

    browser.controller.open_half_frame_dialog.return_value = {"split_x": 0.4}
    browser._on_adjust_half_frame_split("/tmp/scan.tif", "h1")
    browser.controller.open_half_frame_dialog.assert_called_with("/tmp/scan.tif", "h1", initial_scope="current")
    browser.controller.reload_after_half_frame_change.assert_called_once()


def test_reset_half_frame_split_clears_and_reloads(browser, session):
    browser._on_reset_half_frame_split("h1")
    browser.controller.clear_half_frame_override.assert_called_once_with("h1")
    browser.controller.reload_after_half_frame_change.assert_called_once()


def test_unsplit_diptych_needs_the_confirm(browser):
    with patch("negpy.desktop.view.sidebar.files.confirm_undiptych", return_value=False):
        browser.prompt_undiptych()
    browser.controller.request_undiptych.assert_not_called()


def test_context_menu_multi_selection_uses_export_selected(browser, session):
    session.state.selected_indices = [0, 1]
    session.state.selected_file_idx = 0
    labels = _action_labels(browser._build_context_menu())
    assert "Export Selected Frames" in labels
    assert "Export Current Frame" not in labels


def test_context_menu_multi_selection_counts_update_thumbnails(browser, session):
    session.state.selected_indices = [0, 1]
    session.state.selected_file_idx = 0
    labels = _action_labels(browser._build_context_menu())
    assert "Update 2 thumbnails" in labels
    assert "Update Thumbnail" not in labels


def test_context_menu_multi_selection_adds_apply_and_remove_selected(browser, session):
    session.state.selected_indices = [0, 1]
    session.state.selected_file_idx = 0
    labels = _action_labels(browser._build_context_menu())
    assert "Apply Settings…" in labels
    assert "Unload Selected…" in labels
    assert "Unload…" not in labels


def test_apply_dialog_shows_header_scope_and_counts(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=2, roll_count=3)
    assert dlg._scope_radios.sel.text() == "Selected frames (2)"
    assert dlg._scope_radios.sel.isEnabled()
    assert dlg._scope_radios.sel.isChecked()  # selection preferred when it has targets
    assert dlg._scope_radios.roll.text() == "Whole roll (3)"
    assert dlg._scope_radios.roll.isEnabled()


def test_apply_dialog_defaults_to_roll_when_selection_empty(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=0, roll_count=3)
    assert not dlg._scope_radios.sel.isEnabled()
    assert dlg._scope_radios.roll.isChecked()


def test_apply_dialog_check_all_and_none(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    assert dlg.apply_btn.isEnabled()  # rows checked by default
    dlg._set_all_checked(False)
    assert not any(box.isChecked() for box in dlg._all_boxes())
    assert not dlg.apply_btn.isEnabled()
    dlg._set_all_checked(True)
    # unchanged rows stay hidden and unchecked until "Show unchanged settings"
    assert {r.label for r in dlg.selected()} == {"Print Density", "Crop"}
    assert dlg.apply_btn.isEnabled()


def test_apply_dialog_apply_collects_checked_rows_and_scope(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    dlg._scope_radios.roll.setChecked(True)
    dlg._on_apply()
    labels = {r.label for r in dlg.selected()}
    assert "Print Density" in labels  # the edited exposure setting
    assert "Crop" in labels  # the edited geometry setting
    assert dlg.scope() == "roll"


def test_apply_dialog_only_preselects_edited_settings(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    assert {r.label for r in dlg.selected()} == {"Print Density", "Crop"}  # nothing else was non-default
    # the rest are still built, just hidden, so they can be applied on demand (#656)
    assert "Crop Offset" in {row.label for _box, row, _edited, _line in dlg._checks}


def test_open_apply_dialog_routes_rows_bounds_scope_to_session(browser, session):
    session.state.selected_indices = [0, 1]
    session.state.selected_file_idx = 0
    session.sync_selected_settings = MagicMock()

    rows = [object()]
    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.selected.return_value = rows
    mock_dlg.bounds_flags.return_value = (False, False)
    mock_dlg.scope.return_value = "selection"
    with patch("negpy.desktop.view.widgets.granular_settings_dialog.GranularSettingsDialog", return_value=mock_dlg) as ctor:
        browser._open_apply_dialog()

    assert ctor.call_args.args[2] == "IMG_0001.cr2"
    assert ctor.call_args.kwargs["sel_count"] == 1  # 1 other selected
    assert ctor.call_args.kwargs["roll_count"] == 3  # 3 other on roll
    session.sync_selected_settings.assert_called_once_with(rows, (False, False), "selection")


def test_open_apply_dialog_noop_without_active_file(browser, session):
    session.state.selected_file_idx = -1
    session.sync_selected_settings = MagicMock()
    with patch("negpy.desktop.view.widgets.granular_settings_dialog.GranularSettingsDialog") as ctor:
        browser._open_apply_dialog()
    ctor.assert_not_called()
    session.sync_selected_settings.assert_not_called()


def test_open_roll_settings_dialog_routes_rows_and_scope_to_session(browser, session):
    session.state.selected_indices = [0, 1]
    session.state.selected_file_idx = 0
    session.apply_preset_fields = MagicMock(return_value=2)

    rows = [object()]
    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.selected_rows.return_value = rows
    mock_dlg.selected_config.return_value = _edited_cfg()
    mock_dlg.scope.return_value = "selection"
    with patch("negpy.desktop.view.sidebar.files.RollSettingsDialog", return_value=mock_dlg) as ctor:
        browser._open_roll_settings_dialog()

    assert ctor.call_args.kwargs["sel_count"] == 1  # 1 other selected
    assert ctor.call_args.kwargs["roll_count"] == 3  # 3 other on roll
    session.apply_preset_fields.assert_called_once_with(mock_dlg.selected_config.return_value, rows, "selection")
    browser.controller.request_render.assert_called_once()


def test_open_roll_settings_dialog_noop_without_active_file(browser, session):
    session.state.selected_file_idx = -1
    session.apply_preset_fields = MagicMock()
    with patch("negpy.desktop.view.sidebar.files.RollSettingsDialog") as ctor:
        browser._open_roll_settings_dialog()
    ctor.assert_not_called()
    session.apply_preset_fields.assert_not_called()


def test_open_roll_settings_dialog_noop_when_nothing_is_ticked(browser, session):
    session.state.selected_file_idx = 0
    session.apply_preset_fields = MagicMock()
    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.selected_rows.return_value = []
    with patch("negpy.desktop.view.sidebar.files.RollSettingsDialog", return_value=mock_dlg):
        browser._open_roll_settings_dialog()
    session.apply_preset_fields.assert_not_called()


def test_open_roll_settings_dialog_also_prefills_a_gear_match(browser, session):
    """The tag-icon button offers the same suggestion the import-time popup does --
    useful any time you open it, not just the one moment right after import."""
    session.state.selected_file_idx = 0
    detected = MagicMock(any=MagicMock(return_value=True), camera_id="cam1", film_stock_id="")
    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Rejected
    with (
        patch("negpy.desktop.view.sidebar.files.match_gear_for_folder", return_value=detected),
        patch("negpy.desktop.view.sidebar.files.RollSettingsDialog", return_value=mock_dlg),
    ):
        browser._open_roll_settings_dialog()

    mock_dlg.apply_detected_gear.assert_called_once_with(camera_id="cam1", film_stock_id="")


def test_open_roll_settings_dialog_does_not_override_gear_already_set_in_full(browser, session):
    """Both fields already carry something -- there is nothing left to suggest."""
    session.state.selected_file_idx = 0
    session.state.config = replace(
        session.state.config,
        metadata=replace(session.state.config.metadata, camera_id="existing", film_stock_id="existing-film"),
    )
    detected = MagicMock(camera_id="cam1", film_stock_id="film1")
    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Rejected
    with (
        patch("negpy.desktop.view.sidebar.files.match_gear_for_folder", return_value=detected),
        patch("negpy.desktop.view.sidebar.files.RollSettingsDialog", return_value=mock_dlg),
    ):
        browser._open_roll_settings_dialog()

    mock_dlg.apply_detected_gear.assert_not_called()


def test_open_roll_settings_dialog_suggests_only_the_field_not_already_set(browser, session):
    """A camera already tagged (carried from elsewhere, set by hand) must not also block
    a film-stock match that is otherwise free to suggest -- the bug behind a folder like
    '08_penf_gold_200_marbella' matching Kodak Gold 200 but never offering it because an
    unrelated camera happened to already be set."""
    session.state.selected_file_idx = 0
    session.state.config = replace(session.state.config, metadata=replace(session.state.config.metadata, camera_id="existing"))
    detected = MagicMock(camera_id="cam1", film_stock_id="film1")
    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Rejected
    with (
        patch("negpy.desktop.view.sidebar.files.match_gear_for_folder", return_value=detected),
        patch("negpy.desktop.view.sidebar.files.RollSettingsDialog", return_value=mock_dlg),
    ):
        browser._open_roll_settings_dialog()

    mock_dlg.apply_detected_gear.assert_called_once_with(camera_id="", film_stock_id="film1")


def test_folder_name_for_gear_suggestion_prefers_the_active_folder_roll(browser, session, tmp_path):
    from negpy.services.assets.rolls import recognize_folder

    store: dict = {}
    session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    roll_id = recognize_folder(session.repo, str(tmp_path / "08_penf_gold_marbella"))
    session.state.active_roll_id = roll_id

    assert browser._folder_name_for_gear_suggestion() == "08_penf_gold_marbella"


def test_folder_name_for_gear_suggestion_falls_back_to_the_current_files_folder(browser, session):
    session.state.active_roll_id = None
    session.state.selected_file_idx = 0  # session fixture's first file lives under /tmp

    assert browser._folder_name_for_gear_suggestion() == "tmp"


def test_folder_name_for_gear_suggestion_is_empty_without_an_active_file(browser, session):
    session.state.active_roll_id = None
    session.state.selected_file_idx = -1

    assert browser._folder_name_for_gear_suggestion() == ""


def test_maybe_suggest_gear_opens_the_dialog_prefilled_when_something_matches(browser, session):
    session.state.selected_file_idx = 0
    session.apply_preset_fields = MagicMock(return_value=1)
    detected = MagicMock(any=MagicMock(return_value=True), camera_id="cam1", film_stock_id="film1")
    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.selected_rows.return_value = [object()]
    mock_dlg.selected_config.return_value = _edited_cfg()
    mock_dlg.scope.return_value = "roll"
    with (
        patch("negpy.desktop.view.sidebar.files.match_gear_for_folder", return_value=detected),
        patch("negpy.desktop.view.sidebar.files.RollSettingsDialog", return_value=mock_dlg),
    ):
        browser._maybe_suggest_gear("/library/08_penf_gold_marbella")

    mock_dlg.apply_detected_gear.assert_called_once_with(camera_id="cam1", film_stock_id="film1")
    browser.controller.request_render.assert_called_once()


def test_maybe_suggest_gear_does_nothing_without_a_match(browser, session):
    session.state.selected_file_idx = 0
    detected = MagicMock(camera_id="", film_stock_id="")
    with (
        patch("negpy.desktop.view.sidebar.files.match_gear_for_folder", return_value=detected),
        patch("negpy.desktop.view.sidebar.files.RollSettingsDialog") as ctor,
    ):
        browser._maybe_suggest_gear("/library/roll_a")

    ctor.assert_not_called()


def test_maybe_suggest_gear_noop_without_an_active_file(browser, session):
    session.state.selected_file_idx = -1
    detected = MagicMock(any=MagicMock(return_value=True), camera_id="cam1", film_stock_id="")
    with (
        patch("negpy.desktop.view.sidebar.files.match_gear_for_folder", return_value=detected),
        patch("negpy.desktop.view.sidebar.files.RollSettingsDialog") as ctor,
    ):
        browser._maybe_suggest_gear("/library/roll_a")

    ctor.assert_not_called()


def test_context_menu_paste_disabled_without_clipboard(browser, session):
    session.state.clipboard = None
    paste = next(a for a in browser._build_context_menu().actions() if a.text().startswith("Paste"))
    assert not paste.isEnabled()


def test_context_menu_paste_enabled_with_clipboard(browser, session):
    session.state.clipboard = object()
    paste = next(a for a in browser._build_context_menu().actions() if a.text().startswith("Paste"))
    assert paste.isEnabled()


def test_remove_from_menu_routes_single_vs_multi(browser, session):
    session.remove_current_file = MagicMock()
    session.remove_selected_files = MagicMock()

    # confirm_unload opens a blocking QMessageBox — must be patched headless.
    with patch("negpy.desktop.view.sidebar.files.confirm_unload", return_value=True):
        session.state.selected_indices = [1]
        browser._on_remove_from_menu()
        session.remove_current_file.assert_called_once()
        session.remove_selected_files.assert_not_called()

        session.remove_current_file.reset_mock()
        session.state.selected_indices = [0, 1]
        browser._on_remove_from_menu()
        session.remove_selected_files.assert_called_once()
        session.remove_current_file.assert_not_called()


def test_remove_from_menu_cancelled_confirm_removes_nothing(browser, session):
    session.remove_current_file = MagicMock()
    session.remove_selected_files = MagicMock()

    with patch("negpy.desktop.view.sidebar.files.confirm_unload", return_value=False):
        session.state.selected_indices = [1]
        browser._on_remove_from_menu()
        session.state.selected_indices = [0, 1]
        browser._on_remove_from_menu()

    session.remove_current_file.assert_not_called()
    session.remove_selected_files.assert_not_called()


def test_add_files_uses_and_saves_last_folder(browser, session):
    session.repo.get_global_setting.return_value = "/photos/scans"
    with patch(
        "negpy.desktop.view.sidebar.files.QFileDialog.getOpenFileNames",
        return_value=(["/photos/scans/2024/x.cr2"], ""),
    ) as dlg:
        browser.prompt_add_files()
    assert dlg.call_args.args[2] == "/photos/scans"
    session.repo.save_global_setting.assert_called_with("last_open_folder", "/photos/scans/2024")


def test_add_folder_uses_and_saves_parent_of_last_folder(browser, session):
    session.repo.get_global_setting.return_value = "/photos/scans"
    with patch(
        "negpy.desktop.view.sidebar.files.QFileDialog.getExistingDirectory",
        return_value="/photos/scans/2024",
    ) as dlg:
        browser.prompt_add_folder()
    assert dlg.call_args.args[2] == "/photos/scans"
    session.repo.save_global_setting.assert_called_with("last_open_folder", "/photos/scans")


def test_add_files_falls_back_to_empty_dir_when_unset(browser, session):
    session.repo.get_global_setting.return_value = None
    with patch(
        "negpy.desktop.view.sidebar.files.QFileDialog.getOpenFileNames",
        return_value=([], ""),
    ) as dlg:
        browser.prompt_add_files()
    assert dlg.call_args.args[2] == ""
    assert not any(c.args and c.args[0] == "last_open_folder" for c in session.repo.save_global_setting.call_args_list)


def test_clearing_filter_clears_error_stylesheet(browser):
    browser.regex_btn.setChecked(True)
    browser.search_input.setText("[")
    browser._apply_filter()
    assert browser.search_input.styleSheet() != ""

    browser.regex_btn.setChecked(False)
    browser.search_input.setText("")
    browser._apply_filter()
    assert browser.search_input.styleSheet() == ""


def test_thumbnail_grid_defaults_to_one_filling_column_at_min_sidebar_width(browser):
    """The session sidebar can't be dragged below ~240px, so that viewport width is
    the smallest the filmstrip ever lays out at. At the default thumbnail size it
    must show a single column that *fills* it — the previous 180px cell cap left
    25% of the panel empty at that width."""
    view = browser.list_view
    for viewport_w in (214, 240):  # measured: sidebar at minimum, then default width
        assert view.columns_for_width(viewport_w) == 1
        assert view.cell_for_width(viewport_w) / viewport_w > 0.95


def test_thumbnail_grid_stays_single_column_on_a_slightly_wider_sidebar(browser):
    """Regression: with the old 120px target the grid flipped to two columns as soon
    as the panel was nudged past ~260px, which is where users actually sit — the
    reported "two columns by default"."""
    view = browser.list_view
    for viewport_w in (260, 272, 300, 340):
        assert view.columns_for_width(viewport_w) == 1, f"split into columns at {viewport_w}px"


def test_thumbnail_slider_low_end_fits_two_columns(browser):
    """The point of the slider's low end: trade size for a second column at the
    same panel width."""
    view = browser.list_view
    browser.thumb_size_slider.setValue(THUMB_CELL_MIN)
    assert view.columns_for_width(240) == 2


def test_slider_maximum_never_pins_a_widened_sidebar_to_one_oversized_column(browser):
    """Regression: the slider's top end used to hold a widened sidebar at a single
    column, so the cell grew to the full panel width (~500px). Cells are square, so a
    3:2 frame in one left ~165px of empty space above and below it. Even at the
    largest setting a wide panel has to split into columns."""
    view = browser.list_view
    browser.thumb_size_slider.setValue(browser.thumb_size_slider.maximum())
    for viewport_w in (450, 500, 600, 700):
        assert view.columns_for_width(viewport_w) > 1, f"still one column at {viewport_w}px"
        assert view.cell_for_width(viewport_w) <= 300, f"oversized cell at {viewport_w}px"


def test_thumbnail_slider_drives_the_grid_live(browser):
    view = browser.list_view
    browser.thumb_size_slider.setValue(THUMB_CELL_MIN)
    assert view.target_cell == THUMB_CELL_MIN
    browser.thumb_size_slider.setValue(THUMB_CELL_MAX)
    assert view.target_cell == THUMB_CELL_MAX


def test_thumbnail_size_persists_only_on_release(browser, session):
    """Dragging crosses dozens of values; each one must not hit the settings DB."""
    session.repo.save_global_setting.reset_mock()
    browser.thumb_size_slider.setValue(180)
    assert not any(c.args and c.args[0] == "thumbnail_cell_size" for c in session.repo.save_global_setting.call_args_list)

    browser.thumb_size_slider.sliderReleased.emit()
    session.repo.save_global_setting.assert_called_with("thumbnail_cell_size", 180)


def test_thumbnail_size_restored_from_settings(session):
    session.repo.get_global_setting.side_effect = lambda key, default=None: (150 if key == "thumbnail_cell_size" else None)
    controller = MagicMock()
    controller.session = session
    restored = FileBrowser(controller)
    assert restored.thumb_size_slider.value() == 150
    assert restored.list_view.target_cell == 150


def test_thumbnail_size_out_of_range_setting_is_clamped(session):
    """A hand-edited or stale setting must not produce an unusable grid."""
    session.repo.get_global_setting.side_effect = lambda key, default=None: (9999 if key == "thumbnail_cell_size" else None)
    controller = MagicMock()
    controller.session = session
    restored = FileBrowser(controller)
    assert restored.list_view.target_cell == THUMB_CELL_MAX


# --- Session panel scrolling & empty-space menu ---------------------------


def _wheel(angle_y: int = 0, pixel_y: int = 0) -> QWheelEvent:
    return QWheelEvent(
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        QPoint(0, pixel_y),
        QPoint(0, angle_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _scrollable(browser, value: int = 0):
    """Lay the panel out narrow and short so the grid genuinely overflows — the
    scrollbar range is recomputed from the layout, so it can't just be faked."""
    browser.resize(240, 300)
    browser.show()
    QApplication.processEvents()
    view = browser.list_view
    bar = view.verticalScrollBar()
    assert bar.maximum() > 2 * view._row_step(), "fixture grid does not scroll"
    bar.setValue(value)
    QApplication.processEvents()
    return bar


def test_grid_scrolls_per_pixel(browser):
    """ScrollPerItem snaps to whole rows, so no partial offset (and no easing) is
    representable — it is what made the wheel jump several frames at a time."""
    assert browser.list_view.verticalScrollMode() == QAbstractItemView.ScrollMode.ScrollPerPixel


def test_one_wheel_notch_scrolls_exactly_one_row(browser):
    view = browser.list_view
    _scrollable(browser)
    view.wheelEvent(_wheel(angle_y=-120))
    assert view._scroll_target == view._row_step()


def test_wheel_scroll_is_animated_not_instant(browser):
    view = browser.list_view
    bar = _scrollable(browser)
    view.wheelEvent(_wheel(angle_y=-120))
    assert view._scroll_anim.state() == QPropertyAnimation.State.Running
    assert view._scroll_anim.duration() > 0
    assert view._scroll_anim.endValue() == view._scroll_target
    # The jump is eased in, not applied on the spot.
    assert bar.value() != view._scroll_target


def test_consecutive_notches_accumulate_onto_the_running_target(browser):
    """A fast spin must cover the full distance, not restart from wherever the
    easing had reached."""
    view = browser.list_view
    _scrollable(browser)
    view.wheelEvent(_wheel(angle_y=-120))
    assert view._scroll_target == view._row_step()
    view.wheelEvent(_wheel(angle_y=-120))
    assert view._scroll_target == 2 * view._row_step()


def test_wheel_up_scrolls_back(browser):
    view = browser.list_view
    bar = _scrollable(browser, value=2 * view._row_step())
    start = bar.value()  # the layout may clamp the requested offset
    view.wheelEvent(_wheel(angle_y=120))
    assert view._scroll_target == start - view._row_step()


def test_wheel_target_is_clamped_to_the_scroll_range(browser):
    view = browser.list_view
    bar = _scrollable(browser, value=2 * view._row_step())
    for _ in range(20):
        view.wheelEvent(_wheel(angle_y=120))  # up, well past the top
    assert view._scroll_target == bar.minimum()
    assert view._scroll_anim.endValue() == bar.minimum()


def test_trackpad_pixel_delta_scrolls_immediately(browser):
    """Trackpads already deliver continuous deltas; easing them would add lag."""
    view = browser.list_view
    bar = _scrollable(browser, value=100)
    start = bar.value()
    view.wheelEvent(_wheel(pixel_y=-40))
    assert bar.value() == start + 40
    assert view._scroll_anim.state() != QPropertyAnimation.State.Running


def test_session_menu_mirrors_the_toolbar_tools(browser):
    labels = [a.text() for a in browser._build_session_menu().actions() if not a.isSeparator()]
    assert labels == ["Add Files…", "Add Folder…", "Clear All…"]


def test_session_menu_clear_all_disabled_when_nothing_loaded(browser, session):
    session.state.uploaded_files = []
    clear = [a for a in browser._build_session_menu().actions() if a.text() == "Clear All…"][0]
    assert not clear.isEnabled()


def test_session_menu_clear_all_enabled_with_files(browser):
    clear = [a for a in browser._build_session_menu().actions() if a.text() == "Clear All…"][0]
    assert clear.isEnabled()


def test_right_click_on_empty_space_opens_the_session_menu(browser):
    """Previously this returned early, leaving empty space (and an empty session)
    with no context menu at all."""
    with (
        patch.object(browser, "_build_session_menu") as session_menu,
        patch.object(browser, "_build_context_menu") as frame_menu,
    ):
        browser._show_context_menu(QPoint(5, 99999))
    session_menu.assert_called_once()
    frame_menu.assert_not_called()


def test_right_click_on_a_frame_still_opens_the_frame_menu(browser):
    idx = browser.session.asset_model.index(0, 0)
    with (
        patch.object(browser.list_view, "indexAt", return_value=idx),
        patch.object(browser, "_build_context_menu") as frame_menu,
        patch.object(browser, "_build_session_menu") as session_menu,
    ):
        browser._show_context_menu(QPoint(5, 5))
    frame_menu.assert_called_once()
    session_menu.assert_not_called()


def test_session_menu_clear_all_clears_every_frame(browser, session):
    session.clear_files = MagicMock()
    with patch("negpy.desktop.view.sidebar.files.confirm_unload", return_value=True):
        browser._on_clear_all()
    session.clear_files.assert_called_once()


def test_new_roll_menu_action_clears_the_session_like_clear_all(browser, session):
    """Distinct from Unload: this is the deliberate "start over" action, for building a
    roll entirely by drag-drop, so it confirms and clears everything, not the selection."""
    session.clear_files = MagicMock()
    menu = browser.frames_section.actions_btn.menu()
    action = next(a for a in menu.actions() if a.text() == "New Roll…")
    with patch("negpy.desktop.view.sidebar.files.confirm_unload", return_value=True) as confirm:
        action.trigger()
    confirm.assert_called_once_with(browser, clear_all=True)
    session.clear_files.assert_called_once()


def test_reset_roll_menu_action_resets_every_visible_frame(browser, session):
    menu = browser.frames_section.actions_btn.menu()
    action = next(a for a in menu.actions() if a.text() == "Reset Roll to Defaults…")
    with patch("negpy.desktop.view.sidebar.files.confirm_reset_frames", return_value=True) as confirm:
        action.trigger()
    confirm.assert_called_once_with(browser, 4, roll=True)  # the session fixture's 4 uploaded_files
    browser.controller.request_reset_roll.assert_called_once()


def test_reset_roll_menu_action_cancelled_does_nothing(browser, session):
    with patch("negpy.desktop.view.sidebar.files.confirm_reset_frames", return_value=False):
        browser._on_reset_roll()
    browser.controller.request_reset_roll.assert_not_called()


def test_reset_roll_with_nothing_loaded_never_prompts(browser, session):
    session.state.uploaded_files = []
    session.asset_model.refresh()
    with patch("negpy.desktop.view.sidebar.files.confirm_reset_frames") as confirm:
        browser._on_reset_roll()
    confirm.assert_not_called()
    browser.controller.request_reset_roll.assert_not_called()


def test_unload_button_always_targets_the_selection_never_the_whole_roll(browser, session):
    """The toolbar button never falls back to Clear All: opening a different roll already
    replaces the film strip, so a stray click with nothing multi-selected must remove only
    the active frame, not wipe everything."""
    session.remove_current_file = MagicMock()
    session.remove_selected_files = MagicMock()
    session.clear_files = MagicMock()

    with patch("negpy.desktop.view.sidebar.files.confirm_unload", return_value=True):
        session.state.selected_indices = [1]
        browser._on_unload_clicked()
        session.remove_current_file.assert_called_once()
        session.remove_selected_files.assert_not_called()

        session.state.selected_indices = [0, 1]
        browser._on_unload_clicked()
        session.remove_selected_files.assert_called_once()

    session.clear_files.assert_not_called()


def test_unload_button_tooltip_reflects_the_selection(browser, session):
    session.state.selected_indices = [0]
    browser._update_unload_button()
    assert browser.unload_btn.toolTip() == "Unload…"

    session.state.selected_indices = [0, 1]
    browser._update_unload_button()
    assert browser.unload_btn.toolTip() == "Unload Selected…"


def test_save_roll_prompts_for_a_name_and_refreshes_the_tree(browser):
    browser.controller.create_roll_from_session.return_value = "roll-1"
    browser.library_tree = MagicMock()
    with patch("negpy.desktop.view.sidebar.files.QInputDialog.getText", return_value=("Portra", True)):
        browser._on_save_roll_clicked()

    browser.controller.create_roll_from_session.assert_called_once_with("Portra")
    browser.library_tree.reload.assert_called_once()


def test_save_roll_cancelled_does_nothing(browser):
    browser.controller.create_roll_from_session = MagicMock()
    with patch("negpy.desktop.view.sidebar.files.QInputDialog.getText", return_value=("Portra", False)):
        browser._on_save_roll_clicked()

    browser.controller.create_roll_from_session.assert_not_called()


def test_save_roll_rejects_an_invalid_name(browser):
    browser.controller.create_roll_from_session = MagicMock()
    with (
        patch("negpy.desktop.view.sidebar.files.QInputDialog.getText", return_value=("bad/name", True)),
        patch("negpy.desktop.view.sidebar.files.warn_invalid_roll_name"),
    ):
        browser._on_save_roll_clicked()

    browser.controller.create_roll_from_session.assert_not_called()


# --- Composite badges -----------------------------------------------------


def _composite_assets() -> dict:
    return {
        "plain": {"name": "a.cr2", "path": "/tmp/a.cr2", "hash": "h"},
        "stitch": {"name": "a+b (Stitch)", "path": "/tmp/a.cr2", "hash": "h#stitch", "stitch_paths": ("/tmp/b.cr2",)},
        "hdr": {"name": "a +2 (HDR)", "path": "/tmp/a.cr2", "hash": "h#hdr", "hdr_paths": ("/tmp/b.cr2", "/tmp/c.cr2")},
        "rgb": {"name": "a.cr2", "path": "/tmp/a.cr2", "hash": "h", "green_path": "/tmp/g.cr2", "blue_path": "/tmp/b.cr2"},
        "half": {"name": "a [2]", "path": "/tmp/a.cr2", "hash": "h#2", "half": 2},
    }


@pytest.mark.parametrize("key,kind", [("plain", ""), ("stitch", "stitch"), ("hdr", "hdr"), ("rgb", "rgb"), ("half", "half")])
def test_composite_kind_reads_the_asset_dict(key, kind):
    assert composite_kind(_composite_assets()[key]) == kind


def test_stitch_of_triplets_reads_as_a_stitch():
    """A stitch built from triplets carries the primary part's green/blue pair too, so
    a green_path-first test would badge it as a triplet."""
    asset = {**_composite_assets()["stitch"], "green_path": "/tmp/g.cr2", "blue_path": "/tmp/b.cr2"}
    assert composite_kind(asset) == "stitch"


def test_composite_summary_counts_every_source_frame():
    assets = _composite_assets()
    assert composite_summary(assets["stitch"]) == "Stitched composite of 2 frames"
    assert composite_summary(assets["hdr"]) == "HDR merge of 3 exposures"
    assert composite_summary(assets["rgb"]) == "Trichrome triplet"
    assert composite_summary(assets["half"]) == "Half-frame split (2 of 2)"
    assert composite_summary(assets["plain"]) == ""


def test_tooltip_names_what_the_frame_is_built_from(session):
    session.state.uploaded_files = [_composite_assets()["hdr"], _composite_assets()["plain"]]
    session.asset_model.refresh()
    model = session.asset_model
    tips = [model.data(model.index(row, 0), Qt.ItemDataRole.ToolTipRole) for row in range(model.rowCount())]
    merged = [t for t in tips if "HDR merge" in t]
    assert merged == ["/tmp/a.cr2\nHDR merge of 3 exposures"]
    assert tips.count("/tmp/a.cr2") == 1  # the plain frame keeps the path alone


def _render(asset: dict, *, with_thumbnail: bool = True, activity_phase: float | None = None) -> QImage:
    """Paint one delegate cell onto a pixmap. paint() reads only index.data(), so a
    stub index is enough."""
    thumb = QPixmap(60, 40)
    thumb.fill(QColor("#808080"))
    icon = QIcon(thumb) if with_thumbnail else QIcon()
    index = MagicMock()
    index.data.side_effect = lambda role: {
        Qt.ItemDataRole.UserRole: asset,
        Qt.ItemDataRole.DecorationRole: icon,
    }.get(role)

    canvas = QPixmap(120, 120)
    canvas.fill(QColor("#000000"))
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 120, 120)
    painter = QPainter(canvas)
    delegate = _ThumbnailDelegate()
    if activity_phase is not None:
        delegate.set_activity(asset_thumbnail_key(asset))
        delegate._activity_timer.stop()
        delegate._activity_phase = activity_phase
    delegate.paint(painter, option, index)
    painter.end()
    return canvas.toImage()


def test_placeholder_fills_the_square_thumbnail_cell(qapp):
    image = _render({}, with_thumbnail=False)

    assert image.pixelColor(60, 4) != QColor("#000000")
    assert image.pixelColor(4, 60) != QColor("#000000")


def test_placeholder_item_uses_the_full_thumbnail_cell(qapp):
    view = ThumbnailGridView(target_cell=THUMB_CELL_MIN)
    delegate = _ThumbnailDelegate(view)

    assert delegate.sizeHint(QStyleOptionViewItem(), QModelIndex()) == view.iconSize()


def test_active_placeholder_curtain_advances_across_the_glyph(qapp):
    asset = _composite_assets()["plain"]

    early = _render(asset, with_thumbnail=False, activity_phase=0.25)
    late = _render(asset, with_thumbnail=False, activity_phase=0.75)

    assert early != late


def test_placeholder_animation_repaints_only_the_active_cell(session, qapp):
    view = ThumbnailGridView(target_cell=THUMB_CELL_MIN)
    view.resize(320, 240)
    view.setModel(session.asset_model)
    delegate = _ThumbnailDelegate(view, state=session.state)
    view.setItemDelegate(delegate)
    view.show()
    qapp.processEvents()
    delegate.set_activity(asset_thumbnail_key(session.state.uploaded_files[1]))
    delegate._activity_timer.stop()

    with patch.object(view.viewport(), "update") as update:
        delegate._advance_activity()

    update.assert_called_once()
    assert update.call_args.args == (view.visualRect(QModelIndex(delegate._activity_index)),)


def _badge_corner(image: QImage) -> list:
    """The 18px badge box at the bottom-left of the image outline. A 60x40 thumbnail in
    a 120px cell lands at (3, 22, 114, 76), so the chip spans roughly (7, 75)-(25, 93)."""
    return [image.pixel(x, y) for y in range(73, 95) for x in range(5, 27)]


@pytest.mark.parametrize("key", ["stitch", "hdr", "rgb", "half"])
def test_composite_badge_is_painted_bottom_left(key, qapp):
    assets = _composite_assets()
    assert _badge_corner(_render(assets[key])) != _badge_corner(_render(assets["plain"]))


def test_each_composite_kind_draws_its_own_glyph(qapp):
    """The point of per-kind glyphs: a merge must not look like a stitch."""
    assets = _composite_assets()
    corners = [_badge_corner(_render(assets[k])) for k in ("stitch", "hdr", "rgb", "half")]
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            assert a != b


def _scene_menu(menu):
    return next(a.menu() for a in menu.actions() if a.text() == "Scene")


def test_scene_menu_is_absent_without_a_roll(browser, session):
    session.state.selected_indices = [0, 1]
    assert "Scene" not in _action_labels(browser._build_context_menu())


def test_scene_menu_offers_group_and_add_for_loose_frames(browser, session):
    from negpy.services.assets import rolls

    session.state.active_roll_id = "roll-1"
    session.state.selected_indices = [0, 1]
    with patch.object(rolls, "roll_scenes", return_value=[("s1", {"name": "Beach"})]):
        labels = _action_labels(_scene_menu(browser._build_context_menu()))
    assert labels == ["Group as Scene…", "Add to Beach"]


def test_scene_menu_for_members_of_one_scene(browser, session):
    from negpy.services.assets import rolls

    session.state.active_roll_id = "roll-1"
    session.state.selected_indices = [0]
    session.state.selected_file_idx = 0
    session.state.uploaded_files[0]["scene"] = (1, "s1", "Beach")
    with patch.object(rolls, "roll_scenes", return_value=[("s1", {"name": "Beach"}), ("s2", {"name": "Night"})]):
        menu = _scene_menu(browser._build_context_menu())
    assert _action_labels(menu) == ["Add to Night", "Remove from Scene", "Analyze Scene…", "Rename Scene…", "Delete Scene…"]
    next(a for a in menu.actions() if a.text() == "Analyze Scene…").trigger()
    browser.controller.request_scene_analysis.assert_called_once_with("s1")


def test_tooltip_names_the_scene(session):
    session.state.uploaded_files[0]["scene"] = (1, "s1", "Beach")
    session.asset_model.refresh()
    tip = session.asset_model.data(session.asset_model.index(0), Qt.ItemDataRole.ToolTipRole)
    assert "Scene: Beach" in tip


def _stamp_scenes(session, scenes) -> None:
    for f, scene in zip(session.state.uploaded_files, scenes):
        if scene:
            f["scene"] = scene
        else:
            f.pop("scene", None)
    session.asset_model.refresh()
    session.files_changed.emit()


def test_scene_sort_is_offered_only_while_the_roll_has_a_scene(browser, session):
    assert not browser.act_sort_scene.isVisible()

    _stamp_scenes(session, [(1, "s1", "Beach"), None, None, None])
    assert browser.act_sort_scene.isVisible()

    _stamp_scenes(session, [None] * 4)
    assert not browser.act_sort_scene.isVisible()


def test_the_first_scene_switches_the_strip_to_scene_sort(browser, session):
    _stamp_scenes(session, [None, (1, "s1", "Beach"), None, (1, "s1", "Beach")])
    switch = browser.controller.first_scene_created.connect.call_args[0][0]

    switch()

    assert session.asset_model.effective_sort_order == "scene"
    assert browser.act_sort_scene.isChecked()
    names = [session.state.uploaded_files[i]["name"] for i in session.asset_model.visible_actual_indices_ordered()]
    assert names[:2] == ["IMG_0002.cr2", "note.txt"]
    session.repo.save_global_setting.assert_any_call("file_sort_order", "scene")


def test_scene_sort_reads_as_name_without_scenes(browser, session):
    browser._apply_sort_order("scene")

    assert browser.act_sort_name.isChecked()
    assert not browser.act_sort_scene.isVisible()
