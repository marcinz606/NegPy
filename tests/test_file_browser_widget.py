from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QPoint, QPointF, QPropertyAnimation, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QDialog

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.sidebar.files import THUMB_CELL_MAX, THUMB_CELL_MIN, FileBrowser
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository


def _edited_cfg() -> WorkspaceConfig:
    """A config with a couple of non-default settings so the picker renders rows."""
    c = WorkspaceConfig()
    return replace(
        c,
        exposure=replace(c.exposure, density=1.5),
        geometry=replace(c.geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9)),
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
    assert "Export current frame" in labels
    assert "Export selected frames" not in labels
    assert "Reset Settings" in labels
    assert "Unload" in labels
    assert "Apply settings…" in labels


def test_context_menu_multi_selection_uses_export_selected(browser, session):
    session.state.selected_indices = [0, 1]
    session.state.selected_file_idx = 0
    labels = _action_labels(browser._build_context_menu())
    assert "Export selected frames" in labels
    assert "Export current frame" not in labels


def test_context_menu_multi_selection_adds_apply_and_remove_selected(browser, session):
    session.state.selected_indices = [0, 1]
    session.state.selected_file_idx = 0
    labels = _action_labels(browser._build_context_menu())
    assert "Apply settings…" in labels
    assert "Unload Selected" in labels
    assert "Unload" not in labels


def test_apply_dialog_shows_header_scope_and_counts(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=2, roll_count=3)
    assert dlg.sel_radio.text() == "Selected frames (2)"
    assert dlg.sel_radio.isEnabled()
    assert dlg.sel_radio.isChecked()  # selection preferred when it has targets
    assert dlg.roll_radio.text() == "Whole roll (3)"
    assert dlg.roll_radio.isEnabled()


def test_apply_dialog_defaults_to_roll_when_selection_empty(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=0, roll_count=3)
    assert not dlg.sel_radio.isEnabled()
    assert dlg.roll_radio.isChecked()


def test_apply_dialog_check_all_and_none(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    assert dlg.apply_btn.isEnabled()  # rows checked by default
    dlg._set_all_checked(False)
    assert not any(box.isChecked() for box in dlg._all_boxes())
    assert not dlg.apply_btn.isEnabled()
    dlg._set_all_checked(True)
    # unchanged rows stay hidden and unchecked until "Show unchanged settings"
    assert {r.label for r in dlg.selected()} == {"Print Density", "Manual Crop"}
    assert dlg.apply_btn.isEnabled()


def test_apply_dialog_apply_collects_checked_rows_and_scope(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    dlg.roll_radio.setChecked(True)
    dlg._on_apply()
    labels = {r.label for r in dlg.selected()}
    assert "Print Density" in labels  # the edited exposure setting
    assert "Manual Crop" in labels  # the edited geometry setting
    assert dlg.scope() == "roll"


def test_apply_dialog_only_preselects_edited_settings(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    assert {r.label for r in dlg.selected()} == {"Print Density", "Manual Crop"}  # nothing else was non-default
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
    with patch("negpy.desktop.view.sidebar.files.GranularSettingsDialog", return_value=mock_dlg) as ctor:
        browser._open_apply_dialog()

    assert ctor.call_args.args[2] == "IMG_0001.cr2"
    assert ctor.call_args.kwargs["sel_count"] == 1  # 1 other selected
    assert ctor.call_args.kwargs["roll_count"] == 3  # 3 other on roll
    session.sync_selected_settings.assert_called_once_with(rows, (False, False), "selection")


def test_open_apply_dialog_noop_without_active_file(browser, session):
    session.state.selected_file_idx = -1
    session.sync_selected_settings = MagicMock()
    with patch("negpy.desktop.view.sidebar.files.GranularSettingsDialog") as ctor:
        browser._open_apply_dialog()
    ctor.assert_not_called()
    session.sync_selected_settings.assert_not_called()


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
    assert labels == ["Add files…", "Add folder…", "Clear all"]


def test_session_menu_clear_all_disabled_when_nothing_loaded(browser, session):
    session.state.uploaded_files = []
    clear = [a for a in browser._build_session_menu().actions() if a.text() == "Clear all"][0]
    assert not clear.isEnabled()


def test_session_menu_clear_all_enabled_with_files(browser):
    clear = [a for a in browser._build_session_menu().actions() if a.text() == "Clear all"][0]
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
