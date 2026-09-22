"""The two Roll-tab cards that decide how files become frames: Trichrome and Half Frame."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.half_frame import HalfFrameSidebar
from negpy.desktop.view.sidebar.trichrome import TrichromeSidebar
from negpy.features.rgbscan.models import RgbScanConfig


def _controller():
    controller = MagicMock()
    controller.state = AppState()
    controller.session.repo.get_global_setting.return_value = False
    controller.half_frame_mode_for_roll.return_value = False
    controller.active_diptych.return_value = None
    controller.current_base_file.return_value = ("/tmp/scan.tif", "h1")
    controller.selected_base_hashes.return_value = ["h1"]
    return controller


@pytest.fixture
def trichrome(qapp):
    controller = _controller()
    return TrichromeSidebar(controller), controller


@pytest.fixture
def half_frame(qapp):
    controller = _controller()
    controller.state.active_roll_id = "r1"
    controller.state.uploaded_files = [{"path": "/tmp/scan.tif", "hash": "h1"}]
    sidebar = HalfFrameSidebar(controller)
    controller.reset_mock(return_value=False)
    controller.half_frame_mode_for_roll.return_value = False
    controller.active_diptych.return_value = None
    controller.current_base_file.return_value = ("/tmp/scan.tif", "h1")
    controller.selected_base_hashes.return_value = ["h1"]
    return sidebar, controller


def test_trichrome_toggle_drives_the_mode(trichrome):
    """Nothing loaded, so there is no re-discovery to confirm first."""
    sidebar, controller = trichrome

    sidebar.enable_btn.setChecked(True)

    controller.set_rgb_scan_mode.assert_called_once_with(True)


def test_trichrome_asks_before_regrouping_a_loaded_roll(trichrome):
    sidebar, controller = trichrome
    controller.state.uploaded_files = [{"path": "/tmp/r.cr2", "hash": "h1"}]

    with patch("negpy.desktop.view.sidebar.trichrome.confirm_assembly_mode", return_value=False) as ask:
        sidebar.enable_btn.setChecked(True)

    ask.assert_called_once_with(sidebar, "Trichrome", 1)
    controller.set_rgb_scan_mode.assert_not_called()
    assert sidebar.enable_btn.isChecked() is False

    with patch("negpy.desktop.view.sidebar.trichrome.confirm_assembly_mode", return_value=True):
        sidebar.enable_btn.setChecked(True)

    controller.set_rgb_scan_mode.assert_called_once_with(True)


def test_trichrome_off_needs_no_confirmation(trichrome):
    sidebar, controller = trichrome
    controller.state.uploaded_files = [{"path": "/tmp/r.cr2", "hash": "h1"}]
    sidebar._follow_mode(True)

    with patch("negpy.desktop.view.sidebar.trichrome.confirm_assembly_mode") as ask:
        sidebar.enable_btn.setChecked(False)

    ask.assert_not_called()
    controller.set_rgb_scan_mode.assert_called_once_with(False)


def test_trichrome_follows_a_mode_change_it_did_not_make(trichrome):
    """The controller has already applied it; letting toggled through would ask for it a
    second time and re-run discovery."""
    sidebar, controller = trichrome

    sidebar._follow_mode(True)

    assert sidebar.enable_btn.isChecked() is True
    controller.set_rgb_scan_mode.assert_not_called()


def test_trichrome_hint_names_the_frames_exposures(trichrome):
    sidebar, controller = trichrome
    controller.state.config = replace(
        controller.state.config,
        rgbscan=RgbScanConfig(enabled=True, green_path="/tmp/g.cr2", blue_path="/tmp/b.cr2"),
    )

    sidebar.sync_ui()

    assert "g.cr2" in sidebar.hint.text()
    assert "b.cr2" in sidebar.hint.text()


def test_trichrome_edit_needs_a_loaded_frame(trichrome):
    sidebar, controller = trichrome
    assert sidebar.edit_btn.isEnabled() is False

    controller.state.uploaded_files = [{"path": "/tmp/r.cr2", "hash": "h1"}]
    sidebar.sync_ui()

    assert sidebar.edit_btn.isEnabled() is True


def test_trichrome_edit_opens_the_triplet_dialog(trichrome):
    sidebar, controller = trichrome
    controller.state.uploaded_files = [{"path": "/tmp/r.cr2", "hash": "h1"}]
    sidebar.sync_ui()

    with patch("negpy.desktop.view.sidebar.trichrome.open_triplet_dialog") as opener:
        sidebar.edit_btn.click()

    opener.assert_called_once_with(sidebar, controller.session)


def test_half_frame_toggle_on_auto_detects_without_opening_a_dialog(half_frame):
    """A plain toggle: turning it on runs the same batch detection Auto-detect All
    Splits does, never the rectangle editor."""
    sidebar, controller = half_frame

    with patch("negpy.desktop.view.sidebar.half_frame.confirm_assembly_mode", return_value=True):
        sidebar.enable_btn.setChecked(True)

    controller.open_half_frame_dialog.assert_not_called()
    controller.set_half_frame_mode.assert_called_once_with(True)
    controller.auto_detect_all_half_frame_splits.assert_called_once_with()


def test_half_frame_toggle_off_does_not_auto_detect(half_frame):
    sidebar, controller = half_frame

    sidebar._on_toggled(False)

    controller.set_half_frame_mode.assert_called_once_with(False)
    controller.auto_detect_all_half_frame_splits.assert_not_called()


def test_half_frame_toggle_on_skips_auto_detect_with_nothing_loaded(half_frame):
    """Nothing to split means nothing to re-discover, so no confirmation either."""
    sidebar, controller = half_frame
    controller.state.uploaded_files = []

    with patch("negpy.desktop.view.sidebar.half_frame.confirm_assembly_mode") as ask:
        sidebar._on_toggled(True)

    ask.assert_not_called()

    controller.set_half_frame_mode.assert_called_once_with(True)
    controller.auto_detect_all_half_frame_splits.assert_not_called()


def test_half_frame_asks_before_splitting_a_loaded_roll(half_frame):
    sidebar, controller = half_frame

    with patch("negpy.desktop.view.sidebar.half_frame.confirm_assembly_mode", return_value=False) as ask:
        sidebar.enable_btn.setChecked(True)

    ask.assert_called_once_with(sidebar, "Half Frame", 1)
    controller.set_half_frame_mode.assert_not_called()
    controller.auto_detect_all_half_frame_splits.assert_not_called()
    assert sidebar.enable_btn.isChecked() is False


def test_half_frame_off_needs_no_confirmation(half_frame):
    sidebar, controller = half_frame
    controller.half_frame_mode_for_roll.return_value = True
    sidebar._follow_mode(True)

    with patch("negpy.desktop.view.sidebar.half_frame.confirm_assembly_mode") as ask:
        controller.half_frame_mode_for_roll.return_value = False
        sidebar.enable_btn.setChecked(False)

    ask.assert_not_called()
    controller.set_half_frame_mode.assert_called_once_with(False)


def test_half_frame_actions_share_one_row(half_frame):
    """Three buttons side by side, each taking a third of the card's width."""
    sidebar, _ = half_frame
    rows = {sidebar.layout.itemAt(i).layout() for i in range(sidebar.layout.count())}
    row = next(r for r in rows if r is not None and r.indexOf(sidebar.adjust_btn) >= 0)

    assert [row.itemAt(i).widget() for i in range(row.count())] == [
        sidebar.adjust_btn,
        sidebar.auto_btn,
        sidebar.unsplit_btn,
    ]
    assert {row.stretch(i) for i in range(row.count())} == {1}


def test_half_frame_follows_the_active_rolls_state_without_retoggling(half_frame):
    """A roll switch drives the toggle, not a click, so it must not run
    set_half_frame_mode a second time."""
    sidebar, controller = half_frame
    controller.half_frame_mode_for_roll.return_value = True

    sidebar._follow_mode(True)

    assert sidebar.enable_btn.isChecked() is True
    controller.set_half_frame_mode.assert_not_called()


def test_half_frame_card_is_dead_without_one_roll(half_frame):
    """A batch with no single active roll (a library-wide search's mixed results) has no
    roll-wide toggle to apply -- nothing in it splits."""
    sidebar, controller = half_frame
    controller.state.active_roll_id = None

    sidebar.sync_ui()

    assert sidebar.enable_btn.isEnabled() is False
    assert sidebar.adjust_btn.isEnabled() is False
    assert sidebar.auto_btn.isEnabled() is False
    assert "roll-wide setting" in sidebar.hint.text()


def test_half_frame_card_is_live_with_an_active_roll(half_frame):
    sidebar, _ = half_frame

    sidebar.sync_ui()

    assert sidebar.enable_btn.isEnabled() is True
    assert sidebar.adjust_btn.isEnabled() is True
    assert sidebar.hint.text() == ""


def test_unsplit_is_enabled_only_for_a_diptych(half_frame):
    sidebar, controller = half_frame
    sidebar.sync_ui()
    assert sidebar.unsplit_btn.isEnabled() is False

    controller.active_diptych.return_value = ({}, (None, None))
    sidebar.sync_ui()

    assert sidebar.unsplit_btn.isEnabled() is True


def test_unsplit_needs_the_confirm(half_frame):
    sidebar, controller = half_frame

    with patch("negpy.desktop.view.sidebar.half_frame.confirm_undiptych", return_value=False):
        sidebar._on_unsplit()
    controller.request_undiptych.assert_not_called()

    with patch("negpy.desktop.view.sidebar.half_frame.confirm_undiptych", return_value=True):
        sidebar._on_unsplit()
    controller.request_undiptych.assert_called_once_with()


def test_adjust_offers_the_selection_and_reloads_only_on_apply(half_frame):
    sidebar, controller = half_frame
    controller.open_half_frame_dialog.return_value = None

    sidebar._on_adjust()

    controller.open_half_frame_dialog.assert_called_once_with("/tmp/scan.tif", "h1", selected_hashes=["h1"])
    controller.reload_after_half_frame_change.assert_not_called()

    controller.open_half_frame_dialog.return_value = {"split_x": 0.4}
    sidebar._on_adjust()

    controller.reload_after_half_frame_change.assert_called_once_with()


def test_adjust_does_nothing_without_a_frame(half_frame):
    sidebar, controller = half_frame
    controller.current_base_file.return_value = (None, None)

    sidebar._on_adjust()

    controller.open_half_frame_dialog.assert_not_called()
