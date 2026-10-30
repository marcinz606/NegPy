from unittest.mock import MagicMock
import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.mark.parametrize("locale,noun", [("zh_CN", "帧"), ("ja", "フレーム"), ("ko", "프레임")])
@pytest.mark.parametrize("count", [1, 2])
def test_sequential_preview_uses_language_count_forms(qapp, locale: str, noun: str, count: int) -> None:
    from negpy.desktop.view.widgets.granular_settings_dialog import SequentialFrameDialog
    from negpy.kernel.system import i18n

    previous = i18n.language()
    try:
        i18n.configure(locale)
        dialog = SequentialFrameDialog(None, count, count, 10)
        assert dialog.preview.text().endswith(f"({count} {noun})")
    finally:
        i18n.configure(previous)


@pytest.mark.parametrize("count,noun", [(1, "Bild"), (2, "Bilder")])
def test_sequential_preview_uses_german_count_forms(qapp, count: int, noun: str) -> None:
    from negpy.desktop.view.widgets.granular_settings_dialog import SequentialFrameDialog
    from negpy.kernel.system import i18n

    previous = i18n.language()
    try:
        i18n.configure("de")
        dialog = SequentialFrameDialog(None, count, count, 10)
        assert dialog.preview.text().endswith(f"({count} {noun})")
    finally:
        i18n.configure(previous)


def _session() -> DesktopSessionManager:
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.get_max_history_index.return_value = 0
    mgr = DesktopSessionManager(repo)
    mgr.state.uploaded_files = [{"name": f"f{i}.tif", "path": f"/tmp/f{i}.tif", "hash": f"h{i}"} for i in range(3)]
    mgr.asset_model.refresh()
    mgr.state.selected_file_idx = 0
    mgr.state.selected_indices = [0, 1]
    mgr.state.current_file_hash = "h0"
    mgr.state.current_file_path = "/tmp/f0.tif"
    return mgr


def _saved(mgr: DesktopSessionManager) -> dict:
    return {c.args[0]: c.args[1] for c in mgr.repo.save_file_settings.call_args_list}


def test_assign_sequential_selection_ascending(qapp):
    mgr = _session()
    assert mgr.assign_sequential_frame_numbers(10, 1, "asc", "selection") == 2
    # The active frame (index 0) takes the first number via update_config.
    assert mgr.state.config.metadata.capture_frame == 10
    saved = _saved(mgr)
    assert saved["h0"].metadata.capture_frame == 10
    assert saved["h1"].metadata.capture_frame == 11


def test_assign_sequential_roll_descending(qapp):
    mgr = _session()
    assert mgr.assign_sequential_frame_numbers(10, 2, "desc", "roll") == 3
    assert mgr.state.config.metadata.capture_frame == 10
    saved = _saved(mgr)
    assert saved["h1"].metadata.capture_frame == 8
    assert saved["h2"].metadata.capture_frame == 6


def test_assign_sequential_includes_the_active_frame(qapp):
    mgr = _session()
    mgr.assign_sequential_frame_numbers(5, 1, "asc", "selection")
    # The active frame is a target too: it is written through update_config, not skipped.
    assert "h0" in _saved(mgr)
    assert mgr.state.config.metadata.capture_frame == 5


def test_assign_sequential_writes_every_selected_half(qapp):
    mgr = _session()
    mgr.state.selected_indices = [0, 1, 2]
    assert mgr.assign_sequential_frame_numbers(20, 1, "asc", "selection") == 3
    saved = _saved(mgr)
    assert [saved["h0"].metadata.capture_frame, saved["h1"].metadata.capture_frame, saved["h2"].metadata.capture_frame] == [20, 21, 22]


def test_assign_sequential_no_active_frame_returns_zero(qapp):
    mgr = _session()
    mgr.state.selected_file_idx = -1
    assert mgr.assign_sequential_frame_numbers(10, 1, "asc", "selection") == 0
    mgr.repo.save_file_settings.assert_not_called()
    mgr.repo.save_history_step.assert_not_called()


def test_assign_sequential_offscreen_excludes_the_active_frame(qapp):
    mgr = _session()
    offscreen: list[list[str]] = []
    mgr.frames_edited_offscreen.connect(offscreen.append)
    mgr.assign_sequential_frame_numbers(10, 1, "asc", "roll")
    # h0 is the active frame, rendered live; only the offscreen targets flag staleness.
    assert offscreen == [["h1", "h2"]]


def test_assign_sequential_pushes_external_history_for_undo(qapp):
    mgr = _session()
    mgr.assign_sequential_frame_numbers(10, 1, "asc", "selection")
    # Each non-active target writes two history steps (pre, then post) so Ctrl+Z
    # restores the pre value. h1 is the only non-active target here.
    h1_configs = [c.args[2] for c in mgr.repo.save_history_step.call_args_list if c.args[0] == "h1"]
    assert len(h1_configs) == 2
    assert any(cfg.metadata.capture_frame == 11 for cfg in h1_configs)
    assert any(cfg.metadata.capture_frame != 11 for cfg in h1_configs)


def test_assign_sequential_skips_rejected(qapp):
    mgr = _session()
    mgr.state.uploaded_files[1]["excluded"] = True
    # The rejected middle frame leaves the run; the numbers fill the frames that remain.
    assert mgr.assign_sequential_frame_numbers(10, 1, "asc", "roll", skip_rejected=True) == 2
    assert mgr.state.config.metadata.capture_frame == 10
    saved = _saved(mgr)
    assert "h1" not in saved
    assert saved["h2"].metadata.capture_frame == 11


def test_assign_sequential_skips_rejected_active_frame(qapp):
    mgr = _session()
    mgr.state.uploaded_files[0]["excluded"] = True
    # The active frame is rejected too, so the run starts at the next frame.
    assert mgr.assign_sequential_frame_numbers(20, 1, "asc", "roll", skip_rejected=True) == 2
    saved = _saved(mgr)
    assert saved["h1"].metadata.capture_frame == 20
    assert saved["h2"].metadata.capture_frame == 21


def test_dialog_reads_its_direction_and_scope_choices(qapp):
    from negpy.desktop.view.widgets.granular_settings_dialog import SequentialFrameDialog

    dlg = SequentialFrameDialog(None, sel_count=3, roll_count=5, start=10)
    assert dlg.result() == (10, 1, "asc", "selection", False)
    dlg.direction_choice.setCurrentIndex(1)
    dlg._scope_choice.setCurrentIndex(1)
    assert dlg.result() == (10, 1, "desc", "roll", False)
    assert dlg.preview.text().startswith("10, 9, 8 …")


def _session_shuffled(active_idx: int = 0, n: int = 6) -> DesktopSessionManager:
    """Like `_session`, but `uploaded_files` is in a load order that differs from name
    order, so the Film Strip display order is not the internal index order. The active
    frame is `uploaded_files[active_idx]`, which need not be first in display order."""
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.get_max_history_index.return_value = 0
    mgr = DesktopSessionManager(repo)
    load_order = [3, 0, 5, 1, 4, 2][:n]
    mgr.state.uploaded_files = [{"name": f"f{i}.tif", "path": f"/tmp/f{i}.tif", "hash": f"h{i}"} for i in load_order]
    mgr.asset_model.refresh()
    mgr.state.selected_file_idx = active_idx
    mgr.state.selected_indices = list(range(n))
    mgr.state.current_file_hash = mgr.state.uploaded_files[active_idx]["hash"]
    mgr.state.current_file_path = mgr.state.uploaded_files[active_idx]["path"]
    return mgr


def _display_numbers(mgr: DesktopSessionManager) -> list[int]:
    """Capture-frame numbers read in Film Strip display order."""
    saved = _saved(mgr)
    display = mgr.asset_model.visible_actual_indices_ordered()
    return [saved[mgr.state.uploaded_files[i]["hash"]].metadata.capture_frame for i in display]


def test_assign_sequential_selection_follows_display_order(qapp):
    # The load order differs from name order; the run must number in display order.
    mgr = _session_shuffled()
    assert mgr.assign_sequential_frame_numbers(0, 1, "asc", "selection") == 6
    assert _display_numbers(mgr) == [0, 1, 2, 3, 4, 5]


def test_assign_sequential_selection_descending_display(qapp):
    mgr = _session_shuffled()
    mgr.asset_model.set_sort_descending(True)
    assert mgr.assign_sequential_frame_numbers(0, 1, "asc", "selection") == 6
    # The first frame in the descending display order takes the start value.
    assert _display_numbers(mgr)[0] == 0


def test_assign_sequential_hidden_selected_after_visible(qapp):
    # The active frame (f4) is itself hidden by the filter; a filtered-out selected
    # frame stays a target, numbered after the visible ones in index order.
    mgr = _session_shuffled(active_idx=4)
    mgr.asset_model.set_filter("f[0-3]", regex=True)
    assert mgr.assign_sequential_frame_numbers(0, 1, "asc", "selection") == 6
    saved = _saved(mgr)
    # The four visible frames (f0-f3) take the first numbers in display order.
    assert _display_numbers(mgr) == [0, 1, 2, 3]
    # The two hidden selected frames follow, in index order (f5 before f4).
    assert saved["h5"].metadata.capture_frame == 4
    assert saved["h4"].metadata.capture_frame == 5
