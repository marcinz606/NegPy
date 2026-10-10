from unittest.mock import MagicMock

from negpy.desktop.session import DesktopSessionManager
from negpy.infrastructure.storage.repository import StorageRepository


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
