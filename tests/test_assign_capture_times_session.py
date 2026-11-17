from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

from negpy.desktop.session import DesktopSessionManager
from negpy.features.metadata.capture_times import Anchor, Segment, assign_times
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
    mgr.state.selected_indices = [0]
    mgr.state.current_file_hash = "h0"
    mgr.state.current_file_path = "/tmp/f0.tif"
    return mgr


def _saved(mgr: DesktopSessionManager) -> dict:
    return {c.args[0]: c.args[1] for c in mgr.repo.save_file_settings.call_args_list}


def _fields(text: str, source: str = "day") -> dict:
    return {"capture_date": text, "capture_timezone": "Africa/Cairo", "capture_time_source": source}


def test_assign_writes_the_active_frame_and_the_others(qapp) -> None:
    mgr = _session()
    changes = {"h0": _fields("2026-10-25 12:00:00"), "h2": _fields("2026-10-25 12:01:00")}
    assert mgr.assign_capture_times(changes) == 2
    assert mgr.state.config.metadata.capture_date == "2026-10-25 12:00:00"
    saved = _saved(mgr)
    assert saved["h2"].metadata.capture_date == "2026-10-25 12:01:00"
    assert saved["h2"].metadata.capture_timezone == "Africa/Cairo"
    assert "h1" not in saved


def test_assign_flags_only_offscreen_frames(qapp) -> None:
    mgr = _session()
    offscreen: list[list[str]] = []
    mgr.frames_edited_offscreen.connect(offscreen.append)
    mgr.assign_capture_times({"h0": _fields("2026-10-25 12:00:00"), "h1": _fields("2026-10-25 12:01:00")})
    assert offscreen == [["h1"]]


def test_assign_keeps_undo_steps_for_other_frames(qapp) -> None:
    mgr = _session()
    mgr.assign_capture_times({"h1": _fields("2026-10-25 12:01:00")})
    steps = [c.args[2] for c in mgr.repo.save_history_step.call_args_list if c.args[0] == "h1"]
    assert len(steps) == 2
    assert {cfg.metadata.capture_date for cfg in steps} == {"", "2026-10-25 12:01:00"}


def test_assign_is_one_undoable_batch(qapp) -> None:
    mgr = _session()
    batches: list[bool] = []
    mgr.batch_recorded.connect(lambda: batches.append(True))
    mgr.assign_capture_times({"h1": _fields("2026-10-25 12:01:00"), "h2": _fields("2026-10-25 12:02:00")})
    assert batches == [True]


def test_assign_without_changes_writes_nothing(qapp) -> None:
    mgr = _session()
    assert mgr.assign_capture_times({}) == 0
    mgr.repo.save_file_settings.assert_not_called()


def test_logic_results_flow_into_the_frames(qapp) -> None:
    mgr = _session()
    photo = Anchor(local=datetime(2026, 10, 25, 11, 9, 21), offset=timedelta(hours=3), latitude=27.14, longitude=33.88, ref="IMG_2514.HEIC")
    segment = Segment(("h0", "h1", "h2"), day=date(2026, 10, 25), zone="Africa/Cairo", anchors={"h1": photo})
    results, issues = assign_times(segment)
    assert issues == []
    assert mgr.assign_capture_times({r.frame: r.metadata_fields() for r in results}) == 3
    saved = _saved(mgr)
    assert [saved[h].metadata.capture_date for h in ("h0", "h1", "h2")] == [
        "2026-10-25 11:07:21",
        "2026-10-25 11:09:21",
        "2026-10-25 11:11:21",
    ]
    assert saved["h1"].metadata.capture_anchor_ref == "IMG_2514.HEIC"
    assert saved["h2"].metadata.capture_time_source == "after"
    assert (saved["h0"].metadata.gps_latitude, saved["h0"].metadata.gps_longitude) == (27.14, 33.88)
