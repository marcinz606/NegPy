"""Scan as Roll: the folder a scanner or camera writes to becomes a Library roll, open
before discovery runs, so Half Frame and the roll-scoped cards apply to the new frames."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.services.assets.rolls import folder_roll_id_for_path, recognize_folder, roll_for_id, saved_rolls


def _controller(as_roll=False, capture_req=None):
    store: dict = {}
    c = MagicMock()
    c.state = AppState()
    c.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    c.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    c._pending_scanned_file = None
    c._pending_capture_imports = {}
    c._scan_as_roll = as_roll
    c._batch_frame_selected = False
    c._last_capture_req = capture_req
    c._discover_scanned = MethodType(AppController._discover_scanned, c)
    return c


def test_first_scan_opens_its_folder_as_a_roll_before_discovery():
    c = _controller(as_roll=True)

    AppController._on_scan_frame_done(c, 1, "/out/Roll001/a.tif")

    roll_id = folder_roll_id_for_path(c.session.repo, "/out/Roll001")
    assert roll_id is not None and c.state.active_roll_id == roll_id
    c.half_frame_mode_changed.emit.assert_called_once()
    c.library_cleared.emit.assert_called_once()
    c.request_asset_discovery.assert_called_once_with(
        ["/out/Roll001"], auto_open=True, replace_existing=True, reselect_path="/out/Roll001/a.tif", restore_triplets=None
    )
    assert c._pending_scanned_file is None


def test_later_batch_frames_load_without_taking_the_selection():
    c = _controller(as_roll=True)
    AppController._on_scan_frame_done(c, 1, "/out/Roll001/a.tif")
    c.request_asset_discovery.reset_mock()

    AppController._on_scan_frame_done(c, 2, "/out/Roll001/b.tif")

    c.request_asset_discovery.assert_called_once_with(["/out/Roll001/b.tif"], restore_triplets=None)
    assert c._pending_scanned_file is None
    c.scan_frame_done.emit.assert_called_with(2, "/out/Roll001/b.tif")


def test_batch_end_loads_nothing_more():
    c = _controller(as_roll=True)
    AppController._on_scan_batch_finished(c, ["/out/Roll001/a.tif"])
    c.request_asset_discovery.assert_not_called()
    c.scan_batch_finished.emit.assert_called_once_with(["/out/Roll001/a.tif"])


def test_next_scan_into_the_open_roll_appends():
    c = _controller(as_roll=True)
    roll_id = recognize_folder(c.session.repo, "/out/Roll001")
    c.state.active_roll_id = roll_id

    AppController._on_scan_finished(c, "/out/Roll001/c.tif")

    assert c.state.active_roll_id == roll_id
    c.request_asset_discovery.assert_called_once_with(["/out/Roll001/c.tif"], restore_triplets=None)
    assert c._pending_scanned_file == "/out/Roll001/c.tif"


def test_a_new_roll_name_opens_a_separate_roll():
    c = _controller(as_roll=True)
    first = recognize_folder(c.session.repo, "/out/Roll001")
    c.state.active_roll_id = first

    AppController._on_scan_finished(c, "/out/Roll002/a.tif")

    assert c.state.active_roll_id not in (None, first)
    assert roll_for_id(c.session.repo, first)["extra_paths"] == []
    assert c.request_asset_discovery.call_args.kwargs["replace_existing"] is True


def test_scan_without_as_roll_creates_no_roll():
    c = _controller(as_roll=False)

    AppController._on_scan_finished(c, "/out/a.tif")

    assert saved_rolls(c.session.repo) == {}
    assert c.state.active_roll_id is None
    c.request_asset_discovery.assert_called_once_with(["/out/a.tif"], restore_triplets=None)


def test_capture_triplet_reaches_the_roll_open():
    req = SimpleNamespace(white_mode=False, rgb_mode=True, white_process_mode="auto", roll_name="R1", frame_number=1, as_roll=True)
    c = _controller(capture_req=req)
    paths = ["/hot/R1/r.ARW", "/hot/R1/g.ARW", "/hot/R1/b.ARW"]

    AppController._on_capture_finished(c, paths)

    c.request_asset_discovery.assert_called_once_with(
        ["/hot/R1"],
        auto_open=True,
        replace_existing=True,
        reselect_path=paths[0],
        restore_triplets={paths[0]: paths[1:]},
    )


def test_start_scan_remembers_as_roll():
    c = _controller()
    c._batch_frame_selected = True
    AppController.start_batch(c, SimpleNamespace(as_roll=True))
    assert c._scan_as_roll is True and c._batch_frame_selected is False
    AppController.start_scan(c, SimpleNamespace(as_roll=False))
    assert c._scan_as_roll is False
