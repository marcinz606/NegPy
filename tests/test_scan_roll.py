from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.services.assets.rolls import folder_roll_id_for_path, recognize_folder, roll_defaults, roll_for_id, saved_rolls


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
    c._save_rgb_scan_mode = MethodType(AppController._save_rgb_scan_mode, c)
    c._set_roll_sensor_profile = MethodType(AppController._set_roll_sensor_profile, c)
    c._RGB_SCAN_MODE_BY_ROLL_KEY = AppController._RGB_SCAN_MODE_BY_ROLL_KEY
    c._store = store
    return c


def test_first_scan_opens_its_folder_as_a_roll_before_discovery():
    c = _controller(as_roll=True)

    AppController._on_scan_frame_done(c, 1, "/out/Roll001/a.tif")

    roll_id = folder_roll_id_for_path(c.session.repo, "/out/Roll001")
    assert roll_id is not None and c.state.active_roll_id == roll_id
    c._announce_roll_modes.assert_called_once_with(roll_id)
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


def test_capture_records_trichrome_mode_on_the_roll_it_lands_in():
    req = SimpleNamespace(white_mode=False, rgb_mode=True, white_process_mode="auto", roll_name="R1", frame_number=1, as_roll=True)
    c = _controller(capture_req=req)
    c._store["rgbscan_mode_by_roll"] = {"other": False}

    AppController._on_capture_finished(c, ["/hot/R1/r.ARW", "/hot/R1/g.ARW", "/hot/R1/b.ARW"])

    roll_id = folder_roll_id_for_path(c.session.repo, "/hot/R1")
    assert c._store["rgbscan_mode_by_roll"] == {"other": False, roll_id: True}
    assert c._store["rgbscan_mode"] is True


def test_capture_without_as_roll_leaves_an_open_roll_in_another_folder_alone():
    req = SimpleNamespace(white_mode=False, rgb_mode=True, white_process_mode="auto", roll_name="R2", frame_number=1, as_roll=False)
    c = _controller(capture_req=req)
    open_roll = recognize_folder(c.session.repo, "/film/Roll1")
    c.state.active_roll_id = open_roll
    c._store["rgbscan_mode_by_roll"] = {open_roll: False}

    AppController._on_capture_finished(c, ["/hot/R2/r.ARW", "/hot/R2/g.ARW", "/hot/R2/b.ARW"])

    assert c._store["rgbscan_mode_by_roll"] == {open_roll: False}
    assert c._store["rgbscan_mode"] is True


def test_capture_without_as_roll_records_on_its_folders_own_roll():
    req = SimpleNamespace(white_mode=True, rgb_mode=False, white_process_mode="auto", roll_name="R1", frame_number=1, as_roll=False)
    c = _controller(capture_req=req)
    roll_id = recognize_folder(c.session.repo, "/hot/R1")
    c._store["rgbscan_mode_by_roll"] = {roll_id: True}

    AppController._on_capture_finished(c, ["/hot/R1/w.ARW"])

    assert c._store["rgbscan_mode_by_roll"] == {roll_id: False}


def _rgb_req(roll_name, **kw):
    return SimpleNamespace(
        white_mode=False, rgb_mode=True, white_process_mode="auto", roll_name=roll_name, frame_number=1, as_roll=True, **kw
    )


def test_single_capture_records_trichrome_mode_off_on_its_roll():
    c = _controller(capture_req=_rgb_req("R1", single_capture=True))

    AppController._on_capture_finished(c, ["/hot/R1/R1_Frame001.ARW"])

    roll_id = folder_roll_id_for_path(c.session.repo, "/hot/R1")
    assert c._store["rgbscan_mode_by_roll"] == {roll_id: False}
    assert c._store["rgbscan_mode"] is False
    assert c.request_asset_discovery.call_args.kwargs["restore_triplets"] is None


def test_triplet_and_single_capture_rolls_each_keep_their_own_trichrome_mode():
    c = _controller(capture_req=_rgb_req("Triplet"))
    AppController._on_capture_finished(c, ["/hot/Triplet/r.ARW", "/hot/Triplet/g.ARW", "/hot/Triplet/b.ARW"])
    c._last_capture_req = _rgb_req("Single", single_capture=True)
    AppController._on_capture_finished(c, ["/hot/Single/Single_Frame001.ARW"])

    triplet = folder_roll_id_for_path(c.session.repo, "/hot/Triplet")
    single = folder_roll_id_for_path(c.session.repo, "/hot/Single")
    assert c._store["rgbscan_mode_by_roll"] == {triplet: True, single: False}
    assert AppController.rgb_scan_mode_for_roll(c, triplet) is True
    assert AppController.rgb_scan_mode_for_roll(c, single) is False


_UNMIX = [1.0, -0.1, 0.0, -0.1, 1.0, -0.3, 0.0, -0.3, 1.0]


def _profiles(monkeypatch, **matrices):
    import negpy.desktop.controller as controller_module

    monkeypatch.setattr(controller_module.SensorProfiles, "get_matrix", staticmethod(lambda name: matrices.get(name)))


def test_single_capture_gives_its_roll_the_presets_sensor_profile(monkeypatch):
    _profiles(monkeypatch, Portra=_UNMIX)
    c = _controller(capture_req=_rgb_req("R1", single_capture=True, sensor_profile="Portra"))

    AppController._on_capture_finished(c, ["/hot/R1/R1_Frame001.ARW"])

    defaults = roll_defaults(c.session.repo, folder_roll_id_for_path(c.session.repo, "/hot/R1"))
    assert defaults["sensor_profile"] == "Portra" and tuple(defaults["sensor_matrix"]) == tuple(_UNMIX)
    assert defaults["linear_raw"] is True  # the unmix is blocked without it
    pending = next(iter(c._pending_capture_imports.values()))
    assert pending.sensor_profile == "Portra" and pending.sensor_matrix == tuple(_UNMIX)


def test_a_missing_sensor_profile_leaves_the_roll_alone(monkeypatch):
    _profiles(monkeypatch)
    c = _controller(capture_req=_rgb_req("R1", single_capture=True, sensor_profile="Gone"))

    AppController._on_capture_finished(c, ["/hot/R1/R1_Frame001.ARW"])

    assert "sensor_profile" not in roll_defaults(c.session.repo, folder_roll_id_for_path(c.session.repo, "/hot/R1"))
    assert next(iter(c._pending_capture_imports.values())).sensor_matrix is None


def test_a_triplet_capture_never_takes_a_sensor_profile(monkeypatch):
    _profiles(monkeypatch, Portra=_UNMIX)
    c = _controller(capture_req=_rgb_req("R1", sensor_profile="Portra"))

    AppController._on_capture_finished(c, ["/hot/R1/r.ARW", "/hot/R1/g.ARW", "/hot/R1/b.ARW"])

    assert "sensor_profile" not in roll_defaults(c.session.repo, folder_roll_id_for_path(c.session.repo, "/hot/R1"))
