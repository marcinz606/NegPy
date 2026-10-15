"""Remove from Roll: a frame leaves a virtual roll's member list, a folder roll's extras, or
its own folder through excluded_paths, which every walk of that folder skips."""

import os
from unittest.mock import MagicMock

from negpy.desktop.workers.render import AssetDiscoveryTask, AssetDiscoveryWorker
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.composites import source_files


def _repo() -> MagicMock:
    repo = MagicMock(spec=StorageRepository)
    store: dict = {}
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    return repo


def test_a_virtual_roll_drops_the_member():
    repo = _repo()
    roll_id = rolls.create_virtual_roll(repo, "Picks", ["/a/1.tif", "/b/2.tif"])
    rolls.remove_members(repo, roll_id, ["/a/1.tif"], ["h1"])
    assert rolls.roll_for_id(repo, roll_id)["member_paths"] == ["/b/2.tif"]
    assert rolls.rolls_containing_path(repo, "/a/1.tif") == []


def test_a_folder_roll_drops_an_extra_and_excludes_its_own_file():
    repo = _repo()
    roll_id = rolls.recognize_folder(repo, "/scans/roll_a")
    rolls.add_extra_members(repo, roll_id, ["/elsewhere/c.nef"])
    rolls.remove_members(repo, roll_id, ["/elsewhere/c.nef", "/scans/roll_a/1.tif"], [])
    entry = rolls.roll_for_id(repo, roll_id)
    assert entry["extra_paths"] == []
    assert entry["excluded_paths"] == ["/scans/roll_a/1.tif"]
    assert os.path.normcase(os.path.normpath("/scans/roll_a/1.tif")) in rolls.excluded_paths(repo, ["/scans/roll_a/"])
    assert rolls.excluded_paths(repo, ["/scans/roll_b"]) == frozenset()
    assert rolls.rolls_containing_path(repo, "/scans/roll_a/1.tif") == []
    assert rolls.rolls_containing_path(repo, "/scans/roll_a/2.tif") == [roll_id]


def test_adding_an_excluded_file_back_lifts_the_exclusion():
    repo = _repo()
    roll_id = rolls.recognize_folder(repo, "/scans/roll_a")
    rolls.remove_members(repo, roll_id, ["/scans/roll_a/1.tif"], [])
    rolls.add_extra_members(repo, roll_id, ["/scans/roll_a/1.tif"])
    entry = rolls.roll_for_id(repo, roll_id)
    assert entry["excluded_paths"] == []
    assert entry["extra_paths"] == []


def test_adding_a_removed_stitch_back_lifts_its_parts_without_adding_them():
    repo = _repo()
    roll_id = rolls.recognize_folder(repo, "/scans/roll_a")
    stitch = {"path": "/scans/roll_a/1.tif", "stitch_paths": ["/scans/roll_a/1.tif", "/scans/roll_a/2.tif", "/elsewhere/3.tif"]}
    rolls.remove_members(repo, roll_id, source_files(stitch), [])
    assert rolls.roll_for_id(repo, roll_id)["excluded_paths"] == ["/scans/roll_a/1.tif", "/scans/roll_a/2.tif"]

    rolls.add_extra_members(repo, roll_id, [stitch["path"]], parts=source_files(stitch))

    entry = rolls.roll_for_id(repo, roll_id)
    assert entry["excluded_paths"] == []
    assert entry["extra_paths"] == []


def test_removal_drops_the_frames_locks_scene_and_fork():
    repo = _repo()
    roll_id = rolls.create_virtual_roll(repo, "Picks", ["/a/1.tif", "/a/2.tif"])
    rolls.set_frame_override(repo, roll_id, "h1", "film", True)
    rolls.set_frame_override(repo, roll_id, "h2", "film", True)
    rolls.create_scene(repo, roll_id, "Beach", ["h1"])
    rolls.fork_edit(repo, roll_id, "h1", "/a/1.tif", MagicMock())

    forked = rolls.remove_members(repo, roll_id, ["/a/1.tif"], ["h1"])

    assert forked == ["h1"]
    assert rolls.frame_override_cards(repo, roll_id, "h1") == set()
    assert rolls.frame_override_cards(repo, roll_id, "h2") == {"film"}
    assert rolls.roll_scenes(repo, roll_id) == []
    assert not rolls.is_forked(repo, roll_id, "h1")


def test_the_folder_walk_skips_excluded_files_but_not_named_ones(tmp_path):
    for name in ("1.tif", "2.tif"):
        (tmp_path / name).write_bytes(b"x" * 64)
    skip = frozenset({os.path.normcase(os.path.normpath(str(tmp_path / "1.tif")))})
    worker = AssetDiscoveryWorker()
    seen: list = []
    worker.finished.connect(seen.append)

    worker.process(AssetDiscoveryTask(paths=[str(tmp_path)], supported_extensions=(".tif",), skip_paths=skip))
    assert [a["name"] for a in seen.pop()] == ["2.tif"]

    worker.process(AssetDiscoveryTask(paths=[str(tmp_path / "1.tif")], supported_extensions=(".tif",), skip_paths=skip))
    assert [a["name"] for a in seen.pop()] == ["1.tif"]


def test_source_files_names_every_file_of_a_frame():
    stitch = {"path": "/r/a.tif", "stitch_paths": ["/r/a.tif", "/r/b.tif"], "stitch_triplets": [("/r/ag.tif", "/r/ab.tif")]}
    assert source_files(stitch) == ["/r/a.tif", "/r/b.tif", "/r/ag.tif", "/r/ab.tif"]
    assert source_files({"path": "/r/r.tif", "green_path": "/r/g.tif", "blue_path": "/r/b.tif"}) == ["/r/r.tif", "/r/g.tif", "/r/b.tif"]


def _controller():
    from negpy.desktop.controller import AppController

    ctrl = AppController.__new__(AppController)
    ctrl.session = MagicMock()
    store: dict = {}
    repo = ctrl.session.repo
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    ctrl.set_status = MagicMock()
    ctrl.state = MagicMock()
    return ctrl, store


def test_the_controller_removes_both_halves_and_deletes_the_fork_after_the_unload():
    from negpy.desktop.controller import AppController

    ctrl, store = _controller()
    repo = ctrl.session.repo
    roll_id = rolls.create_virtual_roll(repo, "Picks", ["/a/1.tif", "/a/2.tif"])
    store[rolls.ROLLS_KEY][roll_id]["forked_hashes"] = ["h1#1"]
    ctrl.state.active_roll_id = roll_id
    ctrl.state.uploaded_files = [
        {"path": "/a/1.tif", "hash": "h1#1#roll:" + roll_id, "half": 1},
        {"path": "/a/1.tif", "hash": "h1#2", "half": 2},
        {"path": "/a/2.tif", "hash": "h2"},
    ]
    order: list = []
    ctrl.session.remove_files.side_effect = lambda drop: order.append(("unload", drop))
    repo.delete_file_settings.side_effect = lambda h: order.append(("delete", h))

    assert AppController.roll_removal(ctrl, [0]) == [0, 1]
    AppController.request_remove_from_roll(ctrl, [0])

    assert order == [("unload", [0, 1]), ("delete", rolls.roll_edit_hash("h1#1", roll_id))]
    assert rolls.roll_for_id(repo, roll_id)["member_paths"] == ["/a/2.tif"]


def test_nothing_goes_without_an_open_roll_or_a_frame():
    from negpy.desktop.controller import AppController

    ctrl, _ = _controller()
    ctrl.state.uploaded_files = [{"path": "/a/1.tif", "hash": "h1"}]
    ctrl.state.active_roll_id = None
    assert AppController.roll_removal(ctrl, [0]) == []
    ctrl.state.active_roll_id = "r1"
    assert AppController.roll_removal(ctrl, [-1]) == []


def test_hot_folder_does_not_offer_a_removed_file_again(tmp_path):
    from negpy.desktop.view.sidebar.files import FileBrowser

    for name in ("1.tif", "2.tif", "3.tif"):
        (tmp_path / name).write_bytes(b"x")
    repo = _repo()
    roll_id = rolls.recognize_folder(repo, str(tmp_path))
    rolls.remove_members(repo, roll_id, [str(tmp_path / "1.tif")], [])
    view = FileBrowser.__new__(FileBrowser)
    view.session = MagicMock()
    view.session.repo = repo
    view.session.state.uploaded_files = [{"path": str(tmp_path / "2.tif")}]
    view.session.state.duplicate_paths = set()
    view.controller = MagicMock()

    FileBrowser._scan_folder(view)

    view.controller.request_asset_discovery.assert_called_once_with([str(tmp_path / "3.tif")], hot_folder=True)
