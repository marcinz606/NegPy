"""A trichrome grouping found once is not found again: reopening a roll skips the raw read."""

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.workers.render import AssetDiscoveryTask, AssetDiscoveryWorker
from negpy.features.rgbscan import logic
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.triplets import TRIPLETS_KEY, remember_triplets, saved_triplets
from negpy.services.rendering.preview_manager import PreviewManager

_CHANNEL = {"r": 0, "g": 1, "b": 2}


def _repo() -> MagicMock:
    repo = MagicMock(spec=StorageRepository)
    store: dict = {}
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    return repo


def _triplet(red: str, green: str, blue: str) -> dict:
    return {
        "name": "x (RGB)",
        "path": red,
        "hash": f"h{red}",
        "green_path": green,
        "blue_path": blue,
        "green_hash": f"h{green}",
        "blue_hash": f"h{blue}",
    }


def test_remember_stores_members_and_their_hashes():
    repo = _repo()
    remember_triplets(repo, [_triplet("/a/r", "/a/g", "/a/b"), {"name": "loose", "path": "/a/x", "hash": "hx"}])
    assert saved_triplets(repo) == {"/a/r": ["/a/g", "/a/b", True, ["h/a/r", "h/a/g", "h/a/b"]]}


def test_a_new_triplet_replaces_one_that_shares_an_exposure():
    repo = _repo()
    remember_triplets(repo, [_triplet("/a/r1", "/a/g1", "/a/b1")])
    remember_triplets(repo, [_triplet("/a/r2", "/a/g1", "/a/b2")])
    assert list(saved_triplets(repo)) == ["/a/r2"]


def test_a_stitch_of_triplets_is_not_stored_as_a_triplet():
    repo = _repo()
    remember_triplets(repo, [{**_triplet("/a/r", "/a/g", "/a/b"), "stitch_paths": ("/a/r2",)}])
    assert repo.get_global_setting(TRIPLETS_KEY) is None


def _discover(tmp_path, monkeypatch, names, restore):
    probed: list = []

    def probe(path):
        probed.append(path)
        name = os.path.basename(path)
        rng = np.random.default_rng(sum(map(ord, name.split("_")[0])))
        return logic.FrameProbe(
            means=tuple(3.0 if i == _CHANNEL[name[-5]] else 1.0 for i in range(3)), signature=rng.normal(size=(8, 8)).astype(np.float32)
        )

    monkeypatch.setattr(logic, "probe_frame", probe)
    monkeypatch.setattr(logic, "capture_timestamp", lambda path: "")
    worker = AssetDiscoveryWorker()
    seen: list = []
    worker.finished.connect(seen.append)
    paths = [str(tmp_path / n) for n in names]
    worker.process(AssetDiscoveryTask(paths=paths, supported_extensions=(".raw",), rgb_scan=True, restore_triplets=restore))
    return seen.pop(), probed


def test_a_remembered_grouping_reopens_without_reading_the_files(tmp_path, monkeypatch):
    names = ["f1_r.raw", "f1_g.raw", "f1_b.raw", "f2_r.raw", "f2_g.raw", "f2_b.raw"]
    for n in names:
        (tmp_path / n).write_bytes(n.encode() * 64)
    repo = _repo()

    first, probed = _discover(tmp_path, monkeypatch, names, None)
    assert len(first) == 2 and len(probed) == 6
    remember_triplets(repo, first)

    again, probed = _discover(tmp_path, monkeypatch, names, saved_triplets(repo))
    assert sorted(a["name"] for a in again) == ["f1_r (RGB)", "f2_r (RGB)"]
    assert probed == []


def test_a_member_changed_on_disk_is_grouped_again(tmp_path, monkeypatch):
    names = ["f1_r.raw", "f1_g.raw", "f1_b.raw", "f2_r.raw", "f2_g.raw", "f2_b.raw"]
    for n in names:
        (tmp_path / n).write_bytes(n.encode() * 64)
    repo = _repo()
    first, _ = _discover(tmp_path, monkeypatch, names, None)
    remember_triplets(repo, first)

    (tmp_path / "f2_g.raw").write_bytes(b"recaptured" * 64)
    again, probed = _discover(tmp_path, monkeypatch, names, saved_triplets(repo))
    assert sorted(os.path.basename(p) for p in probed) == ["f2_b.raw", "f2_g.raw", "f2_r.raw"]
    assert len(again) == 2


class TestDiscoveryReadsTheTripletStore(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock(spec=DesktopSessionManager)
        self.session.state = AppState()
        self.session.repo = _repo()
        self.session.asset_model = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as rw,
            patch("negpy.desktop.controller.PreviewManager") as pm,
        ):
            rw.return_value = MagicMock()
            pm.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.session)
        self.controller.asset_discovery_requested.disconnect(self.controller.discovery_worker.process)
        self.tasks: list = []
        self.controller.asset_discovery_requested.connect(self.tasks.append)
        remember_triplets(self.session.repo, [_triplet("/roll/r", "/roll/g", "/roll/b")])

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _open(self, rgb_scan: bool):
        self.session.repo.save_global_setting("rgbscan_mode", rgb_scan)
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            self.controller.open_library_folder("/roll")
        return self.tasks[-1]

    def test_trichrome_on_reattaches_the_stored_grouping(self):
        self.assertIn("/roll/r", self._open(True).restore_triplets)

    def test_trichrome_off_leaves_the_files_loose(self):
        self.assertIsNone(self._open(False).restore_triplets)

    def test_a_named_pair_keeps_the_stored_hashes_and_a_different_pair_wins(self):
        self.session.repo.save_global_setting("rgbscan_mode", True)
        self.controller.request_asset_discovery(["/roll"], restore_triplets={"/roll/r": ["/roll/g", "/roll/b", True]})
        self.assertEqual(len(self.tasks[-1].restore_triplets["/roll/r"]), 4)
        self.controller._discovery_running = False
        self.controller._active_batch = None
        self.controller.request_asset_discovery(["/roll"], restore_triplets={"/roll/r": ["/roll/x", "/roll/b", True]})
        self.assertEqual(self.tasks[-1].restore_triplets["/roll/r"][0], "/roll/x")


def test_a_roll_fork_stores_the_file_hash():
    from negpy.services.assets.rolls import roll_edit_hash

    repo = _repo()
    remember_triplets(repo, [{**_triplet("/a/r", "/a/g", "/a/b"), "hash": roll_edit_hash("h/a/r", "roll1")}])
    assert saved_triplets(repo)["/a/r"][3][0] == "h/a/r"


def test_a_changed_red_exposure_is_grouped_again(tmp_path, monkeypatch):
    names = ["f1_r.raw", "f1_g.raw", "f1_b.raw"]
    for n in names:
        (tmp_path / n).write_bytes(n.encode() * 64)
    repo = _repo()
    first, _ = _discover(tmp_path, monkeypatch, names, None)
    remember_triplets(repo, first)

    (tmp_path / "f1_r.raw").write_bytes(b"recaptured" * 64)
    _, probed = _discover(tmp_path, monkeypatch, names, saved_triplets(repo))
    assert len(probed) == 3


def test_a_changed_member_outside_the_discovery_is_still_caught(tmp_path):
    from negpy.kernel.image.logic import calculate_file_hash

    names = ["f1_r.raw", "f1_g.raw", "f1_b.raw"]
    for n in names:
        (tmp_path / n).write_bytes(n.encode() * 64)
    r, g, b = (str(tmp_path / n) for n in names)
    record = {r: [g, b, True, [calculate_file_hash(r), calculate_file_hash(g), calculate_file_hash(b)]]}
    red = {"name": "f1_r.raw", "path": r, "hash": calculate_file_hash(r)}
    worker = AssetDiscoveryWorker()

    assert worker._attach_restored_triplets([red], record)[0]["name"] == "f1_r (RGB)"
    (tmp_path / "f1_g.raw").write_bytes(b"recaptured" * 64)
    assert worker._attach_restored_triplets([red], record)[0]["name"] == "f1_r.raw"
