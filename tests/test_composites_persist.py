"""A stitch or an HDR merge must outlive the file list it was made in."""

import unittest
from unittest.mock import MagicMock, patch

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.workers.render import AssetDiscoveryWorker
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.composites import (
    COMPOSITES_KEY,
    forget_composite,
    remember_composites,
    restore_maps,
)


def _repo() -> MagicMock:
    """A repository whose global settings live in a dict, so a write is readable back."""
    repo = MagicMock(spec=StorageRepository)
    store: dict = {}
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    repo.settings = store
    return repo


def _stitch_asset() -> dict:
    return {
        "name": "a+b (Stitch)",
        "path": "/roll_a/a.nef",
        "hash": "digest#stitch",
        "stitch_paths": ("/roll_a/b.nef",),
        "stitch_transforms": ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
        "stitch_canvas": (120, 80),
        "stitch_sizes": ((80, 80), (80, 80)),
    }


def _hdr_asset() -> dict:
    return {
        "name": "e1 +2 (HDR)",
        "path": "/roll_b/e1.nef",
        "hash": "digest#hdr",
        "hdr_paths": ("/roll_b/e2.nef", "/roll_b/e3.nef"),
        "hdr_ratios": (1.0, 2.0, 4.0),
    }


def test_loading_another_folder_keeps_the_composite():
    """The bug: the registration was derived from the open files, so opening a folder
    that holds none of them erased it."""
    repo = _repo()
    remember_composites(repo, [_stitch_asset()])
    remember_composites(repo, [{"name": "x.nef", "path": "/roll_b/x.nef", "hash": "hx"}])

    stitches, _ = restore_maps(repo)
    assert "/roll_a/a.nef" in stitches
    assert stitches["/roll_a/a.nef"]["paths"] == ["/roll_a/b.nef"]


def test_stitch_and_hdr_are_stored_apart():
    repo = _repo()
    remember_composites(repo, [_stitch_asset(), _hdr_asset()])
    stitches, merges = restore_maps(repo)
    assert list(stitches) == ["/roll_a/a.nef"]
    assert list(merges) == ["/roll_b/e1.nef"]
    assert merges["/roll_b/e1.nef"]["ratios"] == [1.0, 2.0, 4.0]


def test_dissolving_forgets_it():
    repo = _repo()
    remember_composites(repo, [_stitch_asset(), _hdr_asset()])
    forget_composite(repo, "/roll_a/a.nef")
    stitches, merges = restore_maps(repo)
    assert stitches == {}
    assert list(merges) == ["/roll_b/e1.nef"]


def test_legacy_session_entries_are_promoted():
    """Composites made before the store existed must survive the upgrade."""
    repo = _repo()
    repo.settings["session_stitches"] = {
        "/roll_a/a.nef": {"paths": ["/roll_a/b.nef"], "transforms": [], "canvas": [1, 1], "sizes": [], "hash": "d#stitch"}
    }
    repo.settings["session_hdr_merges"] = {"/roll_b/e1.nef": {"paths": ["/roll_b/e2.nef"], "ratios": [1.0, 2.0], "hash": "d#hdr"}}

    stitches, merges = restore_maps(repo)
    assert list(stitches) == ["/roll_a/a.nef"]
    assert list(merges) == ["/roll_b/e1.nef"]
    # Promotion writes through, so the legacy keys are read once and never again.
    assert COMPOSITES_KEY in repo.settings
    repo.settings.pop("session_stitches")
    repo.settings.pop("session_hdr_merges")
    assert list(restore_maps(repo)[0]) == ["/roll_a/a.nef"]


def test_discovery_drops_the_parts_of_a_restored_stitch(tmp_path):
    """A folder walk finds the parts as well as the primary. Re-attaching without
    dropping them puts the composite on the sheet beside the frames it is made of."""
    primary = tmp_path / "a.nef"
    part = tmp_path / "b.nef"
    other = tmp_path / "c.nef"
    for f in (primary, part, other):
        f.write_bytes(b"x")
    assets = [
        {"name": "a.nef", "path": str(primary), "hash": "ha"},
        {"name": "b.nef", "path": str(part), "hash": "hb"},
        {"name": "c.nef", "path": str(other), "hash": "hc"},
    ]
    stitches = {
        str(primary): {
            "paths": [str(part)],
            "transforms": [[1, 0, 0, 0, 1, 0], [1, 0, 40, 0, 1, 0]],
            "canvas": [120, 80],
            "sizes": [[80, 80], [80, 80]],
            "hash": "digest#stitch",
        }
    }
    out = AssetDiscoveryWorker()._attach_restored_stitches(assets, stitches)
    assert [a["path"] for a in out] == [str(primary), str(other)]
    assert out[0]["hash"] == "digest#stitch"


def test_discovery_keeps_the_parts_when_the_stitch_cannot_restore(tmp_path):
    """A composite with a part missing from disk restores as a plain asset, so the
    parts that are there must stay on the sheet."""
    primary = tmp_path / "a.nef"
    part = tmp_path / "b.nef"
    for f in (primary, part):
        f.write_bytes(b"x")
    assets = [
        {"name": "a.nef", "path": str(primary), "hash": "ha"},
        {"name": "b.nef", "path": str(part), "hash": "hb"},
    ]
    stitches = {
        str(primary): {
            "paths": [str(part), str(tmp_path / "gone.nef")],
            "transforms": [],
            "canvas": [1, 1],
            "sizes": [],
            "hash": "digest#stitch",
        }
    }
    out = AssetDiscoveryWorker()._attach_restored_stitches(assets, stitches)
    assert [a["path"] for a in out] == [str(primary), str(part)]
    assert "stitch_paths" not in out[0]


def test_discovery_drops_the_frames_of_a_restored_merge(tmp_path):
    e1, e2 = tmp_path / "e1.nef", tmp_path / "e2.nef"
    for f in (e1, e2):
        f.write_bytes(b"x")
    assets = [
        {"name": "e1.nef", "path": str(e1), "hash": "h1"},
        {"name": "e2.nef", "path": str(e2), "hash": "h2"},
    ]
    merges = {str(e1): {"paths": [str(e2)], "ratios": [1.0, 2.0], "hash": "digest#hdr"}}
    out = AssetDiscoveryWorker()._attach_restored_hdr(assets, merges)
    assert [a["path"] for a in out] == [str(e1)]
    assert out[0]["hdr_paths"] == (str(e2),)


def test_triplet_exposures_of_a_stitch_are_dropped_too(tmp_path):
    """Each part of an RGB-scan stitch is three files; all of them belong to the composite."""
    names = ["a_r.nef", "a_g.nef", "a_b.nef", "b_r.nef", "b_g.nef", "b_b.nef"]
    paths = []
    for n in names:
        f = tmp_path / n
        f.write_bytes(b"x")
        paths.append(str(f))
    assets = [{"name": n, "path": p, "hash": n} for n, p in zip(names, paths)]
    stitches = {
        paths[0]: {
            "paths": [paths[3]],
            "transforms": [[1, 0, 0, 0, 1, 0], [1, 0, 40, 0, 1, 0]],
            "canvas": [120, 80],
            "sizes": [[80, 80], [80, 80]],
            "triplets": [[paths[1], paths[2]], [paths[4], paths[5]]],
            "hash": "digest#stitch",
        }
    }
    out = AssetDiscoveryWorker()._attach_restored_stitches(assets, stitches)
    assert [a["path"] for a in out] == [paths[0]]


class TestDiscoveryReadsTheStore(unittest.TestCase):
    """Every discovery re-attaches composites, not only the one that restores a session."""

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
        self.tasks = []
        self.controller.asset_discovery_requested.connect(self.tasks.append)

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

    def test_opening_a_folder_reattaches_a_composite_from_another_folder(self):
        remember_composites(self.session.repo, [_stitch_asset(), _hdr_asset()])
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            self.controller.open_library_folder("/roll_a")

        self.assertEqual(len(self.tasks), 1)
        task = self.tasks[0]
        self.assertIn("/roll_a/a.nef", task.restore_stitches)
        self.assertIn("/roll_b/e1.nef", task.restore_hdr)

    def test_unstitching_forgets_it(self):
        asset = _stitch_asset()
        remember_composites(self.session.repo, [asset])
        self.session.state.uploaded_files = [asset]
        self.session.state.selected_file_idx = 0
        with patch.object(self.controller, "request_asset_discovery"):
            self.controller.request_unstitch()

        self.assertEqual(restore_maps(self.session.repo)[0], {})

    def test_unmerging_forgets_it(self):
        asset = _hdr_asset()
        remember_composites(self.session.repo, [asset])
        self.session.state.uploaded_files = [asset]
        self.session.state.selected_file_idx = 0
        with patch.object(self.controller, "request_asset_discovery"):
            self.controller.request_unmerge_hdr()

        self.assertEqual(restore_maps(self.session.repo)[1], {})


def test_round_trips_through_the_real_database(tmp_path):
    """The store is JSON in SQLite: tuples must arrive back as the lists discovery reads."""

    def open_db():
        repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
        repo.initialize()
        return repo

    remember_composites(open_db(), [_stitch_asset(), _hdr_asset()])

    stitches, merges = restore_maps(open_db())
    assert stitches["/roll_a/a.nef"]["paths"] == ["/roll_a/b.nef"]
    assert stitches["/roll_a/a.nef"]["canvas"] == [120, 80]
    assert merges["/roll_b/e1.nef"]["ratios"] == [1.0, 2.0, 4.0]


def test_the_reported_scenario(tmp_path):
    """The bug as reported: stitch in one roll, work in another roll, come back.

    Runs the real discovery worker over real folders and the real database, so the
    store, the re-attach and the part-dropping are exercised as one.
    """
    from negpy.desktop.workers.render import AssetDiscoveryTask

    roll_a, roll_b = tmp_path / "roll_a", tmp_path / "roll_b"
    for folder, names in ((roll_a, ("a.tif", "b.tif")), (roll_b, ("c.tif",))):
        folder.mkdir()
        for n in names:
            (folder / n).write_bytes(n.encode() * 64)

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    worker = AssetDiscoveryWorker()
    seen: list = []
    worker.finished.connect(seen.append)

    def open_folder(folder) -> list:
        stitches, merges = restore_maps(repo)
        worker.process(
            AssetDiscoveryTask(
                paths=[str(folder)],
                supported_extensions=(".tif",),
                restore_stitches=stitches,
                restore_hdr=merges,
            )
        )
        return sorted(seen.pop(), key=lambda a: a["name"])

    parts = open_folder(roll_a)
    assert [a["name"] for a in parts] == ["a.tif", "b.tif"]

    stitched = {
        **parts[0],
        "name": "a+b (Stitch)",
        "hash": "digest#stitch",
        "stitch_paths": (parts[1]["path"],),
        "stitch_transforms": ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
        "stitch_canvas": (120, 80),
        "stitch_sizes": ((80, 80), (80, 80)),
    }
    remember_composites(repo, [stitched])

    assert [a["name"] for a in open_folder(roll_b)] == ["c.tif"]

    back = open_folder(roll_a)
    assert [a["name"] for a in back] == ["a+b (Stitch)"]
    assert back[0]["hash"] == "digest#stitch"
    assert back[0]["stitch_paths"] == (parts[1]["path"],)
