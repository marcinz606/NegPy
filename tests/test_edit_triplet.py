"""Edit Trichrome, Split Trichrome and Merge Exposures (Trichrome): a triplet made by hand is one Film Strip frame."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager, trichrome_merge_indices
from negpy.features.rgbscan.logic import FrameProbe, place_triplet
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def scans(tmp_path):
    def make(*names):
        paths = []
        for n in names:
            p = tmp_path / f"{n}.raf"
            p.write_bytes(n.encode())
            paths.append(str(p))
        return paths

    return make


def _plain(path):
    return {"name": path.rsplit("/", 1)[-1], "path": path, "hash": f"h:{path}", "legacy_hash": "", "mtime": 1.0}


def _triplet(red, green, blue):
    return {**_plain(red), "green_path": green, "blue_path": blue, "green_hash": f"h:{green}", "blue_hash": f"h:{blue}"}


def _controller(files, active=0, selected=None):
    c = MagicMock()
    c.state = AppState()
    c.session.state = c.state
    c.state.uploaded_files = files
    c.state.selected_file_idx = active
    c.state.selected_indices = selected if selected is not None else [active]
    c._batch_busy.return_value = False
    c._discovery_busy.return_value = False
    c._begin_batch.return_value = 1
    return c


def _set(c, *args, **kwargs):
    with (
        patch("negpy.desktop.controller.forget_triplets") as forget,
        patch("negpy.kernel.image.logic.calculate_file_hash", side_effect=lambda p: f"h:{p}"),
    ):
        AppController.set_triplet(c, *args, **kwargs)
    return forget


def test_the_three_exposures_become_one_frame(scans):
    r, g, b, x = scans("r", "g", "b", "x")
    c = _controller([_plain(r), _plain(g), _plain(b), _plain(x)])
    _set(c, r, g, b, True, index=0)

    indices, asset = c.session.apply_composite.call_args.args
    assert indices == [0, 1, 2]
    assert (asset["path"], asset["green_path"], asset["blue_path"]) == (r, g, b)
    assert asset["hash"] == f"h:{r}" and asset["triplet_by_hand"] is True
    c._apply_roll_forks.assert_called_once_with([asset])
    c.request_asset_discovery.assert_not_called()


def test_opened_on_the_green_frame_leaves_no_duplicate_red(scans):
    r, g, b = scans("r", "g", "b")
    c = _controller([_plain(r), _plain(g), _plain(b)], active=1)
    _set(c, r, g, b, True, index=1)
    assert c.session.apply_composite.call_args.args[0] == [0, 1, 2]


def test_an_exposure_taken_out_returns_as_its_own_frame(scans):
    r, g1, b, g2 = scans("r", "g1", "b", "g2")
    c = _controller([_triplet(r, g1, b), _plain(g2)])
    forget = _set(c, r, g2, b, True, index=0)

    assert c.session.apply_composite.call_args.args[0] == [0, 1]
    c.request_asset_discovery.assert_called_once_with([g1], rgb_scan=False)
    forget.assert_called_once_with(c.session.repo, [])


def test_a_replaced_red_forgets_the_old_record(scans):
    r1, g1, b1, r2, g2, b2 = scans("r1", "g1", "b1", "r2", "g2", "b2")
    c = _controller([_triplet(r1, g1, b1), _plain(r2), _plain(g2), _plain(b2)])
    forget = _set(c, r2, g2, b2, True, index=0)

    forget.assert_called_once_with(c.session.repo, [r1])
    assert sorted(c.request_asset_discovery.call_args.args[0]) == sorted([r1, g1, b1])


def test_the_same_file_twice_is_refused(scans):
    r, g = scans("r", "g")
    c = _controller([_plain(r), _plain(g)])
    _set(c, r, g, g, True, index=0)
    c.session.apply_composite.assert_not_called()
    assert c.set_status.call_args.kwargs["kind"] == "warning"


def test_a_missing_file_is_refused_before_anything_changes(scans):
    r, g = scans("r", "g")
    c = _controller([_plain(r), _plain(g)])
    forget = _set(c, r, g, r + ".missing", True, index=0)
    c.session.save_active_edit.assert_not_called()
    forget.assert_not_called()
    c.session.apply_composite.assert_not_called()


def test_nothing_changes_while_files_load(scans):
    r, g, b = scans("r", "g", "b")
    c = _controller([_plain(r), _plain(g), _plain(b)], selected=[0, 1, 2])
    c._discovery_busy.return_value = True
    _set(c, r, g, b, True, index=0)
    AppController.request_trichrome_merge(c)
    c.session.apply_composite.assert_not_called()
    c.trichrome_merge_requested.emit.assert_not_called()


def test_the_discovery_guard_says_why():
    c = _controller([])
    c._discovery_running = True
    c._pending_asset_discoveries = []
    assert AppController._discovery_busy(c) is True
    assert c.set_status.call_args.kwargs["kind"] == "warning"


def test_a_failed_placement_still_ends_the_batch():
    from negpy.desktop.workers.triplet_probe import TripletProbeWorker

    worker = TripletProbeWorker()
    seen: list = []
    worker.placed.connect(seen.append)
    with (
        patch("negpy.desktop.workers.triplet_probe.probe_frame", return_value=None),
        patch("negpy.desktop.workers.triplet_probe.place_triplet", side_effect=ValueError("odd signature")),
    ):
        worker.run(("/a", "/b", "/c"))
    assert seen == [{"paths": ("/a", "/b", "/c"), "triplet": None, "reason": "channels"}]


def test_a_stitch_is_never_rebuilt_as_a_triplet(scans):
    a, b, g, bl = scans("a", "b", "g", "bl")
    stitch = {**_plain(a), "hash": "s#stitch", "stitch_paths": (b,)}
    c = _controller([stitch, _plain(g), _plain(bl)])
    _set(c, a, g, bl, True, index=0)
    c.session.apply_composite.assert_not_called()


def test_a_file_inside_a_stitch_is_refused(scans):
    a, b, r, g = scans("a", "b", "r", "g")
    stitch = {**_plain(a), "hash": "s#stitch", "stitch_paths": (b,)}
    c = _controller([stitch, _plain(r), _plain(g)], active=1)
    _set(c, r, g, b, True, index=1)
    c.session.apply_composite.assert_not_called()


def test_split_returns_the_three_exposures_ungrouped(scans):
    r, g, b = scans("r", "g", "b")
    c = _controller([_triplet(r, g, b)])
    with patch("negpy.desktop.controller.forget_triplets") as forget:
        AppController.request_split_triplet(c)

    forget.assert_called_once_with(c.session.repo, [r])
    c._drop_dissolved_composite.assert_called_once_with(0)
    c.request_asset_discovery.assert_called_once_with([r, g, b], rgb_scan=False)
    assert c._pending_scanned_file == r


def test_split_ignores_a_stitch(scans):
    a, b = scans("a", "b")
    c = _controller([{**_plain(a), "stitch_paths": (b,), "green_path": "/g", "blue_path": "/b"}])
    AppController.request_split_triplet(c)
    c.request_asset_discovery.assert_not_called()


def test_merge_needs_three_single_frames_of_three_files():
    files = [_plain(f"/{n}") for n in "abcd"]
    assert trichrome_merge_indices(files, [0, 1, 2]) == [0, 1, 2]
    assert trichrome_merge_indices(files, [0, 1]) == []
    assert trichrome_merge_indices(files, [0, 1, 2, 3]) == []
    assert trichrome_merge_indices([*files[:2], _triplet("/c", "/g", "/b")], [0, 1, 2]) == []
    halves = [{**_plain("/a"), "half": 1}, {**_plain("/a"), "half": 2}, _plain("/b")]
    assert trichrome_merge_indices(halves, [0, 1, 2]) == []


def test_merge_reads_the_selection_in_the_background():
    c = _controller([_plain(f"/{n}") for n in "abc"], selected=[0, 1, 2])
    AppController.request_trichrome_merge(c)
    c.trichrome_merge_requested.emit.assert_called_once_with(("/a", "/b", "/c"))


@pytest.mark.parametrize("reason, words", [("channels", "red, green and blue apart"), ("picture", "same picture")])
def test_a_refused_merge_says_why(reason, words):
    c = _controller([_plain(f"/{n}") for n in "abc"])
    AppController._on_triplet_placed(c, {"paths": ("/a", "/b", "/c"), "triplet": None, "reason": reason})
    c.set_triplet.assert_not_called()
    assert words in c.set_status.call_args.args[0]


def test_a_placed_merge_joins_by_path():
    from negpy.features.rgbscan.logic import Triplet

    c = _controller([_plain(f"/{n}") for n in "abc"])
    AppController._on_triplet_placed(c, {"paths": ("/a", "/b", "/c"), "triplet": Triplet("/b", "/c", "/a", True), "reason": ""})
    c.set_triplet.assert_called_once_with("/b", "/c", "/a")


def test_a_merge_whose_frames_changed_does_nothing():
    from negpy.features.rgbscan.logic import Triplet

    c = _controller([_plain("/a"), _plain("/b")])
    AppController._on_triplet_placed(c, {"paths": ("/a", "/b", "/c"), "triplet": Triplet("/a", "/b", "/c", True), "reason": ""})
    c.set_triplet.assert_not_called()


def _probe(channel, seed=0):
    sig = np.random.default_rng(seed).normal(size=(8, 8)).astype(np.float32)
    return FrameProbe(means=tuple(3.0 if i == channel else 1.0 for i in range(3)), signature=sig)


def test_place_finds_each_color_in_any_order():
    triplet, reason = place_triplet({"/x": _probe(2), "/y": _probe(0), "/z": _probe(1)})
    assert reason == "" and (triplet.red, triplet.green, triplet.blue) == ("/y", "/z", "/x")


def test_place_refuses_two_of_one_color_or_an_unread_file():
    assert place_triplet({"/x": _probe(1), "/y": _probe(1), "/z": _probe(2)}) == (None, "channels")
    assert place_triplet({"/x": _probe(0), "/y": None, "/z": _probe(2)}) == (None, "channels")


def test_place_refuses_three_different_pictures():
    assert place_triplet({"/x": _probe(0, 1), "/y": _probe(1, 2), "/z": _probe(2, 3)}) == (None, "picture")


def _session(files, marks=None):
    repo = MagicMock(spec=StorageRepository)
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.load_file_settings_many.return_value = {}
    repo.get_global_setting.side_effect = lambda key, default=None: default
    repo.load_file_marks.return_value = marks or {}
    repo.get_max_history_index.return_value = 0
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = files
    return session


def test_a_forked_triplet_keeps_its_mark():
    session = _session([_plain("/r"), _plain("/g"), _plain("/b")], marks={"h:/r": "keeper"})
    session.apply_composite([0, 1, 2], {**_triplet("/r", "/g", "/b"), "hash": "h:/r#roll:x"})
    assert session.state.uploaded_files[0]["keeper"] is True


def test_the_session_manifest_keeps_the_by_hand_flag():
    session = _session([{**_triplet("/r", "/g", "/b"), "triplet_by_hand": True}])
    session._persist_session()
    saved = {c.args[0]: c.args[1] for c in session.repo.save_global_setting.call_args_list}
    assert saved["session_triplets"]["/r"] == ["/g", "/b", True, ["h:/r", "h:/g", "h:/b"], True]


def test_a_stitch_of_triplets_keeps_a_short_manifest_record():
    stitch = {
        **_triplet("/r", "/g", "/b"),
        "hash": "s#stitch",
        "stitch_paths": ("/r2",),
        "stitch_transforms": ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
        "stitch_canvas": (120, 80),
        "stitch_sizes": ((80, 80), (80, 80)),
    }
    session = _session([stitch])
    session._persist_session()
    saved = {c.args[0]: c.args[1] for c in session.repo.save_global_setting.call_args_list}
    assert saved["session_triplets"]["/r"] == ["/g", "/b", True]
