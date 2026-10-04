import os
from dataclasses import replace

import numpy as np
import pytest
import tifffile

from negpy.domain.models import WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.stitch.models import StitchConfig
from negpy.features.process.sensor import effective_sensor_matrix
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.sidecar import load_sidecar, sidecar_path_for, write_sidecar
from negpy.services.assets.frame_merge import carry_edit, carry_sidecar, merged_edit
from negpy.services.export import frame_merge
from negpy.services.export.frame_merge import (
    MergeCancelled,
    MergeVerifyError,
    can_merge,
    decode_params,
    describes_a_merge,
    is_merge_description,
    is_merged_source,
    merged_path_for,
    needs_camera_matrix,
    part_files,
    registration_complete,
    write_merged_frame,
)
from negpy.services.rendering.image_processor import ImageProcessor

_MATRIX = (1.0, -0.1, 0.0, -0.05, 1.0, -0.1, 0.0, -0.2, 1.0)


def _triplet(tmp_path) -> tuple[str, str, str]:
    paths = tuple(str(tmp_path / f"IMG_{i}.ARW") for i in (1, 2, 3))
    for p in paths:
        open(p, "wb").close()
    return paths  # type: ignore[return-value]


def _exposures(red: str, green: str, blue: str) -> dict:
    rng = np.random.default_rng(7)
    return {p: rng.integers(0, 65535, size=(24, 36, 3), dtype=np.uint16) for p in (red, green, blue)}


def _fake_sensor_decode(buffers: dict, orientation: int = 1):
    def decode(path, linear_raw, fast=False, wb_override=None, demosaic="Auto", positive_source=False, highlight_mode=0, **_kw):
        return buffers[path], {"orientation": orientation, "cam_xyz": None, "camera_wb": [1.0, 1.0, 1.0]}

    return decode


def _triplet_config(green: str, blue: str) -> WorkspaceConfig:
    return replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False))


def _repo(tmp_path) -> StorageRepository:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


@pytest.mark.parametrize("orientation", [1, 6])
def test_merged_tiff_decodes_to_the_source_the_triplet_renders_from(tmp_path, orientation):
    red, green, blue = _triplet(tmp_path)
    cfg = _triplet_config(green, blue)
    triplet = ImageProcessor()
    triplet._decode_sensor_rgb = _fake_sensor_decode(_exposures(red, green, blue), orientation)
    expected, _ir, _cs = triplet._decode_oriented_f32(red, cfg)

    params = decode_params(cfg, "rgb")
    merged, _ir, _cs = triplet._load_source_f32(red, params)
    out = merged_path_for(red, "rgb")
    write_merged_frame(merged, red, out, params)

    reread, _ir, _cs = ImageProcessor()._decode_oriented_f32(out, merged_edit(cfg, "rgb"))
    assert np.array_equal(reread, expected)


def test_decode_params_leave_flat_field_to_the_edit():
    cfg = replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path="g", blue_path="b"))
    cfg = replace(cfg, flatfield=replace(cfg.flatfield, apply=True, profile_id="abc"))
    assert decode_params(cfg, "rgb").flatfield.apply is False
    with pytest.raises(ValueError):
        decode_params(WorkspaceConfig(), "")


def test_merged_edit_drops_the_triplet_and_the_sensor_unmix():
    cfg = _triplet_config("g", "b")
    cfg = replace(cfg, process=replace(cfg.process, linear_raw=True, sensor_matrix=_MATRIX, sensor_profile="rig"))
    edit = merged_edit(cfg, "rgb")
    assert edit.rgbscan == RgbScanConfig()
    assert effective_sensor_matrix(edit.process) is None
    assert edit.process.sensor_profile == "None"


def test_write_is_atomic_and_recognized_as_a_merge(tmp_path):
    red, _g, _b = _triplet(tmp_path)
    out = merged_path_for(red, "rgb")
    f32 = np.full((8, 10, 3), 0.25, dtype=np.float32)
    write_merged_frame(f32, red, out, _triplet_config("g", "b"))

    assert out.endswith("IMG_1_RGB.tif")
    assert tifffile.imread(out).dtype == np.uint16
    assert is_merged_source(out)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]


def test_an_ordinary_tiff_is_not_a_merge(tmp_path):
    path = str(tmp_path / "scan.tif")
    tifffile.imwrite(path, np.zeros((4, 4, 3), dtype=np.uint16), description="scanner")
    assert not is_merged_source(path)
    assert not is_merged_source(str(tmp_path / "IMG_1.ARW"))


def test_a_file_that_reads_back_wrong_is_never_kept(tmp_path, monkeypatch):
    red, _g, _b = _triplet(tmp_path)
    out = merged_path_for(red, "rgb")
    monkeypatch.setattr(frame_merge.tifffile, "imread", lambda _p: np.zeros((8, 10, 3), dtype=np.uint16))

    with pytest.raises(MergeVerifyError):
        write_merged_frame(np.full((8, 10, 3), 0.5, dtype=np.float32), red, out, _triplet_config("g", "b"))
    assert not os.path.exists(out)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]


def test_an_existing_file_is_never_replaced(tmp_path):
    red, _g, _b = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    with pytest.raises(FileExistsError):
        write_merged_frame(np.zeros((4, 4, 3), dtype=np.float32), red, out, _triplet_config("g", "b"))
    assert os.path.getsize(out) == 0


def test_merged_names_skip_files_on_disk_and_names_taken_in_the_batch(tmp_path):
    red, _g, _b = _triplet(tmp_path)
    open(tmp_path / "IMG_1_RGB.tif", "wb").close()
    second = merged_path_for(red, "rgb")
    assert second.endswith("IMG_1_RGB_2.tif")
    assert merged_path_for(red, "rgb", frozenset({second})).endswith("IMG_1_RGB_3.tif")


def test_can_merge_needs_every_exposure_on_disk_and_camera_raw(tmp_path):
    red, green, blue = _triplet(tmp_path)
    assert can_merge(_asset(red, green, blue), "rgb")
    os.remove(blue)
    assert not can_merge(_asset(red, green, blue), "rgb")
    tif = str(tmp_path / "a.tif")
    open(tif, "wb").close()
    assert not can_merge(_asset(tif, green, red), "rgb")
    assert not can_merge(_asset(red, green, tif), "rgb")


def test_carry_edit_copies_everything_and_keeps_the_original(tmp_path):
    repo = _repo(tmp_path)
    red, green, blue = _triplet(tmp_path)
    new_path = str(tmp_path / "IMG_1_RGB.tif")
    cfg = _triplet_config(green, blue)
    repo.save_file_settings("red", cfg, file_path=red)
    repo.save_history_step("red", 0, cfg)
    repo.save_work_print("red", "Warm", cfg)
    repo.save_file_mark("red", "keeper", file_path=red)
    roll_id = rolls.create_virtual_roll(repo, "Roll", [red, green, blue, "/other.ARW"])
    rolls.fork_edit(repo, roll_id, "red", red, cfg)
    rolls.set_frame_override(repo, roll_id, "red", "film", True)
    scene = rolls.create_scene(repo, roll_id, "Beach", ["red"])

    carry_edit(repo, "red", red, "new", new_path, [green, blue], cfg, "rgb")

    assert repo.load_file_settings("new").rgbscan == RgbScanConfig()
    assert repo.load_all_history("new")[0][1].rgbscan == RgbScanConfig()
    assert repo.load_work_print("new", "Warm").rgbscan == RgbScanConfig()
    assert repo.load_file_marks()["new"] == "keeper"
    assert repo.load_file_marks_by_path()[new_path] == "keeper"
    assert repo.load_file_settings(rolls.roll_edit_hash("new", roll_id)).rgbscan == RgbScanConfig()
    entry = rolls.roll_for_id(repo, roll_id)
    assert entry["member_paths"] == [new_path, "/other.ARW"]
    assert rolls.is_forked(repo, roll_id, "new")
    assert rolls.frame_override_cards(repo, roll_id, "new") == {"film"}
    assert "new" in dict(rolls.roll_scenes(repo, roll_id))[scene]["member_hashes"]
    assert repo.load_file_settings("red").rgbscan == cfg.rgbscan
    assert repo.load_file_marks()["red"] == "keeper"


def test_carry_edit_never_overwrites_an_edit_the_new_file_already_has(tmp_path):
    repo = _repo(tmp_path)
    repo.save_file_settings("red", _triplet_config("g", "b"))
    own = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, crosstalk_strength=0.5))
    repo.save_file_settings("new", own)
    assert repo.copy_file_edits("red", "new", "/x.tif", lambda c: merged_edit(c, "rgb")) is False
    assert repo.load_file_settings("new").process.crosstalk_strength == 0.5


def test_a_roll_sensor_matrix_is_locked_out_of_the_merged_frame(tmp_path):
    repo = _repo(tmp_path)
    folder = tmp_path / "roll"
    folder.mkdir()
    red, green, blue = _triplet(folder)
    new_path = str(folder / "IMG_1_RGB.tif")
    roll_id = rolls.recognize_folder(repo, str(folder))
    rolls.set_roll_defaults(repo, roll_id, sensor_matrix=_MATRIX, linear_raw=True, narrowband_scan=True)
    cfg = _triplet_config(green, blue)

    carry_edit(repo, "red", red, "new", new_path, [green, blue], cfg, "rgb")

    assert "sensor" in rolls.frame_override_cards(repo, roll_id, "new")
    resolved = rolls.resolve_roll_config(repo, roll_id, "new", repo.load_file_settings("new"))
    assert resolved.process.sensor_matrix is None
    assert resolved.process.narrowband_scan is True


def test_a_sidecar_moves_only_when_the_red_exposure_has_one(tmp_path):
    red, green, blue = _triplet(tmp_path)
    new_path = str(tmp_path / "IMG_1_RGB.tif")
    cfg = _triplet_config(green, blue)
    assert carry_sidecar(red, new_path, cfg, "rgb") is False
    assert not os.path.exists(sidecar_path_for(new_path))

    write_sidecar(red, cfg)
    assert carry_sidecar(red, new_path, cfg, "rgb") is True
    assert load_sidecar(new_path).rgbscan == RgbScanConfig()


def _finish(tmp_path, monkeypatch, results, trash=True):
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController

    trashed: list = []
    monkeypatch.setattr("negpy.desktop.controller._move_to_trash", lambda p: trashed.append(p) or True)
    ctrl = MagicMock()
    ctrl.session.repo = _repo(tmp_path)
    ctrl._frame_merge_trash = trash
    ctrl._apply_roll_forks = lambda assets: None
    # A real config, not a mock: merged_edit runs on it.
    ctrl.session.config_for_asset = lambda asset: WorkspaceConfig()
    ctrl._already_merged_to = lambda r: AppController._already_merged_to(ctrl, r)
    return ctrl, trashed, AppController._on_frame_merge_finished


def _asset(red, green, blue, file_hash="red"):
    return {"name": "IMG_1 (RGB)", "path": red, "hash": file_hash, "green_path": green, "blue_path": blue}


def _stitch_asset(primary, *parts, triplets=(), file_hash="digest#stitch"):
    n = 1 + len(parts)
    return {
        "name": "stitch",
        "path": primary,
        "hash": file_hash,
        "stitch_paths": tuple(parts),
        "stitch_transforms": tuple((1.0, 0.0, 0.0, 0.0, 1.0, 0.0) for _ in range(n)),
        "stitch_canvas": (72, 24),
        "stitch_sizes": tuple((36, 24) for _ in range(n)),
        "stitch_triplets": tuple(triplets) or tuple(("", "") for _ in range(n)),
        "stitch_align": False,
    }


def _stitch_config(primary, *parts, triplets=()):
    a = _stitch_asset(primary, *parts, triplets=triplets)
    return replace(
        WorkspaceConfig(),
        stitch=StitchConfig(
            stitch_enabled=True,
            stitch_paths=a["stitch_paths"],
            stitch_transforms=a["stitch_transforms"],
            stitch_canvas=a["stitch_canvas"],
            stitch_sizes=a["stitch_sizes"],
            stitch_triplets=a["stitch_triplets"],
            stitch_align=False,
        ),
    )


def test_finish_swaps_the_frame_and_trashes_its_exposures(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult

    red, green, blue = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    write_sidecar(red, _triplet_config(green, blue))
    asset = _asset(red, green, blue)
    results = [FrameMergeResult(asset, out, kind="rgb", new_hash="new")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [{"path": "/a.ARW", "hash": "a"}, asset]
    ctrl.session.repo.save_file_settings("red", _triplet_config(green, blue), file_path=red)

    finish(ctrl, results, False)

    swapped = ctrl.session.replace_assets.call_args.args[0]
    assert list(swapped) == [1]
    assert swapped[1]["path"] == out and swapped[1]["hash"] == "new"
    assert ctrl.session.repo.load_file_settings("new") is not None
    assert trashed == [red, green, blue, sidecar_path_for(red)]


def test_finish_keeps_the_exposures_of_a_frame_that_failed(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult

    red, green, blue = _triplet(tmp_path)
    asset = _asset(red, green, blue)
    results = [FrameMergeResult(asset, str(tmp_path / "IMG_1_RGB.tif"), kind="rgb", error="disk full")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [asset]

    finish(ctrl, results, False)

    assert trashed == []
    assert ctrl.session.replace_assets.call_args.args[0] == {}
    assert "1 failed" in ctrl.set_status.call_args.args[0]


def test_finish_without_trash_keeps_the_triplet_in_the_film_strip(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult

    red, green, blue = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    asset = _asset(red, green, blue)
    results = [FrameMergeResult(asset, out, kind="rgb", new_hash="new")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results, trash=False)
    ctrl.state.uploaded_files = [asset]

    finish(ctrl, results, False)

    assert trashed == []
    ctrl.session.replace_assets.assert_not_called()
    inserted = ctrl.session.insert_assets.call_args.args[0]
    assert inserted[0]["path"] == out


def test_finish_copies_a_forked_frames_shared_edit(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult

    red, green, blue = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    asset = _asset(red, green, blue, file_hash=rolls.roll_edit_hash("red", "r1"))
    results = [FrameMergeResult(asset, out, kind="rgb", new_hash="new")]
    ctrl, _trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [asset]
    ctrl.session.repo.save_file_settings("red", _triplet_config(green, blue), file_path=red)

    finish(ctrl, results, False)

    assert ctrl.session.repo.load_file_settings("new") is not None


def _plan(uploaded, indices=None, params=None):
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController

    from negpy.desktop.session import resolve_asset_rgbscan, resolve_asset_stitch

    ctrl = MagicMock()
    ctrl.state.uploaded_files = uploaded

    # As _batch_params_for does: the assembly comes off the asset, not the saved edit.
    def batch_params(f):
        return resolve_asset_stitch(resolve_asset_rgbscan(params or WorkspaceConfig(), f), f)

    ctrl._batch_params_for = batch_params
    return AppController.frame_merge_plan(ctrl, indices)


def test_plan_merges_a_stitch_and_refuses_a_bracket(tmp_path):
    red, green, blue = _triplet(tmp_path)
    part = str(tmp_path / "part.ARW")
    open(part, "wb").close()
    indices, skipped = _plan(
        [
            {"name": "plain.ARW", "path": str(tmp_path / "plain.ARW")},
            _asset(red, green, blue),
            _stitch_asset(red, part),
            {**_asset(red, green, blue), "name": "bracket", "hdr_paths": ("/b.ARW",)},
            {**_asset(red, green, blue), "name": "halved", "half": 1},
            {**_asset(red, green, str(tmp_path / "gone.ARW")), "name": "gone"},
        ]
    )
    assert indices == [1, 2]
    assert skipped == [
        "bracket: a bracket would lose its shadow detail in a TIFF",
        "halved: a half-frame scan cannot merge",
        "gone: a source file is missing or not a camera RAW",
    ]


def test_plan_refuses_a_stitch_whose_registration_is_incomplete(tmp_path):
    red, _g, _b = _triplet(tmp_path)
    part = str(tmp_path / "part.ARW")
    open(part, "wb").close()
    broken = {**_stitch_asset(red, part), "stitch_transforms": ()}
    indices, skipped = _plan([broken])
    assert indices == []
    assert skipped == ["stitch: a source file is missing or not a camera RAW"]
    assert not registration_complete(broken)


def test_plan_honours_the_scope(tmp_path):
    red, green, blue = _triplet(tmp_path)
    uploaded = [_asset(red, green, blue), {**_asset(red, green, blue), "name": "second"}]
    assert _plan(uploaded, None)[0] == [0, 1]
    assert _plan(uploaded, [1])[0] == [1]
    assert _plan(uploaded, [9])[0] == []


def test_decode_params_keep_a_stitch_and_its_flat_field(tmp_path):
    cfg = _stitch_config("/a.ARW", "/b.ARW")
    cfg = replace(cfg, flatfield=replace(cfg.flatfield, apply=True, profile_id="abc"))
    out = decode_params(cfg, "stitch")
    assert out.flatfield.apply is True and out.flatfield.profile_id == "abc"
    assert out.stitch == cfg.stitch
    assert out.hdr == HdrConfig()


def test_merged_edit_clears_the_stitch_its_flat_field_and_the_unmix():
    cfg = _stitch_config("/a.ARW", "/b.ARW")
    cfg = replace(
        cfg,
        flatfield=replace(cfg.flatfield, apply=True, profile_id="abc"),
        process=replace(cfg.process, sensor_matrix=_MATRIX, sensor_profile="rig"),
    )
    edit = merged_edit(cfg, "stitch")
    assert edit.stitch == StitchConfig()
    assert edit.flatfield == FlatFieldConfig()
    assert effective_sensor_matrix(edit.process) is None
    assert merged_edit(cfg, "rgb").flatfield.apply is True


def test_part_files_walks_every_part_and_its_triplet():
    a = _stitch_asset("/p0.ARW", "/p1.ARW", triplets=(("/g0.ARW", "/b0.ARW"), ("/g1.ARW", "/b1.ARW")))
    assert part_files(a, "stitch") == ["/p1.ARW", "/g0.ARW", "/b0.ARW", "/g1.ARW", "/b1.ARW"]
    assert part_files({"path": "/r.ARW", "green_path": "/g.ARW", "blue_path": "/b.ARW"}, "rgb") == ["/g.ARW", "/b.ARW"]
    assert part_files({"path": "/x.ARW"}, "") == []


def test_a_merged_stitch_is_named_and_recognized(tmp_path):
    primary, part, _b = _triplet(tmp_path)
    cfg = _stitch_config(primary, part)
    out = merged_path_for(primary, "stitch")
    assert out.endswith("IMG_1_STITCH.tif")

    write_merged_frame(np.full((24, 72, 3), 0.4, dtype=np.float32), primary, out, cfg)
    assert is_merged_source(out)
    with tifffile.TiffFile(out) as tif:
        assert "camera RAW (stitch 2-part)" in (tif.pages[0].description or "")
    assert merged_path_for(primary, "stitch").endswith("IMG_1_STITCH_2.tif")


def test_a_roll_flat_field_is_locked_out_of_a_merged_stitch(tmp_path):
    repo = _repo(tmp_path)
    folder = tmp_path / "roll"
    folder.mkdir()
    primary, part, _b = _triplet(folder)
    new_path = str(folder / "IMG_1_STITCH.tif")
    roll_id = rolls.recognize_folder(repo, str(folder))
    rolls.set_roll_defaults(repo, roll_id, apply=True, profile_id="ff", sensor_matrix=_MATRIX, narrowband_scan=True)

    carry_edit(repo, "digest#stitch", primary, "new", new_path, [part], _stitch_config(primary, part), "stitch")

    assert {"sensor", "flatfield"} <= rolls.frame_override_cards(repo, roll_id, "new")
    resolved = rolls.resolve_roll_config(repo, roll_id, "new", repo.load_file_settings("new"))
    assert resolved.flatfield.apply is False
    assert resolved.process.sensor_matrix is None
    assert resolved.process.narrowband_scan is True


def test_carry_edit_writes_a_row_even_when_the_composite_had_none(tmp_path):
    repo = _repo(tmp_path)
    primary, part, _b = _triplet(tmp_path)
    new_path = str(tmp_path / "IMG_1_STITCH.tif")
    cfg = replace(_stitch_config(primary, part), flatfield=replace(FlatFieldConfig(), apply=True, profile_id="ff"))

    carry_edit(repo, "digest#stitch", primary, "new", new_path, [part], cfg, "stitch")

    saved = repo.load_file_settings("new")
    assert saved is not None and saved.flatfield == FlatFieldConfig()


def test_a_composite_hash_keeps_its_stitch_suffix():
    assert rolls.unforked_hash("digest#stitch") == "digest#stitch"
    assert rolls.unforked_hash(rolls.roll_edit_hash("digest#stitch", "r1")) == "digest#stitch"


def test_finish_dissolves_the_composite_and_trashes_every_part(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult
    from negpy.services.assets.composites import remember_composites, saved_composites

    primary, part, _b = _triplet(tmp_path)
    green, blue = str(tmp_path / "g.ARW"), str(tmp_path / "b.ARW")
    for p in (green, blue):
        open(p, "wb").close()
    out = str(tmp_path / "IMG_1_STITCH.tif")
    open(out, "wb").close()
    write_sidecar(primary, WorkspaceConfig())
    write_sidecar(part, WorkspaceConfig())
    asset = _stitch_asset(primary, part, triplets=(("", ""), (green, blue)))
    asset["process_mode"] = "Slide"
    results = [FrameMergeResult(asset, out, kind="stitch", new_hash="new")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [asset]
    remember_composites(ctrl.session.repo, [asset])

    finish(ctrl, results, False)

    assert primary not in saved_composites(ctrl.session.repo)
    assert trashed == [primary, part, green, blue, sidecar_path_for(primary), sidecar_path_for(part)]
    swapped = ctrl.session.replace_assets.call_args.args[0]
    assert swapped[0]["path"] == out
    assert swapped[0]["process_mode"] == "Slide"


def test_finish_keeps_the_composite_of_a_stitch_that_failed(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult
    from negpy.services.assets.composites import remember_composites, saved_composites

    primary, part, _b = _triplet(tmp_path)
    asset = _stitch_asset(primary, part)
    results = [FrameMergeResult(asset, str(tmp_path / "x.tif"), kind="stitch", error="disk full")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [asset]
    remember_composites(ctrl.session.repo, [asset])

    finish(ctrl, results, False)

    assert primary in saved_composites(ctrl.session.repo)
    assert trashed == []


def test_a_part_left_on_disk_stays_in_its_roll(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult
    from negpy.services.assets.composites import remember_composites, saved_composites

    primary, part, _b = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_STITCH.tif")
    open(out, "wb").close()
    asset = _stitch_asset(primary, part)
    results = [FrameMergeResult(asset, out, kind="stitch", new_hash="new")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results, trash=False)
    ctrl.state.uploaded_files = [asset]
    roll_id = rolls.create_virtual_roll(ctrl.session.repo, "Roll", [primary, part])
    remember_composites(ctrl.session.repo, [asset])

    finish(ctrl, results, False)

    assert trashed == []
    members = rolls.roll_for_id(ctrl.session.repo, roll_id)["member_paths"]
    assert part in members and primary in members and out in members
    assert primary in saved_composites(ctrl.session.repo)


def test_a_merged_stitch_reads_back_as_the_composite_it_replaced(tmp_path):
    from negpy.features.stitch.models import StitchConfig

    rng = np.random.default_rng(3)
    scene = (rng.random((80, 120, 3)) * 60000).astype(np.uint16)
    p0, p1 = str(tmp_path / "part0.ARW"), str(tmp_path / "part1.ARW")
    for p in (p0, p1):
        open(p, "wb").close()
    buffers = {p0: scene[:, :80], p1: scene[:, 40:]}

    cfg = replace(
        WorkspaceConfig(),
        stitch=StitchConfig(
            stitch_enabled=True,
            stitch_paths=(p1,),
            stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
            stitch_canvas=(120, 80),
            stitch_sizes=((80, 80), (80, 80)),
            stitch_triplets=(("", ""), ("", "")),
        ),
    )
    proc = ImageProcessor()
    proc._decode_sensor_rgb = _fake_sensor_decode(buffers)
    composite, _ir, _cs = proc._load_source_f32(p0, cfg)

    params = decode_params(cfg, "stitch")
    proc.release_source_cache()
    merged, _ir, _cs = proc._load_source_f32(p0, params)
    assert np.array_equal(merged, composite)

    out = merged_path_for(p0, "stitch")
    write_merged_frame(merged, p0, out, params)
    reread, _ir, _cs = ImageProcessor()._decode_oriented_f32(out, merged_edit(cfg, "stitch"))
    assert np.allclose(reread, composite, atol=1.0 / 65535.0)


def test_inserting_a_merged_frame_keeps_the_selection_on_the_same_frame():
    from unittest.mock import MagicMock

    from negpy.desktop.session import DesktopSessionManager

    session = MagicMock()
    session.state.uploaded_files = [{"path": "/a", "hash": "a"}, {"path": "/b", "hash": "b"}, {"path": "/c", "hash": "c"}]
    session.state.selected_file_idx = 2
    session.state.selected_indices = [0, 2]
    session.repo.load_file_marks.return_value = {}

    DesktopSessionManager.insert_assets(session, {0: {"path": "/a_RGB.tif", "hash": "m"}})

    assert [f["path"] for f in session.state.uploaded_files] == ["/a", "/a_RGB.tif", "/b", "/c"]
    assert session.state.selected_file_idx == 3
    assert session.state.selected_indices == [0, 3]


def test_a_saved_row_is_still_resolved_against_the_roll(tmp_path):
    repo = _repo(tmp_path)
    folder = tmp_path / "roll"
    folder.mkdir()
    red, green, blue = _triplet(folder)
    new_path = str(folder / "IMG_1_RGB.tif")
    roll_id = rolls.recognize_folder(repo, str(folder))
    rolls.set_roll_defaults(repo, roll_id, sensor_matrix=_MATRIX, narrowband_scan=True, linear_raw=True)
    # A row that predates the roll defaults.
    cfg = _triplet_config(green, blue)
    repo.save_file_settings("red", cfg, file_path=red)

    carry_edit(repo, "red", red, "new", new_path, [green, blue], cfg, "rgb")

    resolved = rolls.resolve_roll_config(repo, roll_id, "new", repo.load_file_settings("new"))
    assert resolved.process.sensor_matrix is None
    assert resolved.process.narrowband_scan is True
    assert resolved.process.linear_raw is True


def test_a_slide_is_refused_because_a_tiff_carries_no_camera_matrix(tmp_path):
    from negpy.features.process.models import ProcessMode

    red, green, blue = _triplet(tmp_path)
    slide = replace(_triplet_config(green, blue), process=replace(WorkspaceConfig().process, process_mode=ProcessMode.E6))
    indices, skipped = _plan([{**_asset(red, green, blue), "name": "slide"}], params=slide)
    assert indices == []
    assert skipped == ["slide: a TIFF cannot carry a slide's camera color matrix"]
    assert needs_camera_matrix(slide) and not needs_camera_matrix(_triplet_config(green, blue))


def test_a_source_whose_label_names_no_assembly_is_refused(tmp_path):
    assert is_merge_description("camera RAW (RGB triplet)")
    assert is_merge_description("camera RAW (stitch 4-part, RGB triplet)")
    assert not is_merge_description("DNG LinearRaw")
    assert not is_merge_description("TIFF")

    red, green, blue = _triplet(tmp_path)
    cfg = _triplet_config(green, blue)
    assert describes_a_merge(red, cfg)


def test_request_merges_by_path_so_a_stale_index_cannot_merge_the_wrong_frame(tmp_path):
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController

    red, green, blue = _triplet(tmp_path)
    ctrl = MagicMock()
    ctrl._batch_busy.return_value = False
    ctrl._begin_batch.return_value = None  # stop before emitting
    # The list the dialog was built against is gone: a discovery replaced it.
    ctrl.state.uploaded_files = [{"path": "/elsewhere.ARW", "hash": "x"}]

    AppController.request_frame_merge(ctrl, [red], True)

    ctrl.frame_merge_requested.emit.assert_not_called()


def _thumb_session(uploaded):
    from unittest.mock import MagicMock

    from negpy.desktop.session import DesktopSessionManager

    session = MagicMock()
    session.state.uploaded_files = uploaded
    session.state.thumbnails = {}
    session.state.rendered_thumbnails = set()
    session.state.stale_thumbnails = set()
    session.state.selected_file_idx = 0
    session.state.selected_indices = [0]
    session.repo.load_file_marks.return_value = {}
    # Bound explicitly: a MagicMock would otherwise answer these with a mock of its own.
    session._carry_thumbnail = lambda old, new: DesktopSessionManager._carry_thumbnail(session, old, new)
    session._drop_thumbnail = lambda asset: DesktopSessionManager._drop_thumbnail(session, asset)
    session.state.embeddings = {}
    return session, DesktopSessionManager


def test_a_merged_frame_keeps_the_rendered_thumbnail_of_the_frame_it_replaces():
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    triplet = {"path": "/r.ARW", "hash": "red", "green_path": "/g.ARW", "blue_path": "/b.ARW"}
    merged = {"path": "/r_RGB.tif", "hash": "new"}
    session, cls = _thumb_session([triplet])
    old_key = asset_thumbnail_key(triplet)
    session.state.thumbnails[old_key] = "icon"
    session.state.rendered_thumbnails.add(old_key)

    cls.replace_assets(session, {0: merged})

    new_key = asset_thumbnail_key(merged)
    assert session.state.thumbnails[new_key] == "icon"
    assert new_key in session.state.rendered_thumbnails
    assert old_key not in session.state.thumbnails


def test_an_inserted_merged_frame_carries_the_thumbnail_and_the_source_keeps_its_own():
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    triplet = {"path": "/r.ARW", "hash": "red", "green_path": "/g.ARW", "blue_path": "/b.ARW"}
    merged = {"path": "/r_RGB.tif", "hash": "new"}
    session, cls = _thumb_session([triplet])
    old_key = asset_thumbnail_key(triplet)
    session.state.thumbnails[old_key] = "icon"
    session.state.rendered_thumbnails.add(old_key)

    cls.insert_assets(session, {0: merged})

    assert session.state.thumbnails[asset_thumbnail_key(merged)] == "icon"
    assert session.state.thumbnails[old_key] == "icon"


def test_abort_during_the_write_leaves_the_frame_untouched(tmp_path):
    red, green, blue = _triplet(tmp_path)
    out = merged_path_for(red, "rgb")

    with pytest.raises(MergeCancelled):
        write_merged_frame(
            np.full((8, 10, 3), 0.25, dtype=np.float32),
            red,
            out,
            _triplet_config(green, blue),
            should_cancel=lambda: True,
        )

    assert not os.path.exists(out)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]
    assert sorted(os.path.basename(p) for p in (red, green, blue)) == sorted(f for f in os.listdir(tmp_path) if f.endswith(".ARW"))


def test_a_cancelled_frame_is_neither_merged_nor_failed(tmp_path, monkeypatch):
    import negpy.desktop.workers.frame_merge as worker_mod

    red, green, blue = _triplet(tmp_path)
    w = worker_mod.FrameMergeWorker()
    task = worker_mod.FrameMergeTask(
        asset=_asset(red, green, blue),
        params=_triplet_config(green, blue),
        out_path=merged_path_for(red, "rgb"),
        compression=WorkspaceConfig().export.tiff_compression,
        kind="rgb",
    )

    # Abort lands while the decode runs.
    def decode(path, params, fast_decode=False):
        w.cancel()
        return np.zeros((8, 10, 3), dtype=np.float32), None, "srgb"

    monkeypatch.setattr(w._processor, "_load_source_f32", decode)
    seen = []
    w.finished.connect(lambda results, aborted: seen.append((results, aborted)))
    w.run([task])

    results, aborted = seen[0]
    assert aborted is True
    assert results == []
    assert not os.path.exists(task.out_path)
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".ARW")) == [
        "IMG_1.ARW",
        "IMG_2.ARW",
        "IMG_3.ARW",
    ]


def test_path_for_file_hash_finds_the_file_an_edit_was_saved_against(tmp_path):
    repo = _repo(tmp_path)
    assert repo.path_for_file_hash("nothing") is None
    repo.save_file_settings("h", WorkspaceConfig(), file_path="/negs/a_RGB.tif")
    assert repo.path_for_file_hash("h") == "/negs/a_RGB.tif"
    repo.save_file_settings("h2", WorkspaceConfig())
    assert repo.path_for_file_hash("h2") is None


def test_a_second_merge_is_refused_and_changes_nothing(tmp_path, monkeypatch):
    from negpy.desktop.workers.frame_merge import FrameMergeResult

    red, green, blue = _triplet(tmp_path)
    original = str(tmp_path / "IMG_1_RGB.tif")
    duplicate = str(tmp_path / "IMG_1_RGB_2.tif")
    for f in (original, duplicate):
        open(f, "wb").close()
    asset = _asset(red, green, blue)
    results = [FrameMergeResult(asset, duplicate, kind="rgb", new_hash="same")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [asset]
    first = replace(_triplet_config(green, blue), geometry=replace(WorkspaceConfig().geometry, rotation=2))
    ctrl.session.repo.save_file_settings("same", first, file_path=original)

    finish(ctrl, results, False)

    assert not os.path.exists(duplicate), "the byte-identical second write is discarded"
    assert os.path.exists(original)
    assert trashed == [], "the frame keeps its exposures"
    assert ctrl.session.replace_assets.call_args.args[0] == {}, "the film strip is untouched"
    assert ctrl.session.repo.load_file_settings("same").geometry.rotation == 2
    msg = ctrl.set_status.call_args.args[0]
    assert "already merged" in msg and "delete the negative" in msg
    assert "unmerge" not in msg.lower()
