import os

import numpy as np
import pytest
import tifffile
from PIL import Image

import negpy.desktop.headless as headless
from negpy.desktop.launch import LaunchArgsError, parse_launch_args
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import ProcessConfig
from negpy.services.export.launch_recipe import LaunchRecipe


def test_render_flags_parse(tmp_path):
    request = parse_launch_args(["--render", str(tmp_path), "--format", "png", "--bit-depth", "16", "--long-edge", "800", "--auto-crop"])
    assert request.render is True
    assert request.paths == (str(tmp_path),)
    assert (request.recipe.export_fmt, request.recipe.export_bit_depth, request.recipe.long_edge) == ("PNG", 16, 800)
    assert request.recipe.auto_crop is True


@pytest.mark.parametrize(
    "argv",
    [["--render"], ["--render", "a.tif", "--intent", "linear"], ["--render", "a.tif", "--quit-after-export"]],
)
def test_render_rejects_what_it_cannot_do(argv):
    with pytest.raises(LaunchArgsError):
        parse_launch_args(argv)


def test_a_saved_edit_is_used_and_keeps_its_film_fact():
    saved = WorkspaceConfig(process=ProcessConfig(source_color=False))
    detect = pytest.fail  # a stored film fact is never measured again
    assert headless.render_config(saved, LaunchRecipe(), detect) is saved


def test_defaults_take_the_first_open_film_check():
    config = headless.render_config(None, LaunchRecipe(), lambda: False)
    assert config.process.source_color is False
    assert config.geometry == WorkspaceConfig().geometry


def test_auto_crop_arms_only_a_frame_without_a_crop():
    armed = headless.render_config(None, LaunchRecipe(auto_crop=True), lambda: True)
    assert armed.geometry.crop_from_auto is True
    cropped = WorkspaceConfig(geometry=GeometryConfig(crop_rect=(0.1, 0.1, 0.9, 0.9)))
    kept = headless.render_config(cropped, LaunchRecipe(auto_crop=True), lambda: True)
    assert kept.geometry.crop_rect == (0.1, 0.1, 0.9, 0.9)
    assert kept.geometry.crop_from_auto is False


def test_folders_expand_to_their_supported_files(tmp_path):
    (tmp_path / "b.tif").write_bytes(b"")
    (tmp_path / "a.TIF").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.tif").write_bytes(b"")
    scans, missing = headless.expand_inputs((str(tmp_path), str(tmp_path / "gone.tif")))
    assert [os.path.basename(s) for s in scans] == ["a.TIF", "b.tif"]
    assert missing == [str(tmp_path / "gone.tif")]


@pytest.fixture
def scans(tmp_path, monkeypatch):
    """Two synthetic negatives and an empty data folder, so no saved edit or database exists."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(headless.APP_CONFIG, "edits_db_path", str(data / "edits.db"))
    monkeypatch.setattr(headless.APP_CONFIG, "settings_db_path", str(data / "settings.db"))
    rng = np.random.default_rng(3)
    folder = tmp_path / "scans"
    folder.mkdir()
    for name in ("one.tif", "two.tif"):
        tifffile.imwrite(folder / name, (rng.random((240, 360, 3)) * 40000 + 9000).astype(np.uint16), photometric="rgb")
    return folder, data


def test_render_writes_previews_at_the_requested_size(scans, tmp_path):
    folder, data = scans
    out = tmp_path / "out"
    request = parse_launch_args(["--render", str(folder), "--format", "jpeg", "--long-edge", "180", "--output-dir", str(out)])
    assert headless.run_render(request, emit=lambda m: None) == headless.EXIT_OK
    for name in ("one.jpg", "two.jpg"):
        with Image.open(out / name) as im:
            assert max(im.size) == 180
    assert os.listdir(data) == []


def test_a_16_bit_tiff_and_a_missing_scan(scans, tmp_path):
    folder, _ = scans
    out = tmp_path / "out16"
    messages = []
    request = parse_launch_args(
        [
            "--render",
            str(folder / "one.tif"),
            str(folder / "gone.tif"),
            "--format",
            "tiff",
            "--bit-depth",
            "16",
            "--long-edge",
            "120",
            "--output-dir",
            str(out),
        ]
    )
    assert headless.run_render(request, emit=messages.append) == headless.EXIT_FAILED
    written = tifffile.imread(out / "one.tiff")
    assert written.dtype == np.uint16 and max(written.shape[:2]) == 120
    assert any("gone.tif" in m for m in messages)
