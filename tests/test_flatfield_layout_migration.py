import os

import numpy as np
import pytest

from negpy.features.flatfield import logic as ff
from negpy.kernel.image.logic import apply_exif_orientation
from negpy.services.assets import flatfield as ffstore
from negpy.services.assets.flatfield import FlatFieldProfiles
from negpy.services.assets.migrations import flatfield_layout
from negpy.services.assets.migrations.flatfield_layout import migrate_flatfield_sensor_layout


def _sensor_gain(h: int = 24, w: int = 36) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 1.0 + 0.5 * (xx / w) + 0.25 * (yy / h)
    return np.repeat(g[:, :, None], 3, axis=2).astype(np.float32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    directory = tmp_path / "flatfield"
    directory.mkdir()
    monkeypatch.setattr(ffstore.APP_CONFIG, "flatfield_dir", str(directory), raising=False)
    return directory


def _exif_tag(monkeypatch, tag: int) -> None:
    """The reference's tag as EXIF carries it, which every build has read."""
    monkeypatch.setattr(flatfield_layout, "read_orientation", lambda path: tag)
    monkeypatch.setattr(flatfield_layout, "read_exif_from_file", lambda path: {"0th": {274: tag}})


def _libraw_tag(monkeypatch, tag: int, sensor_hw: tuple[int, int]) -> None:
    """The reference's tag as only LibRaw reads it (CR3, RAF), over a sensor of *sensor_hw*."""
    from types import SimpleNamespace

    monkeypatch.setattr(flatfield_layout, "read_orientation", lambda path: tag)
    monkeypatch.setattr(flatfield_layout, "read_exif_from_file", lambda path: None)
    raw = SimpleNamespace(sizes=SimpleNamespace(height=sensor_hw[0], width=sensor_hw[1]))
    fake = SimpleNamespace()

    class _Raw:
        def __enter__(self):
            return raw

        def __exit__(self, *exc):
            return False

    fake.imread = lambda path: _Raw()
    monkeypatch.setitem(__import__("sys").modules, "rawpy", fake)


def _write_legacy(directory, profile_id: str, gain: np.ndarray, source: str) -> str:
    """A profile as written before the layout marker: gain and check copy as the reference decoded."""
    path = str(directory / f"{profile_id}.npz")
    np.savez_compressed(
        path,
        gain=gain,
        token=ff.gain_token(gain),
        name="rig",
        k1=0.0,
        source=source,
        reference=(gain * 0.4).astype(np.float16),
        check=np.array([-0.01, 0.01, 0.005, 0.0]),
    )
    return path


def test_a_profile_from_a_turned_reference_moves_to_sensor_layout(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.arw"
    source.write_bytes(b"")
    _exif_tag(monkeypatch, 6)
    sensor = _sensor_gain()
    _write_legacy(store, "p", apply_exif_orientation(sensor, 6), str(source))

    migrate_flatfield_sensor_layout()

    gain, token, sensor_layout = FlatFieldProfiles.load_gain("p")
    np.testing.assert_array_equal(gain, sensor)
    assert token == ff.gain_token(sensor)
    assert sensor_layout
    assert FlatFieldProfiles.load_check("p").reference.shape[:2] == sensor.shape[:2]
    assert FlatFieldProfiles.get("p").name == "rig"
    assert str(FlatFieldProfiles._read("p", ("layout",))["layout"]) == ffstore.SENSOR_LAYOUT


def test_an_upright_profile_keeps_its_gain_and_token(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.arw"
    source.write_bytes(b"")
    _exif_tag(monkeypatch, 1)
    gain = _sensor_gain()
    _write_legacy(store, "p", gain, str(source))

    migrate_flatfield_sensor_layout()

    entry = FlatFieldProfiles.load_gain("p")
    np.testing.assert_array_equal(entry.gain, gain)
    assert entry.token == ff.gain_token(gain)
    assert entry.sensor_layout


def test_a_profile_without_its_reference_stays_as_baked_until_the_reference_is_back(store, tmp_path, monkeypatch):
    source = tmp_path / "offline" / "flat.arw"
    _exif_tag(monkeypatch, 6)
    turned = apply_exif_orientation(_sensor_gain(), 6)
    _write_legacy(store, "p", turned, str(source))

    migrate_flatfield_sensor_layout()

    entry = FlatFieldProfiles.load_gain("p")
    np.testing.assert_array_equal(entry.gain, turned)
    assert not entry.sensor_layout

    source.parent.mkdir()
    source.write_bytes(b"")
    migrate_flatfield_sensor_layout()

    entry = FlatFieldProfiles.load_gain("p")
    np.testing.assert_array_equal(entry.gain, _sensor_gain())
    assert entry.sensor_layout


def test_an_unmarked_profile_applies_as_baked_whatever_the_frame_orientation(store, tmp_path):
    from negpy.features.flatfield.models import FlatFieldConfig

    turned = apply_exif_orientation(_sensor_gain(), 6)
    _write_legacy(store, "p", turned, str(tmp_path / "gone.arw"))
    frame = np.full(turned.shape, 0.5, np.float32)
    ff.set_gain_provider(FlatFieldProfiles.load_gain)
    try:
        out = ff.apply_flatfield(frame, FlatFieldConfig(apply=True, profile_id="p"), orientation=6)
        np.testing.assert_allclose(out, frame * turned)
        assert ff.flatfield_token(FlatFieldConfig(apply=True, profile_id="p")).endswith(":ref")
    finally:
        ff.set_gain_provider(None)


def test_a_marked_profile_is_left_alone(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.arw"
    source.write_bytes(b"")
    _exif_tag(monkeypatch, 6)
    _write_legacy(store, "p", apply_exif_orientation(_sensor_gain(), 6), str(source))
    migrate_flatfield_sensor_layout()
    first = FlatFieldProfiles.load_gain("p").gain

    migrate_flatfield_sensor_layout()

    second = FlatFieldProfiles.load_gain("p").gain
    np.testing.assert_array_equal(first, second)


def test_a_write_cut_short_is_never_listed_as_a_profile(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.arw"
    source.write_bytes(b"")
    _exif_tag(monkeypatch, 6)
    gain = apply_exif_orientation(_sensor_gain(), 6)
    _write_legacy(store, "p", gain, str(source))

    def crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(flatfield_layout.os, "replace", crash)
    migrate_flatfield_sensor_layout()

    assert [i for i, _ in FlatFieldProfiles.list_profiles()] == ["p"]
    assert sorted(os.listdir(store)) == ["p.npz"]
    np.testing.assert_array_equal(FlatFieldProfiles.load_gain("p").gain, gain)


def test_a_libraw_only_tag_the_bake_never_read_leaves_the_gain_in_place(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.cr3"
    source.write_bytes(b"")
    _libraw_tag(monkeypatch, 8, sensor_hw=(4000, 6000))
    sensor = _sensor_gain()  # landscape like the sensor: the bake made no quarter turn
    _write_legacy(store, "p", sensor, str(source))

    migrate_flatfield_sensor_layout()

    entry = FlatFieldProfiles.load_gain("p")
    np.testing.assert_array_equal(entry.gain, sensor)
    assert entry.sensor_layout


def test_a_libraw_only_tag_the_bake_applied_is_undone(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.cr3"
    source.write_bytes(b"")
    _libraw_tag(monkeypatch, 8, sensor_hw=(4000, 6000))
    _write_legacy(store, "p", apply_exif_orientation(_sensor_gain(), 8), str(source))

    migrate_flatfield_sensor_layout()

    np.testing.assert_array_equal(FlatFieldProfiles.load_gain("p").gain, _sensor_gain())


def test_a_libraw_only_half_turn_cannot_be_told_and_stays_as_baked(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.cr3"
    source.write_bytes(b"")
    _libraw_tag(monkeypatch, 3, sensor_hw=(4000, 6000))
    _write_legacy(store, "p", _sensor_gain(), str(source))

    migrate_flatfield_sensor_layout()

    assert not FlatFieldProfiles.load_gain("p").sensor_layout


def test_a_jxl_reference_was_baked_upright_whatever_its_exif_says(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.jxl"
    source.write_bytes(b"")
    _exif_tag(monkeypatch, 6)
    sensor = _sensor_gain()
    _write_legacy(store, "p", sensor, str(source))

    migrate_flatfield_sensor_layout()

    entry = FlatFieldProfiles.load_gain("p")
    np.testing.assert_array_equal(entry.gain, sensor)
    assert entry.sensor_layout


def test_an_unreadable_reference_stays_as_baked(store, tmp_path, monkeypatch):
    source = tmp_path / "flat.arw"
    source.write_bytes(b"")
    _exif_tag(monkeypatch, 6)
    _write_legacy(store, "p", apply_exif_orientation(_sensor_gain(), 6), str(source))
    real_open = open

    def refuse(path, *args, **kwargs):
        if str(path) == str(source):
            raise PermissionError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", refuse)
    migrate_flatfield_sensor_layout()
    monkeypatch.setattr("builtins.open", real_open)

    assert not FlatFieldProfiles.load_gain("p").sensor_layout
