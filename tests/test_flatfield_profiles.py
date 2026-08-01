import os

import numpy as np
import pytest

from negpy.features.flatfield import logic as ff
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.services.assets import flatfield as ffstore
from negpy.services.assets.flatfield import FlatFieldProfiles


def _gain(h=24, w=32) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 1.0 + 0.5 * (xx / w) + 0.25 * (yy / h)
    return np.repeat(g[:, :, None], 3, axis=2).astype(np.float32)


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """Point the profile store at a temp dir and stub the RAW decode/bake."""
    monkeypatch.setattr(ffstore.APP_CONFIG, "flatfield_dir", str(tmp_path / "flatfield"), raising=False)
    monkeypatch.setattr(FlatFieldProfiles, "_bake_gain", staticmethod(lambda path: _gain()))
    return tmp_path


def test_import_and_load_round_trip(store_dir):
    gain = _gain()
    pid = FlatFieldProfiles.import_gain(gain, name="Rig A", k1=-0.08, source="/refs/flat.dng")

    loaded, token = FlatFieldProfiles.load_gain(pid)
    assert np.array_equal(loaded, gain)
    assert token == ff.gain_token(gain)

    prof = FlatFieldProfiles.get(pid)
    assert (prof.id, prof.name, prof.k1, prof.source) == (pid, "Rig A", -0.08, "/refs/flat.dng")


def test_opaque_ids_are_unique_per_profile(store_dir):
    a = FlatFieldProfiles.import_gain(_gain(), name="dup")
    b = FlatFieldProfiles.import_gain(_gain(), name="dup")
    assert a != b  # same display name, distinct ids
    assert {n for _, n in FlatFieldProfiles.list_profiles()} == {"dup"}
    assert sorted(i for i, _ in FlatFieldProfiles.list_profiles()) == sorted([a, b])


def test_list_profiles_sorted_by_name(store_dir):
    FlatFieldProfiles.import_gain(_gain(), name="Zeta")
    FlatFieldProfiles.import_gain(_gain(), name="alpha")
    assert [n for _, n in FlatFieldProfiles.list_profiles()] == ["alpha", "Zeta"]


def test_set_k1_preserves_gain_and_token(store_dir):
    pid = FlatFieldProfiles.import_gain(_gain(), name="rig", k1=0.0)
    _, tok_before = FlatFieldProfiles.load_gain(pid)

    FlatFieldProfiles.set_k1(pid, 0.12)
    gain_after, tok_after = FlatFieldProfiles.load_gain(pid)
    assert FlatFieldProfiles.get(pid).k1 == 0.12
    assert np.array_equal(gain_after, _gain())  # gain untouched by a k1 edit
    assert tok_after == tok_before  # so the render cache is not invalidated


def test_delete_and_missing(store_dir):
    pid = FlatFieldProfiles.import_gain(_gain(), name="rig")
    FlatFieldProfiles.delete(pid)
    assert FlatFieldProfiles.load_gain(pid) is None
    assert FlatFieldProfiles.get(pid) is None
    FlatFieldProfiles.delete(pid)  # idempotent, no raise


def test_create_bakes_and_activates_via_provider(store_dir):
    pid = FlatFieldProfiles.create("rig", "/refs/flat.dng")
    assert pid is not None

    ff.set_gain_provider(FlatFieldProfiles.load_gain)
    try:
        stored, _ = FlatFieldProfiles.load_gain(pid)
        img = np.full((24, 32, 3), 0.5, dtype=np.float32)
        out = ff.apply_flatfield(img, FlatFieldConfig(apply=True, profile_id=pid))
        assert np.allclose(out, img * stored)  # provider resolved the baked gain and applied it
    finally:
        ff.set_gain_provider(None)


def test_create_returns_none_when_reference_unreadable(store_dir, monkeypatch):
    monkeypatch.setattr(FlatFieldProfiles, "_bake_gain", staticmethod(lambda path: None))
    assert FlatFieldProfiles.create("rig", "/gone.dng") is None


def test_metadata_reads_do_not_require_the_gain(store_dir):
    # A file with metadata but no gain member: get()/list_profiles must still work,
    # proving the sidebar's per-sync list build never decompresses the gain array.
    directory = ffstore.APP_CONFIG.flatfield_dir
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, "meta-only.npz"), name="MetaOnly", k1=0.05, source="/x.dng")

    prof = FlatFieldProfiles.get("meta-only")
    assert prof is not None and prof.name == "MetaOnly" and prof.k1 == 0.05
    assert ("meta-only", "MetaOnly") in FlatFieldProfiles.list_profiles()
    assert FlatFieldProfiles.load_gain("meta-only") is None  # no gain member to resolve
