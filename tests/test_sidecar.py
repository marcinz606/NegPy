import json
import os
from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.sidecar import load_or_promote, load_sidecar, sidecar_path_for, write_sidecar


def _rich_config() -> WorkspaceConfig:
    """A config exercising scalar + crop + local-mask paths, so the round trip is meaningful."""
    return WorkspaceConfig(
        exposure=ExposureConfig(density=0.42, grade=130.0),
        geometry=GeometryConfig(fine_rotation=1.5, manual_crop_rect=(0.1, 0.2, 0.8, 0.9)),
        local=LocalAdjustmentsConfig(masks=(PolygonMask(vertices=((0.0, 0.0), (0.5, 0.5)), strength=0.7, feather=0.05),)),
    )


def test_sidecar_path_for_next_to_source():
    assert sidecar_path_for("/photos/IMG_001.NEF") == os.path.join("/photos", "IMG_001.negpy")


def test_roundtrip_next_to_source(tmp_path):
    src = str(tmp_path / "IMG_001.NEF")
    cfg = _rich_config()
    path = write_sidecar(src, cfg)

    assert path == str(tmp_path / "IMG_001.negpy")
    assert os.path.exists(path)

    loaded = load_sidecar(src)
    assert loaded is not None
    d = loaded.to_dict()
    assert d["density"] == 0.42
    assert d["grade"] == 130.0
    assert tuple(d["manual_crop_rect"]) == (0.1, 0.2, 0.8, 0.9)
    masks = d["local_masks"]["masks"]
    assert len(masks) == 1
    assert masks[0]["strength"] == 0.7
    assert masks[0]["feather"] == 0.05


def test_load_missing_returns_none(tmp_path):
    assert load_sidecar(str(tmp_path / "nope.NEF")) is None


def test_load_malformed_returns_none(tmp_path):
    src = str(tmp_path / "IMG_003.NEF")
    with open(sidecar_path_for(src), "w", encoding="utf-8") as f:
        f.write("{ not valid json")
    assert load_sidecar(src) is None


@pytest.fixture()
def repo(tmp_path):
    r = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    r.initialize()
    return r


def test_load_or_promote_promotes_sidecar_to_db(tmp_path, repo):
    src = str(tmp_path / "IMG_004.NEF")
    write_sidecar(src, _rich_config())

    assert repo.load_file_settings("h4") is None  # DB starts empty
    loaded = load_or_promote(repo, "h4", src)
    assert loaded is not None
    assert loaded.exposure.density == 0.42

    # Promotion: the DB now holds it, so a later load never needs the sidecar.
    promoted = repo.load_file_settings("h4")
    assert promoted is not None
    assert promoted.exposure.density == 0.42


def test_load_or_promote_db_wins(tmp_path, repo):
    src = str(tmp_path / "IMG_005.NEF")
    write_sidecar(src, replace(_rich_config(), exposure=ExposureConfig(density=0.11)))
    repo.save_file_settings("h5", replace(_rich_config(), exposure=ExposureConfig(density=0.99)))

    loaded = load_or_promote(repo, "h5", src)
    assert loaded is not None
    assert loaded.exposure.density == 0.99  # DB value, sidecar ignored


def test_load_or_promote_none_when_neither(tmp_path, repo):
    assert load_or_promote(repo, "h6", str(tmp_path / "IMG_006.NEF")) is None


def test_write_payload_is_to_dict_json(tmp_path):
    src = str(tmp_path / "IMG_007.NEF")
    cfg = _rich_config()
    write_sidecar(src, cfg)
    with open(sidecar_path_for(src), "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == json.loads(json.dumps(cfg.to_dict(), default=str))


def test_save_file_settings_stores_path(repo):
    """save_file_settings with file_path persists the path for later recovery."""
    cfg = _rich_config()
    repo.save_file_settings("h7", cfg, file_path="/photos/IMG_007.NEF")
    result = repo.load_file_settings_by_path("/photos/IMG_007.NEF")
    assert result is not None
    old_hash, loaded = result
    assert old_hash == "h7"
    assert loaded.exposure.density == 0.42


def test_path_fallback_recovers_orphaned_settings(tmp_path, repo):
    """When EXIF changes the hash, path-based fallback re-homes the settings."""
    src = str(tmp_path / "IMG_008.NEF")
    write_sidecar(src, _rich_config())

    # First load: promotes sidecar to DB under hash "h8"
    loaded1 = load_or_promote(repo, "h8", src)
    assert loaded1 is not None
    assert loaded1.exposure.density == 0.42

    # Simulate EXIF change: new hash "h9" misses DB, but path matches
    # load_or_promote should find by path, re-home to "h9"
    loaded2 = load_or_promote(repo, "h9", src)
    assert loaded2 is not None
    assert loaded2.exposure.density == 0.42

    # Old hash "h8" should be deleted
    assert repo.load_file_settings("h8") is None

    # New hash "h9" should have the settings
    promoted = repo.load_file_settings("h9")
    assert promoted is not None
    assert promoted.exposure.density == 0.42


def test_rehome_file_settings(repo):
    """rehome_file_settings copies and deletes correctly."""
    cfg = _rich_config()
    repo.save_file_settings("h10", cfg, file_path="/photos/IMG_010.NEF")

    # Re-home to new hash
    repo.rehome_file_settings("h10", "h11", "/photos/IMG_010.NEF")

    # Old hash gone
    assert repo.load_file_settings("h10") is None

    # New hash has the data
    loaded = repo.load_file_settings("h11")
    assert loaded is not None
    assert loaded.exposure.density == 0.42

    # Path lookup also finds it
    result = repo.load_file_settings_by_path("/photos/IMG_010.NEF")
    assert result is not None
    assert result[0] == "h11"


def test_load_file_settings_by_path_empty_or_missing(repo):
    """Empty path and non-existent path both return None."""
    assert repo.load_file_settings_by_path("") is None
    assert repo.load_file_settings_by_path("/nonexistent/file.NEF") is None
