import json
import sqlite3

import numpy as np
import pytest

from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import flatfield as ffstore
from negpy.services.assets.flatfield import FlatFieldProfiles
from negpy.services.assets.flatfield_migration import migrate_legacy_flatfield_profiles


@pytest.fixture
def legacy_repo(tmp_path, monkeypatch):
    """A repo carrying a legacy flatfield_profiles table + a per-image edit pointing at one."""
    monkeypatch.setattr(ffstore.APP_CONFIG, "flatfield_dir", str(tmp_path / "flatfield"), raising=False)
    monkeypatch.setattr(FlatFieldProfiles, "_bake_gain", staticmethod(lambda path: np.ones((8, 8, 3), dtype=np.float32)))

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()

    with sqlite3.connect(repo.edits_db_path) as conn:
        conn.execute("CREATE TABLE flatfield_profiles (name TEXT PRIMARY KEY, path TEXT, k1 REAL DEFAULT 0.0)")
        conn.executemany(
            "INSERT INTO flatfield_profiles (name, path, k1) VALUES (?, ?, ?)",
            [("rig-a", str(tmp_path / "a.dng"), -0.05), ("rig-b", str(tmp_path / "b.dng"), 0.0)],
        )
        edit = {"apply": True, "reference_path": str(tmp_path / "a.dng"), "k1": -0.05, "density": 1.0}
        conn.execute(
            "INSERT INTO file_settings (file_hash, settings_json) VALUES (?, ?)",
            ("hash1", json.dumps(edit)),
        )
    # References were baked from real files; touch them so create() sees them present.
    (tmp_path / "a.dng").write_bytes(b"x")
    (tmp_path / "b.dng").write_bytes(b"x")
    repo.save_global_setting("flatfield_active_profile", "rig-a")
    return repo


def _config(repo, file_hash):
    with sqlite3.connect(repo.edits_db_path) as conn:
        row = conn.execute("SELECT settings_json FROM file_settings WHERE file_hash = ?", (file_hash,)).fetchone()
    return json.loads(row[0])


def test_migration_bakes_and_repoints(legacy_repo):
    migrate_legacy_flatfield_profiles(legacy_repo)

    # Both legacy profiles are now npz files, keyed by opaque id.
    profiles = dict((name, pid) for pid, name in FlatFieldProfiles.list_profiles())
    assert set(profiles) == {"rig-a", "rig-b"}
    assert FlatFieldProfiles.get(profiles["rig-a"]).k1 == -0.05

    # The per-image edit now references rig-a by id, with the legacy path dropped.
    cfg = _config(legacy_repo, "hash1")
    assert cfg["profile_id"] == profiles["rig-a"]
    assert "reference_path" not in cfg

    # Active-profile setting remapped name -> id; legacy table gone; flag set.
    assert legacy_repo.get_global_setting("flatfield_active_profile") == profiles["rig-a"]
    with sqlite3.connect(legacy_repo.edits_db_path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='flatfield_profiles'").fetchone() is None
    assert legacy_repo.get_global_setting("flatfield_migrated_v2") is True


def test_migration_is_idempotent(legacy_repo):
    migrate_legacy_flatfield_profiles(legacy_repo)
    count_after_first = len(FlatFieldProfiles.list_profiles())

    migrate_legacy_flatfield_profiles(legacy_repo)  # second run must not re-bake
    assert len(FlatFieldProfiles.list_profiles()) == count_after_first


def test_migration_no_legacy_table_just_sets_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(ffstore.APP_CONFIG, "flatfield_dir", str(tmp_path / "flatfield"), raising=False)
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()

    migrate_legacy_flatfield_profiles(repo)
    assert repo.get_global_setting("flatfield_migrated_v2") is True
    assert FlatFieldProfiles.list_profiles() == []
