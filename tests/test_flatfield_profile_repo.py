import os
import sqlite3

from negpy.infrastructure.storage.repository import StorageRepository


def _repo(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def test_profile_k1_round_trip(tmp_path):
    repo = _repo(tmp_path)
    repo.save_flatfield_profile("rig-a", "/refs/flat.dng", k1=-0.08)
    assert repo.get_flatfield_profile("rig-a") == ("/refs/flat.dng", -0.08)

    # Default k1 when omitted.
    repo.save_flatfield_profile("rig-b", "/refs/b.dng")
    assert repo.get_flatfield_profile("rig-b") == ("/refs/b.dng", 0.0)

    assert repo.get_flatfield_profile("missing") is None


def test_k1_column_migration_on_legacy_db(tmp_path):
    """A DB created before the k1 column must gain it (defaulting to 0.0) on init."""
    edits = str(tmp_path / "edits.db")
    conn = sqlite3.connect(edits)
    try:
        conn.execute("CREATE TABLE flatfield_profiles (name TEXT PRIMARY KEY, path TEXT)")
        conn.execute("INSERT INTO flatfield_profiles (name, path) VALUES (?, ?)", ("legacy", "/refs/old.dng"))
        conn.commit()
    finally:
        conn.close()

    repo = StorageRepository(edits, str(tmp_path / "settings.db"))
    repo.initialize()  # runs the ALTER TABLE ADD COLUMN migration

    assert repo.get_flatfield_profile("legacy") == ("/refs/old.dng", 0.0)
    repo.save_flatfield_profile("legacy", "/refs/old.dng", k1=0.15)
    assert repo.get_flatfield_profile("legacy") == ("/refs/old.dng", 0.15)
    assert os.path.exists(edits)


def test_save_global_settings_batch_round_trip(tmp_path):
    repo = _repo(tmp_path)
    values = {"a": 1, "b": [1, 2], "c": {"x": "y"}, "d": True, "e": "text"}
    repo.save_global_settings(values)
    for key, value in values.items():
        assert repo.get_global_setting(key) == value

    # INSERT OR REPLACE semantics, same as the single-key path.
    repo.save_global_settings({"a": 99})
    assert repo.get_global_setting("a") == 99


def test_save_global_settings_matches_single_write_path(tmp_path):
    repo_batch = _repo(tmp_path / "batch")
    repo_single = _repo(tmp_path / "single")
    values = {"mode": "C41", "clip": 0.01, "matrix": [[1, 0], [0, 1]], "flag": False}

    repo_batch.save_global_settings(values)
    for key, value in values.items():
        repo_single.save_global_setting(key, value)

    def rows(repo):
        conn = sqlite3.connect(repo.settings_db_path)
        try:
            return sorted(conn.execute("SELECT key, value_json FROM global_settings").fetchall())
        finally:
            conn.close()

    assert rows(repo_batch) == rows(repo_single)


def test_initialize_enables_wal(tmp_path):
    repo = _repo(tmp_path)
    for path in (repo.edits_db_path, repo.settings_db_path):
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()
