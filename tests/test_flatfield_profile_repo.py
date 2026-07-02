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
