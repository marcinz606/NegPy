import json
import sqlite3
from contextlib import closing

import pytest

from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.migrations.normalization_roll import migrate_legacy_normalization_rolls


@pytest.fixture
def legacy_repo(tmp_path):
    """A repo carrying a legacy normalization_rolls table, name-matched to two library
    rolls plus one orphan name no roll owns."""
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()

    rolls.create_virtual_roll(repo, "Tri-X", [])
    rolls.create_virtual_roll(repo, "Portra 400", [])

    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        conn.execute("CREATE TABLE normalization_rolls (name TEXT PRIMARY KEY, floors_json TEXT, ceils_json TEXT, cast_json TEXT)")
        conn.executemany(
            "INSERT INTO normalization_rolls (name, floors_json, ceils_json, cast_json) VALUES (?, ?, ?, ?)",
            [
                ("Tri-X", json.dumps([0.1, 0.1, 0.1]), json.dumps([0.9, 0.9, 0.9]), json.dumps([0.01, 0.0, -0.01])),
                ("Portra 400", json.dumps([0.05, 0.05, 0.05]), json.dumps([0.95, 0.95, 0.95]), None),
                ("Deleted Roll", json.dumps([0.2, 0.2, 0.2]), json.dumps([0.8, 0.8, 0.8]), None),
            ],
        )
    return repo


def _roll_id(repo, name):
    return next(rid for rid, entry in rolls.all_rolls_sorted(repo) if entry.get("name") == name)


def test_migration_copies_each_row_onto_its_matching_roll(legacy_repo):
    migrate_legacy_normalization_rolls(legacy_repo)

    tri_x = rolls.roll_normalization(legacy_repo, _roll_id(legacy_repo, "Tri-X"))
    assert tri_x == {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.01, 0.0, -0.01), "outliers": ()}


def test_migration_defaults_a_missing_cast_to_zero(legacy_repo):
    migrate_legacy_normalization_rolls(legacy_repo)

    portra = rolls.roll_normalization(legacy_repo, _roll_id(legacy_repo, "Portra 400"))
    assert portra["cast"] == (0.0, 0.0, 0.0)


def test_migration_creates_no_roll_for_a_name_that_matches_none(legacy_repo):
    migrate_legacy_normalization_rolls(legacy_repo)

    for _roll, entry in rolls.all_rolls_sorted(legacy_repo):
        if entry.get("name") == "Deleted Roll":
            pytest.fail("orphan row must not create a roll")


def test_migration_keeps_an_unmatched_row_and_stays_pending(legacy_repo):
    """Rolls are recognized on import, so a name that matches none today can match one
    tomorrow. The row waits; dropping it would discard the only copy of that baseline."""
    migrate_legacy_normalization_rolls(legacy_repo)

    with closing(sqlite3.connect(legacy_repo.edits_db_path)) as conn:
        rows = conn.execute("SELECT name FROM normalization_rolls").fetchall()
    assert [r[0] for r in rows] == ["Deleted Roll"]
    assert legacy_repo.get_global_setting("normalization_rolls_migrated_v1") is None


def test_migration_survives_a_first_launch_with_no_rolls_yet(tmp_path):
    """The upgrade path: nothing has been imported as a roll, so nothing matches and
    every baseline must still be there once the user imports the folders."""
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        conn.execute("CREATE TABLE normalization_rolls (name TEXT PRIMARY KEY, floors_json TEXT, ceils_json TEXT, cast_json TEXT)")
        conn.execute(
            "INSERT INTO normalization_rolls VALUES (?, ?, ?, ?)",
            ("Tri-X", json.dumps([0.1, 0.1, 0.1]), json.dumps([0.9, 0.9, 0.9]), None),
        )

    migrate_legacy_normalization_rolls(repo)

    rolls.create_virtual_roll(repo, "Tri-X", [])
    migrate_legacy_normalization_rolls(repo)

    tri_x = rolls.roll_normalization(repo, _roll_id(repo, "Tri-X"))
    assert tri_x["floors"] == (0.1, 0.1, 0.1)
    assert repo.get_global_setting("normalization_rolls_migrated_v1") is True
    with closing(sqlite3.connect(repo.edits_db_path)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='normalization_rolls'").fetchone() is None


def test_migration_is_idempotent(legacy_repo):
    migrate_legacy_normalization_rolls(legacy_repo)
    migrate_legacy_normalization_rolls(legacy_repo)  # second run must not raise or re-copy a migrated row

    tri_x = rolls.roll_normalization(legacy_repo, _roll_id(legacy_repo, "Tri-X"))
    assert tri_x["floors"] == (0.1, 0.1, 0.1)


def test_migration_no_legacy_table_just_sets_the_flag(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()

    migrate_legacy_normalization_rolls(repo)

    assert repo.get_global_setting("normalization_rolls_migrated_v1") is True


def test_migration_closes_the_connection_it_opens(legacy_repo, monkeypatch):
    """Mirrors test_flatfield_migration's leak guard: sqlite3.connect() used as a bare
    context manager commits on exit but never closes."""
    edits_db = legacy_repo.edits_db_path
    real_connect = sqlite3.connect
    opened = []

    def tracking_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if str(database) == str(edits_db):
            opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    migrate_legacy_normalization_rolls(legacy_repo)
    monkeypatch.undo()

    assert opened, "migration opened no connection to the edits DB"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
