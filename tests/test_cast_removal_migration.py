import json
import sqlite3
from contextlib import closing

import pytest

from negpy.features.exposure.models import ExposureConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.migrations.cast_removal import _SHIPPED_CAST_STRENGTH, migrate_legacy_slide_cast_removal


@pytest.fixture
def repo(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def _insert(conn, table, **row):
    cols = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", tuple(row.values()))


def _settings(repo, table, **key):
    where = " AND ".join(f"{k} = ?" for k in key)
    with closing(sqlite3.connect(repo.edits_db_path)) as conn:
        row = conn.execute(f"SELECT settings_json FROM {table} WHERE {where}", tuple(key.values())).fetchone()
    return json.loads(row[0])


def test_the_mirrored_default_matches_the_dataclass():
    assert _SHIPPED_CAST_STRENGTH == float(ExposureConfig.cast_removal_strength)


def test_legacy_slide_at_shipped_default_is_zeroed(repo):
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        _insert(
            conn,
            "file_settings",
            file_hash="hash1",
            settings_json=json.dumps({"process_mode": "Transparency", "cast_removal_strength": 0.5}),
        )

    migrate_legacy_slide_cast_removal(repo)

    assert _settings(repo, "file_settings", file_hash="hash1")["cast_removal_strength"] == 0.0


def test_a_chosen_slide_strength_is_left_alone(repo):
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        _insert(
            conn,
            "file_settings",
            file_hash="hash1",
            settings_json=json.dumps({"process_mode": "Transparency", "cast_removal_strength": 0.8}),
        )

    migrate_legacy_slide_cast_removal(repo)

    assert _settings(repo, "file_settings", file_hash="hash1")["cast_removal_strength"] == 0.8


def test_a_negative_at_the_shipped_default_is_untouched(repo):
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        _insert(
            conn,
            "file_settings",
            file_hash="hash1",
            settings_json=json.dumps({"process_mode": "Color Negative", "cast_removal_strength": 0.5}),
        )

    migrate_legacy_slide_cast_removal(repo)

    assert _settings(repo, "file_settings", file_hash="hash1")["cast_removal_strength"] == 0.5


def test_the_legacy_e6_mode_name_is_swept_too(repo):
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        _insert(
            conn,
            "file_settings",
            file_hash="hash1",
            settings_json=json.dumps({"process_mode": "E-6", "cast_removal_strength": 0.5}),
        )

    migrate_legacy_slide_cast_removal(repo)

    assert _settings(repo, "file_settings", file_hash="hash1")["cast_removal_strength"] == 0.0


def test_edit_history_and_work_prints_are_swept_too(repo):
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        _insert(
            conn,
            "edit_history",
            file_hash="hash1",
            step_index=0,
            settings_json=json.dumps({"process_mode": "Transparency", "cast_removal_strength": 0.5}),
        )
        _insert(
            conn,
            "work_prints",
            file_hash="hash1",
            name="v1",
            created_at=0.0,
            settings_json=json.dumps({"process_mode": "Transparency", "cast_removal_strength": 0.5}),
        )

    migrate_legacy_slide_cast_removal(repo)

    assert _settings(repo, "edit_history", file_hash="hash1", step_index=0)["cast_removal_strength"] == 0.0
    assert _settings(repo, "work_prints", file_hash="hash1", name="v1")["cast_removal_strength"] == 0.0


def test_migration_sets_the_done_flag(repo):
    migrate_legacy_slide_cast_removal(repo)
    assert repo.get_global_setting("cast_removal_slide_migrated_v1") is True


def test_migration_does_not_rerun_on_a_later_deliberate_save(repo):
    """The whole point: once migrated, a value the user saves afterwards must survive
    every future load, unlike the per-load equality check this replaces."""
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        _insert(
            conn,
            "file_settings",
            file_hash="hash1",
            settings_json=json.dumps({"process_mode": "Transparency", "cast_removal_strength": 0.5}),
        )
    migrate_legacy_slide_cast_removal(repo)
    assert _settings(repo, "file_settings", file_hash="hash1")["cast_removal_strength"] == 0.0

    # The user dials it back to exactly 0.5 after the migration has already run.
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        conn.execute(
            "UPDATE file_settings SET settings_json = ? WHERE file_hash = ?",
            (json.dumps({"process_mode": "Transparency", "cast_removal_strength": 0.5}), "hash1"),
        )

    migrate_legacy_slide_cast_removal(repo)  # e.g. a second app launch

    assert _settings(repo, "file_settings", file_hash="hash1")["cast_removal_strength"] == 0.5
