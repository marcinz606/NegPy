import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from negpy.features.exposure.normalization import BOUNDS_VERSION, analyze_log_exposure_bounds_from_log, prefilter_log_grid
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.migrations import bounds
from negpy.services.assets.migrations.bounds import migrate_stale_frame_bounds

_MEASURED = {"local_floors": [-1.1, -1.2, -1.0], "local_ceils": [-0.2, -0.4, -0.6]}
_ZERO = [0.0, 0.0, 0.0]

# Bounds the analysis measures on _synthetic_negative() at each (luma_range_clip, color_range_clip),
# under BOUNDS_VERSION. A change to them needs a bump; then record the new values here.
_RECORDED_VERSION = 3
_RECORDED = {
    (0.0, 1.0): ([-1.135058, -1.168505, -1.038996], [-0.247402, -0.451715, -0.607022]),
    (-0.3, 0.5): ([-1.512205, -1.48816, -1.244211], [0.135708, -0.13491, -0.408953]),
}


@pytest.fixture
def repo(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def _insert(repo, table, settings, **key):
    row = {**key, "settings_json": json.dumps(settings)}
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", tuple(row.values()))


def _settings(repo, table, **key):
    where = " AND ".join(f"{k} = ?" for k in key)
    with closing(sqlite3.connect(repo.edits_db_path)) as conn:
        return json.loads(conn.execute(f"SELECT settings_json FROM {table} WHERE {where}", tuple(key.values())).fetchone()[0])


def test_saved_bounds_are_dropped_in_every_edit_table(repo):
    _insert(repo, "file_settings", _MEASURED, file_hash="h1")
    _insert(repo, "edit_history", _MEASURED, file_hash="h1", step_index=0)
    _insert(repo, "work_prints", _MEASURED, file_hash="h1", name="v1", created_at=0.0)

    migrate_stale_frame_bounds(repo)

    for table, key in (("file_settings", {}), ("edit_history", {"step_index": 0}), ("work_prints", {"name": "v1"})):
        saved = _settings(repo, table, file_hash="h1", **key)
        assert saved["local_floors"] == _ZERO and saved["local_ceils"] == _ZERO
    assert repo.get_global_setting("bounds_version") == BOUNDS_VERSION


def test_locked_bounds_are_kept(repo):
    _insert(repo, "file_settings", {**_MEASURED, "lock_bounds": True}, file_hash="h1")

    migrate_stale_frame_bounds(repo)

    assert _settings(repo, "file_settings", file_hash="h1")["local_floors"] == _MEASURED["local_floors"]


def test_bounds_saved_under_the_current_version_are_kept(repo):
    migrate_stale_frame_bounds(repo)
    _insert(repo, "file_settings", _MEASURED, file_hash="h1")

    migrate_stale_frame_bounds(repo)

    assert _settings(repo, "file_settings", file_hash="h1")["local_floors"] == _MEASURED["local_floors"]


def test_a_version_bump_drops_them_again(repo, monkeypatch):
    migrate_stale_frame_bounds(repo)
    _insert(repo, "file_settings", _MEASURED, file_hash="h1")
    monkeypatch.setattr(bounds, "BOUNDS_VERSION", BOUNDS_VERSION + 1)

    migrate_stale_frame_bounds(repo)

    assert _settings(repo, "file_settings", file_hash="h1")["local_floors"] == _ZERO


def _synthetic_negative() -> np.ndarray:
    """A thin-to-dense ramp with texture, each channel over a different transmittance range."""
    y, x = np.mgrid[0:120, 0:180].astype(np.float32)
    ramp = (x / 179.0)[..., None]
    texture = (0.5 + 0.5 * np.sin(y / 7.0) * np.cos(x / 11.0))[..., None]
    lo = np.array([0.02, 0.04, 0.08], np.float32)
    hi = np.array([0.55, 0.35, 0.25], np.float32)
    return lo + (hi - lo) * (0.7 * ramp + 0.3 * texture)


def test_a_changed_bounds_analysis_bumps_bounds_version():
    assert BOUNDS_VERSION == _RECORDED_VERSION, "BOUNDS_VERSION moved: record the new bounds in _RECORDED"
    grid = prefilter_log_grid(_synthetic_negative(), None, 0.05)
    for (clip, color), (floors, ceils) in _RECORDED.items():
        measured = analyze_log_exposure_bounds_from_log(grid, None, 0.0, percentile_clip=clip, color_clip=color)
        np.testing.assert_allclose(
            [measured.floors, measured.ceils], [floors, ceils], atol=1e-5, err_msg="bounds analysis changed: bump BOUNDS_VERSION"
        )
