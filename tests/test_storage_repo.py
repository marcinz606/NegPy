import sqlite3
from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.models import MetadataConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.features.process.models import ProcessMode


def _repo(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


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
    values = {"mode": ProcessMode.C41, "clip": 0.01, "matrix": [[1, 0], [0, 1]], "flag": False}

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


def _config(film: str) -> WorkspaceConfig:
    return replace(WorkspaceConfig(), metadata=MetadataConfig(film=film))


def test_load_file_settings_many_returns_only_saved_hashes(tmp_path):
    repo = _repo(tmp_path)
    repo.save_file_settings("h1", _config("Portra"), file_path="/a/1.nef")
    repo.save_file_settings("h2", _config("Velvia"), file_path="/a/2.nef")

    loaded = repo.load_file_settings_many(["h1", "h2", "missing"])
    assert set(loaded) == {"h1", "h2"}
    assert loaded["h1"].metadata.film == "Portra"
    assert repo.load_file_settings_many([]) == {}


def test_load_file_settings_many_handles_more_than_one_chunk(tmp_path):
    repo = _repo(tmp_path)
    hashes = [f"h{i}" for i in range(1200)]  # over the 500-per-query chunk
    for h in hashes:
        repo.save_file_settings(h, _config("Portra"), file_path=f"/a/{h}.nef")

    assert len(repo.load_file_settings_many(hashes)) == 1200


def test_load_settings_by_path_skips_rows_without_a_path(tmp_path):
    repo = _repo(tmp_path)
    repo.save_file_settings("h1", _config("Portra"), file_path="/a/1.nef")
    repo.save_file_settings("h2", _config("Velvia"))  # legacy row, no path

    by_path = repo.load_settings_by_path()
    assert set(by_path) == {"/a/1.nef"}
    assert by_path["/a/1.nef"].metadata.film == "Portra"


def test_delete_file_settings_takes_the_edit_history_and_work_prints(tmp_path):
    repo = _repo(tmp_path)
    for h in ("h1", "h2"):
        repo.save_file_settings(h, _config("Portra"), file_path="/a/1.nef")
        repo.save_history_step(h, 0, _config("Portra"))
        repo.save_work_print(h, "print", _config("Portra"))
    repo.save_file_mark("h1", "keeper", file_path="/a/1.nef")

    repo.delete_file_settings("h1")

    assert repo.load_file_settings("h1") is None
    assert repo.load_history_step("h1", 0) is None
    assert repo.list_work_prints("h1") == []
    assert repo.load_file_marks() == {"h1": "keeper"}  # a triage mark is not an edit
    assert repo.load_file_settings("h2") is not None
    assert repo.list_work_prints("h2") == ["print"]


def test_file_marks_are_resolvable_by_path(tmp_path):
    repo = _repo(tmp_path)
    repo.save_file_mark("h1", "keeper", file_path="/a/1.nef")
    repo.save_file_mark("h2", "excluded", file_path="/a/2.nef")
    repo.save_file_mark("h3", "keeper")  # written without a path

    assert repo.load_file_marks_by_path() == {"/a/1.nef": "keeper", "/a/2.nef": "excluded"}
    assert repo.load_file_marks() == {"h1": "keeper", "h2": "excluded", "h3": "keeper"}

    repo.save_file_mark("h1", None)
    assert repo.load_file_marks_by_path() == {"/a/2.nef": "excluded"}


def test_initialize_enables_wal(tmp_path):
    repo = _repo(tmp_path)
    for path in (repo.edits_db_path, repo.settings_db_path):
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()
