import sqlite3
from dataclasses import replace

import numpy as np

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


def test_load_embeddings_for_returns_only_saved_hashes(tmp_path):
    repo = _repo(tmp_path)
    repo.save_embedding("h1", np.array([1.0, 2.0], dtype=np.float32), "v1")
    repo.save_embedding("h2", np.array([3.0, 4.0], dtype=np.float32), "v1")

    loaded = repo.load_embeddings_for(["h1", "h2", "missing"], "v1")
    assert set(loaded) == {"h1", "h2"}
    assert np.array_equal(loaded["h1"], np.array([1.0, 2.0], dtype=np.float32))
    assert repo.load_embeddings_for([], "v1") == {}


def test_load_embeddings_for_ignores_a_different_model_version(tmp_path):
    """A model swap must not score a frame against a vector computed by the model
    being retired -- the version is part of the identity, not metadata on the side."""
    repo = _repo(tmp_path)
    repo.save_embedding("h1", np.array([1.0, 2.0], dtype=np.float32), "v1")

    assert repo.load_embeddings_for(["h1"], "v2") == {}


def test_load_embeddings_for_handles_more_than_one_chunk(tmp_path):
    repo = _repo(tmp_path)
    hashes = [f"h{i}" for i in range(1200)]
    for h in hashes:
        repo.save_embedding(h, np.zeros(4, dtype=np.float32), "v1")

    assert len(repo.load_embeddings_for(hashes, "v1")) == 1200


def test_save_embedding_overwrites_the_previous_vector(tmp_path):
    repo = _repo(tmp_path)
    repo.save_embedding("h1", np.array([1.0, 0.0], dtype=np.float32), "v1")
    repo.save_embedding("h1", np.array([0.0, 1.0], dtype=np.float32), "v1")

    loaded = repo.load_embeddings_for(["h1"], "v1")
    assert np.array_equal(loaded["h1"], np.array([0.0, 1.0], dtype=np.float32))


def test_delete_file_settings_also_takes_the_embedding(tmp_path):
    repo = _repo(tmp_path)
    repo.save_file_settings("h1", _config("Portra"), file_path="/a/1.nef")
    repo.save_embedding("h1", np.array([1.0, 2.0], dtype=np.float32), "v1")

    repo.delete_file_settings("h1")

    assert repo.load_embeddings_for(["h1"], "v1") == {}


def test_load_all_embeddings_returns_path_and_vector_for_every_row(tmp_path):
    repo = _repo(tmp_path)
    repo.save_embedding("h1", np.array([1.0, 0.0], dtype=np.float32), "v1", "/a/1.nef")
    repo.save_embedding("h2", np.array([0.0, 1.0], dtype=np.float32), "v1", "/a/2.nef")

    loaded = repo.load_all_embeddings("v1")

    assert set(loaded) == {"h1", "h2"}
    path, vec = loaded["h1"]
    assert path == "/a/1.nef"
    assert np.array_equal(vec, np.array([1.0, 0.0], dtype=np.float32))


def test_load_all_embeddings_excludes_other_model_versions(tmp_path):
    repo = _repo(tmp_path)
    repo.save_embedding("h1", np.array([1.0, 0.0], dtype=np.float32), "v1", "/a/1.nef")
    repo.save_embedding("h2", np.array([0.0, 1.0], dtype=np.float32), "v2", "/a/2.nef")

    assert set(repo.load_all_embeddings("v1")) == {"h1"}


def test_load_all_embeddings_is_empty_for_an_unknown_version(tmp_path):
    repo = _repo(tmp_path)
    assert repo.load_all_embeddings("v1") == {}


def test_save_embedding_without_a_path_still_round_trips(tmp_path):
    """Callers that predate the file_path column (or the in-session path, which never
    needs one back) can still just not pass it."""
    repo = _repo(tmp_path)
    repo.save_embedding("h1", np.array([1.0, 0.0], dtype=np.float32), "v1")

    path, vec = repo.load_all_embeddings("v1")["h1"]
    assert path == ""
    assert np.array_equal(vec, np.array([1.0, 0.0], dtype=np.float32))


def test_initialize_enables_wal(tmp_path):
    repo = _repo(tmp_path)
    for path in (repo.edits_db_path, repo.settings_db_path):
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()


def test_global_setting_reads_come_from_memory_after_the_first(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    repo.save_global_setting("a", 1)
    assert repo.get_global_setting("a") == 1
    connects = []
    real = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: connects.append(a) or real(*a, **k))
    assert repo.get_global_setting("a") == 1
    assert repo.get_global_setting("missing", default=7) == 7
    assert connects == []


def test_global_setting_cache_follows_writes_and_persists(tmp_path):
    repo = _repo(tmp_path)
    assert repo.get_global_setting("rolls") is None
    repo.save_global_setting("rolls", {"r1": {"members": ["a"]}})
    repo.save_global_settings({"rolls": {"r1": {"members": ["a", "b"]}}, "x": 2})
    assert repo.get_global_setting("rolls") == {"r1": {"members": ["a", "b"]}}
    reopened = StorageRepository(repo.edits_db_path, repo.settings_db_path)
    assert reopened.get_global_setting("rolls") == {"r1": {"members": ["a", "b"]}}
    assert reopened.get_global_setting("x") == 2


def test_mutating_a_global_setting_result_does_not_reach_the_cache(tmp_path):
    repo = _repo(tmp_path)
    repo.save_global_setting("rolls", {"r1": {"members": ["a"]}})
    got = repo.get_global_setting("rolls")
    got["r1"]["members"].append("b")
    assert repo.get_global_setting("rolls") == {"r1": {"members": ["a"]}}


def test_reset_everything_empties_the_global_setting_cache(tmp_path):
    repo = _repo(tmp_path)
    repo.save_global_setting("a", 1)
    assert repo.get_global_setting("a") == 1
    repo.reset_everything()
    assert repo.get_global_setting("a") is None


def test_load_all_history_reuses_parsed_steps(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = WorkspaceConfig()
    for i in range(3):
        repo.save_history_step("h", i, replace(base, exposure=replace(base.exposure, density=1.0 + i)))
    first = repo.load_all_history("h")
    parses = []
    real = WorkspaceConfig.from_flat_dict
    monkeypatch.setattr(WorkspaceConfig, "from_flat_dict", staticmethod(lambda d: parses.append(1) or real(d)))
    repo.save_history_step("h", 3, replace(base, exposure=replace(base.exposure, density=9.0)))
    second = repo.load_all_history("h")
    assert len(parses) == 1
    assert [c for _, c in second[:3]] == [c for _, c in first]
    assert second[3][1].exposure.density == 9.0
