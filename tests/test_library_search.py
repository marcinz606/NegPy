import os
from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.models import MetadataConfig
from negpy.services.assets.library import LibraryWalkCache, iter_library_files, search_library
from negpy.services.assets.search import parse_query


@pytest.fixture
def tree(tmp_path):
    """Two roll folders plus a nested one, a stray non-image and an IR sidecar pair."""
    roll_a = tmp_path / "roll_a"
    roll_b = tmp_path / "roll_b" / "nested"
    roll_a.mkdir()
    roll_b.mkdir(parents=True)
    (roll_a / "IMG_0001.NEF").write_bytes(b"a")
    (roll_a / "IMG_0002.NEF").write_bytes(b"b")
    (roll_b / "scan.tif").write_bytes(b"c")
    (roll_b / "scan_ir.tif").write_bytes(b"d")  # sidecar of scan.tif
    (roll_a / "notes.txt").write_text("not an image")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "IMG_9999.NEF").write_bytes(b"e")
    return tmp_path


def _names(entries) -> set:
    return {e["name"] for e in entries}


def test_walk_finds_images_recursively(tree):
    assert _names(iter_library_files([str(tree)])) == {"IMG_0001.NEF", "IMG_0002.NEF", "scan.tif"}


def test_walk_skips_ir_sidecars_and_non_images(tree):
    names = _names(iter_library_files([str(tree)]))
    assert "scan_ir.tif" not in names
    assert "notes.txt" not in names


def test_walk_skips_dot_directories(tree):
    assert "IMG_9999.NEF" not in _names(iter_library_files([str(tree)]))


def test_walk_carries_stat_data(tree):
    entry = next(e for e in iter_library_files([str(tree)]) if e["name"] == "IMG_0001.NEF")
    assert entry["size"] == 1
    assert entry["mtime"] > 0
    assert os.path.isabs(entry["path"]) or entry["path"].startswith(str(tree))


def test_nested_root_is_not_walked_twice(tree):
    entries = list(iter_library_files([str(tree), str(tree / "roll_a")]))
    assert len(entries) == len({e["path"] for e in entries})


def test_missing_root_is_skipped_not_fatal(tree):
    assert _names(iter_library_files([str(tree / "does_not_exist"), str(tree / "roll_a")])) == {"IMG_0001.NEF", "IMG_0002.NEF"}


def test_search_matches_filename_without_any_edits(tree):
    files = list(iter_library_files([str(tree)]))
    hits = search_library(files, parse_query("img_0001"), {})
    assert [os.path.basename(p) for p in hits] == ["IMG_0001.NEF"]


def test_search_joins_edit_metadata_by_path(tree):
    files = list(iter_library_files([str(tree)]))
    edited = str(tree / "roll_a" / "IMG_0002.NEF")
    configs = {edited: replace(WorkspaceConfig(), metadata=MetadataConfig(film="Portra 400", film_iso=400))}

    assert search_library(files, parse_query("film:portra"), configs) == [edited]
    assert search_library(files, parse_query("iso:>=400"), configs) == [edited]
    assert search_library(files, parse_query("-film:portra"), configs) == [p for p in (f["path"] for f in files) if p != edited]


def test_search_resolves_marks_by_path(tree):
    files = list(iter_library_files([str(tree)]))
    keeper = str(tree / "roll_a" / "IMG_0001.NEF")

    assert search_library(files, parse_query("keeper:"), {}, {keeper: "keeper"}) == [keeper]
    assert search_library(files, parse_query("rejected:"), {}, {keeper: "keeper"}) == []


def test_empty_query_returns_nothing_not_the_whole_archive(tree):
    files = list(iter_library_files([str(tree)]))
    assert search_library(files, parse_query(""), {}) == []


def test_walk_cache_reuses_one_traversal_until_roots_change(tree):
    cache = LibraryWalkCache()
    first = cache.files([str(tree)])
    assert cache.files([str(tree)]) is first

    assert cache.files([str(tree / "roll_a")]) is not first

    cache.invalidate()
    assert cache.files([str(tree / "roll_a")]) is not first


def test_walk_cache_sees_new_files_after_invalidate(tree):
    cache = LibraryWalkCache()
    before = len(cache.files([str(tree)]))
    (tree / "roll_a" / "IMG_0003.NEF").write_bytes(b"f")

    assert len(cache.files([str(tree)])) == before  # cached on purpose
    cache.invalidate()
    assert len(cache.files([str(tree)])) == before + 1


def test_folder_counts_are_one_listing_deep(tree):
    from negpy.services.assets.library import folder_counts

    assert folder_counts(str(tree / "roll_a")) == (2, 0)
    assert folder_counts(str(tree / "roll_b")) == (0, 1)  # the nested dir, no images of its own
    assert folder_counts(str(tree)) == (0, 2)


def test_folder_counts_ignore_ir_sidecars_and_dot_dirs(tree):
    from negpy.services.assets.library import folder_counts

    assert folder_counts(str(tree / "roll_b" / "nested")) == (1, 0)  # scan.tif, not scan_ir.tif
    assert ".hidden" not in str(folder_counts(str(tree)))


def test_folder_counts_on_a_missing_path_are_zero_not_fatal(tree):
    from negpy.services.assets.library import folder_counts

    assert folder_counts(str(tree / "nope")) == (0, 0)


def test_count_summaries_read_naturally():
    from negpy.services.assets.library import summarize_counts

    assert summarize_counts(1, 0) == "1 photo"
    assert summarize_counts(36, 2) == "36 photos · 2 folders"
    assert summarize_counts(0, 0) == "empty"
