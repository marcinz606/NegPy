"""Interior-sampled content hash and the rehome of edits saved under the old digest."""

import hashlib
import os
from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.workers.render import AssetDiscoveryWorker
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.image.logic import calculate_file_hash, file_hashes
from negpy.services.assets.hash_migration import blank_ambiguous_legacy_hashes

MIB = 1024 * 1024


def _legacy_digest(path: str) -> str:
    """The pre-interior algorithm, spelled out so the compatibility slot is pinned
    against an independent implementation rather than against itself."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(MIB))
        if size > 2 * MIB:
            f.seek(-MIB, os.SEEK_END)
            h.update(f.read(MIB))
    return h.hexdigest()


def _twin(tmp_path, name: str, filler: bytes) -> str:
    """A 5 MiB file with a fixed 1 MiB head and 1 MiB tail — only the middle varies.
    Mirrors samples/conflict/: two scans of one frame share header and trailer."""
    p = tmp_path / name
    p.write_bytes(b"H" * MIB + filler * (3 * MIB) + b"T" * MIB)
    return str(p)


def test_same_head_tail_different_middle_now_distinct(tmp_path):
    a = _twin(tmp_path, "a.dng", b"\x01")
    b = _twin(tmp_path, "b.dng", b"\x02")

    a_cur, a_legacy = file_hashes(a)
    b_cur, b_legacy = file_hashes(b)

    assert a_legacy == b_legacy  # the bug: old digest could not tell them apart
    assert a_cur != b_cur


def test_small_files_keep_their_legacy_identity(tmp_path):
    p = tmp_path / "small.tif"
    p.write_bytes(b"x" * (2 * MIB))  # no interior to sample

    current, legacy = file_hashes(str(p))
    assert current == legacy == _legacy_digest(str(p))


def test_legacy_slot_matches_the_old_algorithm(tmp_path):
    path = _twin(tmp_path, "big.dng", b"\x03")
    current, legacy = file_hashes(path)

    assert legacy == _legacy_digest(path)
    assert current != legacy  # interior samples changed the identity


def test_hash_is_deterministic_and_sha256_shaped(tmp_path):
    path = _twin(tmp_path, "det.dng", b"\x04")
    assert file_hashes(path) == file_hashes(path)
    assert calculate_file_hash(path) == file_hashes(path)[0]
    assert len(calculate_file_hash(path)) == 64


def test_missing_file_yields_err_sentinel_and_empty_legacy(tmp_path):
    current, legacy = file_hashes(str(tmp_path / "nope.dng"))
    assert current.startswith("err_")
    assert legacy == ""


@pytest.fixture()
def repo(tmp_path):
    r = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    r.initialize()
    return r


def _session(repo) -> DesktopSessionManager:
    return DesktopSessionManager(repo)


def test_rehome_moves_settings_history_and_mark(repo):
    cfg = WorkspaceConfig(exposure=ExposureConfig(density=0.42))
    repo.save_file_settings("v1", cfg, file_path="/p/a.dng")
    repo.save_history_step("v1", 0, cfg)
    repo.save_history_step("v1", 1, cfg)
    repo.save_file_mark("v1", "keeper")

    _session(repo).add_files([], validated_info=[{"name": "a.dng", "path": "/p/a.dng", "hash": "v2", "legacy_hash": "v1"}])

    moved = repo.load_file_settings("v2")
    assert moved is not None and moved.exposure.density == 0.42
    assert repo.load_file_settings("v1") is None
    assert [i for i, _ in repo.load_all_history("v2")] == [0, 1]
    assert repo.load_all_history("v1") == []
    assert repo.load_file_marks() == {"v2": "keeper"}


def test_migrated_mark_lands_on_the_asset(repo):
    repo.save_file_settings("v1", WorkspaceConfig(), file_path="/p/a.dng")
    repo.save_file_mark("v1", "excluded")

    session = _session(repo)
    session.add_files([], validated_info=[{"name": "a.dng", "path": "/p/a.dng", "hash": "v2", "legacy_hash": "v1"}])

    # The overlay reads marks after the rehome, so the migrated flag is picked up
    # in the same pass rather than one import later.
    assert session.state.uploaded_files[0]["excluded"] is True


def test_rehome_never_clobbers_a_live_edit(repo):
    repo.save_file_settings("v1", WorkspaceConfig(exposure=ExposureConfig(density=0.11)), file_path="/p/a.dng")
    repo.save_file_settings("v2", WorkspaceConfig(exposure=ExposureConfig(density=0.99)), file_path="/p/a.dng")

    _session(repo).add_files([], validated_info=[{"name": "a.dng", "path": "/p/a.dng", "hash": "v2", "legacy_hash": "v1"}])

    kept = repo.load_file_settings("v2")
    assert kept is not None and kept.exposure.density == 0.99


def test_rehome_of_one_half_leaves_the_sibling_alone(repo):
    repo.save_file_settings("v1#1", WorkspaceConfig(exposure=ExposureConfig(density=0.11)), file_path="/p/a.tif")
    repo.save_file_settings("v1#2", WorkspaceConfig(exposure=ExposureConfig(density=0.22)), file_path="/p/a.tif")

    _session(repo).add_files(
        [],
        validated_info=[
            {"name": "a.tif [1]", "path": "/p/a.tif", "hash": "v2#1", "legacy_hash": "v1#1", "half": 1},
            {"name": "a.tif [2]", "path": "/p/a.tif", "hash": "v2#2", "legacy_hash": "v1#2", "half": 2},
        ],
    )

    first, second = repo.load_file_settings("v2#1"), repo.load_file_settings("v2#2")
    assert first is not None and first.exposure.density == 0.11
    assert second is not None and second.exposure.density == 0.22


def test_expand_half_frames_suffixes_the_legacy_hash(monkeypatch):
    monkeypatch.setattr("negpy.services.assets.half_frame.detect_split_x_for_file", lambda _p: 0.5)
    out = AssetDiscoveryWorker()._expand_half_frames([{"name": "a.tif", "path": "/p/a.tif", "hash": "v2", "legacy_hash": "v1"}])

    assert [a["hash"] for a in out] == ["v2#1", "v2#2"]
    assert [a["legacy_hash"] for a in out] == ["v1#1", "v1#2"]


def test_restored_stitch_carries_no_legacy_hash(tmp_path):
    part = tmp_path / "part.tif"
    part.write_bytes(b"x")
    entry = {
        "hash": "composite",
        "paths": [str(part)],
        "transforms": [(1.0, 0.0, 0.0)],
        "canvas": (10, 10),
        "sizes": [(5, 5)],
    }
    primary = {"name": "a.tif", "path": "/p/a.tif", "hash": "v2", "legacy_hash": "v1"}

    out = AssetDiscoveryWorker()._attach_restored_stitches([primary], {"/p/a.tif": entry})

    # Inheriting the part's legacy digest would rehome that part's edit onto the composite.
    assert out[0]["hash"] == "composite"
    assert out[0]["legacy_hash"] == ""


def test_ambiguous_legacy_hash_is_dropped():
    assets = [
        {"path": "/p/a.dng", "hash": "v2a", "legacy_hash": "shared"},
        {"path": "/p/b.dng", "hash": "v2b", "legacy_hash": "shared"},
        {"path": "/p/c.dng", "hash": "v2c", "legacy_hash": "own"},
    ]
    blank_ambiguous_legacy_hashes(assets)

    # "shared" can't say whose edits it holds, so neither file claims them.
    assert [a["legacy_hash"] for a in assets] == ["", "", "own"]


def test_conflict_samples_get_a_blank_legacy_hash(tmp_path):
    a = _twin(tmp_path, "ice-Normal.dng", b"\x01")
    b = _twin(tmp_path, "ice-Fine.dng", b"\x02")
    assets = [{"path": p, "hash": file_hashes(p)[0], "legacy_hash": file_hashes(p)[1]} for p in (a, b)]

    assert assets[0]["hash"] != assets[1]["hash"]
    blank_ambiguous_legacy_hashes(assets)
    assert [x["legacy_hash"] for x in assets] == ["", ""]


def test_identical_content_still_dedups_but_is_logged(caplog):
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.side_effect = lambda key, default=None: default
    repo.load_file_marks.return_value = {}
    session = DesktopSessionManager(repo)

    with caplog.at_level("INFO"):
        session.add_files(
            [],
            validated_info=[
                {"name": "a.dng", "path": "/p/a.dng", "hash": "same", "legacy_hash": ""},
                {"name": "copy.dng", "path": "/q/copy.dng", "hash": "same", "legacy_hash": ""},
            ],
        )

    assert [f["path"] for f in session.state.uploaded_files] == ["/p/a.dng"]
    assert "/q/copy.dng" in caplog.text
