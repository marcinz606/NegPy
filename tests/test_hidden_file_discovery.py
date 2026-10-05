"""Dot-prefixed files never become assets.

`.DS_Store` and a macOS `._frame.dng` AppleDouble fork both carry the source's extension,
so an extension-only filter lets them through. The fork is readable, so it hashes as a
valid asset and only fails later at decode, as "Unsupported file format or not RAW file".
"""

from __future__ import annotations

import os

import pytest
from PIL import Image

from negpy.desktop.workers.render import AssetDiscoveryTask, AssetDiscoveryWorker
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS, is_hidden_path
from negpy.services.assets.library import folder_counts, iter_library_files


def _write_tiff(path) -> None:
    Image.new("RGB", (4, 4)).save(path)


def _discover(paths: list[str]) -> list[dict]:
    worker = AssetDiscoveryWorker()
    out: list[list[dict]] = []
    worker.finished.connect(out.append)
    worker.process(AssetDiscoveryTask(paths=list(paths), supported_extensions=tuple(SUPPORTED_RAW_EXTENSIONS)))
    assert out, "discovery finished without emitting"
    return out[0]


@pytest.mark.parametrize(
    "name,hidden",
    [
        ("frame.tif", False),
        ("frame_ir.tif", False),
        (".DS_Store", True),
        ("._frame.dng", True),
        (".hidden.tif", True),
    ],
)
def test_hidden_path_predicate(name: str, hidden: bool) -> None:
    assert is_hidden_path(name) is hidden


def test_discovery_skips_hidden_files_in_a_folder(tmp_path) -> None:
    _write_tiff(tmp_path / "frame.tif")
    (tmp_path / "._frame.tif").write_bytes(b"AppleDouble")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")

    assert [os.path.basename(a["path"]) for a in _discover([str(tmp_path)])] == ["frame.tif"]


def test_discovery_skips_a_hidden_file_named_directly(tmp_path) -> None:
    (tmp_path / "._frame.dng").write_bytes(b"AppleDouble")

    assert _discover([str(tmp_path / "._frame.dng")]) == []


def test_watcher_skips_hidden_files(tmp_path) -> None:
    _write_tiff(tmp_path / "frame.tif")
    (tmp_path / "._frame.tif").write_bytes(b"AppleDouble")

    found = FolderWatchService.scan_for_new_files(str(tmp_path), set())

    assert sorted(os.path.basename(p) for p in found) == ["frame.tif"]


def test_library_walk_skips_hidden_files(tmp_path) -> None:
    _write_tiff(tmp_path / "frame.tif")
    (tmp_path / "._frame.tif").write_bytes(b"AppleDouble")

    assert {e["name"] for e in iter_library_files([str(tmp_path)])} == {"frame.tif"}


def test_folder_counts_ignore_hidden_files(tmp_path) -> None:
    _write_tiff(tmp_path / "frame.tif")
    (tmp_path / "._frame.tif").write_bytes(b"AppleDouble")

    assert folder_counts(str(tmp_path)) == (1, 0)
