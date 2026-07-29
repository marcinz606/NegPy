from __future__ import annotations

import pytest

from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.constants import is_ir_sidecar_path


def _touch(tmp_path, *names: str) -> None:
    for name in names:
        (tmp_path / name).write_bytes(b"")


@pytest.mark.parametrize(
    "name,hidden",
    [
        ("frame1.tif", False),
        ("frame1_IR.tif", True),
        ("frame1_IR_VALID.tif", True),
        ("frame1_ir.tiff", True),
        ("frame1_IR.jpg", False),
        ("lonely_IR.tif", False),
        ("_IR.tif", False),
    ],
)
def test_ir_sidecar_predicate(tmp_path, name: str, hidden: bool) -> None:
    _touch(tmp_path, "frame1.tif", "frame1_IR.tif", "frame1_IR_VALID.tif", "frame1_ir.tiff", "frame1_IR.jpg", "lonely_IR.tif", "_IR.tif")
    assert is_ir_sidecar_path(str(tmp_path / name)) is hidden


def test_main_file_case_mismatch_still_hides_sidecar(tmp_path) -> None:
    _touch(tmp_path, "Frame2.TIF", "frame2_ir.tif")
    assert is_ir_sidecar_path(str(tmp_path / "frame2_ir.tif")) is True


def test_watcher_skips_ir_sidecars(tmp_path) -> None:
    _touch(tmp_path, "frame1.tif", "frame1_IR.tif", "frame1_IR_VALID.tif", "orphan_IR.tif")

    found = FolderWatchService.scan_for_new_files(str(tmp_path), set())

    assert sorted(p.rsplit("/", 1)[-1] for p in found) == ["frame1.tif", "orphan_IR.tif"]
