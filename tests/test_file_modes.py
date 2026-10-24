"""Files NegPy writes through a temporary name take the umask's mode, not tempfile's 0600."""

import os
import pathlib
import stat
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from negpy.kernel.system import paths
from negpy.kernel.system.paths import apply_umask_mode

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX modes")

EXPECTED = 0o666 & ~paths._UMASK


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_apply_umask_mode_opens_a_tempfile_file(tmp_path):
    import tempfile

    with tempfile.NamedTemporaryFile(dir=tmp_path, delete=False) as tmp:
        assert _mode(tmp.name) == 0o600
    apply_umask_mode(tmp.name)
    assert _mode(tmp.name) == EXPECTED


def test_an_export_takes_the_umask_mode(tmp_path, monkeypatch):
    import negpy.desktop.workers.export as export

    worker = export.ExportWorker.__new__(export.ExportWorker)
    worker._processor = SimpleNamespace(encode_export=lambda *a, **k: (b"jpeg bytes", ""))
    worker._written = set()
    worker._names_lock = threading.Lock()
    monkeypatch.setattr(export, "_export_resolution", lambda task: None)
    monkeypatch.setattr(export, "resolve_export_naming", lambda task: (str(tmp_path), "frame", "jpg"))
    monkeypatch.setattr(export, "_export_target_is_a_source", lambda path, task: False)
    monkeypatch.setattr(export, "sync_export_filesystem_dates", lambda *a: None)
    task = SimpleNamespace(
        metadata_config=None,
        export_settings=SimpleNamespace(overwrite=True),
        working_color_space="",
        file_info={"path": "", "name": "frame"},
    )

    worker._encode_and_write(task, np.zeros((2, 2, 3), np.float32), "sRGB", None)

    assert _mode(tmp_path / "frame.jpg") == EXPECTED


def test_a_film_scan_takes_the_umask_mode(tmp_path):
    from negpy.infrastructure.scanners.result import ScanResult
    from negpy.services.scanning.writer import write_tiff_16bit

    rgb = np.zeros((8, 8, 3), np.uint16)
    result = ScanResult(rgb=rgb, ir=np.zeros((8, 8), np.uint16), dpi=3600, device_model="Test", ir_valid_mask=np.ones((8, 8), bool))
    path = write_tiff_16bit(result, str(tmp_path / "scan"))

    for written in (path, path.replace(".tif", "_IR.tif"), path.replace(".tif", "_IR_VALID.tif")):
        assert _mode(written) == EXPECTED


def test_a_sidecar_takes_the_umask_mode(tmp_path):
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.assets.sidecar import write_sidecar

    assert _mode(write_sidecar(str(tmp_path / "IMG_001.NEF"), WorkspaceConfig())) == EXPECTED


# App-internal files that stay private to their owner.
_PRIVATE = {"kernel/system/user_directory.py"}


def test_every_tempfile_rename_applies_the_umask_mode():
    """NamedTemporaryFile and mkstemp make 0600 files; mkdtemp and gettempdir make none."""
    root = pathlib.Path(paths.__file__).resolve().parents[2]
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if str(p.relative_to(root)) not in _PRIVATE
        and any(maker in (text := p.read_text(encoding="utf-8")) for maker in ("NamedTemporaryFile(", "mkstemp("))
        and "os.replace(" in text
        and "apply_umask_mode(" not in text
    ]
    assert offenders == [], "call apply_umask_mode before renaming a tempfile file into place"
