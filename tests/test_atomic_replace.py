import errno
import os
import stat
import sys
import threading
from types import SimpleNamespace
import numpy as np
import pytest

from negpy.kernel.system.paths import replace_file, safe_unlink
import negpy.desktop.workers.export as export


def test_replace_file_creates_new_file(tmp_path):
    src = str(tmp_path / "src.txt")
    dst = str(tmp_path / "dst.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("content")

    replace_file(src, dst)

    assert not os.path.exists(src)
    assert os.path.exists(dst)
    with open(dst, "r", encoding="utf-8") as f:
        assert f.read() == "content"


def test_replace_file_overwrites_existing(tmp_path):
    src = str(tmp_path / "src.txt")
    dst = str(tmp_path / "dst.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("new")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("old")

    replace_file(src, dst)

    assert not os.path.exists(src)
    with open(dst, "r", encoding="utf-8") as f:
        assert f.read() == "new"


def test_replace_file_overwrites_readonly(tmp_path):
    src = str(tmp_path / "src.txt")
    dst = str(tmp_path / "dst.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("new")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("old")
    os.chmod(dst, stat.S_IREAD)

    replace_file(src, dst)

    assert not os.path.exists(src)
    with open(dst, "r", encoding="utf-8") as f:
        assert f.read() == "new"


def test_replace_file_retries_transient_lock(tmp_path, monkeypatch):
    src = str(tmp_path / "src.txt")
    dst = str(tmp_path / "dst.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("data")

    real_replace = os.replace
    attempts = 0

    def mock_replace(s, d):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            err = PermissionError(13, "The process cannot access the file")
            err.winerror = 32
            raise err
        return real_replace(s, d)

    monkeypatch.setattr(os, "replace", mock_replace)
    monkeypatch.setattr(sys, "platform", "win32")

    replace_file(src, dst, timeout=1.0, interval=0.01)

    assert attempts == 3
    assert os.path.exists(dst)


def test_replace_file_retries_transient_access_denied(tmp_path, monkeypatch):
    src = str(tmp_path / "src.txt")
    dst = str(tmp_path / "dst.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("data")

    real_replace = os.replace
    attempts = 0

    def mock_replace(s, d):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            err = PermissionError(errno.EACCES, "Access is denied")
            err.winerror = 5
            raise err
        return real_replace(s, d)

    monkeypatch.setattr(os, "replace", mock_replace)
    monkeypatch.setattr(sys, "platform", "win32")

    replace_file(src, dst, timeout=1.0, interval=0.01)

    assert attempts == 3
    assert os.path.exists(dst)


def test_replace_file_times_out_on_persistent_lock(tmp_path, monkeypatch):
    src = str(tmp_path / "src.txt")
    dst = str(tmp_path / "dst.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("data")

    def mock_replace(s, d):
        err = PermissionError(13, "The process cannot access the file")
        err.winerror = 32
        raise err

    monkeypatch.setattr(os, "replace", mock_replace)
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(PermissionError):
        replace_file(src, dst, timeout=0.05, interval=0.01)


def test_replace_file_raises_immediately_for_missing_source(tmp_path):
    src = str(tmp_path / "missing.txt")
    dst = str(tmp_path / "dst.txt")

    with pytest.raises(FileNotFoundError):
        replace_file(src, dst, timeout=0.5)


def test_safe_unlink_removes_file(tmp_path):
    target = str(tmp_path / "temp.part")
    with open(target, "w", encoding="utf-8") as f:
        f.write("temp")

    safe_unlink(target)
    assert not os.path.exists(target)


def test_safe_unlink_ignores_missing_file(tmp_path):
    target = str(tmp_path / "nonexistent.part")
    safe_unlink(target)
    safe_unlink(None)


def test_safe_unlink_removes_readonly_file(tmp_path):
    target = str(tmp_path / "readonly.part")
    with open(target, "w", encoding="utf-8") as f:
        f.write("temp")
    os.chmod(target, stat.S_IREAD)

    safe_unlink(target)
    assert not os.path.exists(target)


def test_export_worker_retries_and_succeeds_when_folder_watched(tmp_path, monkeypatch):
    worker = export.ExportWorker.__new__(export.ExportWorker)
    worker._processor = SimpleNamespace(encode_export=lambda *a, **k: (b"jpeg bits", ""))
    worker._written = set()
    worker._names_lock = threading.Lock()

    monkeypatch.setattr(export, "_export_resolution", lambda task: None)
    monkeypatch.setattr(export, "resolve_export_naming", lambda task: (str(tmp_path), "output", "jpg"))
    monkeypatch.setattr(export, "_export_target_is_a_source", lambda path, task: False)
    monkeypatch.setattr(export, "sync_export_filesystem_dates", lambda *a: None)

    task = SimpleNamespace(
        metadata_config=None,
        export_settings=SimpleNamespace(overwrite=True),
        working_color_space="",
        file_info={"path": "", "name": "frame"},
    )

    real_replace = os.replace
    attempts = 0

    def mock_replace(s, d):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            err = PermissionError(13, "The process cannot access the file")
            err.winerror = 32
            raise err
        return real_replace(s, d)

    monkeypatch.setattr(os, "replace", mock_replace)
    monkeypatch.setattr(sys, "platform", "win32")

    err = worker._encode_and_write(task, np.zeros((2, 2, 3), np.float32), "sRGB", None)

    assert err is None
    assert attempts == 3
    final_output = tmp_path / "output.jpg"
    assert final_output.exists()
    assert final_output.read_bytes() == b"jpeg bits"
    # Verify no leftover .part files in the directory
    part_files = list(tmp_path.glob("*.part"))
    assert part_files == []
