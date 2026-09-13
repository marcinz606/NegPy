"""Tests for the export filesystem-date sync (negpy.features.metadata.fsdate)."""

import ctypes
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

from negpy.features.metadata import fsdate
from negpy.features.metadata.fsdate import sync_export_filesystem_dates


def test_syncs_mtime_and_atime_to_source(tmp_path) -> None:
    src = tmp_path / "source.arw"
    dst = tmp_path / "export.jpg"
    src.write_bytes(b"source")
    dst.write_bytes(b"export")

    old_time = time.time() - 5 * 24 * 3600
    os.utime(src, (old_time, old_time))

    sync_export_filesystem_dates(str(dst), str(src))

    result = os.stat(dst)
    assert abs(result.st_mtime - old_time) < 1
    assert abs(result.st_atime - old_time) < 1


def test_missing_source_is_a_noop(tmp_path) -> None:
    dst = tmp_path / "export.jpg"
    dst.write_bytes(b"export")
    before = os.stat(dst).st_mtime

    sync_export_filesystem_dates(str(dst), str(tmp_path / "does-not-exist.arw"))

    assert os.stat(dst).st_mtime == before


def test_missing_destination_does_not_raise(tmp_path) -> None:
    src = tmp_path / "source.arw"
    src.write_bytes(b"source")
    # No exception even though the "export" was never written (e.g. caller passed
    # the wrong path) -- this only ever protects an export that already succeeded.
    sync_export_filesystem_dates(str(tmp_path / "does-not-exist.jpg"), str(src))


def test_macos_creation_date_uses_source_birthtime(tmp_path) -> None:
    """On Darwin, the creation date is set via setattrlist(2) using the source's
    st_birthtime. Exercised here with a mocked libc since this suite also runs on
    Linux/CI -- see the NEEDS VERIFICATION note in fsdate.py for the one thing this
    mock can't confirm: that the real syscall lands as expected on an actual Mac."""
    dst = tmp_path / "export.jpg"
    dst.write_bytes(b"export")

    birthtime = time.time() - 10 * 24 * 3600
    fake_stat = SimpleNamespace(st_atime=birthtime, st_mtime=birthtime, st_birthtime=birthtime)

    calls = []

    class _FakeLibc:
        def setattrlist(self, path, attrs_ptr, ts_ptr, ts_size, options):
            attrs = ctypes.cast(attrs_ptr, ctypes.POINTER(fsdate._AttrList)).contents
            ts = ctypes.cast(ts_ptr, ctypes.POINTER(fsdate._TimeSpec)).contents
            calls.append((path, attrs.commonattr, ts.tv_sec, options))
            return 0

    with (
        patch.object(fsdate.os, "stat", return_value=fake_stat),
        patch.object(fsdate.platform, "system", return_value="Darwin"),
        patch.object(fsdate, "_get_libc", return_value=_FakeLibc()),
    ):
        sync_export_filesystem_dates(str(dst), "/irrelevant/source/path")

    assert len(calls) == 1
    path, commonattr, tv_sec, options = calls[0]
    assert path == os.fsencode(str(dst))
    assert commonattr == fsdate._ATTR_CMN_CRTIME
    assert tv_sec == int(birthtime)


def test_non_macos_skips_creation_date_call(tmp_path) -> None:
    """On a non-Darwin platform, only mtime/atime are touched -- _get_libc must
    never even be consulted."""
    dst = tmp_path / "export.jpg"
    src = tmp_path / "source.arw"
    dst.write_bytes(b"export")
    src.write_bytes(b"source")

    with (
        patch.object(fsdate.platform, "system", return_value="Linux"),
        patch.object(fsdate, "_get_libc") as get_libc,
    ):
        sync_export_filesystem_dates(str(dst), str(src))
        get_libc.assert_not_called()
