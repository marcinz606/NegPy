"""Filesystem mtime and creation date for exported files.

An export writes a brand-new file, so the OS stamps its filesystem dates with the write
time. The embedded EXIF/XMP dates are right (writer.py), but tools that sort or filter on
the filesystem date instead then show the export time, not the capture time.

The source file's own mtime is mirrored onto the output, not a resolved capture date: the
source mtime is what the write loses, and it is available for every source.
"""

import ctypes
import ctypes.util
import os
import platform
from typing import Optional


def sync_export_filesystem_dates(output_path: str, source_path: str) -> None:
    """Stamp *output_path*'s mtime/atime, and on macOS its creation date, from *source_path*.

    Never raises: the pixels and the metadata are already written by the time this runs, so
    an unreadable source or a filesystem that rejects the call must not fail the export.
    """
    try:
        src_stat = os.stat(source_path)
    except OSError:
        return

    try:
        os.utime(output_path, (src_stat.st_atime, src_stat.st_mtime))
    except OSError:
        return

    if platform.system() == "Darwin":
        _sync_macos_creation_date(output_path, src_stat)


# APFS/HFS+ creation date (birthtime) has no stdlib entry point: os.utime touches atime and
# mtime only, on every platform. setattrlist(2) is what macOS itself uses, and avoids a
# dependency on SetFile, which ships with the Xcode Command Line Tools rather than the OS.
_ATTR_BIT_MAP_COUNT = 5
_ATTR_CMN_CRTIME = 0x00000200


class _AttrList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


class _TimeSpec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


_libc: Optional[ctypes.CDLL] = None


def _get_libc() -> Optional[ctypes.CDLL]:
    global _libc
    if _libc is None:
        try:
            path = ctypes.util.find_library("c")
            _libc = ctypes.CDLL(path, use_errno=True) if path else None
        except OSError:
            _libc = None
    return _libc


def _sync_macos_creation_date(output_path: str, src_stat: os.stat_result) -> None:
    birthtime = getattr(src_stat, "st_birthtime", None)
    if birthtime is None:
        return

    libc = _get_libc()
    if libc is None:
        return

    attrs = _AttrList(
        bitmapcount=_ATTR_BIT_MAP_COUNT, reserved=0, commonattr=_ATTR_CMN_CRTIME, volattr=0, dirattr=0, fileattr=0, forkattr=0
    )
    whole_seconds = int(birthtime)
    ts = _TimeSpec(tv_sec=whole_seconds, tv_nsec=int((birthtime - whole_seconds) * 1_000_000_000))

    try:
        rc = libc.setattrlist(os.fsencode(output_path), ctypes.byref(attrs), ctypes.byref(ts), ctypes.sizeof(ts), 0)
    except OSError:
        return
    if rc != 0:
        # exFAT and network volumes carry no creation date. The mtime above still stands.
        return
