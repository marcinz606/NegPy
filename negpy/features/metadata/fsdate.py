"""Filesystem-level date sync for exported files.

EXIF/XMP capture and modify dates are preserved faithfully through the metadata
writer (see writer.py) -- that part of the pipeline is correct even with
"Protect Original Metadata" off. What's missing is the exported file's own
*filesystem* mtime/creation date: every export path writes a brand-new file, so
the OS stamps it with "now" and nothing downstream corrects it. Apps that sort
or filter by file date rather than embedded EXIF (Capture One's default "Date"
capture-date column reads EXIF, but its Finder-style file list and many other
tools read the filesystem date) then show the export/build time instead of when
the frame was actually shot.

This module mirrors the *source* file's own filesystem mtime -- and, best
effort, its macOS creation date -- onto the exported file right after it's
written. It deliberately does not try to resolve "the" capture date (a
user-typed override, a value carried in EXIF, etc.): the source RAW/scan
file's own mtime is what's actually being lost, it's available unconditionally,
and matching it is enough to fix sort order and "when was this shot" filesystem
metadata in any tool that reads it.
"""

import ctypes
import ctypes.util
import os
import platform
from typing import Optional


def sync_export_filesystem_dates(output_path: str, source_path: str) -> None:
    """Best-effort: stamp *output_path*'s mtime/atime -- and, on macOS, creation
    date -- to match *source_path*. Never raises: a missing/unreadable source, a
    destination filesystem that rejects the call, or (on macOS) the absence of
    the low-level creation-date API should not fail the export itself, since the
    export's actual pixel/metadata content already succeeded by the time this
    runs."""
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


# APFS/HFS+ "creation date" (birthtime) has no stdlib entry point -- os.utime()
# only ever touches atime/mtime, on every platform. The only way to set it
# without shelling out to a tool that may not be installed (SetFile, part of
# the Xcode Command Line Tools) is the setattrlist(2) syscall macOS itself uses
# for this. NEEDS VERIFICATION: this talks to a C syscall via ctypes and has
# only been checked against the public setattrlist/attrlist header definitions,
# not run on a real Mac (this patch was written from a Linux sandbox) -- please
# confirm `mdls -name kMDItemFSCreationDate` on an exported file actually shows
# the source's date after this ships. It fails silently either way, so a wrong
# offset just means the creation date stays unfixed, same as before this patch.
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
        # Best-effort only: e.g. exFAT/network volumes don't support a creation
        # date at all. The mtime fix above already landed regardless.
        return
