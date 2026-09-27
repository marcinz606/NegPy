"""nkscan's diagnostics, captured into nkscan.log for a bug report.

nkscan writes its `tracing` output from Rust straight to the process's stdout (0.12 documents
stderr), which Python's logging never sees and a packaged app has no terminal for. The capture
points both descriptors at pipes once per process: nkscan's lines go to the log file at the
chosen level and nowhere else, every other line to the stream it was written to.
"""

from __future__ import annotations

import atexit
import os
import platform
import re
import sys
import threading
import time
from importlib import metadata
from typing import BinaryIO, TextIO

from negpy.kernel.system.paths import get_default_user_dir

LOG_NAME = "nkscan.log"
# Debug shows each scan's decisions; trace adds every command sent to the unit.
LEVELS = ("off", "debug", "trace")
_MAX_BYTES = 20_000_000
# nkscan's level is fixed at its first init, so it starts at trace and the pump drops what the
# chosen level excludes. RUST_LOG overrides it.
_NKSCAN_LINE = re.compile(rb"^\S+Z\s+(TRACE|DEBUG|INFO|WARN|ERROR)\s")

_lock = threading.Lock()
_level = "off"
_installed = False
_file: BinaryIO | None = None
_crash_stream: TextIO | None = None


def log_path() -> str:
    return os.path.join(get_default_user_dir(), LOG_NAME)


def set_level(level: str) -> bool:
    """Write nkscan's diagnostics to nkscan.log at `level`, one of LEVELS. False when nkscan is not installed."""
    global _level
    if level not in LEVELS:
        level = "off"
    with _lock:
        if level != "off" and not _installed and not _install():
            return False
        if level != _level and level != "off":
            _write(_header(level))
        _level = level
        return True


def _install() -> bool:
    global _installed
    try:
        import nkscan
    except ImportError:
        return False
    streams = []
    for fd, std_handle in ((1, -11), (2, -12)):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
        read_fd, write_fd = os.pipe()
        try:
            original: int | None = os.dup(fd)
        except OSError:
            original = None  # A windowed Windows build has no console stream to copy to.
        _point(fd, std_handle, write_fd)
        os.close(write_fd)
        if fd == 2 and original is not None:
            _keep_crash_tracebacks(original)
        pump = threading.Thread(target=_pump, args=(read_fd, original), name=f"nkscan-log-{fd}", daemon=True)
        pump.start()
        streams.append((fd, std_handle, original, pump))
    atexit.register(_restore, streams)
    nkscan.init_logging("trace")
    _installed = True
    return True


def _point(fd: int, std_handle: int, target: int) -> None:
    os.dup2(target, fd)
    if sys.platform == "win32":
        # Rust asks Windows for the standard handle, not the C runtime for the descriptor.
        import ctypes
        import msvcrt

        ctypes.windll.kernel32.SetStdHandle(std_handle, msvcrt.get_osfhandle(fd))


def _restore(streams: list[tuple[int, int, int | None, threading.Thread]]) -> None:
    """Give each stream its descriptor back and let its pump copy what is left.

    Daemon pumps die with the process, which cuts off whatever was written last, such as the
    traceback of an error at exit that Python prints before atexit runs.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    for fd, std_handle, original, pump in streams:
        try:
            if original is not None:
                _point(fd, std_handle, original)
            else:
                null = os.open(os.devnull, os.O_WRONLY)
                _point(fd, std_handle, null)
                os.close(null)
        except OSError:
            continue
        pump.join(timeout=2)


def _keep_crash_tracebacks(original: int) -> None:
    """faulthandler writes from a dying process, which the pipe's reader does not outlive."""
    import faulthandler

    if faulthandler.is_enabled():
        global _crash_stream
        _crash_stream = os.fdopen(os.dup(original), "w")
        faulthandler.enable(file=_crash_stream)


def _pump(read_fd: int, original: int | None) -> None:
    # Must never stop: a full pipe with no reader blocks every later write to the stream.
    with os.fdopen(read_fd, "rb") as pipe:
        for line in pipe:
            match = _NKSCAN_LINE.match(line)
            if match is not None:
                with _lock:
                    if _wanted(match.group(1), _level):
                        _write(line)
            elif original is not None:
                try:
                    os.write(original, line)
                except OSError:
                    pass


def _wanted(line_level: bytes, level: str) -> bool:
    """Whether an nkscan line at `line_level` belongs in a log at `level`."""
    return level == "trace" or (level == "debug" and line_level != b"TRACE")


def _write(data: bytes) -> None:
    global _file
    try:
        if _file is None:
            _file = open(log_path(), "ab")
        _file.write(data)
        _file.flush()
        if _file.tell() > _MAX_BYTES:
            _file.close()
            _file = None
            os.replace(log_path(), log_path() + ".1")
    except OSError:
        _file = None


def _header(level: str) -> bytes:
    def version(dist: str) -> str:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return "unknown"

    from negpy import __version__

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"\n--- {stamp} NegPy {__version__}, nkscan {version('nkscan')}, "
        f"{platform.system()} {platform.release()} {platform.machine()}, level {os.environ.get('RUST_LOG', level)}\n"
    ).encode()
