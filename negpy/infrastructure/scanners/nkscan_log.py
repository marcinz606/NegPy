"""The nkscan diagnostics. nkscan writes them to nkscan.log at the selected level."""

from __future__ import annotations

import os
import platform
import threading
import time
from importlib import metadata

from negpy.kernel.system.paths import get_default_user_dir

LOG_NAME = "nkscan.log"
LEVELS = ("off", "debug", "trace")
_MAX_BYTES = 20_000_000

_lock = threading.Lock()
_level = "off"


def log_path() -> str:
    return os.path.join(get_default_user_dir(), LOG_NAME)


def set_level(level: str) -> bool:
    """Write nkscan's diagnostics to nkscan.log at `level`, one of LEVELS. False when nkscan is not installed."""
    global _level
    if level not in LEVELS:
        level = "off"
    try:
        import nkscan
    except ImportError:
        return level == "off"
    with _lock:
        if level == _level:
            return True
        # Close the file before it moves to the .1 file or gets the header.
        nkscan.init_logging("off")
        if level != "off":
            path = log_path()
            _roll(path)
            _append(path, _header(level))
            nkscan.init_logging(level, path)
        _level = level
        return True


def _roll(path: str) -> None:
    try:
        if os.path.getsize(path) > _MAX_BYTES:
            os.replace(path, path + ".1")
    except OSError:
        pass


def _append(path: str, data: bytes) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "ab") as file:
            file.write(data)
    except OSError:
        pass


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
