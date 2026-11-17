"""The nkscan diagnostics. The crate writes them to stderr at the selected level."""

from __future__ import annotations

import os
import threading

from negpy.kernel.system.paths import get_default_user_dir

LOG_NAME = "nkscan.log"
LEVELS = ("off", "debug", "trace")

_lock = threading.Lock()
_level = "off"


def log_path() -> str:
    return os.path.join(get_default_user_dir(), LOG_NAME)


def set_level(level: str) -> bool:
    """Enable nkscan diagnostics at `level`, one of LEVELS, on stderr. False when nkscan is not installed."""
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
        # The crate sends diagnostics to stderr; the first call wins, so set the level once.
        nkscan.init_logging(level)
        _level = level
        return True
