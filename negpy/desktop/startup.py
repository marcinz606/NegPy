"""Windows data-folder recovery before imports bind application paths."""

import ctypes
import os
from pathlib import Path
import sqlite3
import sys

from negpy.kernel.system.paths import get_default_user_dir
from negpy.kernel.system.user_directory import ensure_writable, local_data_root, recover_user_directory, saved_user_directory


def _message(text: str, flags: int) -> int:
    message_box = ctypes.windll.user32.MessageBoxW
    message_box.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint)
    message_box.restype = ctypes.c_int
    return message_box(None, text, "NegPy data folder", flags | 0x10000)


def prepare_user_directory() -> bool:
    """Offer a persistent, non-destructive recovery for an unwritable default folder."""
    if sys.platform != "win32":
        return True
    try:
        selected = Path(get_default_user_dir())
        try:
            ensure_writable(selected)
            return True
        except OSError:
            if os.environ.get("NEGPY_USER_DIR") or saved_user_directory() is not None:
                raise
        root = local_data_root()
        answer = _message(
            f"NegPy cannot write to its data folder:\n{selected}\n\n"
            "Windows may protect this folder. Close any other NegPy windows before continuing.\n\n"
            f"Copy your edits, settings and presets to a new folder under:\n{root}\n\n"
            "NegPy will use the new location on future starts. Original files stay in place. "
            "Caches and exported images are not copied.\n\nContinue?",
            0x24,
        )
        if answer != 6:
            return False
        recovered = recover_user_directory(selected)
        _message(f"NegPy will use:\n{recovered}\n\nYour original files remain in:\n{selected}", 0x40)
        return True
    except (OSError, ValueError, sqlite3.Error) as error:
        _message(
            f"NegPy could not prepare its data folder:\n\n{error}\n\n"
            "No original data was removed. Check folder access and available disk space, then try again. "
            "If NEGPY_USER_DIR is set, check that it names a writable folder.",
            0x10,
        )
        return False
