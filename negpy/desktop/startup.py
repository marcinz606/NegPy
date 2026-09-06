"""Windows data-folder selection before imports bind application paths."""

import ctypes
import os
from pathlib import Path
import sys

from negpy.kernel.system.paths import get_default_user_dir
from negpy.kernel.system.user_directory import ensure_writable, local_data_root, select_user_directory


def _message(text: str, flags: int) -> int:
    message_box = ctypes.windll.user32.MessageBoxW
    message_box.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint)
    message_box.restype = ctypes.c_int
    return message_box(None, text, "NegPy data folder", flags | 0x10000)


def _choose_directory(source: Path, suggested: Path) -> Path | None:
    # Native controls leave QApplication creation until after the saved UI scale is loaded.
    from negpy.desktop.windows_data_dialog import choose_directory

    return choose_directory(source, suggested)


def prepare_user_directory() -> bool:
    if sys.platform != "win32":
        return True
    try:
        selected = Path(get_default_user_dir())
        try:
            ensure_writable(selected)
            return True
        except OSError:
            if os.environ.get("NEGPY_USER_DIR"):
                raise
        # A saved folder must not be replaced by an automatic return to Documents.
        from negpy.kernel.system.user_directory import saved_user_directory

        if saved_user_directory() is not None:
            raise OSError(f"Cannot write to the saved data folder: {selected}. Restore access and start NegPy again.")
        suggested = local_data_root() / "data"
        while True:
            choice = _choose_directory(selected, suggested)
            if choice is None:
                return False
            try:
                select_user_directory(choice)
                return True
            except (OSError, ValueError) as error:
                _message(f"NegPy could not use this data folder:\n{choice}\n\n{error}\n\nSelect another folder or cancel.", 0x10)
                suggested = choice
    except (OSError, ValueError) as error:
        _message(
            f"NegPy could not prepare its data folder:\n\n{error}\n\n"
            "Check folder access and available disk space, then try again. "
            "If NEGPY_USER_DIR is set, check that it names a writable folder.",
            0x10,
        )
        return False
