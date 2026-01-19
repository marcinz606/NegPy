import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    elif getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        # this file is in src/kernel/system/paths.py
        # Root is 3 levels up
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )

    return os.path.join(base_path, relative_path)


def get_default_user_dir() -> str:
    """Resolve the user directory, defaulting to Documents/NegPy with platform-native detection."""
    env_path = os.getenv("NEGPY_USER_DIR")
    if env_path:
        return os.path.abspath(env_path)

    docs_dir: Optional[Path] = None

    # 1. Windows: Use Shell API to get the "Personal" (Documents) folder
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PERSONAL = 5
            ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
            if buf.value:
                docs_dir = Path(buf.value)
        except Exception:
            pass

    # 2. Linux: Check XDG environment and utility
    elif sys.platform == "linux":
        xdg_docs = os.getenv("XDG_DOCUMENTS_DIR")
        if xdg_docs:
            docs_dir = Path(xdg_docs)
        else:
            try:
                out = subprocess.check_output(
                    ["xdg-user-dir", "DOCUMENTS"], stderr=subprocess.DEVNULL
                )
                path_str = out.decode("utf-8").strip()
                if path_str:
                    docs_dir = Path(path_str)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

    # 3. macOS: Standard location is ~/Documents, managed by System
    elif sys.platform == "darwin":
        docs_dir = Path.home() / "Documents"

    # 4. Global Fallback
    if not docs_dir or not docs_dir.exists():
        try:
            docs_dir = Path.home() / "Documents"
        except RuntimeError:
            docs_dir = Path(os.path.expanduser("~")) / "Documents"

    return str((docs_dir / "NegPy").absolute())
