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
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    return os.path.join(base_path, relative_path)


def _usable_user_dir(base: Path) -> Optional[str]:
    """
    The NegPy dir under `base` if `base` is usable, else None.

    When `base` already exists this touches nothing (the app creates its
    subdirectories at startup). When it's missing, creatability has to be proven
    by actually creating — a registered-but-absent folder is indistinguishable
    from a creatable one until CreateDirectory runs.
    """
    target = base / "NegPy"
    try:
        if os.path.isdir(base):
            return str(target.absolute())
        os.makedirs(target, exist_ok=True)
        return str(target.absolute())
    except OSError:
        return None


def get_default_user_dir() -> str:
    """Resolve the user directory, defaulting to Documents/NegPy with platform-native detection."""
    env_path = os.getenv("NEGPY_USER_DIR")
    if env_path:
        return os.path.abspath(env_path)

    docs_dir: Optional[Path] = None

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PERSONAL = 5
            res = ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
            if res == 0 and buf.value:
                docs_dir = Path(buf.value)
        except Exception:
            pass

    elif sys.platform == "linux":
        xdg_docs = os.getenv("XDG_DOCUMENTS_DIR")
        if xdg_docs:
            docs_dir = Path(xdg_docs)

        # fallback to xdg-user-dir
        if not docs_dir:
            try:
                out = subprocess.check_output(["xdg-user-dir", "DOCUMENTS"], stderr=subprocess.DEVNULL)
                path_str = out.decode("utf-8").strip()
                if path_str:
                    docs_dir = Path(path_str)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        # fallback to user-dirs.dirs
        if not docs_dir:
            config_home = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
            user_dirs_file = os.path.join(config_home, "user-dirs.dirs")
            if os.path.exists(user_dirs_file):
                try:
                    with open(user_dirs_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("XDG_DOCUMENTS_DIR="):
                                # Line format: XDG_DOCUMENTS_DIR="$HOME/doc"
                                path = line.split("=", 1)[1].strip().strip('"')
                                path = path.replace("$HOME", os.path.expanduser("~"))
                                if os.path.isabs(path):
                                    docs_dir = Path(path)
                                    break
                except Exception:
                    pass

    elif sys.platform == "darwin":
        docs_dir = Path.home() / "Documents"

    home = Path(os.path.expanduser("~"))
    if not docs_dir:
        docs_dir = home / "Documents"

    # The registered Documents folder can point at a location that does not exist
    # on disk — most commonly a OneDrive-backed Documents (...\OneDrive\Documents)
    # after OneDrive is unlinked, signed out, or not yet synced. Trusting it blindly
    # made the startup os.makedirs die with WinError 2 (#441). Validate the
    # candidate and fall back to plain local locations that always exist.
    candidates = list(dict.fromkeys([docs_dir, home / "Documents", home]))
    for base in candidates:
        usable = _usable_user_dir(base)
        if usable is not None:
            return usable

    # Last resort: home always exists in practice; let startup surface any error.
    return str((home / "NegPy").absolute())
