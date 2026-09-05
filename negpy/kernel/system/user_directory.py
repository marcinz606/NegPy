"""Persistent Windows data-directory recovery without modifying the source data."""

from contextlib import closing, contextmanager, ExitStack
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile


_DATABASES = {"edits.db", "settings.db"}
_DIRECTORIES = ("presets", "cache", "icc", "crosstalk", "sensor", "flatfield", "gear", "contact_sheets", "export")


def local_data_root() -> Path:
    """Return the Windows application-data location, not a Documents fallback."""
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if not base.is_absolute():
        raise OSError("LOCALAPPDATA must be an absolute path.")
    return base / "NegPy"


def saved_user_directory() -> Path | None:
    """Read the committed location; an invalid saved location must not select old data."""
    root = local_data_root()
    try:
        record = json.loads((root / "data-location.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"Invalid NegPy data location in {root / 'data-location.json'}.")
    name = record.get("directory")
    if record.get("version") != 1 or not isinstance(name, str) or not name.startswith("data-") or Path(name).name != name:
        raise ValueError(f"Invalid NegPy data location in {root / 'data-location.json'}.")
    target = root / name
    if not target.is_dir():
        raise OSError(f"The saved NegPy data folder is missing: {target}")
    return target


def _probe_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as probe:
        probe.write(b"NegPy")
        probe.flush()


def ensure_writable(path: Path) -> None:
    """Probe directory writes and existing database access before loading app configuration."""
    _probe_directory(path)
    for name in _DIRECTORIES:
        _probe_directory(path / name)
    for name in _DATABASES:
        database = path / name
        if database.exists():
            with database.open("r+b"):
                pass


@contextmanager
def _locked_database_file(path: Path):
    if os.name != "nt":
        with path.open("rb") as content:
            yield content
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    # Read access with FILE_SHARE_READ denies existing and new writers until the copy is complete.
    handle = create_file(str(path), 0x80000000, 1, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error in (32, 33):
            raise OSError(f"Database in use: {path}. Close other NegPy instances and try again.")
        raise ctypes.WinError(error)
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "rb") as content:
        yield content


def _backup_database(source: Path, target: Path) -> None:
    # SQLite opens only the private snapshot; protected sources need no WAL/SHM writes.
    with tempfile.TemporaryDirectory(prefix="database-", dir=target.parent) as temporary:
        snapshot = Path(temporary) / source.name
        with ExitStack() as locks:
            for suffix in ("", "-wal", "-journal"):
                try:
                    content = locks.enter_context(_locked_database_file(Path(str(source) + suffix)))
                except FileNotFoundError:
                    if suffix:
                        continue
                    raise
                with Path(str(snapshot) + suffix).open("wb") as copied:
                    shutil.copyfileobj(content, copied)
        with closing(sqlite3.connect(snapshot)) as original:
            with closing(sqlite3.connect(target)) as copied_db:
                original.backup(copied_db)


def _copy_entry(source: Path, target: Path) -> None:
    if source.is_symlink() or source.is_junction():
        raise OSError(f"Cannot automatically copy a linked data path: {source}")
    if source.is_dir():
        target.mkdir()
        for child in source.iterdir():
            _copy_entry(child, target / child.name)
    else:
        shutil.copyfile(source, target)


def _publish_location(root: Path, target: Path, source: Path) -> Path:
    record = {"version": 1, "directory": target.name, "source": str(source)}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(record, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        # A hard link publishes a complete record without replacing another process's choice.
        os.link(temporary_path, root / "data-location.json")
        return target
    except FileExistsError:
        chosen = saved_user_directory()
        if chosen is None:
            raise OSError("The saved data location changed during recovery. Try again.")
        return chosen
    finally:
        try:
            temporary_path.unlink()
        except OSError:
            pass


def recover_user_directory(source: Path) -> Path:
    """Copy persistent data into a new folder, then commit its location exactly once."""
    root = local_data_root()
    _probe_directory(root)
    existing = saved_user_directory()
    if existing is not None:
        ensure_writable(existing)
        return existing
    target = Path(tempfile.mkdtemp(prefix="data-", dir=root))
    committed = False
    try:
        try:
            entries = list(source.iterdir())
        except FileNotFoundError:
            entries = []
        for entry in entries:
            name = entry.name
            if name in {"cache", "export", "negpy.log", "negpy.log.1", "negpy.log.2"}:
                continue
            if any(name == database + suffix for database in _DATABASES for suffix in ("-wal", "-shm", "-journal")):
                continue
            if name in _DATABASES:
                if entry.is_symlink() or entry.is_junction():
                    raise OSError(f"Cannot automatically copy a linked database: {entry}")
                _backup_database(entry, target / name)
            else:
                _copy_entry(entry, target / name)
        ensure_writable(target)
        chosen = _publish_location(root, target, source)
        committed = chosen == target
        return chosen
    finally:
        if not committed:
            # This directory is private to this recovery attempt; source data is never removed.
            shutil.rmtree(target)
