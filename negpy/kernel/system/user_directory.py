"""Persistent Windows data-directory recovery without modifying the source data."""

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time


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


def _backup_database(source: Path, target: Path) -> None:
    started = time.monotonic()

    def progress(status: int, remaining: int, total: int) -> None:
        if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) and time.monotonic() - started > 5:
            raise OSError("A data database is busy. Close other NegPy instances and try again.")

    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=5)) as original:
        with closing(sqlite3.connect(target)) as copied:
            original.backup(copied, pages=256, progress=progress)


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
