"""Persistent Windows data-directory selection."""

import json
import os
from pathlib import Path
import tempfile


_DATABASES = ("edits.db", "settings.db")
_DIRECTORIES = ("presets", "cache", "icc", "crosstalk", "sensor", "flatfield", "gear", "contact_sheets", "export")


def local_data_root() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if not base.is_absolute():
        raise OSError("LOCALAPPDATA must be an absolute path.")
    return base / "NegPy"


def saved_user_directory() -> Path | None:
    root = local_data_root()
    record_path = root / "data-location.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict) or not isinstance(record.get("directory"), str):
        raise ValueError(f"Invalid NegPy data location in {record_path}.")
    name = record["directory"]
    if record.get("version") == 1 and name.startswith("data-") and Path(name).name == name:
        target = root / name
    elif record.get("version") == 2 and Path(name).is_absolute():
        target = Path(name)
    else:
        raise ValueError(f"Invalid NegPy data location in {record_path}.")
    if not target.is_dir():
        raise OSError(f"The saved NegPy data folder is missing: {target}")
    return target


def _probe_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as probe:
        probe.write(b"NegPy")
        probe.flush()


def ensure_writable(path: Path) -> None:
    """Probe directory writes and existing databases before loading app configuration."""
    _probe_directory(path)
    for name in _DIRECTORIES:
        _probe_directory(path / name)
    for name in _DATABASES:
        database = path / name
        if database.exists():
            with database.open("r+b"):
                pass


def select_user_directory(target: Path) -> None:
    """Validate and remember a folder without moving or copying user data."""
    if not target.is_absolute():
        raise ValueError("Select an absolute data-folder path.")
    ensure_writable(target)
    root = local_data_root()
    _probe_directory(root)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump({"version": 2, "directory": str(target)}, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        # Publish the complete record without replacing another instance's selection.
        os.link(temporary_path, root / "data-location.json")
    except FileExistsError:
        if saved_user_directory() != target:
            raise OSError("Another NegPy instance selected a data folder. Close NegPy and start it again.") from None
    finally:
        try:
            temporary_path.unlink()
        except OSError:
            pass
