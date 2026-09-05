import json
import os
from pathlib import Path
import runpy
import shutil
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from negpy.desktop import startup
from negpy.kernel.system import paths, user_directory as data


@pytest.fixture
def locations(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.delenv("NEGPY_USER_DIR", raising=False)
    source = tmp_path / "Documents" / "NegPy"
    source.mkdir(parents=True)
    return source, data.local_data_root()


def _database(path, value):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE example (value TEXT)")
    connection.execute("INSERT INTO example VALUES (?)", (value,))
    connection.commit()
    connection.close()


def _value(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT value FROM example").fetchone()[0]
    finally:
        connection.close()


def _save_record(root, name="data-existing"):
    target = root / name
    target.mkdir(parents=True)
    (root / "data-location.json").write_text(json.dumps({"version": 1, "directory": name}), encoding="utf-8")
    return target


def test_copy_preserves_data_without_copying_cache_or_exports(locations):
    source, root = locations
    for name in ("edits.db", "settings.db"):
        _database(source / name, name)
    (source / "presets").mkdir()
    (source / "presets" / "custom.json").write_text('{"value": 3}')
    (source / "override.toml").write_text('[rendering]\nbackend="cpu"')
    for name in ("cache", "export"):
        (source / name).mkdir()
        (source / name / "image.tif").write_bytes(b"image")
    (source / "negpy.log").write_text("old log")
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}

    target = data.recover_user_directory(source)

    assert target.parent == root
    assert data.saved_user_directory() == target
    assert _value(target / "edits.db") == "edits.db"
    assert _value(target / "settings.db") == "settings.db"
    assert (target / "presets" / "custom.json").read_text() == '{"value": 3}'
    assert (target / "override.toml").read_text() == '[rendering]\nbackend="cpu"'
    assert list((target / "cache").iterdir()) == []
    assert list((target / "export").iterdir()) == []
    assert not (target / "negpy.log").exists()
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_backup_includes_committed_wal_data(locations, tmp_path):
    source, _ = locations
    live = tmp_path / "live.db"
    connection = sqlite3.connect(live)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE example (value TEXT)")
        connection.execute("INSERT INTO example VALUES ('wal edit')")
        connection.commit()
        assert Path(str(live) + "-wal").stat().st_size > 0
        shutil.copyfile(live, source / "edits.db")
        shutil.copyfile(Path(str(live) + "-wal"), source / "edits.db-wal")
    finally:
        connection.close()
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    target = data.recover_user_directory(source)
    assert _value(target / "edits.db") == "wal edit"
    assert not (target / "edits.db-wal").exists()
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing flags")
def test_open_database_writer_prevents_recovery(locations):
    source, root = locations
    connection = sqlite3.connect(source / "edits.db")
    try:
        connection.execute("CREATE TABLE example (value TEXT)")
        connection.commit()
        with pytest.raises(OSError, match="Close other NegPy"):
            data.recover_user_directory(source)
        assert not (root / "data-location.json").exists()
    finally:
        connection.close()


def test_sqlite_never_opens_the_protected_source(monkeypatch, locations):
    source, _ = locations
    _database(source / "edits.db", "keep")
    connect = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        assert str(source) not in str(database), "SQLite must not create sidecars in the source"
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(data.sqlite3, "connect", guarded_connect)
    target = data.recover_user_directory(source)
    assert _value(target / "edits.db") == "keep"


def test_read_only_source_files_become_writable_copies(locations):
    source, _ = locations
    preset = source / "preset.json"
    preset.write_text("{}")
    preset.chmod(0o444)
    try:
        target = data.recover_user_directory(source)
        (target / preset.name).write_text('{"changed": true}')
        assert preset.read_text() == "{}"
    finally:
        preset.chmod(0o666)


def test_missing_source_creates_fresh_data(locations):
    source, _ = locations
    target = data.recover_user_directory(source / "not-created")
    assert (target / "presets").is_dir()
    assert data.saved_user_directory() == target


def test_existing_appdata_is_not_overwritten(locations):
    source, root = locations
    _database(source / "edits.db", "Documents")
    root.mkdir(parents=True)
    _database(root / "edits.db", "older local copy")
    target = data.recover_user_directory(source)
    assert _value(target / "edits.db") == "Documents"
    assert _value(root / "edits.db") == "older local copy"


def test_existing_selection_is_not_recopied(locations):
    source, root = locations
    selected = _save_record(root)
    _database(selected / "edits.db", "new edits")
    _database(source / "edits.db", "old edits")
    assert data.recover_user_directory(source) == selected
    assert _value(selected / "edits.db") == "new edits"


@pytest.mark.parametrize("failure", ["copy", "publish", "database"])
def test_failed_recovery_keeps_source_and_does_not_select_partial_copy(monkeypatch, locations, failure):
    source, root = locations
    (source / "preset.json").write_text("keep")

    def fail(*args, **kwargs):
        raise OSError("simulated disk or permission failure")

    if failure == "copy":
        monkeypatch.setattr(data, "_copy_entry", fail)
    elif failure == "publish":
        monkeypatch.setattr(data.os, "link", fail)
    else:
        (source / "edits.db").write_text("invalid database")
    with pytest.raises((OSError, sqlite3.Error)):
        data.recover_user_directory(source)
    assert (source / "preset.json").read_text() == "keep"
    assert not (root / "data-location.json").exists()
    assert list(root.glob("data-*")) == []


def test_unreadable_source_is_not_treated_as_empty(monkeypatch, locations):
    source, root = locations
    original = Path.iterdir

    def deny(path):
        if path == source:
            raise PermissionError("source cannot be read")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", deny)
    with pytest.raises(PermissionError):
        data.recover_user_directory(source)
    assert not (root / "data-location.json").exists()


def test_linked_content_does_not_get_followed(monkeypatch, locations):
    source, root = locations
    linked = source / "linked"
    linked.mkdir()
    real = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda p: p == linked or real(p))
    with pytest.raises(OSError, match="linked"):
        data.recover_user_directory(source)
    assert not (root / "data-location.json").exists()


def test_concurrent_publication_keeps_the_first_location(monkeypatch, locations):
    source, root = locations

    def race(*args):
        _save_record(root, "data-winner")
        raise FileExistsError("another process committed first")

    monkeypatch.setattr(data.os, "link", race)
    assert data.recover_user_directory(source) == root / "data-winner"
    assert set(root.glob("data-*")) == {
        root / "data-winner",
        root / "data-location.json",
    }


@pytest.mark.parametrize("record", [[], {}, {"version": 2, "directory": "data-a"}, {"version": 1, "directory": "../other"}])
def test_invalid_record_does_not_fall_back(locations, record):
    _, root = locations
    root.mkdir(parents=True)
    (root / "data-location.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        data.saved_user_directory()


def test_missing_saved_directory_does_not_fall_back(locations):
    _, root = locations
    target = _save_record(root)
    target.rmdir()
    with pytest.raises(OSError, match="missing"):
        data.saved_user_directory()


def test_recovered_location_stays_selected_after_documents_is_writable(monkeypatch, locations):
    source, _ = locations
    target = data.recover_user_directory(source)
    monkeypatch.setattr(paths, "sys", SimpleNamespace(platform="win32"))
    assert paths.get_default_user_dir() == str(target)
    (target / "new-edit.json").write_text("new")
    assert paths.get_default_user_dir() == str(target)
    assert not (source / "new-edit.json").exists()


def test_explicit_override_wins_even_with_corrupt_record(monkeypatch, locations):
    source, root = locations
    root.mkdir(parents=True)
    (root / "data-location.json").write_text("broken")
    monkeypatch.setenv("NEGPY_USER_DIR", str(source))
    monkeypatch.setattr(paths, "sys", SimpleNamespace(platform="win32"))
    assert paths.get_default_user_dir() == str(source)


@pytest.fixture
def startup_probe(monkeypatch, locations):
    source, root = locations
    monkeypatch.setattr(startup, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(startup, "get_default_user_dir", lambda: str(source))
    messages = []
    monkeypatch.setattr(startup, "_message", lambda text, flags: messages.append((text, flags)) or 6)
    return source, root, messages


def _block_source(monkeypatch, source):
    real = startup.ensure_writable

    def check(path):
        if path == source:
            raise FileNotFoundError(2, "Controlled Folder Access denied the write")
        real(path)

    monkeypatch.setattr(startup, "ensure_writable", check)


def test_writable_startup_has_no_prompt_or_record(startup_probe):
    _, root, messages = startup_probe
    assert startup.prepare_user_directory()
    assert messages == []
    assert not (root / "data-location.json").exists()


def test_blocked_default_startup_recovers_after_confirmation(monkeypatch, startup_probe):
    source, _, messages = startup_probe
    _database(source / "edits.db", "keep this edit")
    _block_source(monkeypatch, source)
    assert startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x24, 0x40]
    assert _value(data.saved_user_directory() / "edits.db") == "keep this edit"


def test_declining_recovery_changes_no_location(monkeypatch, startup_probe):
    source, root, _ = startup_probe
    _block_source(monkeypatch, source)
    monkeypatch.setattr(startup, "_message", lambda *args: 7)
    assert not startup.prepare_user_directory()
    assert not root.exists()


def test_blocked_explicit_override_is_not_replaced(monkeypatch, startup_probe):
    source, root, messages = startup_probe
    _block_source(monkeypatch, source)
    monkeypatch.setenv("NEGPY_USER_DIR", str(source))
    assert not startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x10]
    assert not root.exists()


def test_blocked_saved_location_is_not_replaced(monkeypatch, startup_probe):
    source, root, messages = startup_probe
    chosen = _save_record(root)
    _block_source(monkeypatch, source)
    assert not startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x10]
    assert data.saved_user_directory() == chosen


def test_copy_failure_is_visible_and_does_not_activate_location(monkeypatch, startup_probe):
    source, root, messages = startup_probe
    _block_source(monkeypatch, source)
    (source / "edits.db").write_text("corrupt database")
    assert not startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x24, 0x10]
    assert not (root / "data-location.json").exists()


def test_non_windows_startup_is_unchanged(monkeypatch, startup_probe):
    source, _, messages = startup_probe
    _block_source(monkeypatch, source)
    monkeypatch.setattr(startup, "sys", SimpleNamespace(platform="linux"))
    assert startup.prepare_user_directory()
    assert messages == []


def test_saved_record_error_is_visible_without_a_second_recovery(monkeypatch, startup_probe):
    _, root, messages = startup_probe
    root.mkdir(parents=True)
    (root / "data-location.json").write_text("invalid json")
    monkeypatch.setattr(startup, "get_default_user_dir", lambda: str(data.saved_user_directory()))
    assert not startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x10]


def test_failed_startup_preflight_does_not_import_main(monkeypatch):
    import builtins

    monkeypatch.setattr(startup, "prepare_user_directory", lambda: False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "negpy.desktop.main"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "desktop.py"), run_name="__main__")
    assert result.value.code == 1


def test_successful_startup_preflight_runs_before_main_import(monkeypatch):
    import builtins

    events = []
    monkeypatch.setattr(startup, "prepare_user_directory", lambda: events.append("prepared") or True)
    monkeypatch.setitem(sys.modules, "negpy.desktop.main", SimpleNamespace(main=lambda: events.append("main")))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "negpy.desktop.main":
            assert events == ["prepared"]
            events.append("imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "desktop.py"), run_name="__main__")
    assert events == ["prepared", "imported", "main"]
