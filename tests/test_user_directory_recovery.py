import json
from pathlib import Path
import runpy
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


def _save_record(root, name="data-existing"):
    target = root / name
    target.mkdir(parents=True)
    (root / "data-location.json").write_text(json.dumps({"version": 1, "directory": name}), encoding="utf-8")
    return target


def test_selection_preserves_both_folders_without_copying(locations, tmp_path):
    source, _ = locations
    (source / "edits.db").write_bytes(b"original data")
    target = tmp_path / "custom Å & data"
    target.mkdir()
    (target / "settings.db").write_bytes(b"existing data")
    data.select_user_directory(target)
    assert data.saved_user_directory() == target
    assert (source / "edits.db").read_bytes() == b"original data"
    assert not (target / "edits.db").exists()
    assert (target / "settings.db").read_bytes() == b"existing data"


def test_select_new_directory_and_keep_choice_when_documents_is_writable(monkeypatch, locations):
    _, root = locations
    target = root / "data"
    data.select_user_directory(target)
    monkeypatch.setattr(paths, "sys", SimpleNamespace(platform="win32"))
    assert paths.get_default_user_dir() == str(target)
    assert all((target / name).is_dir() for name in data._DIRECTORIES)
    assert json.loads((root / "data-location.json").read_text())["version"] == 2


def test_existing_preview_record_remains_usable(locations):
    _, root = locations
    target = _save_record(root)
    assert data.saved_user_directory() == target


def test_relative_selection_is_rejected(locations):
    _, root = locations
    with pytest.raises(ValueError, match="absolute"):
        data.select_user_directory(Path("relative"))
    assert not root.exists()


def test_unwritable_selection_is_not_saved(monkeypatch, locations):
    _, root = locations

    def deny(path):
        raise PermissionError("blocked")

    monkeypatch.setattr(data, "ensure_writable", deny)
    with pytest.raises(PermissionError):
        data.select_user_directory(root / "data")
    assert not root.exists()


def test_publish_failure_keeps_existing_files_and_no_record(monkeypatch, locations, tmp_path):
    _, root = locations
    target = tmp_path / "chosen"
    target.mkdir()
    (target / "preset.json").write_text("keep")

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(data.os, "link", fail)
    with pytest.raises(OSError):
        data.select_user_directory(target)
    assert (target / "preset.json").read_text() == "keep"
    assert not list(root.iterdir())


def test_concurrent_selection_is_not_overwritten(monkeypatch, locations, tmp_path):
    _, root = locations

    def race(*args):
        _save_record(root, "data-winner")
        raise FileExistsError("already selected")

    monkeypatch.setattr(data.os, "link", race)
    with pytest.raises(OSError, match="Another NegPy"):
        data.select_user_directory(tmp_path / "loser")
    assert data.saved_user_directory() == root / "data-winner"


def test_same_selection_is_idempotent(locations):
    _, root = locations
    target = root / "data"
    data.select_user_directory(target)
    before = (root / "data-location.json").read_bytes()
    data.select_user_directory(target)
    assert (root / "data-location.json").read_bytes() == before


@pytest.mark.parametrize(
    "record",
    [
        [],
        {},
        {"version": 3, "directory": "data-a"},
        {"version": 1, "directory": "../other"},
        {"version": 2, "directory": "relative"},
        {"version": 2, "directory": None},
    ],
)
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
    choices = []
    monkeypatch.setattr(startup, "_message", lambda text, flags: messages.append((text, flags)) or 1)

    def choose(old, suggested):
        choices.append((old, suggested))
        return suggested

    monkeypatch.setattr(startup, "_choose_directory", choose)
    return source, root, messages, choices


def _block_source(monkeypatch, source):
    real = startup.ensure_writable

    def check(path):
        if path == source:
            raise FileNotFoundError(2, "Controlled Folder Access denied the write")
        real(path)

    monkeypatch.setattr(startup, "ensure_writable", check)


def test_writable_startup_has_no_prompt_or_record(startup_probe):
    _, root, messages, choices = startup_probe
    assert startup.prepare_user_directory()
    assert messages == choices == []
    assert not (root / "data-location.json").exists()


def test_blocked_default_suggests_appdata_without_copying(monkeypatch, startup_probe):
    source, root, messages, choices = startup_probe
    (source / "edits.db").write_bytes(b"keep this edit")
    _block_source(monkeypatch, source)
    assert startup.prepare_user_directory()
    assert choices == [(source, root / "data")]
    assert messages == []
    assert data.saved_user_directory() == root / "data"
    assert not (root / "data" / "edits.db").exists()
    assert (source / "edits.db").read_bytes() == b"keep this edit"


def test_custom_choice_is_saved(monkeypatch, startup_probe, tmp_path):
    source, _, _, _ = startup_probe
    _block_source(monkeypatch, source)
    target = tmp_path / "custom data"
    monkeypatch.setattr(startup, "_choose_directory", lambda *args: target)
    assert startup.prepare_user_directory()
    assert data.saved_user_directory() == target


def test_cancel_changes_no_location(monkeypatch, startup_probe):
    source, root, _, _ = startup_probe
    _block_source(monkeypatch, source)
    monkeypatch.setattr(startup, "_choose_directory", lambda *args: None)
    assert not startup.prepare_user_directory()
    assert not root.exists()


def test_failed_choice_can_be_changed(monkeypatch, startup_probe, tmp_path):
    source, root, messages, _ = startup_probe
    _block_source(monkeypatch, source)
    invalid = tmp_path / "file"
    invalid.write_text("do not overwrite")
    choices = iter([invalid, root / "data"])
    monkeypatch.setattr(startup, "_choose_directory", lambda *args: next(choices))
    assert startup.prepare_user_directory()
    assert len(messages) == 1
    assert data.saved_user_directory() == root / "data"
    assert invalid.read_text() == "do not overwrite"


def test_blocked_explicit_override_is_not_replaced(monkeypatch, startup_probe):
    source, root, messages, choices = startup_probe
    _block_source(monkeypatch, source)
    monkeypatch.setenv("NEGPY_USER_DIR", str(source))
    assert not startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x10]
    assert choices == []
    assert not root.exists()


def test_blocked_saved_location_is_not_replaced(monkeypatch, startup_probe):
    source, root, messages, choices = startup_probe
    chosen = _save_record(root)
    _block_source(monkeypatch, source)
    assert not startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x10]
    assert choices == []
    assert data.saved_user_directory() == chosen


def test_non_windows_startup_is_unchanged(monkeypatch, startup_probe):
    source, _, messages, choices = startup_probe
    _block_source(monkeypatch, source)
    monkeypatch.setattr(startup, "sys", SimpleNamespace(platform="linux"))
    assert startup.prepare_user_directory()
    assert messages == choices == []


def test_saved_record_error_is_visible_without_selection(monkeypatch, startup_probe):
    _, root, messages, choices = startup_probe
    root.mkdir(parents=True)
    (root / "data-location.json").write_text("invalid json")
    monkeypatch.setattr(startup, "get_default_user_dir", lambda: str(data.saved_user_directory()))
    assert not startup.prepare_user_directory()
    assert [flags for _, flags in messages] == [0x10]
    assert choices == []


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
