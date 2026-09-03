"""
User-dir resolution must survive a registered Documents folder that does not
exist on disk — e.g. a OneDrive-backed Documents after OneDrive is unlinked or
signed out (issue #441: startup makedirs died with WinError 2).
"""

import os
from pathlib import Path

from negpy.kernel.system import paths
from negpy.kernel.system.paths import _usable_user_dir, get_default_user_dir


def test_usable_dir_existing_base_returns_without_creating(tmp_path):
    result = _usable_user_dir(tmp_path)

    assert result == str((tmp_path / "NegPy").absolute())
    assert not (tmp_path / "NegPy").exists()


def test_usable_dir_existing_base_creates_and_probes_target_when_requested(tmp_path):
    result = _usable_user_dir(tmp_path, probe_write=True)

    assert result == str((tmp_path / "NegPy").absolute())
    assert (tmp_path / "NegPy").is_dir()
    assert list((tmp_path / "NegPy").iterdir()) == []


def test_usable_dir_missing_but_creatable_base_creates(tmp_path):
    base = tmp_path / "Documents"

    result = _usable_user_dir(base)

    assert result == str((base / "NegPy").absolute())
    assert (base / "NegPy").is_dir()


def test_usable_dir_uncreatable_base_returns_none(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")

    # A path under a regular file can never be created — mirrors the broken
    # OneDrive case where CreateDirectory fails on the registered path.
    assert _usable_user_dir(blocker / "Documents") is None


def test_usable_dir_existing_but_unwritable_target_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "NegPy"
    target.mkdir()

    def denied(*args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(paths.tempfile, "NamedTemporaryFile", denied)

    assert _usable_user_dir(tmp_path, probe_write=True) is None


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("NEGPY_USER_DIR", str(tmp_path / "custom"))

    assert get_default_user_dir() == os.path.abspath(str(tmp_path / "custom"))


def test_env_override_expands_a_leading_tilde(monkeypatch, tmp_path):
    """Without expansion "~/dev" becomes a literal "~" directory under the working
    directory — the app runs, out of a folder inside the checkout."""
    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home), 1) if p.startswith("~") else p)
    monkeypatch.setenv("NEGPY_USER_DIR", "~/negpy-devhome")

    assert get_default_user_dir() == str(home / "negpy-devhome")


def test_broken_documents_falls_back_to_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    blocker = tmp_path / "blocker"
    blocker.write_text("")

    monkeypatch.delenv("NEGPY_USER_DIR", raising=False)
    # Deterministic docs detection on every platform: the linux branch reads
    # XDG_DOCUMENTS_DIR directly, so point it at an uncreatable path.
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(blocker / "OneDrive" / "Documents"))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)

    result = get_default_user_dir()

    # Falls past the broken Documents to home/Documents, created on the spot.
    assert result == str((home / "Documents" / "NegPy").absolute())
    assert Path(result).is_dir()


def test_existing_documents_is_used_without_side_effects(monkeypatch, tmp_path):
    docs = tmp_path / "Docs"
    docs.mkdir()

    monkeypatch.delenv("NEGPY_USER_DIR", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(docs))

    result = get_default_user_dir()

    assert result == str((docs / "NegPy").absolute())
    assert not (docs / "NegPy").exists()


def test_protected_windows_documents_falls_back_to_local_app_data(monkeypatch, tmp_path):
    home = tmp_path / "home"
    local_app_data = tmp_path / "LocalAppData"
    calls = []

    def usable(base, *, probe_write=False):
        base = Path(base)
        calls.append((base, probe_write))
        if base == local_app_data:
            return str((base / "NegPy").absolute())
        return None

    monkeypatch.delenv("NEGPY_USER_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)
    monkeypatch.setattr(paths, "_usable_user_dir", usable)

    result = get_default_user_dir()

    assert result == str((local_app_data / "NegPy").absolute())
    assert (local_app_data, True) in calls
    assert all(probe_write for _base, probe_write in calls)
    assert all(base != home for base, _probe_write in calls)
