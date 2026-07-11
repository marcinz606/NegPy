"""
User-dir resolution must survive a registered Documents folder that does not
exist on disk — e.g. a OneDrive-backed Documents after OneDrive is unlinked or
signed out (issue #441: startup makedirs died with WinError 2).
"""

import os
from pathlib import Path

from negpy.kernel.system.paths import _usable_user_dir, get_default_user_dir


def test_usable_dir_existing_base_returns_without_creating(tmp_path):
    result = _usable_user_dir(tmp_path)

    assert result == str((tmp_path / "NegPy").absolute())
    # Base exists, so resolution must not touch the filesystem.
    assert not (tmp_path / "NegPy").exists()


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


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("NEGPY_USER_DIR", str(tmp_path / "custom"))

    assert get_default_user_dir() == os.path.abspath(str(tmp_path / "custom"))


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


def test_existing_documents_used_without_side_effects(monkeypatch, tmp_path):
    docs = tmp_path / "Docs"
    docs.mkdir()

    monkeypatch.delenv("NEGPY_USER_DIR", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(docs))

    result = get_default_user_dir()

    assert result == str((docs / "NegPy").absolute())
    assert not (docs / "NegPy").exists()
