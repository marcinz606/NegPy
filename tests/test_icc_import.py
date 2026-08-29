"""Importing a user ICC profile into the app's profile folder."""

import os

import pytest

from negpy.infrastructure.display.color_mgmt import ColorService, import_icc_profile


@pytest.fixture
def profile_bytes() -> bytes:
    src = next(p for p in ColorService.get_available_profiles() if p.lower().endswith((".icc", ".icm")))
    with open(src, "rb") as f:
        return f.read()


def _write(path, payload: bytes) -> str:
    with open(path, "wb") as f:
        f.write(payload)
    return str(path)


def test_imports_a_profile(tmp_path, profile_bytes):
    src = _write(tmp_path / "Paper.icc", profile_bytes)
    dest = tmp_path / "icc"
    dest.mkdir()

    stored = import_icc_profile(src, str(dest))

    assert stored == os.path.join(str(dest), "Paper.icc")
    assert open(stored, "rb").read() == profile_bytes


def test_reimporting_the_same_bytes_reuses_the_stored_file(tmp_path, profile_bytes):
    src = _write(tmp_path / "Paper.icc", profile_bytes)
    dest = tmp_path / "icc"
    dest.mkdir()

    first = import_icc_profile(src, str(dest))
    assert import_icc_profile(src, str(dest)) == first
    assert os.listdir(dest) == ["Paper.icc"]


def test_a_different_profile_under_a_taken_name_lands_beside_it(tmp_path, profile_bytes):
    """Overwriting would silently retarget every edit already using the stored profile."""
    dest = tmp_path / "icc"
    dest.mkdir()
    import_icc_profile(_write(tmp_path / "Paper.icc", profile_bytes), str(dest))

    other = tmp_path / "b"
    other.mkdir()
    stored = import_icc_profile(_write(other / "Paper.icc", profile_bytes + b"\x00"), str(dest))

    assert os.path.basename(stored) == "Paper (2).icc"
    assert open(os.path.join(str(dest), "Paper.icc"), "rb").read() == profile_bytes


def test_a_file_that_is_not_a_profile_is_rejected(tmp_path):
    dest = tmp_path / "icc"
    dest.mkdir()

    with pytest.raises(ValueError):
        import_icc_profile(_write(tmp_path / "notes.icc", b"not a profile"), str(dest))

    assert os.listdir(dest) == []
