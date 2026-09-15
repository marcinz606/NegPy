"""A Roll is a navigation layer over a folder or a hand-built set of paths; it never
scopes or duplicates the edits themselves."""

from unittest.mock import MagicMock

from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.rolls import (
    add_extra_member,
    all_rolls_sorted,
    create_virtual_roll,
    delete_roll,
    folder_roll_id_for_path,
    import_subfolders_as_rolls,
    recognize_folder,
    rename_roll,
    roll_for_id,
    saved_rolls,
    virtual_rolls,
)


def _repo() -> MagicMock:
    """A repository whose global settings live in a dict, so a write is readable back."""
    repo = MagicMock(spec=StorageRepository)
    store: dict = {}
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    repo.settings = store
    return repo


def test_recognize_folder_names_it_from_the_path():
    repo = _repo()
    roll_id = recognize_folder(repo, "/scans/2024-10-portra")
    entry = roll_for_id(repo, roll_id)
    assert entry["kind"] == "folder"
    assert entry["folder_path"] == "/scans/2024-10-portra"
    assert entry["name"] == "2024-10-portra"
    assert entry["extra_paths"] == []


def test_recognize_folder_is_idempotent():
    repo = _repo()
    first = recognize_folder(repo, "/scans/roll_a")
    second = recognize_folder(repo, "/scans/roll_a")
    assert first == second
    assert len(saved_rolls(repo)) == 1


def test_folder_roll_id_for_path_finds_a_recognized_folder():
    repo = _repo()
    assert folder_roll_id_for_path(repo, "/scans/roll_a") is None
    roll_id = recognize_folder(repo, "/scans/roll_a")
    assert folder_roll_id_for_path(repo, "/scans/roll_a") == roll_id


def test_create_virtual_roll_stores_the_exact_member_list():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef", "/b.nef"])
    entry = roll_for_id(repo, roll_id)
    assert entry["kind"] == "virtual"
    assert entry["name"] == "Portra"
    assert entry["member_paths"] == ["/a.nef", "/b.nef"]


def test_add_extra_member_extends_a_folder_rolls_extra_paths():
    repo = _repo()
    roll_id = recognize_folder(repo, "/scans/roll_a")
    add_extra_member(repo, roll_id, "/elsewhere/c.nef")
    assert roll_for_id(repo, roll_id)["extra_paths"] == ["/elsewhere/c.nef"]


def test_add_extra_member_extends_a_virtual_rolls_member_paths():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    add_extra_member(repo, roll_id, "/b.nef")
    assert roll_for_id(repo, roll_id)["member_paths"] == ["/a.nef", "/b.nef"]


def test_add_extra_member_does_not_duplicate():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    add_extra_member(repo, roll_id, "/a.nef")
    assert roll_for_id(repo, roll_id)["member_paths"] == ["/a.nef"]


def test_add_extra_member_on_unknown_roll_is_a_noop():
    repo = _repo()
    add_extra_member(repo, "not-a-real-id", "/a.nef")
    assert saved_rolls(repo) == {}


def test_rename_and_delete_roll():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", [])
    rename_roll(repo, roll_id, "Portra 400")
    assert roll_for_id(repo, roll_id)["name"] == "Portra 400"

    delete_roll(repo, roll_id)
    assert roll_for_id(repo, roll_id) is None
    assert saved_rolls(repo) == {}


def test_virtual_rolls_lists_only_virtual_ones_sorted_by_name():
    repo = _repo()
    recognize_folder(repo, "/scans/roll_a")
    create_virtual_roll(repo, "Zebra", [])
    create_virtual_roll(repo, "apple", [])

    names = [entry["name"] for _id, entry in virtual_rolls(repo)]
    assert names == ["apple", "Zebra"]


def test_all_rolls_sorted_lists_every_kind_by_name():
    repo = _repo()
    recognize_folder(repo, "/scans/roll_a", name="Zebra")
    create_virtual_roll(repo, "apple", [])

    names = [entry["name"] for _id, entry in all_rolls_sorted(repo)]
    assert names == ["apple", "Zebra"]


def test_import_subfolders_as_rolls_recognizes_each_immediate_subfolder(tmp_path):
    (tmp_path / "roll_a").mkdir()
    (tmp_path / "roll_b").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "roll_a" / "nested").mkdir()
    repo = _repo()

    roll_ids = import_subfolders_as_rolls(repo, str(tmp_path))

    names = sorted(roll_for_id(repo, rid)["name"] for rid in roll_ids)
    assert names == ["roll_a", "roll_b"]
    # One level only: "nested" inside roll_a is not recognized on its own.
    assert folder_roll_id_for_path(repo, str(tmp_path / "roll_a" / "nested")) is None


def test_import_subfolders_as_rolls_is_idempotent_per_subfolder(tmp_path):
    (tmp_path / "roll_a").mkdir()
    repo = _repo()

    first = import_subfolders_as_rolls(repo, str(tmp_path))
    second = import_subfolders_as_rolls(repo, str(tmp_path))

    assert first == second
    assert len(saved_rolls(repo)) == 1


def test_import_subfolders_as_rolls_on_a_missing_parent_returns_nothing():
    repo = _repo()
    assert import_subfolders_as_rolls(repo, "/does/not/exist") == []
