"""Renaming a roll's folder on disk carries every path-keyed record with it."""

import os

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import composites, rolls
from negpy.services.assets.rehome import rehome_path_prefix


def test_every_path_keyed_store_follows_the_folder(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old, new = str(tmp_path / "roll1"), str(tmp_path / "roll1-portra")
    a, b = os.path.join(old, "a.tif"), os.path.join(old, "b.tif")
    elsewhere = str(tmp_path / "roll10" / "c.tif")
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=a)
    repo.save_file_settings("hc", WorkspaceConfig(), file_path=elsewhere)
    repo.save_file_mark("ha", "keeper", file_path=a)
    vroll = rolls.create_virtual_roll(repo, "picks", [a, elsewhere])
    repo.save_global_setting(composites.COMPOSITES_KEY, {a: {"kind": "stitch", "paths": [a, b], "hash": "hs"}})

    rehome_path_prefix(repo, old, new)

    assert repo.load_file_settings_by_path(os.path.join(new, "a.tif")) is not None
    assert repo.load_file_settings_by_path(elsewhere) is not None  # a sibling sharing the prefix stays
    assert os.path.join(new, "a.tif") in repo.load_file_marks_by_path()
    assert rolls.roll_for_id(repo, vroll)["member_paths"] == [os.path.join(new, "a.tif"), elsewhere]
    saved = composites.saved_composites(repo)
    assert list(saved) == [os.path.join(new, "a.tif")]
    assert saved[os.path.join(new, "a.tif")]["paths"] == [os.path.join(new, "a.tif"), os.path.join(new, "b.tif")]


def test_relink_follows_a_folder_renamed_outside_the_app(tmp_path):
    from negpy.services.assets.rehome import folder_roll_missing, relink_folder_roll

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old, new = tmp_path / "roll1", tmp_path / "renamed"
    old.mkdir()
    (old / "a.tif").write_bytes(b"x")
    roll_id = rolls.recognize_folder(repo, str(old))
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=str(old / "a.tif"))
    old.rename(new)

    assert folder_roll_missing(repo, roll_id)
    assert relink_folder_roll(repo, roll_id, str(new)) == (str(old), "")
    assert not folder_roll_missing(repo, roll_id)
    assert rolls.roll_for_id(repo, roll_id)["folder_path"] == str(new)
    assert repo.load_file_settings_by_path(str(new / "a.tif")) is not None


def test_relink_refuses_a_folder_another_roll_owns(tmp_path):
    from negpy.services.assets.rehome import relink_folder_roll

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    gone = rolls.recognize_folder(repo, str(tmp_path / "gone"))
    rolls.recognize_folder(repo, str(b))

    from negpy.services.assets.rehome import RELINK_FOLDER_TAKEN, RELINK_NOT_A_FOLDER, RELINK_NOT_A_FOLDER_ROLL

    assert relink_folder_roll(repo, gone, str(b)) == (None, RELINK_FOLDER_TAKEN)
    assert relink_folder_roll(repo, gone, str(tmp_path / "nope")) == (None, RELINK_NOT_A_FOLDER)
    assert relink_folder_roll(repo, "no-such-roll", str(b)) == (None, RELINK_NOT_A_FOLDER_ROLL)


def test_a_folder_must_hold_the_rolls_edited_files(tmp_path):
    from negpy.services.assets.rehome import folder_fits_roll

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old, right, wrong = tmp_path / "old", tmp_path / "right", tmp_path / "wrong"
    for folder in (right, wrong):
        folder.mkdir()
    (right / "A.TIF").write_bytes(b"x")
    (wrong / "other.tif").write_bytes(b"x")
    roll_id = rolls.recognize_folder(repo, str(old))
    assert folder_fits_roll(repo, roll_id, str(wrong))  # no saved edits: any folder
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=str(old / "a.tif"))

    assert folder_fits_roll(repo, roll_id, str(right))
    assert not folder_fits_roll(repo, roll_id, str(wrong))


def test_a_default_roll_name_follows_the_folder_and_a_set_name_stays(tmp_path):
    from negpy.services.assets.rehome import relink_folder_roll

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    for name in ("one", "two", "one_new", "two_new"):
        (tmp_path / name).mkdir()
    plain = rolls.recognize_folder(repo, str(tmp_path / "one"))
    named = rolls.recognize_folder(repo, str(tmp_path / "two"))
    rolls.rename_roll(repo, named, "Portra trip")

    relink_folder_roll(repo, plain, str(tmp_path / "one_new"))
    relink_folder_roll(repo, named, str(tmp_path / "two_new"))

    assert rolls.roll_for_id(repo, plain)["name"] == "one_new"
    assert rolls.roll_for_id(repo, named)["name"] == "Portra trip"


def test_a_nested_default_roll_name_follows_its_import_path(tmp_path):
    from negpy.services.assets.rehome import relink_folder_roll

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    parent = tmp_path / "scans"
    (parent / "2024" / "r1").mkdir(parents=True)
    (parent / "2024" / "r1_new").mkdir()
    rolls.import_subfolders_as_rolls(repo, str(parent))
    roll_id = rolls.folder_roll_id_for_path(repo, str(parent / "2024" / "r1"))
    if roll_id is None:
        roll_id = rolls.recognize_folder(repo, str(parent / "2024" / "r1"))
    default = rolls.roll_for_id(repo, roll_id)["name"]

    relink_folder_roll(repo, roll_id, str(parent / "2024" / "r1_new"))

    assert rolls.roll_for_id(repo, roll_id)["name"] == rolls.folder_roll_name(repo, str(parent / "2024" / "r1_new"))
    assert default != rolls.roll_for_id(repo, roll_id)["name"]


def test_every_relink_reason_has_a_dialog_message():
    from negpy.desktop.view.sidebar.library_tree import _RELINK_MESSAGES
    from negpy.services.assets.rehome import RELINK_FOLDER_TAKEN, RELINK_NOT_A_FOLDER, RELINK_NOT_A_FOLDER_ROLL

    assert set(_RELINK_MESSAGES) == {RELINK_FOLDER_TAKEN, RELINK_NOT_A_FOLDER, RELINK_NOT_A_FOLDER_ROLL}
    assert all(_RELINK_MESSAGES.values())
