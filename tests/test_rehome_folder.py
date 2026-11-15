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


def test_edited_file_names_follow_the_platforms_case_rule(tmp_path, monkeypatch):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    stored = os.path.join(str(tmp_path), "Roll", "a.tif")
    repo.save_file_settings("ha", WorkspaceConfig(), file_path=stored)
    asked = os.path.join(str(tmp_path), "roll")

    assert repo.edited_file_names_under(asked) == set()
    monkeypatch.setattr(os.path, "normcase", str.lower)
    assert repo.edited_file_names_under(asked) == {"a.tif"}


def test_a_virtual_rolls_frames_follow_a_located_folder(tmp_path):
    from negpy.services.assets.rehome import gone_folders_for, missing_member_folders, roll_missing

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old, new = tmp_path / "old", tmp_path / "new"
    new.mkdir()
    (new / "a.tif").write_bytes(b"x")
    roll_id = rolls.create_virtual_roll(repo, "picks", [str(old / "a.tif")])
    entry = rolls.roll_for_id(repo, roll_id)

    assert roll_missing(repo, roll_id) and missing_member_folders(entry) == [str(old)]
    assert gone_folders_for(entry, [str(old)], str(new)) == [str(old)]
    assert gone_folders_for(entry, [str(old)], str(tmp_path)) == []
    rehome_path_prefix(repo, str(old), str(new))
    assert not roll_missing(repo, roll_id)


def test_a_folder_roll_never_moves_onto_a_folder_another_roll_holds(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old, new = str(tmp_path / "old"), str(tmp_path / "new")
    stale = rolls.recognize_folder(repo, old)
    held = rolls.recognize_folder(repo, new)
    vroll = rolls.create_virtual_roll(repo, "picks", [os.path.join(old, "a.tif")])

    rehome_path_prefix(repo, old, new)

    assert rolls.roll_for_id(repo, stale)["folder_path"] == old
    assert rolls.roll_for_id(repo, held)["folder_path"] == new
    assert rolls.roll_for_id(repo, vroll)["member_paths"] == [os.path.join(new, "a.tif")]


def test_a_located_folder_replaces_the_gone_folder_whose_files_it_holds(tmp_path):
    from negpy.services.assets.rehome import gone_folders_for

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    a, b, new_b = tmp_path / "A", tmp_path / "b", tmp_path / "moved_b"
    new_b.mkdir()
    (new_b / "b1.tif").write_bytes(b"x")
    roll_id = rolls.create_virtual_roll(repo, "picks", [str(a / "a1.tif"), str(b / "b1.tif")])

    assert gone_folders_for(rolls.roll_for_id(repo, roll_id), [str(a), str(b)], str(new_b)) == [str(b)]


def test_gone_member_paths_lists_only_a_virtual_rolls_gone_files(tmp_path):
    from negpy.services.assets.rehome import gone_member_paths

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    here = tmp_path / "a.tif"
    here.write_bytes(b"x")
    gone = str(tmp_path / "gone" / "b.tif")
    roll_id = rolls.create_virtual_roll(repo, "picks", [str(here), gone])

    assert gone_member_paths(rolls.roll_for_id(repo, roll_id)) == [gone]
    assert gone_member_paths(rolls.roll_for_id(repo, rolls.recognize_folder(repo, str(tmp_path)))) == []


def test_a_tie_prefers_the_folder_of_the_same_name_and_otherwise_returns_both(tmp_path):
    from negpy.services.assets.rehome import gone_folders_for

    z, b = str(tmp_path / "Z_roll"), str(tmp_path / "B_roll")
    entry = {"kind": "virtual", "member_paths": [os.path.join(f, f"Image ({i}).tif") for f in (z, b) for i in (1, 2, 3)]}
    for name in ("Z_roll", "elsewhere"):
        (tmp_path / "new" / name).mkdir(parents=True)
        for i in (1, 2, 3):
            (tmp_path / "new" / name / f"Image ({i}).tif").write_bytes(b"x")

    assert gone_folders_for(entry, sorted([z, b]), str(tmp_path / "new" / "Z_roll")) == [z]
    assert sorted(gone_folders_for(entry, sorted([z, b]), str(tmp_path / "new" / "elsewhere"))) == sorted([z, b])


def test_a_folder_holding_all_of_a_small_folder_beats_part_of_a_large_one(tmp_path):
    from negpy.services.assets.rehome import gone_folders_for

    z, b = str(tmp_path / "Z"), str(tmp_path / "B")
    entry = {
        "kind": "virtual",
        "member_paths": [os.path.join(z, f"{i}.tif") for i in range(3)] + [os.path.join(b, f"{i}.tif") for i in range(5)],
    }
    new = tmp_path / "picked"
    new.mkdir()
    for i in range(3):
        (new / f"{i}.tif").write_bytes(b"x")

    assert gone_folders_for(entry, sorted([z, b]), str(new)) == [z]


def test_a_refused_folder_roll_keeps_its_removed_frames(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old, new = str(tmp_path / "old"), str(tmp_path / "new")
    stale = rolls.recognize_folder(repo, old)
    rolls.recognize_folder(repo, new)
    rolls.remove_members(repo, stale, [os.path.join(old, "x.tif")], [])

    rehome_path_prefix(repo, old, new)

    entry = rolls.roll_for_id(repo, stale)
    assert entry["folder_path"] == old and entry["excluded_paths"] == [os.path.join(old, "x.tif")]


def test_file_hashes_for_path_lists_halves_forks_and_marks(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    path = str(tmp_path / "gone.tif")
    repo.save_file_settings("h#1", WorkspaceConfig(), file_path=path)
    repo.save_file_settings("h#2#roll:r", WorkspaceConfig(), file_path=path)
    repo.save_file_mark("m", "keeper", file_path=path)

    assert repo.file_hashes_for_path(path) == ["h#1", "h#2#roll:r", "m"]


def test_locating_one_folder_finds_its_moved_siblings_by_name_and_files(tmp_path):
    from negpy.services.assets.rehome import folders_moved_with

    old_parent, new_parent = tmp_path / "Downloads", tmp_path / "Slides"
    a, b, c, inner = (str(old_parent / n) for n in ("roll A", "roll B", "roll C", "roll A/rejects"))
    entry = {"kind": "virtual", "member_paths": [os.path.join(f, "x.tif") for f in (a, b, c, inner)]}
    for name, holds in (("roll A", "x.tif"), ("roll A/rejects", "x.tif"), ("roll B", "x.tif"), ("roll C", "other.tif")):
        (new_parent / name).mkdir(parents=True, exist_ok=True)
        (new_parent / name / holds).write_bytes(b"x")

    moves = folders_moved_with(entry, [a, b, c, inner], a, str(new_parent / "roll A"))

    assert sorted(moves) == sorted(
        [(a, str(new_parent / "roll A")), (inner, str(new_parent / "roll A" / "rejects")), (b, str(new_parent / "roll B"))]
    )
