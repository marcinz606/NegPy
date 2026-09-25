"""A Roll is a navigation layer over a folder or a hand-built set of paths; it does not
scope or duplicate the edits themselves, with one exception: roll-wide defaults for a
handful of film, rig and scanning facts (see TestRollDefaults below)."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.domain.models import ProcessConfig, WorkspaceConfig
from negpy.features.process.models import DemosaicMode, ProcessMode
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.rolls import (
    add_to_scene,
    baseline_label,
    create_scene,
    delete_scene,
    next_scene_name,
    remove_from_scenes,
    rename_scene,
    roll_scenes,
    scene_by_hash,
    scene_normalization,
    set_scene_normalization,
    add_extra_members,
    all_rolls_sorted,
    create_virtual_roll,
    delete_roll,
    fork_edit,
    folder_roll_id_for_path,
    frame_override_cards,
    delete_folder_rolls,
    discovery_filters,
    import_sources,
    matches_discovery_filter,
    prune_filtered_rolls,
    set_discovery_filters,
    import_subfolders_as_rolls,
    is_forked,
    recognize_folder,
    rename_folder_roll_disk,
    rename_roll,
    resolve_roll_baseline,
    resolve_roll_config,
    roll_defaults,
    section_push,
    set_section_push,
    roll_edit_hash,
    roll_for_id,
    roll_normalization,
    rolls_containing_path,
    saved_rolls,
    set_frame_override,
    set_roll_defaults,
    set_roll_normalization,
    unfork_edit,
    unforked_hash,
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
    add_extra_members(repo, roll_id, ["/elsewhere/c.nef"])
    assert roll_for_id(repo, roll_id)["extra_paths"] == ["/elsewhere/c.nef"]


def test_add_extra_member_extends_a_virtual_rolls_member_paths():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    add_extra_members(repo, roll_id, ["/b.nef"])
    assert roll_for_id(repo, roll_id)["member_paths"] == ["/a.nef", "/b.nef"]


def test_add_extra_member_does_not_duplicate():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    add_extra_members(repo, roll_id, ["/a.nef"])
    assert roll_for_id(repo, roll_id)["member_paths"] == ["/a.nef"]


def test_add_extra_member_on_unknown_roll_is_a_noop():
    repo = _repo()
    add_extra_members(repo, "not-a-real-id", ["/a.nef"])
    assert saved_rolls(repo) == {}


def test_rename_and_delete_roll():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", [])
    rename_roll(repo, roll_id, "Portra 400")
    assert roll_for_id(repo, roll_id)["name"] == "Portra 400"

    delete_roll(repo, roll_id)
    assert roll_for_id(repo, roll_id) is None
    assert saved_rolls(repo) == {}


def test_rename_folder_roll_disk_renames_and_updates_folder_path(tmp_path):
    repo = _repo()
    old = tmp_path / "roll_a"
    old.mkdir()
    (old / "frame001.tif").write_bytes(b"x")
    roll_id = recognize_folder(repo, str(old))

    new_path = rename_folder_roll_disk(repo, roll_id, "roll_b")

    assert new_path == str(tmp_path / "roll_b")
    assert not old.exists()
    assert (tmp_path / "roll_b" / "frame001.tif").exists()
    assert roll_for_id(repo, roll_id)["folder_path"] == new_path


def test_rename_folder_roll_disk_refuses_a_sibling_collision(tmp_path):
    repo = _repo()
    old = tmp_path / "roll_a"
    old.mkdir()
    (tmp_path / "roll_b").mkdir()
    roll_id = recognize_folder(repo, str(old))

    assert rename_folder_roll_disk(repo, roll_id, "roll_b") is None
    assert old.exists()
    assert roll_for_id(repo, roll_id)["folder_path"] == str(old)


def test_rename_folder_roll_disk_same_name_is_a_noop_success(tmp_path):
    repo = _repo()
    old = tmp_path / "roll_a"
    old.mkdir()
    roll_id = recognize_folder(repo, str(old))

    assert rename_folder_roll_disk(repo, roll_id, "roll_a") == str(old)
    assert old.exists()


def test_rename_folder_roll_disk_on_a_virtual_roll_is_a_noop(tmp_path):
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", [])
    assert rename_folder_roll_disk(repo, roll_id, "anything") is None


def test_rename_folder_roll_disk_missing_folder_is_a_noop(tmp_path):
    repo = _repo()
    roll_id = recognize_folder(repo, str(tmp_path / "gone"))
    assert rename_folder_roll_disk(repo, roll_id, "roll_b") is None


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


def _roll_dir(path, *images):
    path.mkdir(parents=True, exist_ok=True)
    for name in images or ("f1.tif",):
        (path / name).write_bytes(b"x")
    return path


def test_import_subfolders_as_rolls_finds_nested_roll_folders(tmp_path):
    tmp_path = tmp_path / "scans"
    _roll_dir(tmp_path / "2024" / "roll_a")
    _roll_dir(tmp_path / "2024" / "summer" / "roll_b")
    _roll_dir(tmp_path / ".hidden")
    (tmp_path / "empty").mkdir()
    repo = _repo()

    roll_ids = import_subfolders_as_rolls(repo, str(tmp_path))

    names = sorted(roll_for_id(repo, rid)["name"] for rid in roll_ids)
    assert names == ["scans/2024/roll_a", "scans/2024/summer/roll_b"]
    # A folder with no images of its own is not a roll.
    assert folder_roll_id_for_path(repo, str(tmp_path / "2024")) is None


def test_import_subfolders_as_rolls_does_not_walk_into_a_roll_folder(tmp_path):
    _roll_dir(tmp_path / "roll_a")
    _roll_dir(tmp_path / "roll_a" / "TIFF", "f1_export.tif")
    repo = _repo()

    import_subfolders_as_rolls(repo, str(tmp_path))

    assert folder_roll_id_for_path(repo, str(tmp_path / "roll_a")) is not None
    assert folder_roll_id_for_path(repo, str(tmp_path / "roll_a" / "TIFF")) is None


def test_import_subfolders_as_rolls_names_each_roll_by_its_path_from_the_picked_folder(tmp_path):
    day = tmp_path / "20260901"
    _roll_dir(day / "kentmere_400_1")
    _roll_dir(day / "kentmere_400_2")
    repo = _repo()

    roll_ids = import_subfolders_as_rolls(repo, str(day))

    names = sorted(roll_for_id(repo, rid)["name"] for rid in roll_ids)
    assert names == ["20260901/kentmere_400_1", "20260901/kentmere_400_2"]


def test_import_subfolders_as_rolls_names_a_picked_roll_folder_by_its_own_name(tmp_path):
    roll = _roll_dir(tmp_path / "kentmere_400_1")
    repo = _repo()

    [roll_id] = import_subfolders_as_rolls(repo, str(roll))

    assert roll_for_id(repo, roll_id)["name"] == "kentmere_400_1"


def test_import_subfolders_as_rolls_is_idempotent_per_subfolder(tmp_path):
    _roll_dir(tmp_path / "roll_a")
    repo = _repo()

    first = import_subfolders_as_rolls(repo, str(tmp_path))
    second = import_subfolders_as_rolls(repo, str(tmp_path))

    assert first == second
    assert len(saved_rolls(repo)) == 1


def test_import_subfolders_as_rolls_remembers_the_parent_as_a_source(tmp_path):
    _roll_dir(tmp_path / "roll_a")
    repo = _repo()

    import_subfolders_as_rolls(repo, str(tmp_path))
    import_subfolders_as_rolls(repo, str(tmp_path))

    assert import_sources(repo) == [str(tmp_path)]


def test_import_subfolders_as_rolls_can_skip_a_deleted_folder_roll(tmp_path):
    roll_a = _roll_dir(tmp_path / "roll_a")
    _roll_dir(tmp_path / "roll_b")
    repo = _repo()
    import_subfolders_as_rolls(repo, str(tmp_path))
    delete_roll(repo, folder_roll_id_for_path(repo, str(roll_a)))

    import_subfolders_as_rolls(repo, str(tmp_path), skip_dismissed=True)
    assert folder_roll_id_for_path(repo, str(roll_a)) is None
    assert len(saved_rolls(repo)) == 1

    import_subfolders_as_rolls(repo, str(tmp_path))
    assert folder_roll_id_for_path(repo, str(roll_a)) is not None


def test_recognizing_a_deleted_folder_by_hand_lets_discovery_find_it_again(tmp_path):
    roll_a = _roll_dir(tmp_path / "roll_a")
    repo = _repo()
    import_subfolders_as_rolls(repo, str(tmp_path))
    delete_roll(repo, folder_roll_id_for_path(repo, str(roll_a)))
    roll_id = recognize_folder(repo, str(roll_a))

    assert import_subfolders_as_rolls(repo, str(tmp_path), skip_dismissed=True) == [roll_id]


def test_import_subfolders_as_rolls_on_a_missing_parent_returns_nothing():
    repo = _repo()
    assert import_subfolders_as_rolls(repo, "/does/not/exist") == []
    assert import_sources(repo) == []


def test_roll_edit_hash_suffixes_the_roll_id():
    assert roll_edit_hash("abc123", "r1") == "abc123#roll:r1"


def test_roll_edit_hash_preserves_a_half_frame_suffix():
    """A half-frame asset's own suffix (``#1``/``#2``) survives, so each half forks to
    its own identity rather than collapsing together."""
    assert roll_edit_hash("abc123#1", "r1") == "abc123#1#roll:r1"


def test_unforked_hash_strips_only_the_roll_suffix():
    assert unforked_hash("abc123#roll:r1") == "abc123"
    assert unforked_hash("abc123#1#roll:r1") == "abc123#1"
    assert unforked_hash("abc123#1") == "abc123#1"
    assert unforked_hash("abc123") == "abc123"


def test_fork_edit_seeds_the_forked_hash_and_marks_it_forked():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    config = object()

    forked = fork_edit(repo, roll_id, "abc123", "/a.nef", config)

    assert forked == "abc123#roll:" + roll_id
    repo.save_file_settings.assert_called_once_with(forked, config, file_path="/a.nef")
    assert is_forked(repo, roll_id, "abc123")


def test_fork_edit_on_an_unknown_roll_is_a_noop():
    repo = _repo()
    forked = fork_edit(repo, "not-a-real-id", "abc123", "/a.nef", object())
    assert forked == "abc123"
    repo.save_file_settings.assert_not_called()


def test_is_forked_is_false_before_forking():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    assert not is_forked(repo, roll_id, "abc123")


def test_unfork_edit_reverses_fork_edit():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    forked = fork_edit(repo, roll_id, "abc123", "/a.nef", object())

    unfork_edit(repo, roll_id, "abc123")

    assert not is_forked(repo, roll_id, "abc123")
    repo.delete_file_settings.assert_called_once_with(forked)


def test_rolls_containing_path_finds_a_folder_roll_by_prefix(tmp_path):
    repo = _repo()
    folder = tmp_path / "roll_a"
    folder.mkdir()
    roll_id = recognize_folder(repo, str(folder))
    assert rolls_containing_path(repo, str(folder / "frame001.tif")) == [roll_id]


def test_rolls_containing_path_finds_a_folder_rolls_extra_member():
    repo = _repo()
    roll_id = recognize_folder(repo, "/scans/roll_a")
    add_extra_members(repo, roll_id, ["/elsewhere/c.nef"])
    assert rolls_containing_path(repo, "/elsewhere/c.nef") == [roll_id]


def test_rolls_containing_path_finds_a_virtual_rolls_member():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    assert rolls_containing_path(repo, "/a.nef") == [roll_id]


def test_rolls_containing_path_lists_every_matching_roll(tmp_path):
    repo = _repo()
    folder = tmp_path / "roll_a"
    folder.mkdir()
    frame = folder / "frame001.tif"
    folder_roll = recognize_folder(repo, str(folder))
    virtual_roll = create_virtual_roll(repo, "Picks", [str(frame)])
    assert sorted(rolls_containing_path(repo, str(frame))) == sorted([folder_roll, virtual_roll])


def test_rolls_containing_path_is_empty_for_an_unshared_path():
    repo = _repo()
    create_virtual_roll(repo, "Portra", ["/a.nef"])
    assert rolls_containing_path(repo, "/unrelated.nef") == []


class TestRollDefaults:
    """Film, rig and scanning facts a roll shares across every member frame, unless a
    frame has locked the owning card to its own value."""

    def test_a_field_with_no_roll_default_leaves_the_frame_alone(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        config = WorkspaceConfig(process=ProcessConfig(linear_raw=True))

        resolved = resolve_roll_config(repo, roll_id, "h1", config)

        assert resolved is config

    def test_a_roll_default_overrides_the_frames_own_value(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True, narrowband_scan=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        assert resolved.process.linear_raw is True
        assert resolved.process.narrowband_scan is True

    def test_no_roll_id_leaves_the_frame_alone(self):
        repo = _repo()
        assert resolve_roll_config(repo, None, "h1", WorkspaceConfig()) == WorkspaceConfig()

    def test_locking_a_card_keeps_that_frames_own_value(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True, demosaic_preview=DemosaicMode.VNG)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        # sensor (Calibration) is locked, so linear_raw keeps the frame's own value...
        assert resolved.process.linear_raw is False
        # ...but demosaic (a different card) is not locked, so it still takes the roll default.
        assert resolved.process.demosaic_preview == DemosaicMode.VNG

    def test_locking_does_not_affect_a_different_frame_in_the_same_roll(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h2", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        assert resolved.process.linear_raw is True

    def test_unlocking_a_card_reverts_to_the_roll_default(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=False)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(linear_raw=False)))

        assert resolved.process.linear_raw is True
        assert frame_override_cards(repo, roll_id, "h1") == set()

    def test_frame_override_cards_is_empty_for_an_unknown_roll(self):
        repo = _repo()
        assert frame_override_cards(repo, "not-a-real-id", "h1") == set()

    def test_set_roll_defaults_on_unknown_roll_is_a_noop(self):
        repo = _repo()
        set_roll_defaults(repo, "not-a-real-id", linear_raw=True)
        assert saved_rolls(repo) == {}

    def test_roll_defaults_reads_back_what_was_set(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, linear_raw=True)
        set_roll_defaults(repo, roll_id, hue_trim=2.5)

        assert roll_defaults(repo, roll_id) == {"linear_raw": True, "hue_trim": 2.5}

    def test_process_mode_follows_the_roll_unless_the_film_card_is_locked(self):
        """process_mode is a roll default on the "film" card, like everything else --
        locking a different card leaves it following the roll."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, process_mode=ProcessMode.BW)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)
        set_frame_override(repo, roll_id, "h1", "demosaic", locked=True)
        set_frame_override(repo, roll_id, "h1", "process", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.C41)))

        assert resolved.process.process_mode == ProcessMode.BW

    def test_process_mode_can_be_locked_away_on_the_film_card(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, process_mode=ProcessMode.BW)
        set_frame_override(repo, roll_id, "h1", "film", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.C41)))

        assert resolved.process.process_mode == ProcessMode.C41

    def test_positive_source_follows_the_roll_unless_the_film_card_is_locked(self):
        """Positive is a roll default on the same "film" card as process_mode."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, positive_source=True)
        set_frame_override(repo, roll_id, "h1", "sensor", locked=True)
        set_frame_override(repo, roll_id, "h1", "demosaic", locked=True)
        set_frame_override(repo, roll_id, "h1", "process", locked=True)

        resolved = resolve_roll_config(
            repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=False))
        )

        assert resolved.process.positive_source is True

    def test_a_geometry_field_follows_the_roll_unless_its_card_is_locked(self):
        """Roll defaults span config sections: the Auto Crop card's fields live on
        GeometryConfig, not ProcessConfig."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, autocrop_rebate_trim=0.5, distortion_k1=0.02)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig())

        assert resolved.geometry.autocrop_rebate_trim == 0.5
        assert resolved.geometry.distortion_k1 == 0.02

    def test_locking_auto_crop_leaves_lens_following_the_roll(self):
        """Both cards write GeometryConfig, so one lock must not take the other with it."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, autocrop_rebate_trim=0.5, distortion_k1=0.02)
        set_frame_override(repo, roll_id, "h1", "autocrop", locked=True)

        resolved = resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig())

        assert resolved.geometry.autocrop_rebate_trim == 1.0
        assert resolved.geometry.distortion_k1 == 0.02

    def test_flat_field_follows_the_roll_unless_its_card_is_locked(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, apply=True, profile_id="rig-1")

        assert resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig()).flatfield.profile_id == "rig-1"

        set_frame_override(repo, roll_id, "h1", "flatfield", locked=True)
        assert resolve_roll_config(repo, roll_id, "h1", WorkspaceConfig()).flatfield.profile_id == ""

    def test_positive_source_can_be_locked_away_on_the_film_card(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_defaults(repo, roll_id, positive_source=True)
        set_frame_override(repo, roll_id, "h1", "film", locked=True)

        resolved = resolve_roll_config(
            repo, roll_id, "h1", WorkspaceConfig(process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=False))
        )

        assert resolved.process.positive_source is False


class TestSectionPush:
    """What a frame-level card last pushed to the roll: a record, not a binding -- it
    never overlays onto another frame, it only lets the card say whether the frame in
    front of you still agrees with the roll."""

    def test_a_roll_with_no_push_reads_empty(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        assert section_push(repo, roll_id, "tone") == {}

    def test_it_reads_back_what_was_pushed(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_section_push(repo, roll_id, "tone", {"dye_separation": 0.4})
        assert section_push(repo, roll_id, "tone") == {"dye_separation": 0.4}

    def test_a_second_push_merges_rather_than_replaces(self):
        """Applying two of a card's settings in two goes leaves both at the roll."""
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_section_push(repo, roll_id, "tone", {"dye_separation": 0.4})
        set_section_push(repo, roll_id, "tone", {"toe": 0.2})
        assert section_push(repo, roll_id, "tone") == {"dye_separation": 0.4, "toe": 0.2}

    def test_cards_do_not_share_a_record(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_section_push(repo, roll_id, "tone", {"dye_separation": 0.4})
        assert section_push(repo, roll_id, "finish") == {}

    def test_pushing_to_an_unknown_roll_is_a_noop(self):
        repo = _repo()
        set_section_push(repo, "not-a-real-id", "tone", {"dye_separation": 0.4})
        assert saved_rolls(repo) == {}


class TestRollNormalization:
    """A roll's own Roll Analysis baseline: written only by Roll Analysis itself, read
    by any frame's Use Luma/Color Average axes -- unlike ROLL_DEFAULT_FIELDS, this has no
    lock/override of its own."""

    def test_unanalyzed_roll_has_no_baseline(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        assert roll_normalization(repo, roll_id) is None

    def test_reads_back_what_was_set(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (0.01, 0.0, -0.01))

        data = roll_normalization(repo, roll_id)

        assert data == {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.01, 0.0, -0.01), "axis": None, "outliers": ()}

    def test_defaults_cast_to_zero(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

        assert roll_normalization(repo, roll_id)["cast"] == (0.0, 0.0, 0.0)

    def test_overwrites_a_previous_baseline(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        set_roll_normalization(repo, roll_id, (0.2, 0.2, 0.2), (0.8, 0.8, 0.8))

        assert roll_normalization(repo, roll_id)["floors"] == (0.2, 0.2, 0.2)

    def test_set_on_unknown_roll_is_a_noop(self):
        repo = _repo()
        set_roll_normalization(repo, "not-a-real-id", (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        assert saved_rolls(repo) == {}

    def test_normalization_is_isolated_per_roll(self):
        repo = _repo()
        roll_a = create_virtual_roll(repo, "Portra", [])
        roll_b = create_virtual_roll(repo, "Tri-X", [])
        set_roll_normalization(repo, roll_a, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

        assert roll_normalization(repo, roll_a) is not None
        assert roll_normalization(repo, roll_b) is None


class TestScenes:
    def _roll(self):
        repo = _repo()
        return repo, create_virtual_roll(repo, "Portra", [])

    def test_create_lists_scene_with_members_and_no_baseline(self):
        repo, roll_id = self._roll()
        sid = create_scene(repo, roll_id, "Beach", ["a", "b", "a"])

        assert roll_scenes(repo, roll_id) == [(sid, {"name": "Beach", "member_hashes": ["a", "b"], "normalization": None})]
        assert scene_normalization(repo, roll_id, sid) is None

    def test_frame_is_in_one_scene_only(self):
        repo, roll_id = self._roll()
        first = create_scene(repo, roll_id, "Beach", ["a", "b"])
        second = create_scene(repo, roll_id, "Night", ["b", "c"])

        assert scene_by_hash(repo, roll_id) == {"a": (1, first, "Beach"), "b": (2, second, "Night"), "c": (2, second, "Night")}

    def test_emptied_scene_is_dropped(self):
        repo, roll_id = self._roll()
        create_scene(repo, roll_id, "Beach", ["a"])
        night = create_scene(repo, roll_id, "Night", ["a", "b"])

        assert [sid for sid, _e in roll_scenes(repo, roll_id)] == [night]

    def test_add_moves_frame_and_keeps_ordinals(self):
        repo, roll_id = self._roll()
        beach = create_scene(repo, roll_id, "Beach", ["a", "b"])
        night = create_scene(repo, roll_id, "Night", ["c"])
        add_to_scene(repo, roll_id, beach, ["c"])
        add_to_scene(repo, roll_id, beach, ["b"])

        assert [sid for sid, _e in roll_scenes(repo, roll_id)] == [beach]
        assert scene_by_hash(repo, roll_id)["c"] == (1, beach, "Beach")
        assert night not in dict(roll_scenes(repo, roll_id))

    def test_add_to_later_scene_keeps_earlier_ordinal(self):
        repo, roll_id = self._roll()
        beach = create_scene(repo, roll_id, "Beach", ["a", "b"])
        night = create_scene(repo, roll_id, "Night", ["c"])
        add_to_scene(repo, roll_id, night, ["b"])

        assert scene_by_hash(repo, roll_id) == {"a": (1, beach, "Beach"), "c": (2, night, "Night"), "b": (2, night, "Night")}

    def test_remove_rename_dissolve(self):
        repo, roll_id = self._roll()
        sid = create_scene(repo, roll_id, "Beach", ["a", "b"])
        remove_from_scenes(repo, roll_id, ["a"])
        rename_scene(repo, roll_id, sid, "Shore")
        assert scene_by_hash(repo, roll_id) == {"b": (1, sid, "Shore")}

        delete_scene(repo, roll_id, sid)
        assert roll_scenes(repo, roll_id) == []

    def test_normalization_round_trip(self):
        repo, roll_id = self._roll()
        sid = create_scene(repo, roll_id, "Beach", ["a"])
        axis = ((-1.0, -1.1, -1.2), (-0.4, -0.5, -0.6), None, 0.8)
        set_scene_normalization(repo, roll_id, sid, (0.1, 0.2, 0.3), (0.7, 0.8, 0.9), outliers=("a",), axis=axis)

        assert scene_normalization(repo, roll_id, sid) == {
            "floors": (0.1, 0.2, 0.3),
            "ceils": (0.7, 0.8, 0.9),
            "axis": axis,
            "outliers": ("a",),
        }
        assert roll_normalization(repo, roll_id) is None

    def test_roll_record_keeps_its_outliers_and_an_old_record_has_none(self):
        repo, roll_id = self._roll()
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), outliers=("a", "b"))
        assert roll_normalization(repo, roll_id)["outliers"] == ("a", "b")

        store = saved_rolls(repo)
        del store[roll_id]["normalization"]["outliers"]
        repo.save_global_setting("rolls_by_id", store)
        assert roll_normalization(repo, roll_id)["outliers"] == ()

    def test_unknown_roll_is_a_noop(self):
        repo = _repo()
        assert create_scene(repo, "nope", "Beach", ["a"]) is None
        assert roll_scenes(repo, None) == []
        assert scene_by_hash(repo, None) == {}
        assert saved_rolls(repo) == {}

    def test_next_scene_name_skips_taken(self):
        repo, roll_id = self._roll()
        assert next_scene_name(repo, roll_id) == "Scene 1"
        create_scene(repo, roll_id, "Scene 2", ["a"])
        assert next_scene_name(repo, roll_id) == "Scene 3"


class TestResolveRollBaseline:
    def _riding(self, **process):
        cfg = WorkspaceConfig()
        return replace(cfg, process=replace(cfg.process, use_luma_average=True, use_color_average=True, **process))

    def test_frame_without_a_baseline_takes_the_rolls(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

        out = resolve_roll_baseline(repo, roll_id, "h1", self._riding())

        assert out.process.locked_floors == (0.1, 0.1, 0.1)
        assert out.process.locked_ceils == (0.9, 0.9, 0.9)

    def test_scene_member_takes_the_scenes(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        sid = create_scene(repo, roll_id, "Beach", ["h1"])
        set_scene_normalization(repo, roll_id, sid, (0.2, 0.2, 0.2), (0.8, 0.8, 0.8))

        assert resolve_roll_baseline(repo, roll_id, "h1", self._riding()).process.locked_floors == (0.2, 0.2, 0.2)

    def test_existing_baseline_off_axes_and_locked_frames_are_left_alone(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        own = self._riding(locked_floors=(0.3, 0.3, 0.3), locked_ceils=(0.7, 0.7, 0.7))
        off = WorkspaceConfig()
        locked = self._riding(lock_bounds=True)

        for cfg in (own, off, locked):
            assert resolve_roll_baseline(repo, roll_id, "h1", cfg) is cfg

    def test_unanalyzed_roll_leaves_the_frame_alone(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        cfg = self._riding()
        assert resolve_roll_baseline(repo, roll_id, "h1", cfg) is cfg


class TestBaselineLabel:
    def _process(self, source, roll_name=None):
        return replace(ProcessConfig(), baseline_source=source, roll_name=roll_name)

    def test_names_roll_scene_and_frame_live(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        sid = create_scene(repo, roll_id, "Beach", ["h1"])
        rename_scene(repo, roll_id, sid, "Shore")

        assert baseline_label(repo, self._process(f"roll:{roll_id}")) == "Roll “Portra”"
        assert baseline_label(repo, self._process(f"scene:{sid}")) == "Scene “Shore”"
        assert baseline_label(repo, self._process("frame:f003.tif")) == "Frame “f003.tif”"

    def test_legacy_and_deleted_sources(self):
        repo = _repo()
        assert baseline_label(repo, self._process("", roll_name="Tri-X")) == "Roll “Tri-X”"
        assert baseline_label(repo, self._process("")) == "a saved baseline"
        assert baseline_label(repo, self._process("roll:gone")) == "a deleted roll"
        assert baseline_label(repo, self._process("scene:gone")) == "a deleted scene"

    def test_filled_baseline_records_its_source(self):
        repo = _repo()
        roll_id = create_virtual_roll(repo, "Portra", [])
        set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
        sid = create_scene(repo, roll_id, "Beach", ["h1"])
        set_scene_normalization(repo, roll_id, sid, (0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
        cfg = WorkspaceConfig()
        cfg = replace(cfg, process=replace(cfg.process, use_luma_average=True))

        assert resolve_roll_baseline(repo, roll_id, "h1", cfg).process.baseline_source == f"scene:{sid}"
        assert resolve_roll_baseline(repo, roll_id, "h2", cfg).process.baseline_source == f"roll:{roll_id}"


def test_add_extra_members_writes_many_paths_once():
    repo = _repo()
    roll_id = create_virtual_roll(repo, "Portra", ["/a.nef"])
    writes = []
    real = repo.save_global_setting
    repo.save_global_setting = lambda k, v: writes.append(k) or real(k, v)
    add_extra_members(repo, roll_id, ["/b.nef", "/a.nef", "/c.nef", "/b.nef"])
    assert roll_for_id(repo, roll_id)["member_paths"] == ["/a.nef", "/b.nef", "/c.nef"]
    assert len(writes) == 1


def test_same_value_reads_a_stored_list_as_the_tuple_it_was():
    """The store is JSON: a tuple comes back as a list, nested ones too."""
    from negpy.services.assets.rolls import config_value, same_value

    matrix = ((1.0, 0.1), (0.0, 1.0))
    assert same_value(matrix, [[1.0, 0.1], [0.0, 1.0]])
    assert same_value((0.1, 0.9), [0.1, 0.9])
    assert not same_value((0.1, 0.9), [0.1, 0.8])
    assert same_value("C41", "C41")
    assert config_value([[1.0, 0.1], [0.0, 1.0]]) == matrix


def test_folder_roll_lookup_matches_windows_spellings_of_one_folder(monkeypatch):
    """A folder dialog on Windows gives "C:/Scans/Roll", a walk gives "C:\\scans\\roll"."""
    import ntpath
    from types import SimpleNamespace

    monkeypatch.setattr("negpy.services.assets.rolls.os", SimpleNamespace(path=ntpath, sep="\\"))
    repo = _repo()
    roll_id = recognize_folder(repo, "C:/Scans/Roll")

    assert folder_roll_id_for_path(repo, "C:\\scans\\roll") == roll_id
    assert recognize_folder(repo, "C:\\Scans\\Roll\\") == roll_id
    assert len(saved_rolls(repo)) == 1


def test_discovery_filters_default_to_export_and_can_be_replaced():
    repo = _repo()
    assert discovery_filters(repo) == ["export"]

    set_discovery_filters(repo, ["  raw_* ", "", "tmp"])

    assert discovery_filters(repo) == ["raw_*", "tmp"]


def test_discovery_filter_matches_a_phrase_anywhere_or_a_wildcard_on_the_whole_name():
    assert matches_discovery_filter("Exports_TIFF", ["export"])
    assert matches_discovery_filter("raw_scans", ["raw_*"])
    assert not matches_discovery_filter("my_raw_scans", ["raw_*"])
    assert not matches_discovery_filter("roll01", ["export", "raw_*"])


def test_import_subfolders_as_rolls_skips_filtered_folders_and_their_subfolders(tmp_path):
    _roll_dir(tmp_path / "roll_a")
    _roll_dir(tmp_path / "Export")
    _roll_dir(tmp_path / "Export" / "nested_roll")
    repo = _repo()

    import_subfolders_as_rolls(repo, str(tmp_path))

    assert [e["folder_path"] for e in saved_rolls(repo).values()] == [str(tmp_path / "roll_a")]


def test_a_filter_never_skips_the_picked_folder(tmp_path):
    picked = _roll_dir(tmp_path / "export_best")
    repo = _repo()

    assert import_subfolders_as_rolls(repo, str(picked)) != []


def test_prune_filtered_rolls_drops_rolls_a_new_filter_catches_and_a_rescan_restores_them(tmp_path):
    _roll_dir(tmp_path / "roll_a")
    _roll_dir(tmp_path / "scratch" / "roll_b")
    repo = _repo()
    import_subfolders_as_rolls(repo, str(tmp_path))
    assert len(saved_rolls(repo)) == 2

    set_discovery_filters(repo, ["scratch"])
    assert prune_filtered_rolls(repo) == 1
    assert [e["folder_path"] for e in saved_rolls(repo).values()] == [str(tmp_path / "roll_a")]

    set_discovery_filters(repo, [])
    import_subfolders_as_rolls(repo, str(tmp_path), skip_dismissed=True)
    assert len(saved_rolls(repo)) == 2


def test_prune_filtered_rolls_leaves_rolls_outside_import_sources(tmp_path):
    repo = _repo()
    recognize_folder(repo, str(_roll_dir(tmp_path / "export_by_hand")))

    assert prune_filtered_rolls(repo) == 0


def test_delete_folder_rolls_forgets_rolls_and_the_source_under_it(tmp_path):
    _roll_dir(tmp_path / "day" / "roll_a")
    _roll_dir(tmp_path / "day" / "roll_b")
    repo = _repo()
    other = recognize_folder(repo, str(_roll_dir(tmp_path / "elsewhere")))
    import_subfolders_as_rolls(repo, str(tmp_path / "day"))

    delete_folder_rolls(repo, str(tmp_path / "day"))

    assert list(saved_rolls(repo)) == [other]
    assert import_sources(repo) == []
